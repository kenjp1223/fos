"""Semi-NMF for count or adjusted-count data.

The model is designed to work with datasets consisting of the following
quantities:

* ``raw_counts``: non-negative integer counts assumed to follow a Poisson
  distribution.
* ``bg_counts``: background estimate with a roughly Poisson distribution; its
  entries may be negative or positive and need not be integers.
* ``counts``: the adjusted data defined as ``raw_counts - bg_counts``.  This
  array must be non-negative; shifting ``counts`` or ``bg_counts`` is acceptable
  if required.

When a Poisson likelihood is chosen the background term is added to the rate
parameter.  Alternatively, the adjusted counts may be modeled directly with a
Gaussian likelihood via a softplus link.
"""

import dataclasses
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from fastprogress import progress_bar
from jax import grad, hessian, vmap, lax, jit
from jax.nn import softplus, sigmoid
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp

from fos.prox import soft_threshold
from fos.utils import register_pytree_node_dataclass

tfd = tfp.distributions


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SemiNMFParams:
    """Parameters for the semi-NMF model."""
    factors: Float[Array, "num_factors num_columns"]
    loadings: Float[Array, "num_rows num_factors"]
    row_effects: Float[Array, "num_rows"]
    column_effects: Float[Array, "num_columns"]

    @property
    def num_factors(self) -> int:
        return self.factors.shape[0]


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class QuadraticApprox:
    J_counts: Float[Array, "num_rows num_columns"]
    h_counts: Float[Array, "num_rows num_columns"]


# -----------------------------------------------------------------------------
# Core helper functions
# -----------------------------------------------------------------------------

def compute_activations(params: SemiNMFParams) -> Float[Array, "num_rows num_columns"]:
    return (params.row_effects[:, None]
            + params.column_effects
            + jnp.einsum('mk,kn->mn', params.loadings, params.factors))


def smooth_loss(params: SemiNMFParams,
                counts: Float[Array, "num_rows num_columns"],
                mask: Float[Array, "num_rows num_columns"],
                mean_func: str,
                distribution: str = "poisson",
                bg_counts: float = 0.0,
                gaussian_var: float = 1.0) -> Float[Array, ""]:
    g = dict(softplus=softplus)[mean_func]
    activations = compute_activations(params)
    predictions = g(activations)

    if distribution == "poisson":
        rate = predictions + bg_counts
        obs = counts + bg_counts
        ll = tfd.Poisson(rate=rate + 1e-8).log_prob(obs)
    elif distribution == "gaussian":
        ll = tfd.Normal(loc=predictions, scale=jnp.sqrt(gaussian_var)).log_prob(counts)
    else:
        raise ValueError(f"invalid distribution: {distribution}")

    return -jnp.where(mask, ll, 0.0).sum()


penalty = lambda params, s, e: (
    e * s * jnp.sum(jnp.abs(params.loadings))
    + 0.5 * (1 - e) * s * jnp.sum(params.loadings ** 2)
)


def compute_loss(counts: Float[Array, "num_rows num_columns"],
                 mask: Float[Array, "num_rows num_columns"],
                 params: SemiNMFParams,
                 mean_func: str,
                 sparsity_penalty: float,
                 elastic_net_frac: float,
                 distribution: str = "poisson",
                 bg_counts: float = 0.0,
                 gaussian_var: float = 1.0) -> Float[Array, ""]:
    loss = smooth_loss(params, counts, mask, mean_func,
                       distribution=distribution,
                       bg_counts=bg_counts,
                       gaussian_var=gaussian_var)
    loss += penalty(params, sparsity_penalty, elastic_net_frac)
    return loss / counts.size


# -----------------------------------------------------------------------------
# Quadratic approximation and coordinate updates
# -----------------------------------------------------------------------------

def compute_quadratic_approx(counts: Float[Array, "num_rows num_columns"],
                             mask: Float[Array, "num_rows num_columns"],
                             params: SemiNMFParams,
                             mean_func: str,
                             distribution: str = "poisson",
                             bg_counts: float = 0.0,
                             gaussian_var: float = 1.0) -> QuadraticApprox:
    if mean_func.lower() != "softplus":
        raise ValueError(f"invalid mean function: {mean_func}")

    activations = compute_activations(params)
    predictions = softplus(activations)

    if distribution == "poisson":
        rate = predictions + bg_counts
        sigm = sigmoid(activations)
        dg = sigm / rate
        d2g = (sigm * (1 - sigm) * rate - sigm ** 2) / (rate ** 2)
        J = mask * (d2g * (predictions - counts) + (dg ** 2) * rate)
        h = mask * dg * (counts - predictions)
    elif distribution == "gaussian":
        sigm = sigmoid(activations)
        d2g = sigm * (1 - sigm)
        dg = sigm
        J = mask * (d2g * (predictions - counts) / gaussian_var + (dg ** 2) / gaussian_var)
        h = mask * dg * (counts - predictions) / gaussian_var
    else:
        raise ValueError(f"invalid distribution: {distribution}")

    return QuadraticApprox(J, h)


# -----------------------------------------------------------------------------
# Parameter update functions
# -----------------------------------------------------------------------------

def update_loadings(quad: QuadraticApprox,
                    params: SemiNMFParams,
                    sparsity_penalty: float,
                    elastic_net_frac: float):
    def _update_one_loading(h_m, J_m, loading_m):
        def _update_one_coord(h_m, args):
            loading_mk, factor_k = args
            num = jnp.einsum('n,n->', factor_k, (h_m + J_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', J_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / (den + 1e-8)
            h_m += J_m * loading_mk * factor_k
            h_m -= J_m * new_loading_mk * factor_k
            return h_m, new_loading_mk
        h_m, loading_m = lax.scan(_update_one_coord, h_m, (loading_m, params.factors))
        return h_m, loading_m

    h_counts, loadings = vmap(_update_one_loading)(quad.h_counts, quad.J_counts, params.loadings)
    params = dataclasses.replace(params, loadings=loadings)
    quad = dataclasses.replace(quad, h_counts=h_counts)
    return quad, params


def update_factors(quad: QuadraticApprox, params: SemiNMFParams):
    def _update_one_column(h_n, J_n, factor_n):
        def _update_one_coord(h_n, args):
            factor_nk, loading_k = args
            num = jnp.einsum('m,m->', loading_k, (h_n + J_n * factor_nk * loading_k))
            den = jnp.einsum('m,m,m->', J_n, loading_k, loading_k)
            new_factor_nk = jnp.maximum(num, 0.0) / (den + 1e-8)
            h_n += J_n * factor_nk * loading_k
            h_n -= J_n * new_factor_nk * loading_k
            return h_n, new_factor_nk
        h_n, factor_n = lax.scan(_update_one_coord, h_n, (factor_n, params.loadings.T))
        return h_n, factor_n

    h_countsT, factorsT = vmap(_update_one_column)(quad.h_counts.T, quad.J_counts.T, params.factors.T)
    h_counts = h_countsT.T
    factors = factorsT.T
    scale = factors.sum(axis=1) + 1e-8
    factors /= scale[:, None]
    loadings = params.loadings * scale
    params = dataclasses.replace(params, factors=factors, loadings=loadings)
    quad = dataclasses.replace(quad, h_counts=h_counts)
    return quad, params


def update_row_effect(quad: QuadraticApprox, params: SemiNMFParams):
    def _update_one_row(h_m, J_m, row_effect_m):
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m)
        new_row_effect_m = num / den
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m

    h_counts, row_effects = vmap(_update_one_row)(quad.h_counts, quad.J_counts, params.row_effects)
    params = dataclasses.replace(params, row_effects=row_effects)
    quad = dataclasses.replace(quad, h_counts=h_counts)
    return quad, params


def update_column_effect(quad: QuadraticApprox, params: SemiNMFParams):
    def _update_one_column(h_n, J_n, col_effect_n):
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n)
        new_col_effect_n = num / den
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n

    h_countsT, col_effects = vmap(_update_one_column)(quad.h_counts.T, quad.J_counts.T, params.column_effects)
    h_counts = h_countsT.T
    mean = jnp.mean(col_effects)
    col_effects -= mean
    row_effects = params.row_effects + mean
    params = dataclasses.replace(params, row_effects=row_effects, column_effects=col_effects)
    quad = dataclasses.replace(quad, h_counts=h_counts)
    return quad, params


# -----------------------------------------------------------------------------
# Initialization helpers
# -----------------------------------------------------------------------------

def initialize_random(key: jr.PRNGKey, data: Array, num_factors: int, mean_func: str) -> SemiNMFParams:
    m, n = data.shape
    if mean_func.lower() == "softplus":
        data = jnp.maximum(data, 1e-1)
        targets = data + jnp.log(1 - jnp.exp(-data))
    else:
        raise ValueError(f"invalid mean function: {mean_func}")
    row_effects = targets.mean(axis=1)
    col_effects = jnp.zeros(n)
    factors = jr.exponential(key, shape=(num_factors, n))
    factors /= factors.sum(axis=1, keepdims=True)
    loadings = jnp.zeros((m, num_factors))
    return SemiNMFParams(factors, loadings, row_effects, col_effects)


def initialize_prediction(counts: Array,
                          initial_params: SemiNMFParams,
                          mean_func: str) -> SemiNMFParams:
    num_rows, num_cols = counts.shape
    if mean_func.lower() == "softplus":
        pc = jnp.maximum(counts, 1e-1)
        targets = pc + jnp.log(1 - jnp.exp(-pc))
    else:
        raise ValueError(f"invalid mean function: {mean_func}")
    targets -= initial_params.column_effects
    factors = initial_params.factors
    padded = jnp.row_stack((jnp.ones(num_cols), factors))
    loadings = jnp.linalg.solve(jnp.einsum('jn,kn->jk', padded, padded),
                                jnp.einsum('mn,kn->km', targets, padded)).T
    row_effects = loadings[:, 0]
    loadings = loadings[:, 1:]
    return dataclasses.replace(initial_params, row_effects=row_effects, loadings=loadings)


# -----------------------------------------------------------------------------
# Fitting and prediction
# -----------------------------------------------------------------------------

def fit_seminmf(counts: Array,
                initial_params: SemiNMFParams,
                mask: Array | None = None,
                mean_func: str = "softplus",
                num_iters: int = 10,
                sparsity_penalty: float = 1.0,
                elastic_net_frac: float = 0.0,
                num_coord_ascent_iters: int = 20,
                tolerance: float = 1e-1,
                distribution: str = "poisson",
                bg_counts: float = 0.0,
                gaussian_var: float = 1.0):
    mask = jnp.ones_like(counts, dtype=bool) if mask is None else mask

    @jit
    def _step(params, _):
        quad = compute_quadratic_approx(counts, mask, params, mean_func,
                                        distribution=distribution,
                                        bg_counts=bg_counts,
                                        gaussian_var=gaussian_var)
        def _row_step(carry, _):
            quad, params = carry
            quad, params = update_loadings(quad, params, sparsity_penalty, elastic_net_frac)
            quad, params = update_row_effect(quad, params)
            return (quad, params), None
        (quad, params), _ = lax.scan(_row_step, (quad, params), None, length=num_coord_ascent_iters)

        quad = compute_quadratic_approx(counts, mask, params, mean_func,
                                        distribution=distribution,
                                        bg_counts=bg_counts,
                                        gaussian_var=gaussian_var)
        def _col_step(carry, _):
            quad, params = carry
            quad, params = update_factors(quad, params)
            quad, params = update_column_effect(quad, params)
            return (quad, params), None
        (_, params), _ = lax.scan(_col_step, (quad, params), None, length=num_coord_ascent_iters)
        loss = compute_loss(counts, mask, params, mean_func,
                           sparsity_penalty, elastic_net_frac,
                           distribution=distribution,
                           bg_counts=bg_counts,
                           gaussian_var=gaussian_var)
        return params, loss

    params = initial_params
    losses = [compute_loss(counts, mask, params, mean_func,
                           sparsity_penalty, elastic_net_frac,
                           distribution=distribution,
                           bg_counts=bg_counts,
                           gaussian_var=gaussian_var)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss = _step(params, itr)
        losses.append(loss)
        pbar.comment = f"loss: {losses[-1]:.4f}"
        if jnp.abs(losses[-1] - losses[-2]) < tolerance:
            break
    return params, jnp.stack(losses)


def predict_seminmf(counts: Array,
                    params: SemiNMFParams,
                    mean_func: str = "softplus",
                    num_iters: int = 10,
                    sparsity_penalty: float = 1.0,
                    elastic_net_frac: float = 0.0,
                    num_coord_ascent_iters: int = 20,
                    tolerance: float = 1e-1,
                    distribution: str = "poisson",
                    bg_counts: float = 0.0,
                    gaussian_var: float = 1.0):
    params = initialize_prediction(counts, params, mean_func)
    mask = jnp.ones_like(counts, dtype=bool)

    @jit
    def _step(params, _):
        quad = compute_quadratic_approx(counts, mask, params, mean_func,
                                        distribution=distribution,
                                        bg_counts=bg_counts,
                                        gaussian_var=gaussian_var)
        def _row_step(carry, _):
            quad, params = carry
            quad, params = update_loadings(quad, params, sparsity_penalty, elastic_net_frac)
            quad, params = update_row_effect(quad, params)
            return (quad, params), None
        (_, params), _ = lax.scan(_row_step, (quad, params), None, length=num_coord_ascent_iters)
        loss = compute_loss(counts, mask, params, mean_func,
                           sparsity_penalty, elastic_net_frac,
                           distribution=distribution,
                           bg_counts=bg_counts,
                           gaussian_var=gaussian_var)
        return params, loss

    losses = [compute_loss(counts, mask, params, mean_func,
                           sparsity_penalty, elastic_net_frac,
                           distribution=distribution,
                           bg_counts=bg_counts,
                           gaussian_var=gaussian_var)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss = _step(params, itr)
        losses.append(loss)
        pbar.comment = f"loss: {losses[-1]:.4f}"
        if jnp.abs(losses[-1] - losses[-2]) < tolerance:
            break
    return params, jnp.stack(losses)


# Convenience wrappers --------------------------------------------------------

def fit_poisson_seminmf(counts, initial_params, **kwargs):
    return fit_seminmf(counts, initial_params, distribution="poisson", **kwargs)


def predict_poisson_seminmf(counts, params, **kwargs):
    return predict_seminmf(counts, params, distribution="poisson", **kwargs)


def fit_gaussian_seminmf(counts, initial_params, gaussian_var=1.0, **kwargs):
    return fit_seminmf(counts, initial_params, distribution="gaussian", gaussian_var=gaussian_var, **kwargs)


def predict_gaussian_seminmf(counts, params, gaussian_var=1.0, **kwargs):
    return predict_seminmf(counts, params, distribution="gaussian", gaussian_var=gaussian_var, **kwargs)

