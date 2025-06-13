import dataclasses
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from fastprogress import progress_bar
from jax import grad, hessian, vmap, lax, jit
from jax.nn import softplus
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
warnings.filterwarnings("ignore")

from fos.prox import soft_threshold
from fos.utils import register_pytree_node_dataclass, tree_add, tree_dot


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SemiNMFParams:
    """
    Container for the model parameters - count data only version
    """
    factors : Float[Array, "num_factors num_columns"]
    count_loadings : Float[Array, "num_rows num_factors"]
    count_row_effects : Float[Array, "num_rows"]
    count_col_effects : Float[Array, "num_columns"]

    @property
    def num_factors(self):
        return self.factors.shape[0]


def smooth_loss(params, counts, mask, mean_func):
    # -log p(counts | params)
    g = dict(softplus=softplus)[mean_func]
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    loss = jnp.where(mask, -tfd.Poisson(rate=count_means + 1e-8).log_prob(counts), 0.0).sum()
    return loss


grad_smooth_loss = grad(smooth_loss, argnums=0)


def penalty(params, sparsity_penalty, elastic_net_frac):
    loss = elastic_net_frac * sparsity_penalty * jnp.sum(abs(params.count_loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.count_loadings ** 2)
    return loss


@partial(jit, static_argnums=(3,))
def compute_loss(counts : Float[Array, "num_rows num_columns"],
                 mask : Float[Array, "num_rows num_columns"],
                 params : SemiNMFParams,
                 mean_func : str,
                 sparsity_penalty : float,
                 elastic_net_frac: float
                 ):
    loss = smooth_loss(params, counts, mask, mean_func)
    loss += penalty(params, sparsity_penalty, elastic_net_frac)
    return loss / counts.size


def heldout_loglike(counts : Float[Array, "num_rows num_columns"],
                    mask : Float[Array, "num_rows num_columns"],
                    params : SemiNMFParams,
                    mean_func : str):
    # -log p(counts | params)
    g = dict(softplus=softplus)[mean_func]
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    loss = jnp.where(~mask, tfd.Poisson(rate=count_means + 1e-8).log_prob(counts), 0.0).sum()
    return loss / counts.size


def initialize_nnsvd(counts, num_factors, mean_func, drugs=None):
    """Initialize the model with an SVD. Project the right singular vectors
    onto the non-negative orthant.
    """
    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        pseudocounts = jnp.maximum(counts, 1e-1)
        # y = log(1 + e^{x})  ->  x = log(e^y - 1) = y + log(1 - e^{-y})
        targets = pseudocounts + jnp.log(1 - jnp.exp(-pseudocounts))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    num_mice = counts.shape[0]
    shape = counts.shape[1:]

    # Initialize the row- and column-effects
    count_row_effect = jnp.mean(targets, axis=1)
    targets -= count_row_effect[:, None]

    if drugs is not None:
        # !!!!HACK!!!!! Leaking information about drugs into column effect
        count_col_effect = targets[drugs == 10].mean(axis=0)
    else:
        count_col_effect = targets.mean(axis=0)
    targets -= count_col_effect

    # Now run SVD on the residual
    U, S, VT = jnp.linalg.svd(targets.reshape(num_mice, -1), full_matrices=False)

    # flip signs on factors so that each has non-negative mean
    count_loadings = []
    factors = []
    for uk, sk, vk in zip(U.T[:num_factors], S[:num_factors], VT[:num_factors]):
        sign = jnp.sign(vk.mean())
        vk = jnp.clip(vk * sign, a_min=1e-8)
        scale = vk.sum()
        factors.append((vk / scale).reshape(shape))
        count_loadings.append(uk * sk * scale * sign)

    count_loadings = jnp.column_stack(count_loadings)
    factors = jnp.stack(factors)

    return SemiNMFParams(factors=factors,
                        count_loadings=count_loadings,
                        count_row_effects=count_row_effect,
                        count_col_effects=count_col_effect)


def fit_poisson_seminmf(counts,
                        initial_params,
                        mask=None,
                        mean_func="softplus",
                        num_iters=10,
                        sparsity_penalty=1.0,
                        elastic_net_frac=0.0,
                        num_coord_ascent_iters=20,
                        tolerance=1e-1,
                        ):

    # Make mask if necessary
    mask = jnp.ones_like(counts, dtype=bool) if mask is None else mask
    assert mask.shape == counts.shape

    @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        # Update rows
        quad_approx = compute_quadratic_approx(counts, mask, params, mean_func)
        def _row_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            quad_approx, params = update_row_effect(quad_approx, params)
            return (quad_approx, params), None
        (quad_approx, new_params), _ = lax.scan(_row_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        # Update columns
        quad_approx = compute_quadratic_approx(counts, mask, params, mean_func)
        def _column_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_factors(quad_approx, params)
            return (quad_approx, params), None
        (_, new_params), _ = lax.scan(_column_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)
        
        loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        hll = heldout_loglike(counts, mask, params, mean_func)
        return params, loss, hll

    # Run coordinate ascent
    params = initial_params
    losses = [compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)]
    hlls = [heldout_loglike(counts, mask, params, mean_func)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss, hll = _step(params, itr)
        losses.append(loss)
        hlls.append(hll)
        assert jnp.isfinite(loss)
        pbar.comment = "loss: {:.4f}".format(losses[-1])

        if abs(losses[-1] - losses[-2]) < tolerance:
            break

    return params, jnp.stack(losses), jnp.stack(hlls)


def predict_poisson_seminmf(counts,
                            params,
                            mean_func="softplus",
                            num_iters=10,
                            sparsity_penalty=1.0,
                            elastic_net_frac=0.0,
                            num_coord_ascent_iters=20,
                            tolerance=1e-1,
                            ):
    """
    Predict the loadings (row factors) given data and (column) factors.
    """
    # Initialize row parameters
    params = initialize_prediction(counts, params, mean_func)
    mask = jnp.ones_like(counts, dtype=bool) 

    @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        # Update rows
        quad_approx = compute_quadratic_approx(counts, mask, params, mean_func)
        def _row_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            quad_approx, params = update_row_effect(quad_approx, params)
            return (quad_approx, params), None
        (quad_approx, new_params), _ = lax.scan(_row_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        hll = heldout_loglike(counts, mask, params, mean_func)
        return params, loss, hll

    # Run coordinate ascent
    losses = [compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)]
    hlls = [heldout_loglike(counts, mask, params, mean_func)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss, hll = _step(params, itr)
        losses.append(loss)
        hlls.append(hll)
        assert jnp.isfinite(loss)
        pbar.comment = "loss: {:.4f}".format(losses[-1])

        if abs(losses[-1] - losses[-2]) < tolerance:
            break

    return params, jnp.stack(losses), jnp.stack(hlls) 

def compute_quadratic_approx(counts, mask, params, mean_func):
    """Compute quadratic approximation to the loss function."""
    g = dict(softplus=softplus)[mean_func]
    g_inv = dict(softplus=lambda x: jnp.log(jnp.exp(x) - 1))[mean_func]
    
    # Compute means and residuals
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    residuals = counts - count_means
    
    # Compute quadratic approximation
    quad_approx = {
        'count_loadings': {
            'linear': residuals,  # Changed from einsum to just use residuals directly
            'quadratic': jnp.einsum('kn,kn->k', params.factors, params.factors)
        },
        'count_row_effects': {
            'linear': jnp.sum(residuals, axis=1),
            'quadratic': counts.shape[1]
        },
        'count_col_effects': {
            'linear': jnp.sum(residuals, axis=0),
            'quadratic': counts.shape[0]
        }
    }
    
    return quad_approx

def update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac):
    """Update loadings using quadratic approximation."""
    # Debug prints for input shapes
    print("\nDebug shapes in update_loadings:")
    print(f"params.count_loadings shape: {params.count_loadings.shape}")
    print(f"quad_approx['count_loadings']['linear'] shape: {quad_approx['count_loadings']['linear'].shape}")
    print(f"quad_approx['count_loadings']['quadratic'] shape: {quad_approx['count_loadings']['quadratic'].shape}")
    
    linear_term = quad_approx['count_loadings']['linear']
    quadratic_term = quad_approx['count_loadings']['quadratic']
    
    # Compute step size with proper shape for broadcasting
    step_size = 1.0 / (quadratic_term + sparsity_penalty)
    # Reshape step_size to match the number of factors
    step_size = step_size.reshape(-1, 1)
    print(f"step_size shape after reshape: {step_size.shape}")
    
    # Update loadings with soft thresholding and proper broadcasting
    new_loadings = soft_threshold(
        params.count_loadings - jnp.einsum('mk,mn->mk', step_size, linear_term),
        step_size * sparsity_penalty * elastic_net_frac
    )
    print(f"new_loadings shape: {new_loadings.shape}")
    
    # Update parameters
    new_params = dataclasses.replace(params, count_loadings=new_loadings)
    
    return quad_approx, new_params

def update_row_effect(quad_approx, params):
    """Update row effects using quadratic approximation."""
    linear_term = quad_approx['count_row_effects']['linear']
    quadratic_term = quad_approx['count_row_effects']['quadratic']
    
    # Compute step size
    step_size = 1.0 / quadratic_term
    
    # Update row effects
    new_row_effects = params.count_row_effects - step_size * linear_term
    
    # Update parameters
    new_params = dataclasses.replace(params, count_row_effects=new_row_effects)
    
    return quad_approx, new_params

def update_factors(quad_approx, params):
    """Update factors using quadratic approximation."""
    # Debug prints for input shapes
    print("\nDebug shapes in update_factors:")
    print(f"params.factors shape: {params.factors.shape}")
    print(f"params.count_loadings shape: {params.count_loadings.shape}")
    print(f"quad_approx['count_loadings']['linear'] shape: {quad_approx['count_loadings']['linear'].shape}")
    
    # Compute linear term - corrected einsum operation to maintain voxel dimension
    linear_term = jnp.einsum('mk,mn->kn', params.count_loadings, quad_approx['count_loadings']['linear'])
    print(f"linear_term shape after einsum: {linear_term.shape}")
    
    # Compute quadratic term with proper broadcasting
    quadratic_term = jnp.sum(params.count_loadings ** 2, axis=0, keepdims=True)
    print(f"quadratic_term shape: {quadratic_term.shape}")
    
    # Compute step size with proper shape for broadcasting
    step_size = 1.0 / (quadratic_term + 1e-8)
    print(f"step_size shape: {step_size.shape}")
    
    # Update factors with proper broadcasting
    new_factors = params.factors - step_size * linear_term
    print(f"new_factors shape: {new_factors.shape}")
    
    # Project onto non-negative orthant
    new_factors = jnp.maximum(new_factors, 0.0)
    
    # Normalize factors to sum to 1
    scale = new_factors.sum(axis=1, keepdims=True) + 1e-8
    new_factors = new_factors / scale
    
    # Update parameters
    new_params = dataclasses.replace(params, factors=new_factors)
    
    return quad_approx, new_params

def backtracking_line_search(counts, mask, old_params, new_params, mean_func, sparsity_penalty, elastic_net_frac):
    """Perform backtracking line search to ensure sufficient decrease."""
    old_loss = compute_loss(counts, mask, old_params, mean_func, sparsity_penalty, elastic_net_frac)
    c = 0.5  # Armijo condition constant
    
    def cond_fn(state):
        step_size, params, new_loss = state
        return (step_size >= 1e-6) & (new_loss > old_loss + c * step_size * (new_loss - old_loss))
    
    def body_fn(state):
        step_size, params, _ = state
        step_size = step_size * 0.5
        
        # Interpolate between old and new parameters
        params = dataclasses.replace(
            old_params,
            count_loadings=old_params.count_loadings + step_size * (new_params.count_loadings - old_params.count_loadings),
            count_row_effects=old_params.count_row_effects + step_size * (new_params.count_row_effects - old_params.count_row_effects),
            count_col_effects=old_params.count_col_effects + step_size * (new_params.count_col_effects - old_params.count_col_effects),
            factors=old_params.factors + step_size * (new_params.factors - old_params.factors)
        )
        
        new_loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        return step_size, params, new_loss
    
    # Initial state
    init_state = (1.0, old_params, compute_loss(counts, mask, new_params, mean_func, sparsity_penalty, elastic_net_frac))
    
    # Run the loop
    final_step_size, final_params, final_loss = lax.while_loop(cond_fn, body_fn, init_state)
    
    # If we didn't find a good step size, return the old parameters
    return lax.cond(final_step_size < 1e-6,
                   lambda _: old_params,
                   lambda _: final_params,
                   None)

def initialize_prediction(counts, params, mean_func):
    """Initialize parameters for prediction."""
    g = dict(softplus=softplus)[mean_func]
    g_inv = dict(softplus=lambda x: jnp.log(jnp.exp(x) - 1))[mean_func]
    
    # Initialize row effects
    count_row_effects = jnp.mean(g_inv(jnp.maximum(counts, 1e-1)), axis=1)
    
    # Initialize loadings
    count_loadings = jnp.zeros((counts.shape[0], params.num_factors))
    
    return dataclasses.replace(
        params,
        count_loadings=count_loadings,
        count_row_effects=count_row_effects
    ) 