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
    Container for model parameters in a Semi-Nonnegative Matrix Factorization model.
    This model jointly analyzes count data (e.g., number of cells) and intensity data (e.g., fluorescence levels).
    
    The model decomposes the data into:
    - Shared factors (patterns across samples)
    - Separate loadings for counts and intensities (how much each sample uses each pattern)
    - Row and column effects (baseline variations)
    - Intensity-specific variance parameters
    
    This decomposition helps identify common patterns while accounting for different data types
    and systematic variations.
    """
    factors : Float[Array, "num_factors num_columns"]
    count_loadings : Float[Array, "num_rows num_factors"]
    count_row_effects : Float[Array, "num_rows"]
    count_col_effects : Float[Array, "num_columns"]


    @property
    def num_factors(self):
        return self.factors.shape[0]


def soft_threshold(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)


def smooth_loss(params, counts, mask, mean_func):
    """
    Computes the negative log-likelihood loss for both count and intensity data.
    
    This function implements a probabilistic model where:
    1. Count data follows a Poisson distribution with mean determined by factors and count loadings
    2. Intensity data follows a Normal distribution with mean determined by factors and intensity loadings
    
    The loss combines both likelihoods to allow joint optimization of the model parameters.
    The mask parameter allows for handling missing or invalid data points.
    """
    # -log p(counts | params)
    g = dict(softplus=softplus)[mean_func]
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    loss = jnp.where(mask, -tfd.Poisson(rate=count_means + 1e-8).log_prob(counts), 0.0).sum()

    return loss


grad_smooth_loss = grad(smooth_loss, argnums=0)


def penalty(params, sparsity_penalty, elastic_net_frac):
    """
    Implements regularization penalties on the model parameters to prevent overfitting.
    
    Uses a combination of L1 (lasso) and L2 (ridge) penalties through elastic net regularization:
    - L1 penalty promotes sparsity (many zeros) in the loadings
    - L2 penalty prevents extreme parameter values
    - elastic_net_frac controls the balance between L1 and L2 penalties
    
    This helps ensure the model captures meaningful patterns rather than noise.
    """
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
    """
    Computes the total loss function for model evaluation.
    
    Components:
    1. Negative log-likelihood of the data
    2. Regularization penalties on model parameters
    3. Normalized by data size for comparability
    4. Combines both fit quality and model complexity
    
    The loss function guides optimization and helps monitor convergence while
    balancing between data fit and model simplicity through regularization.
    """
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


def backtracking_line_search(counts,
                             mask,
                             params,
                             new_params,
                             mean_func,
                             sparsity_penalty,
                             elastic_net_frac,
                             alpha=0.5,
                             beta=0.5,
                             max_iters=20):
    """
    Implements a backtracking line search algorithm to ensure stable optimization steps.
    
    Key concepts:
    1. Starts with a full step and gradually reduces step size until improvement is found
    2. Uses Armijo condition to check if step size is acceptable
    3. Balances between making progress and maintaining stability
    
    This is crucial for non-convex optimization problems where naive steps might lead
    to divergence or poor solutions. The line search ensures we make reliable progress
    while respecting the model's constraints.
    """
    # Precompute some constants
    dg = grad_smooth_loss(params, counts, mask, mean_func)
    descent_direction = tree_add(new_params, params, -1.0)
    dg_direc = tree_dot(dg, descent_direction)
    baseline = smooth_loss(params, counts, mask, mean_func)
    baseline += (1 - alpha) * penalty(params, sparsity_penalty, elastic_net_frac)

    def cond_fun(state):
        stepsize, itr = state
        new_params = tree_add(params, descent_direction, stepsize)
        new_loss = smooth_loss(new_params, counts, mask, mean_func)
        new_loss += penalty(new_params, sparsity_penalty, elastic_net_frac)
        bound = baseline + alpha * stepsize * dg_direc
        bound += alpha * penalty(new_params, sparsity_penalty, elastic_net_frac)
        return (new_loss > bound) & (itr < max_iters)

    def body_fun(state):
        stepsize, itr = state
        return beta * stepsize, itr + 1

    init_state = (1.0, 0)
    (stepsize, _) = lax.while_loop(cond_fun, body_fun, init_state)
    return tree_add(params, descent_direction, stepsize)


# def backtracking_line_search_scan(counts,
#                              intensity,
#                              mask,
#                              params,
#                              new_params,
#                              mean_func,
#                              sparsity_penalty,
#                              elastic_net_frac,
#                              alpha=0.5,
#                              beta=0.5,
#                              max_iters=20):
#     # Precompute some constants
#     dg = grad_smooth_loss(params, counts, intensity, mask, mean_func)
#     descent_direction = tree_add(new_params, params, -1.0)
#     dg_direc = tree_dot(dg, descent_direction)
#     baseline = smooth_loss(params, counts, intensity, mask, mean_func)
#     baseline += (1 - alpha) * penalty(params, sparsity_penalty, elastic_net_frac)

#     def _step(carry, stepsize):
#         prev_params, prev_criterion_met = carry

#         # Compute new params and check if loss is less than upper bound
#         new_params = tree_add(params, descent_direction, stepsize)
#         new_loss = smooth_loss(new_params, counts, intensity, mask, mean_func)
#         new_loss += penalty(new_params, sparsity_penalty, elastic_net_frac)
#         bound = baseline + alpha * stepsize * dg_direc
#         bound += alpha * penalty(new_params, sparsity_penalty, elastic_net_frac)
#         new_criterion_met = new_loss < bound

#         # If criterion not met on previous iteration, return these params
#         new_carry = lax.cond(
#             prev_criterion_met,
#             lambda: prev_params, prev_criterion_met,
#             lambda: new_params, new_criterion_met)

#         return new_carry, None

#     stepsizes = beta ** jnp.arange(max_iters)
#     (new_params, criterion_met), _ = lax.scan(_step, (params, False), stepsizes)
#     return new_params


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class QuadraticApprox:
    """
    Container for the model parameters
    """
    J_counts : Float[Array, "num_rows num_columns"]
    h_counts : Float[Array, "num_rows num_columns"]



def compute_quadratic_approx(counts, mask, params, mean_func):
    """
    Computes a quadratic approximation to the loss function for optimization.
    
    This is a key component of the coordinate descent algorithm:
    1. For the Poisson loss (counts), uses a second-order Taylor expansion
    2. For the Gaussian loss (intensity), directly uses the quadratic form
    
    The quadratic approximation makes the optimization problem easier to solve
    while still maintaining good convergence properties.
    """

    # Define key functions of the Poisson GLM
    A = jnp.exp                             # shorthand
    d2A = vmap(vmap(hessian(A)))            # want to broadcast scalar function to whole matrix

    if mean_func.lower() == "softplus":
        # Define numerically safe versions of log(softplus) and its gradients
        f = softplus
        log_softplus = lambda a: jnp.log(f(a))
        thresh = -10
        g = lambda a: jnp.where(a > thresh, log_softplus(a), a)
        dg = lambda a: jnp.where(a > thresh, vmap(vmap(grad(log_softplus)))(a), 1.0)
        d2g = lambda a: jnp.where(a > thresh, vmap(vmap(hessian(log_softplus)))(a), 0.0)
    else:
        raise Exception("invalid mean function: {}".format(mean_func))

    # Compute the quadratic approximation for the Poisson loss
    activations = params.count_row_effects[:, None] \
                + params.count_col_effects \
                + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors)
    predictions = f(activations)
    J_counts = mask * (d2g(activations) * (predictions - counts) + (dg(activations))**2 * d2A(g(activations)))
    h_counts = mask * dg(activations) * (counts - predictions)

    
    return QuadraticApprox(J_counts, h_counts)


def update_loadings(quad_approx,
                    params,
                    sparsity_penalty,
                    elastic_net_frac):
    """
    Update the loadings while holding the remaining parameters fixed.
    """
    def _update_one_loading(h_m, J_m, loading_m):
        """
        Coordinate descent to update the m-th loading
        """
        def _update_one_coord(h_m, args):
            """
            Update one coordinate of the m-th loading
            """
            loading_mk, factor_k = args

            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of loading \beta_{mk}
            num = jnp.einsum('n,n->', factor_k, (h_m + J_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', J_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty

            # Apply prox operator
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / (den + 1e-8)

            # Update the weighted residual
            h_m += J_m * loading_mk * factor_k
            h_m -= J_m * new_loading_mk * factor_k
            return h_m, new_loading_mk

        # Scan over the (K,) dimension
        h_m, loading_m = lax.scan(_update_one_coord, h_m, (loading_m, params.factors))
        return h_m, loading_m

    # Update the count loadings
    h_counts, count_loadings = vmap(_update_one_loading)(quad_approx.h_counts, quad_approx.J_counts, params.count_loadings)

    params = dataclasses.replace(params,
                             count_loadings=count_loadings)

    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,
                                      )
    return quad_approx, params


def update_factors(quad_approx, params):
    """
    Updates the shared factors while keeping loadings fixed.
    
    Conceptual approach:
    1. Treats each factor as a pattern of activity across samples
    2. Updates factors using coordinate descent on quadratic approximation
    3. Ensures non-negativity constraints are maintained
    4. Normalizes factors to prevent scale ambiguity
    
    The factors represent interpretable patterns in the data, so this update
    is crucial for finding meaningful decompositions while maintaining
    the model's probabilistic interpretation.
    """
    def _update_one_column(hc_n, Jc_n, factor_n):
        def _update_one_coord(carry, args):
            hc_n = carry  # Simplified carry handling
            factor_nk, count_loading_k = args

            # Compute the numerator and denominator
            num = jnp.sum(count_loading_k * (hc_n + Jc_n * factor_nk * count_loading_k))
            den = jnp.sum(Jc_n * count_loading_k * count_loading_k)

            # Apply non-negativity constraint
            new_factor_nk = jnp.maximum(num, 0.0) / (den + 1e-8)

            # Update residuals
            hc_n = hc_n + Jc_n * factor_nk * count_loading_k
            hc_n = hc_n - Jc_n * new_factor_nk * count_loading_k

            return hc_n, new_factor_nk

        # Scan over factors
        hc_n, factor_n = lax.scan(
            _update_one_coord,
            hc_n,  # Initial carry value
            (factor_n, params.count_loadings.T)  # Args to scan over
        )
        return hc_n, factor_n

    # Map over columns
    h_countsT, factorsT = vmap(_update_one_column)(
        quad_approx.h_counts.T,
        quad_approx.J_counts.T,
        params.factors.T
    )
    
    # Reshape results
    h_counts = h_countsT.T
    factors = factorsT.T

    # Normalize factors
    scale = factors.sum(axis=1, keepdims=True) + 1e-8
    factors = factors / scale
    count_loadings = params.count_loadings * scale.squeeze()

    # Update parameters
    params = dataclasses.replace(
        params,
        factors=factors,
        count_loadings=count_loadings
    )
    quad_approx = dataclasses.replace(
        quad_approx,
        h_counts=h_counts
    )
    return quad_approx, params


def update_row_effect(quad_approx, params):
    """
    Updates the row-specific baseline effects in the model.
    
    Purpose and approach:
    1. Captures systematic variations specific to each sample/row
    2. Removes global offsets that might obscure the underlying patterns
    3. Uses closed-form updates based on quadratic approximation
    4. Helps separate sample-specific effects from shared patterns
    
    These effects are important for handling systematic differences between samples
    that aren't related to the biological patterns of interest.
    """
    def _update_one_row(h_m, J_m, row_effect_m):
        """
        Update the m-th row effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading b_{m}
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m)
        new_row_effect_m = num / den

        # Update residual
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m

    # Update the row effects for the count data
    h_counts, count_row_effects = \
        vmap(_update_one_row)(quad_approx.h_counts, quad_approx.J_counts, params.count_row_effects)


    params = dataclasses.replace(params,
                                 count_row_effects=count_row_effects,)
    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,)
    return quad_approx, params


def update_column_effect(quad_approx, params):
    """
    Updates the column-specific baseline effects in the model.
    
    Purpose and approach:
    1. Captures systematic variations specific to each feature/column
    2. Accounts for baseline differences across measurement locations
    3. Ensures identifiability by centering the effects
    4. Separates technical/systematic variation from biological patterns
    
    These effects help account for spatial or technical biases in the measurements,
    improving the model's ability to find true biological patterns.
    """
    def _update_one_column(h_n, J_n, col_effect_n):
        """
        Update the n-th column effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading c_{n}
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n)
        new_col_effect_n = num / den

        # Update residual
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n

    # Update the column effects for the counts
    h_countsT, count_col_effects = \
        vmap(_update_one_column)(quad_approx.h_counts.T,
                                 quad_approx.J_counts.T,
                                 params.count_col_effects)
    h_counts = h_countsT.T


    # Make sure column effects sum to zero
    mean = jnp.mean(count_col_effects)
    count_col_effects -= mean
    count_row_effects = params.count_row_effects + mean


    params = dataclasses.replace(params,
                                 count_row_effects=count_row_effects,
                                 count_col_effects=count_col_effects,)
    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,)
    return quad_approx, params


'''def update_emission_noise_var(counts, 
                              mask,
                              params,
                              alpha=0.0001, 
                              beta=0.0001):
    """
    Updates the emission noise variance parameters for count data.
    
    Statistical approach:
    1. Uses conjugate prior updates for variance parameters
    2. Incorporates both prior knowledge (alpha, beta) and observed data
    3. Accounts for data uncertainty in a principled way
    4. Helps model overdispersion in count data
    
    This update is crucial for handling heterogeneous noise levels across different
    measurements, making the model more robust to varying data quality.
    """
    # Compute the quadratic loss for the intensity
    predictions = params.intensity_row_effects[:, None] \
                + params.intensity_col_effects \
                + jnp.einsum('mk, kn->mn', params.intensity_loadings, params.factors)
    residual = intensity - predictions

    alpha_post = alpha + 0.5 * jnp.sum((mask * counts) > 0, axis=0)
    beta_post = beta + 0.5 * jnp.sum(mask * counts * residual**2, axis=0)
    intensity_variance = beta_post / alpha_post
    return dataclasses.replace(params, intensity_variance=intensity_variance)'''


def initialize_random(key, data, num_factors, mean_func):
    """
    Provides random initialization for model parameters.
    
    Strategy:
    1. Generates random initial values for factors and loadings
    2. Ensures proper scaling and non-negativity constraints
    3. Initializes baseline effects using data summaries
    4. Provides a starting point for optimization
    
    While simpler than NNSVD initialization, random initialization can be useful
    for assessing robustness of results or when prior knowledge is limited.
    """
    m, n = data.shape

    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        data = jnp.maximum(data, 1e-1)
        targets = data + jnp.log(1 - jnp.exp(-data))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    # Initialize the row and column effects
    row_effects = targets.mean(axis=1)
    col_effects = jnp.zeros(n)

    # initialie the factors randomly
    factors = jr.exponential(key, shape=(num_factors, n))
    factors /= factors.sum(axis=1, keepdims=True)
    loadings = jnp.zeros((m, num_factors))
    return SemiNMFParams(loadings, factors, row_effects, col_effects)


def initialize_nnsvd(counts, num_factors, mean_func, drugs=None):
    """
    Initializes model parameters using Non-negative SVD (Singular Value Decomposition).
    
    This initialization strategy:
    1. Removes row and column effects from the data
    2. Performs SVD on the residuals
    3. Projects the factors onto the non-negative orthant
    4. Scales the factors and loadings appropriately
    
    Good initialization is crucial for the model to converge to a meaningful solution,
    as the optimization problem is non-convex.
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
        # sign = 1.0
        vk = jnp.clip(vk * sign, a_min=1e-8)
        scale = vk.sum()
        factors.append((vk / scale).reshape(shape))
        count_loadings.append(uk * sk * scale * sign )

    count_loadings = jnp.column_stack(count_loadings)
    factors = jnp.stack(factors)

    
    

    return SemiNMFParams(factors,
                         count_loadings,
                         count_row_effect,
                         count_col_effect,)


def initialize_prediction(counts, initial_params, mean_func):
    """
    Initializes parameters for prediction tasks with new data.
    
    Approach:
    1. Uses existing factors from trained model
    2. Estimates new loadings via regression
    3. Computes appropriate baseline effects
    4. Maintains model structure while adapting to new data
    
    This initialization is crucial for transfer learning scenarios where we want
    to apply learned patterns to new samples while preserving the model's
    interpretability.
    """
    num_mice, num_voxels = counts.shape

    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        pseudocounts = jnp.maximum(counts, 1e-1)
        # y = log(1 + e^{x})  ->  x = log(e^y - 1) = y + log(1 - e^{-y})
        targets = pseudocounts + jnp.log(1 - jnp.exp(-pseudocounts))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))
    
    # Initialize the row- and column-effects
    targets -= initial_params.count_col_effects

    # Solve for count loadings using a simple regression
    factors = initial_params.factors
    padded_factors = jnp.row_stack((jnp.ones(num_voxels), factors))
    count_loadings = jnp.linalg.solve(
        jnp.einsum('jn, kn->jk', padded_factors, padded_factors),
        jnp.einsum('mn, kn->km', targets, padded_factors)).T
    assert jnp.all(jnp.isfinite(count_loadings))

    count_row_effects = count_loadings[:,0]
    count_loadings = count_loadings[:,1:]

    
    return dataclasses.replace(initial_params,
                               count_row_effects=count_row_effects,
                               count_loadings=count_loadings)


def fit_poisson_seminmf(counts,
                        initial_params,
                        mask=None,
                        mean_func="softplus",
                        num_iters=10,
                        sparsity_penalty=1.0,
                        elastic_net_frac=0.0,
                        num_coord_ascent_iters=20,
                        tolerance=1e-1):
    
    # Print shapes for debugging
    print("Shapes:")
    print(f"counts: {counts.shape}")
    print(f"factors: {initial_params.factors.shape}")
    print(f"count_loadings: {initial_params.count_loadings.shape}")
    print(f"count_row_effects: {initial_params.count_row_effects.shape}")
    print(f"count_col_effects: {initial_params.count_col_effects.shape}")

    # Make mask if necessary
    mask = jnp.ones_like(counts, dtype=bool) if mask is None else mask
    assert mask.shape == counts.shape, f"Mask shape {mask.shape} doesn't match counts shape {counts.shape}"

    # Validate shapes
    n_rows, n_cols = counts.shape
    n_factors = initial_params.factors.shape[0]
    
    assert initial_params.factors.shape == (n_factors, n_cols), \
        f"Factors shape {initial_params.factors.shape} incorrect, should be ({n_factors}, {n_cols})"
    assert initial_params.count_loadings.shape == (n_rows, n_factors), \
        f"Loadings shape {initial_params.count_loadings.shape} incorrect, should be ({n_rows}, {n_factors})"
    assert initial_params.count_row_effects.shape == (n_rows,), \
        f"Row effects shape {initial_params.count_row_effects.shape} incorrect, should be ({n_rows},)"
    assert initial_params.count_col_effects.shape == (n_cols,), \
        f"Column effects shape {initial_params.count_col_effects.shape} incorrect, should be ({n_cols},)"

    # Add checks for invalid values
    assert jnp.all(jnp.isfinite(counts)), "Input counts contain NaN or infinite values"
    assert jnp.all(jnp.isfinite(initial_params.factors)), "Initial factors contain NaN or infinite values"
    assert jnp.all(jnp.isfinite(initial_params.count_loadings)), "Initial loadings contain NaN or infinite values"

    print("counts dtype:", counts.dtype)
    print("sparsity_penalty type:", type(sparsity_penalty))
    print("elastic_net_frac type:", type(elastic_net_frac))

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

# Add these imports and functions to the TOP of your seminmf_counts_2.py file

import time
import psutil
import os

def print_memory_usage(label=""):
    """Print current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage {label}: {memory_mb:.1f} MB")
    except:
        print(f"Memory usage {label}: Unable to measure")

# Add this MODIFIED version of your fit function with debugging
def fit_poisson_seminmf_debug(counts,
                        initial_params,
                        mask=None,
                        mean_func="softplus",
                        num_iters=10,
                        sparsity_penalty=1.0,
                        elastic_net_frac=0.0,
                        num_coord_ascent_iters=20,
                        tolerance=1e-1,
                        debug_freq=10):
    
    print(f"\n=== Starting fit_poisson_seminmf_debug ===")
    print_memory_usage("initial")
    
    # Print shapes for debugging
    print("Shapes:")
    print(f"counts: {counts.shape}")
    print(f"factors: {initial_params.factors.shape}")
    print(f"count_loadings: {initial_params.count_loadings.shape}")
    print(f"count_row_effects: {initial_params.count_row_effects.shape}")
    print(f"count_col_effects: {initial_params.count_col_effects.shape}")

    # Make mask if necessary
    if mask is None:
        print("Creating mask...")
        mask = jnp.ones_like(counts, dtype=bool)
    assert mask.shape == counts.shape, f"Mask shape {mask.shape} doesn't match counts shape {counts.shape}"

    # Validate shapes
    n_rows, n_cols = counts.shape
    n_factors = initial_params.factors.shape[0]
    
    print(f"Problem size: {n_rows} rows, {n_cols} cols, {n_factors} factors")
    print(f"Total elements: {n_rows * n_cols:,}")
    
    assert initial_params.factors.shape == (n_factors, n_cols), \
        f"Factors shape {initial_params.factors.shape} incorrect, should be ({n_factors}, {n_cols})"
    assert initial_params.count_loadings.shape == (n_rows, n_factors), \
        f"Loadings shape {initial_params.count_loadings.shape} incorrect, should be ({n_rows}, {n_factors})"

    # Add checks for invalid values
    assert jnp.all(jnp.isfinite(counts)), "Input counts contain NaN or infinite values"
    assert jnp.all(jnp.isfinite(initial_params.factors)), "Initial factors contain NaN or infinite values"
    assert jnp.all(jnp.isfinite(initial_params.count_loadings)), "Initial loadings contain NaN or infinite values"

    print("Data validation passed.")
    print_memory_usage("after validation")

    # Test compute_quadratic_approx first
    print("\nTesting compute_quadratic_approx...")
    start_time = time.time()
    try:
        test_quad_approx = compute_quadratic_approx(counts, mask, initial_params, mean_func)
        print(f"compute_quadratic_approx completed in {time.time() - start_time:.2f}s")
        print_memory_usage("after quadratic approx")
    except Exception as e:
        print(f"ERROR in compute_quadratic_approx: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    print("Compiling JIT functions...")
    compilation_start = time.time()
    
    @jit
    def _step_debug(params, iteration):
        """
        One sweep over parameter updates with debug info
        """
        # Update rows
        quad_approx = compute_quadratic_approx(counts, mask, params, mean_func)
        
        def _row_step(carry, inner_iter):
            quad_approx, params = carry
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            quad_approx, params = update_row_effect(quad_approx, params)
            return (quad_approx, params), None
        
        (quad_approx, new_params), _ = lax.scan(_row_step, (quad_approx, params), 
                                               jnp.arange(num_coord_ascent_iters))
        
        # Apply line search
        params = backtracking_line_search(counts, mask, params, new_params, mean_func, 
                                        sparsity_penalty, elastic_net_frac)

        # Update columns
        quad_approx = compute_quadratic_approx(counts, mask, params, mean_func)
        
        def _column_step(carry, inner_iter):
            quad_approx, params = carry
            quad_approx, params = update_factors(quad_approx, params)
            return (quad_approx, params), None
        
        (_, new_params), _ = lax.scan(_column_step, (quad_approx, params), 
                                    jnp.arange(num_coord_ascent_iters))
        
        params = backtracking_line_search(counts, mask, params, new_params, mean_func, 
                                        sparsity_penalty, elastic_net_frac)
        
        loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        hll = heldout_loglike(counts, mask, params, mean_func)
        
        return params, loss, hll

    print(f"JIT compilation took {time.time() - compilation_start:.2f}s")
    print_memory_usage("after JIT compilation")

    # Run coordinate ascent
    params = initial_params
    print("\nComputing initial loss...")
    initial_loss_start = time.time()
    initial_loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
    initial_hll = heldout_loglike(counts, mask, params, mean_func)
    print(f"Initial loss computation took {time.time() - initial_loss_start:.2f}s")
    
    losses = [initial_loss]
    hlls = [initial_hll]
    
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Initial held-out loglike: {initial_hll:.6f}")
    print_memory_usage("before main loop")

    print(f"\nStarting main optimization loop...")
    pbar = progress_bar(range(num_iters))
    
    for itr in pbar:
        iteration_start = time.time()
        
        try:
            params, loss, hll = _step_debug(params, itr)
            losses.append(loss)
            hlls.append(hll)
            
            iteration_time = time.time() - iteration_start
            
            if not jnp.isfinite(loss):
                print(f"ERROR: Non-finite loss at iteration {itr}: {loss}")
                break
                
            pbar.comment = f"loss: {losses[-1]:.4f}, time: {iteration_time:.1f}s"
            
            # Detailed logging every debug_freq iterations
            if itr % debug_freq == 0:
                print(f"\nIteration {itr}:")
                print(f"  Loss: {loss:.6f} (change: {loss - losses[-2]:.6f})")
                print(f"  Held-out loglike: {hll:.6f}")
                print(f"  Iteration time: {iteration_time:.2f}s")
                print_memory_usage(f"iter {itr}")
                
                # Check parameter magnitudes
                print(f"  Factor magnitudes: min={jnp.min(params.factors):.4f}, max={jnp.max(params.factors):.4f}")
                print(f"  Loading magnitudes: min={jnp.min(params.count_loadings):.4f}, max={jnp.max(params.count_loadings):.4f}")

            # Early stopping
            if abs(losses[-1] - losses[-2]) < tolerance:
                print(f"\nConverged at iteration {itr} (change < {tolerance})")
                break
                
        except Exception as e:
            print(f"ERROR at iteration {itr}: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"\nOptimization completed. Final loss: {losses[-1]:.6f}")
    print_memory_usage("final")
    
    return params, jnp.stack(losses), jnp.stack(hlls)