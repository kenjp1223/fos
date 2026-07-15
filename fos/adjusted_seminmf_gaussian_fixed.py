# ===== COMPLETE GAUSSIAN + SOFTPLUS SEMINMF IMPLEMENTATION =====

import dataclasses
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from fastprogress import progress_bar
from jax import grad, hessian, vmap, lax, jit
from jax.nn import sigmoid, softplus
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
warnings.filterwarnings("ignore")

from fos.prox import soft_threshold
from fos.utils import register_pytree_node_dataclass, tree_add, tree_dot
from jax.tree_util import register_pytree_node_class


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SemiNMFParams:
    factors : Float[Array, "num_factors num_columns"]
    count_loadings : Float[Array, "num_rows num_factors"]
    count_row_effects : Float[Array, "num_rows"]
    count_col_effects : Float[Array, "num_columns"]

    @property
    def num_factors(self):
        return self.factors.shape[0]


def smooth_loss(params, counts, mask, mean_func, data_variance=1.0):
    """Compute negative log-likelihood using softplus link for Gaussian model"""
    if mean_func.lower() == "softplus":
        g = softplus
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))
    
    # Compute means through softplus link function
    linear_pred = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    means = g(linear_pred)
    
    # Gaussian likelihood with estimated variance
    var = data_variance
    
    # Negative log-likelihood (to be minimized)
    loss = jnp.where(mask, -tfd.Normal(loc=means, scale=jnp.sqrt(var)).log_prob(counts), 0.0).sum()
    return loss


grad_smooth_loss = grad(smooth_loss, argnums=0)


def penalty(params, sparsity_penalty, elastic_net_frac):
    """L1/L2 penalty on loadings"""
    loss = elastic_net_frac * sparsity_penalty * jnp.sum(jnp.abs(params.count_loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.count_loadings ** 2)
    return loss


@partial(jit, static_argnums=(3,))
def compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac, data_variance=1.0):
    """Total loss: negative log-likelihood + penalty"""
    loss = smooth_loss(params, counts, mask, mean_func, data_variance)
    loss += penalty(params, sparsity_penalty, elastic_net_frac)
    return loss / counts.size


def heldout_loglike(counts, mask, params, mean_func, data_variance=1.0):
    """Compute held-out log-likelihood on unmasked entries"""
    if mean_func.lower() == "softplus":
        g = softplus
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))
    
    linear_pred = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    means = g(linear_pred)
    var = data_variance
    
    # Evaluate on held-out (unmasked) entries
    loglike = jnp.where(~mask, tfd.Normal(loc=means, scale=jnp.sqrt(var)).log_prob(counts), 0.0).sum()
    return loglike / jnp.sum(~mask) if jnp.sum(~mask) > 0 else 0.0


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class QuadraticApprox:
    J_counts: Float[Array, "num_rows num_columns"]
    h_counts: Float[Array, "num_rows num_columns"]

    def tree_flatten(self):
        return ((self.J_counts, self.h_counts), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        J_counts, h_counts = children
        return cls(J_counts=J_counts, h_counts=h_counts)


@partial(jit, static_argnums=(3,))
def compute_quadratic_approx(counts, mask, params, mean_func, data_variance=1.0):
    """Compute quadratic approximation for Gaussian likelihood with softplus link"""
    
    if mean_func.lower() == "softplus":
        f = softplus
        # For numerical stability in gradients
        thresh = -10
        # First derivative of softplus
        df = lambda a: jnp.where(a > thresh, sigmoid(a), jnp.exp(a))
        # Second derivative of softplus  
        d2f = lambda a: jnp.where(a > thresh, sigmoid(a) * (1 - sigmoid(a)), jnp.exp(a))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))
    
    # Compute linear activations
    activations = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    
    # Apply link function to get predictions
    predictions = f(activations)
    
    var = data_variance
    residuals = counts - predictions
    
    df_vals = df(activations)
    d2f_vals = d2f(activations)
    
    # Compute the quadratic approximation terms
    # For softplus, we need to be careful about the curvature
    # When activations are large, d2f_vals approaches 0, so we need to ensure
    # the quadratic term doesn't become too small
    J_counts = mask * ((df_vals**2) / var + jnp.maximum(d2f_vals * (-residuals) / var, 1e-6))
    h_counts = mask * df_vals * residuals / var
    
    return QuadraticApprox(J_counts, h_counts)


def initialize_nnsvd(counts, num_factors, mean_func=None, drugs=None):
    """Initialize using NNSVD with robust handling for shifted Gaussian data"""
    num_mice, num_voxels = counts.shape
    
    # Check for problematic data
    if jnp.any(jnp.isnan(counts)) or jnp.any(jnp.isinf(counts)):
        raise ValueError("Input counts contain NaN or infinite values!")
    
    if mean_func and mean_func.lower() == "softplus":
        # For shifted Gaussian data with softplus link
        # Work with log-transformed data for initialization
        # Add small constant to avoid log(0)
        safe_counts = jnp.maximum(counts, 1e-3)
        targets = jnp.log(safe_counts)
    else:
        targets = counts
    
    # Check targets for issues
    if jnp.any(jnp.isnan(targets)) or jnp.any(jnp.isinf(targets)):
        print("⚠️  Transformation produced problematic values, using safer approach")
        targets = jnp.log(jnp.maximum(counts, 1e-3))
    
    # Robust mean removal
    row_effects = jnp.nanmean(targets, axis=1)
    row_effects = jnp.where(jnp.isnan(row_effects), 0.0, row_effects)
    
    col_centered = targets - row_effects[:, None]
    col_effects = jnp.nanmean(col_centered, axis=0)
    col_effects = jnp.where(jnp.isnan(col_effects), 0.0, col_effects)
    
    residuals = col_centered - col_effects
    
    # Robust SVD with fallback
    try:
        U, S, VT = jnp.linalg.svd(residuals, full_matrices=False)
        
        # Check SVD results
        if jnp.any(jnp.isnan(U)) or jnp.any(jnp.isnan(S)) or jnp.any(jnp.isnan(VT)):
            raise ValueError("SVD produced NaN values")
        
        # Extract factors with robust handling
        count_loadings = []
        factors = []
        
        for k in range(min(num_factors, len(S))):
            if k >= len(S) or S[k] < 1e-10:
                # Use random factor for small singular values
                key = jr.PRNGKey(42 + k)
                factor = jr.uniform(key, (num_voxels,), minval=0.01, maxval=0.1)
                loading = jnp.zeros(num_mice)
            else:
                uk, sk, vk = U[:, k], S[k], VT[k]
                
                # Ensure non-negativity for factors
                vk_pos = jnp.maximum(vk, 0)
                vk_neg = jnp.maximum(-vk, 0)
                
                # Choose the one with larger norm
                if jnp.linalg.norm(vk_pos) > jnp.linalg.norm(vk_neg):
                    factor = vk_pos
                    sign = 1.0
                else:
                    factor = vk_neg
                    sign = -1.0
                
                # Normalize
                scale = jnp.sum(factor) + 1e-10
                factor = factor / scale
                loading = uk * sk * scale * sign
            
            factors.append(factor)
            count_loadings.append(loading)
        
        # Fill remaining factors with random values if needed
        key = jr.PRNGKey(999)
        while len(factors) < num_factors:
            k = len(factors)
            subkey = jr.fold_in(key, k)
            factor = jr.uniform(subkey, (num_voxels,), minval=0.01, maxval=0.1)
            factor = factor / jnp.sum(factor)
            factors.append(factor)
            count_loadings.append(jnp.zeros(num_mice))
        
        count_loadings = jnp.column_stack(count_loadings)
        factors = jnp.stack(factors)
        
    except Exception as e:
        print(f"❌ SVD failed ({e}), using random initialization")
        # Fallback to random initialization
        key = jr.PRNGKey(123)
        factors = jr.uniform(key, (num_factors, num_voxels), minval=0.01, maxval=0.1)
        factors = factors / jnp.sum(factors, axis=1, keepdims=True)
        count_loadings = jnp.zeros((num_mice, num_factors))
        row_effects = jnp.zeros(num_mice)
        col_effects = jnp.zeros(num_voxels)
    
    # Final safety check
    if jnp.any(jnp.isnan(factors)) or jnp.any(jnp.isnan(count_loadings)):
        print("❌ NaN in final initialization, using safe fallback")
        key = jr.PRNGKey(456)
        factors = jr.uniform(key, (num_factors, num_voxels), minval=0.01, maxval=0.1)
        factors = factors / jnp.sum(factors, axis=1, keepdims=True)
        count_loadings = jnp.zeros((num_mice, num_factors))
        row_effects = jnp.zeros(num_mice)
        col_effects = jnp.zeros(num_voxels)
    
    return SemiNMFParams(
        factors=factors,
        count_loadings=count_loadings,
        count_row_effects=row_effects,
        count_col_effects=col_effects
    )


def update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac):
    """Update loadings using coordinate descent with soft thresholding"""
    def _update_one_loading(h_m, J_m, loading_m):
        def _update_one_coord(h_m, args):
            loading_mk, factor_k = args
            num = jnp.einsum('n,n->', factor_k, (h_m + J_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', J_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / (den + 1e-8)
            
            # Update the residual term only once with the change in loading
            loading_change = new_loading_mk - loading_mk
            h_m = h_m - J_m * loading_change * factor_k
            
            return h_m, new_loading_mk
        h_m, loading_m = lax.scan(_update_one_coord, h_m, (loading_m, params.factors))
        return h_m, loading_m
    
    h_counts, count_loadings = vmap(_update_one_loading)(quad_approx.h_counts, quad_approx.J_counts, params.count_loadings)
    
    # Clip loadings to prevent extreme values
    count_loadings = jnp.clip(count_loadings, -1e6, 1e6)
    
    params = dataclasses.replace(params, count_loadings=count_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_factors(quad_approx, params):
    """Update factors with non-negativity constraint"""
    def _update_one_column(hc_n, Jc_n, factor_n):
        def _update_one_coord(hc_n, args):
            factor_nk, count_loading_k = args
            # Compute the gradient term separately
            grad_term = jnp.einsum('m,m->', count_loading_k, hc_n)
            # Compute the quadratic term separately
            quad_term = jnp.einsum('m,m,m->', Jc_n, count_loading_k, count_loading_k)
            # Add a small regularization to prevent denominator from being too small
            quad_term = jnp.maximum(quad_term, 1e-6)
            
            # Compute the update with a step size
            step_size = 0.1  # Small step size to prevent overshooting
            new_factor_nk = factor_nk + step_size * grad_term / quad_term
            # Ensure non-negativity
            new_factor_nk = jnp.maximum(new_factor_nk, 0.0)
            
            # Update the residual term
            factor_change = new_factor_nk - factor_nk
            hc_n = hc_n - Jc_n * factor_change * count_loading_k
            
            return hc_n, new_factor_nk
        hc_n, factor_n = lax.scan(_update_one_coord, hc_n, (factor_n, params.count_loadings.T))
        return hc_n, factor_n
    
    h_countsT, factorsT = vmap(_update_one_column)(quad_approx.h_counts.T, quad_approx.J_counts.T, params.factors.T)
    h_counts = h_countsT.T
    factors = factorsT.T
    
    # Normalize factors and rescale loadings with minimum scale
    raw_scale = factors.sum(axis=1) + 1e-8
    min_scale = 0.1  # Minimum scale factor to prevent factors from becoming too small
    scale = jnp.maximum(raw_scale, min_scale)
    factors /= scale[:, None]
    count_loadings = params.count_loadings * scale
    
    # Clip to prevent extreme values
    factors = jnp.clip(factors, 1e-12, 1e6)
    count_loadings = jnp.clip(count_loadings, -1e6, 1e6)
    
    params = dataclasses.replace(params, factors=factors, count_loadings=count_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_row_effect(quad_approx, params):
    """Update row effects"""
    def _update_one_row(h_m, J_m, row_effect_m):
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m) + 1e-8
        new_row_effect_m = num / den
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m
    
    h_counts, count_row_effects = vmap(_update_one_row)(quad_approx.h_counts, quad_approx.J_counts, params.count_row_effects)
    
    # Clip to prevent extreme values
    count_row_effects = jnp.clip(count_row_effects, -1e6, 1e6)
    
    params = dataclasses.replace(params, count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_column_effect(quad_approx, params):
    """Update column effects with identifiability constraint"""
    def _update_one_column(h_n, J_n, col_effect_n):
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n) + 1e-8
        new_col_effect_n = num / den
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n
    
    h_countsT, count_col_effects = vmap(_update_one_column)(quad_approx.h_counts.T, quad_approx.J_counts.T, params.count_col_effects)
    h_counts = h_countsT.T
    
    # Maintain identifiability: column effects sum to zero
    mean_col_effect = jnp.mean(count_col_effects)
    count_col_effects -= mean_col_effect
    count_row_effects = params.count_row_effects + mean_col_effect
    
    # Clip to prevent extreme values
    count_col_effects = jnp.clip(count_col_effects, -1e6, 1e6)
    count_row_effects = jnp.clip(count_row_effects, -1e6, 1e6)
    
    params = dataclasses.replace(params, 
                                count_col_effects=count_col_effects,
                                count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def backtracking_line_search(counts, mask, params, new_params, mean_func, 
                           sparsity_penalty, elastic_net_frac, data_variance=1.0, 
                           alpha=0.5, beta=0.5, max_iters=30):
    """Backtracking line search with improved numerical stability"""
    
    # Compute descent direction
    descent_direction = tree_add(new_params, params, -1.0)
    
    # Check if this is actually a descent direction
    dg = grad_smooth_loss(params, counts, mask, mean_func, data_variance)
    dg_direc = tree_dot(dg, descent_direction)
    
    # Use JAX's conditional instead of Python's if statement
    def descent_case():
        baseline = smooth_loss(params, counts, mask, mean_func, data_variance)
        baseline += penalty(params, sparsity_penalty, elastic_net_frac)
        
        def cond_fun(state):
            stepsize, itr = state
            # Use JAX comparison for all conditions
            too_small = stepsize < 1e-10
            
            new_params_try = tree_add(params, descent_direction, stepsize)
            new_loss = smooth_loss(new_params_try, counts, mask, mean_func, data_variance)
            new_loss += penalty(new_params_try, sparsity_penalty, elastic_net_frac)
            
            # Armijo condition
            bound = baseline + alpha * stepsize * dg_direc
            needs_reduction = new_loss > bound
            
            # Continue loop only if not too small, needs reduction, and not exceeded max iters
            return (~too_small) & needs_reduction & (itr < max_iters)
        
        def body_fun(state):
            stepsize, itr = state
            return beta * stepsize, itr + 1
        
        init_state = (1.0, 0)
        stepsize, _ = lax.while_loop(cond_fun, body_fun, init_state)
        
        # Ensure minimum step size
        stepsize = jnp.maximum(stepsize, 1e-8)
        
        return tree_add(params, descent_direction, stepsize)
    
    def no_descent_case():
        return params
    
    # Use lax.cond for the main branching logic
    return lax.cond(dg_direc < 0, descent_case, no_descent_case)


def estimate_data_variance_proper(counts, mean_func="softplus", max_factors=20):
    """
    Estimate variance in the correct space for Gaussian + softplus model
    """
    print(f"Estimating variance for {mean_func} link function...")
    
    # Input validation
    if jnp.any(jnp.isnan(counts)) or jnp.any(jnp.isinf(counts)):
        raise ValueError("Input counts contain NaN or infinite values!")
    
    # For Gaussian likelihood, we need variance in the ORIGINAL data space
    print("Estimating variance in original data space")
    
    # Method 1: Remove row/column effects first
    row_effects = jnp.mean(counts, axis=1, keepdims=True)
    col_centered = counts - row_effects
    col_effects = jnp.mean(col_centered, axis=0, keepdims=True)
    residuals = col_centered - col_effects
    
    # Method 2: Use MAD (Median Absolute Deviation) for robustness
    mad = jnp.median(jnp.abs(residuals - jnp.median(residuals)))
    robust_std = 1.4826 * mad  # MAD to std conversion
    robust_var = robust_std ** 2
    
    # Method 3: Remove low-rank structure using SVD
    try:
        U, S, VT = jnp.linalg.svd(residuals, full_matrices=False)
        k = min(max_factors, min(residuals.shape[0], residuals.shape[1]) - 1)
        
        if k > 0:
            # Keep only the noise (remove signal)
            signal_reconstruction = (U[:, :k] * S[:k]) @ VT[:k]
            noise_residuals = residuals - signal_reconstruction
            noise_var = float(jnp.var(noise_residuals))
            
            # Use the more conservative estimate
            final_var = max(noise_var, robust_var)
            
            print(f"Noise variance (SVD): {noise_var:.6f}")
            print(f"Robust variance (MAD): {robust_var:.6f}")
            print(f"Final variance: {final_var:.6f}")
            
            return final_var
            
    except Exception as e:
        print(f"SVD failed ({e}), using robust estimate")
        return max(robust_var, 0.1)


def check_data_requirements(counts, mask=None):
    """Check if the data meets requirements for factorization.
    
    This function performs several checks on the input data to ensure it's suitable
    for factorization. It checks for:
    - NaN and Inf values
    - Negative values
    - Value ranges and statistics
    - Data sparsity
    - Constant rows/columns
    - Mask statistics (if provided)
    
    Args:
        counts: Array of count data
        mask: Optional mask for held-out data
        
    Returns:
        bool: True if data meets requirements, False otherwise
        
    Example:
        >>> import jax.numpy as jnp
        >>> from fos.adjusted_seminmf_gaussian_fixed import check_data_requirements
        >>> data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> check_data_requirements(data)
    """
    print("\nChecking data requirements:")
    
    # Check for NaN and Inf values
    has_nan = jnp.isnan(counts).any()
    has_inf = jnp.isinf(counts).any()
    print(f"Contains NaN values: {has_nan}")
    print(f"Contains Inf values: {has_inf}")
    
    # Check value ranges
    min_val = float(jnp.min(counts))
    max_val = float(jnp.max(counts))
    mean_val = float(jnp.mean(counts))
    std_val = float(jnp.std(counts))
    print(f"Value range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"Mean: {mean_val:.2f}")
    print(f"Std: {std_val:.2f}")
    
    # Check for negative values
    has_neg = (counts < 0).any()
    print(f"Contains negative values: {has_neg}")
    
    # Check sparsity
    sparsity = float(jnp.mean(counts == 0))
    print(f"Data sparsity: {sparsity:.2%}")
    
    # Check mask if provided
    if mask is not None:
        mask_sparsity = float(jnp.mean(~mask))
        print(f"Mask sparsity: {mask_sparsity:.2%}")
    
    # Check for constant rows/columns
    row_std = jnp.std(counts, axis=1)
    col_std = jnp.std(counts, axis=0)
    has_const_rows = (row_std == 0).any()
    has_const_cols = (col_std == 0).any()
    print(f"Contains constant rows: {has_const_rows}")
    print(f"Contains constant columns: {has_const_cols}")
    
    # Return True if all checks pass
    return not (has_nan or has_inf or has_neg)


def fit_gaussian_seminmf(counts, initial_params, mask=None, mean_func="softplus", num_iters=100,
                        sparsity_penalty=0.0, elastic_net_frac=0.5, num_coord_ascent_iters=10,
                        tolerance=1e-5, data_variance=None):
    """Fit the semi-NMF model with Gaussian noise.
    
    Args:
        counts: Array of count data
        initial_params: Initial parameters for the model
        mask: Optional mask for the data
        mean_func: Function to compute mean (default: "softplus")
        num_iters: Number of iterations to run
        sparsity_penalty: Penalty for sparsity
        elastic_net_frac: Fraction of elastic net penalty
        num_coord_ascent_iters: Number of coordinate ascent iterations
        tolerance: Convergence tolerance
        data_variance: Optional data variance
        
    Returns:
        tuple: (params, losses, heldout_loglikes)
    """
    # Initialize parameters
    params = initial_params
    if data_variance is None:
        data_variance = jnp.ones(counts.shape[1:])
    
    # Initialize lists to track progress
    losses = []
    heldout_loglikes = []
    
    # Compute initial loss
    initial_loss = compute_loss(counts, mask, params, mean_func, sparsity_penalty, elastic_net_frac, data_variance)
    losses.append(initial_loss)
    
    # Main optimization loop
    for itr in range(num_iters):
        # Take a step
        params, loss, hll = _step(params, counts, mask, mean_func, sparsity_penalty, 
                                 elastic_net_frac, num_coord_ascent_iters, data_variance)
        losses.append(loss)
        heldout_loglikes.append(hll)
        
        # Check for convergence
        relative_change = abs(loss - losses[-2]) / (abs(losses[-2]) + 1e-10)
        
        # Print detailed debugging information
        print(f"\nIteration {itr+1}:")
        print(f"Loss: {loss:.6f}")
        print(f"Relative change: {relative_change:.6f}")
        if isinstance(hll, dict):
            print("Heldout log-likelihood components:")
            for key, value in hll.items():
                print(f"  {key}: {value:.6f}")
        else:
            print(f"Heldout log-likelihood: {hll:.6f}")
        
        if relative_change < tolerance:
            print(f"Converged after {itr+1} iterations")
            break
    
    return params, jnp.array(losses), jnp.array(heldout_loglikes)


# ===== EXECUTION SCRIPT INTEGRATION =====

def run_gaussian_seminmf_experiment(counts, masks, analysis_resultpath, WANDB_PROJECT, DATA_FILE, MASK_KEY, MASK_SIZE, NUM_MASKS_PER_MOUSE):
    """
    Complete execution script for Gaussian + softplus SemiNMF
    """
    
    # Hyperparameter settings
    mean_func = "softplus"
    elastic_net_frac = 1.0
    num_iters = 15
    num_coord_ascent_iters = 1
    subsample_frac = 0.3

    # Subsample voxels for faster computation in hyperparameter search
    key = jr.PRNGKey(42)
    n_voxels = int(counts.shape[1] * subsample_frac)
    voxel_idx = jr.choice(key, counts.shape[1], (n_voxels,), replace=False)

    sub_counts = counts[:, voxel_idx]
    sub_masks = masks[:, voxel_idx]

    # Estimate data variance on subsampled data
    max_num_factors = 25
    print("=== ESTIMATING DATA VARIANCE ===")
    
    try:
        estimated_data_variance = estimate_data_variance_proper(sub_counts, mean_func, max_num_factors)
        
        # Validation
        data_var = float(jnp.var(sub_counts))
        ratio = estimated_data_variance / data_var
        print(f"Data variance: {data_var:.6f}")
        print(f"Estimated variance: {estimated_data_variance:.6f}")
        print(f"Ratio: {ratio:.4f}")
        
        if ratio < 0.01 or ratio > 10:
            print("⚠️  Variance estimate seems problematic, using conservative fallback")
            estimated_data_variance = data_var * 0.2  # 20% of data variance
            estimated_data_variance = max(estimated_data_variance, 1.0)
        
    except Exception as e:
        print(f"❌ Variance estimation failed: {e}")
        estimated_data_variance = float(jnp.var(sub_counts)) * 0.1
        estimated_data_variance = max(estimated_data_variance, 1.0)
    
    print(f"=== FINAL VARIANCE ESTIMATE: {estimated_data_variance:.6f} ===")
    
    return estimated_data_variance


def analyze_constant_columns(counts):
    """Analyze which columns are constant and their values.
    
    Args:
        counts: Array of count data
        
    Returns:
        tuple: (constant_col_indices, constant_col_values)
    """
    col_std = jnp.std(counts, axis=0)
    constant_cols = (col_std == 0)
    constant_col_indices = jnp.where(constant_cols)[0]
    constant_col_values = counts[0, constant_col_indices]  # All values in constant columns are the same
    
    print("\nConstant Column Analysis:")
    print(f"Number of constant columns: {len(constant_col_indices)}")
    print("Constant column indices and their values:")
    for idx, val in zip(constant_col_indices, constant_col_values):
        print(f"Column {idx}: value = {val:.4f}")
    
    return constant_col_indices, constant_col_values


def preprocess_data(counts, remove_constant_cols=True, handle_negatives=True):
    """Preprocess data for factorization.
    
    Args:
        counts: Array of count data
        remove_constant_cols: Whether to remove constant columns
        handle_negatives: Whether to handle negative values
        
    Returns:
        tuple: (processed_data, removed_col_indices)
    """
    print("\nPreprocessing data:")
    removed_col_indices = []
    
    # Handle negative values
    if handle_negatives:
        print("Handling negative values...")
        counts = jnp.abs(counts)
        print(f"New value range: [{float(jnp.min(counts)):.2f}, {float(jnp.max(counts)):.2f}]")
    
    # Remove constant columns
    if remove_constant_cols:
        print("Removing constant columns...")
        col_std = jnp.std(counts, axis=0)
        non_constant_cols = col_std > 0
        removed_col_indices = jnp.where(~non_constant_cols)[0]
        counts = counts[:, non_constant_cols]
        print(f"Removed {len(removed_col_indices)} constant columns")
        print(f"New shape: {counts.shape}")
    
    return counts, removed_col_indices


if __name__ == "__main__":
    # Example usage of check_data_requirements
    import jax.numpy as jnp
    
    # Create some example data
    # Good data example
    good_data = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Bad data example with issues
    bad_data = jnp.array([
        [1.0, jnp.nan, 3.0],
        [4.0, -1.0, 6.0],
        [7.0, 8.0, jnp.inf]
    ])
    
    # Example with mask
    mask = jnp.array([
        [True, True, False],
        [True, True, True],
        [False, True, True]
    ])
    
    print("\nTesting with good data:")
    check_data_requirements(good_data)
    
    print("\nTesting with bad data:")
    check_data_requirements(bad_data)
    
    print("\nTesting with mask:")
    check_data_requirements(good_data, mask)
    
    # Example of how to use it in practice:
    print("\nExample of practical usage:")
    try:
        # This will pass
        if check_data_requirements(good_data):
            print("Data is valid, proceeding with factorization...")
        
        # This will raise an error
        if check_data_requirements(bad_data):
            print("This won't be printed because bad_data will fail the check")
    except ValueError as e:
        print(f"Error: {e}")