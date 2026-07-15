# ===== ADJUSTED COUNTS SEMI-NMF IMPLEMENTATION =====
# For data like: counts = raw_counts - bg_counts (can be negative)

import dataclasses
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from fastprogress import progress_bar
from jax import grad, hessian, vmap, lax, jit
from jax.nn import sigmoid
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
warnings.filterwarnings("ignore")

# If fos.prox is not available, define soft_threshold locally
try:
    from fos.prox import soft_threshold
except ImportError:
    def soft_threshold(x, thresh):
        """Soft thresholding operator for L1 regularization"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)

try:
    from fos.utils import register_pytree_node_dataclass, tree_add, tree_dot
    from jax.tree_util import register_pytree_node_class
except ImportError:
    # Define minimal versions if fos.utils not available
    import jax
    from jax.tree_util import register_pytree_node_class
    
    def register_pytree_node_dataclass(cls):
        return dataclasses.dataclass(frozen=True)(cls)
    
    def tree_add(x, y, alpha=1.0):
        """Add two pytrees: x + alpha * y"""
        return jax.tree_map(lambda a, b: a + alpha * b, x, y)
    
    def tree_dot(x, y):
        """Dot product of two pytrees (flattened)"""
        flat_x = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(x)])
        flat_y = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(y)])
        return jnp.dot(flat_x, flat_y)


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class AdjustedCountsSemiNMFParams:
    factors : Float[Array, "num_factors num_columns"]
    count_loadings : Float[Array, "num_rows num_factors"]
    count_row_effects : Float[Array, "num_rows"]
    count_col_effects : Float[Array, "num_columns"]

    @property
    def num_factors(self):
        return self.factors.shape[0]


def normal_loss(params, counts, mask, data_variance=1.0):
    """Compute negative log-likelihood using identity link for Normal model"""
    # Identity link function - linear predictor can be negative
    linear_pred = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    
    # Normal likelihood with estimated variance
    var = data_variance
    
    # Negative log-likelihood (to be minimized)
    loss = jnp.where(mask, -tfd.Normal(loc=linear_pred, scale=jnp.sqrt(var)).log_prob(counts), 0.0).sum()
    return loss


grad_normal_loss = grad(normal_loss, argnums=0)


def penalty(params, sparsity_penalty, elastic_net_frac):
    """L1/L2 penalty on loadings"""
    loss = elastic_net_frac * sparsity_penalty * jnp.sum(jnp.abs(params.count_loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.count_loadings ** 2)
    return loss


@partial(jit, static_argnums=())
def compute_loss(counts, mask, params, sparsity_penalty, elastic_net_frac, data_variance=1.0):
    """Total loss: negative log-likelihood + penalty"""
    loss = normal_loss(params, counts, mask, data_variance)
    loss += penalty(params, sparsity_penalty, elastic_net_frac)
    return loss / counts.size


def heldout_loglike(counts, mask, params, data_variance=1.0):
    """Compute held-out log-likelihood on unmasked entries"""
    linear_pred = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    var = data_variance
    
    # Evaluate on held-out (unmasked) entries
    loglike = jnp.where(~mask, tfd.Normal(loc=linear_pred, scale=jnp.sqrt(var)).log_prob(counts), 0.0).sum()
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


@partial(jit, static_argnums=())
def compute_quadratic_approx(counts, mask, params, data_variance=1.0):
    """Compute quadratic approximation for Normal likelihood with identity link - FIXED"""
    
    # Identity link function - linear predictor
    linear_pred = (params.count_row_effects[:, None] + 
                   params.count_col_effects + 
                   jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    
    var = data_variance
    residuals = counts - linear_pred
    
    # For coordinate descent on Normal likelihood:
    # We want to minimize: (1/2var) * sum((counts - pred)^2)
    # 
    # For identity link f(x) = x:
    # - Curvature (Hessian diagonal): mask / var
    # - Gradient times coordinate: mask * residuals / var
    # 
    # The h_counts represents the "working response" in the coordinate descent
    J_counts = mask / var
    h_counts = mask * residuals / var
    
    return QuadraticApprox(J_counts, h_counts)


def initialize_adjusted_counts_svd(counts, num_factors, drugs=None):
    """Initialize using SVD for adjusted counts data (can be negative) - IMPROVED"""
    num_mice, num_voxels = counts.shape
    
    # Check for problematic data
    if jnp.any(jnp.isnan(counts)) or jnp.any(jnp.isinf(counts)):
        raise ValueError("Input counts contain NaN or infinite values!")
    
    print(f"Initializing with data range: [{float(jnp.min(counts)):.3f}, {float(jnp.max(counts)):.3f}]")
    
    # For adjusted counts, work directly with the data (no transformation needed)
    targets = counts
    
    # Robust mean removal
    row_effects = jnp.mean(targets, axis=1)
    col_centered = targets - row_effects[:, None]
    col_effects = jnp.mean(col_centered, axis=0)
    residuals = col_centered - col_effects
    
    # SVD with proper handling of negative values
    try:
        U, S, VT = jnp.linalg.svd(residuals, full_matrices=False)
        
        # Check SVD results
        if jnp.any(jnp.isnan(U)) or jnp.any(jnp.isnan(S)) or jnp.any(jnp.isnan(VT)):
            raise ValueError("SVD produced NaN values")
        
        # Extract factors - ensure they're not too small
        count_loadings = []
        factors = []
        
        for k in range(min(num_factors, len(S))):
            if k >= len(S) or S[k] < 1e-10:
                # Use larger random factor for small singular values
                key = jr.PRNGKey(42 + k)
                factor = jnp.abs(jr.normal(key, (num_voxels,))) * 0.1  # Larger initialization
                loading = jr.normal(jr.fold_in(key, 1), (num_mice,)) * 0.1
            else:
                uk, sk, vk = U[:, k], S[k], VT[k]
                
                # For Semi-NMF: factors must be non-negative, loadings can be any sign
                # Take absolute value of factors and adjust loadings accordingly
                factor = jnp.abs(vk)
                loading = uk * sk * jnp.sign(vk)
                
                # Ensure factors aren't too small
                factor = jnp.maximum(factor, 1e-6)
                
                # Normalize factor but keep reasonable scale
                scale = jnp.sum(factor) + 1e-10
                factor = factor / scale
                loading = loading * scale
            
            factors.append(factor)
            count_loadings.append(loading)
        
        # Fill remaining factors with reasonable random values
        key = jr.PRNGKey(999)
        while len(factors) < num_factors:
            k = len(factors)
            subkey = jr.fold_in(key, k)
            factor = jnp.abs(jr.normal(subkey, (num_voxels,))) * 0.1
            factor = jnp.maximum(factor, 1e-6)  # Ensure not too small
            factor = factor / (jnp.sum(factor) + 1e-10)
            factors.append(factor)
            count_loadings.append(jr.normal(jr.fold_in(subkey, 1), (num_mice,)) * 0.1)
        
        count_loadings = jnp.column_stack(count_loadings)
        factors = jnp.stack(factors)
        
    except Exception as e:
        print(f"❌ SVD failed ({e}), using robust random initialization")
        # Fallback to robust random initialization
        key = jr.PRNGKey(123)
        factors = jnp.abs(jr.normal(key, (num_factors, num_voxels))) * 0.1
        factors = jnp.maximum(factors, 1e-6)  # Ensure not too small
        factors = factors / jnp.sum(factors, axis=1, keepdims=True)
        count_loadings = jr.normal(jr.fold_in(key, 1), (num_mice, num_factors)) * 0.1
        row_effects = jnp.mean(targets, axis=1)
        col_effects = jnp.mean(targets - row_effects[:, None], axis=0)
    
    # Final safety check - ensure no values are too small
    factors = jnp.maximum(factors, 1e-6)
    if jnp.any(jnp.isnan(factors)) or jnp.any(jnp.isnan(count_loadings)):
        print("❌ NaN in final initialization, using safe fallback")
        key = jr.PRNGKey(456)
        factors = jnp.abs(jr.normal(key, (num_factors, num_voxels))) * 0.1
        factors = jnp.maximum(factors, 1e-6)
        factors = factors / jnp.sum(factors, axis=1, keepdims=True)
        count_loadings = jr.normal(jr.fold_in(key, 1), (num_mice, num_factors)) * 0.1
        row_effects = jnp.mean(targets, axis=1)
        col_effects = jnp.mean(targets - row_effects[:, None], axis=0)
    
    print(f"Initialized factors range: [{float(jnp.min(factors)):.6f}, {float(jnp.max(factors)):.6f}]")
    print(f"Initialized loadings range: [{float(jnp.min(count_loadings)):.6f}, {float(jnp.max(count_loadings)):.6f}]")
    print(f"Factors min value: {float(jnp.min(factors)):.8f} (should be > 1e-6)")
    
    return AdjustedCountsSemiNMFParams(
        factors=factors,
        count_loadings=count_loadings,
        count_row_effects=row_effects,
        count_col_effects=col_effects
    )


def update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac):
    """Update loadings using coordinate descent - IMPROVED NUMERICAL STABILITY"""
    def _update_one_loading(h_m, J_m, loading_m):
        def _update_one_coord(h_m, args):
            loading_mk, factor_k = args
            
            # Coordinate descent update with better numerical stability
            num = jnp.einsum('n,n->', factor_k, (h_m + J_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', J_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty
            
            # More robust denominator check - if factors are too small, skip update
            den = jnp.maximum(den, 1e-6)  # Larger minimum to prevent numerical issues
            
            # Apply soft thresholding (prox operator)
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / den
            
            # Check for NaN and clip if necessary
            new_loading_mk = jnp.where(jnp.isnan(new_loading_mk), loading_mk, new_loading_mk)
            new_loading_mk = jnp.clip(new_loading_mk, -1e3, 1e3)  # Prevent explosion
            
            # Update residual
            h_m += J_m * loading_mk * factor_k
            h_m -= J_m * new_loading_mk * factor_k
            
            return h_m, new_loading_mk
            
        h_m, loading_m = lax.scan(_update_one_coord, h_m, (loading_m, params.factors))
        return h_m, loading_m
    
    h_counts, count_loadings = vmap(_update_one_loading)(quad_approx.h_counts, quad_approx.J_counts, params.count_loadings)
    

    
    params = dataclasses.replace(params, count_loadings=count_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_factors(quad_approx, params):
    """Update factors with non-negativity constraint - IMPROVED NUMERICAL STABILITY"""
    def _update_one_column(hc_n, Jc_n, factor_n):
        def _update_one_coord(hc_n, args):
            factor_nk, count_loading_k = args
            
            # Coordinate descent update with better numerical stability
            numerator = jnp.einsum('m,m->', count_loading_k, (hc_n + Jc_n * factor_nk * count_loading_k))
            denominator = jnp.einsum('m,m,m->', Jc_n, count_loading_k, count_loading_k)
            
            # More robust denominator - ensure it's not too small
            denominator = jnp.maximum(denominator, 1e-6)
            
            # Apply non-negativity constraint
            new_factor_nk = jnp.maximum(numerator, 0.0) / denominator
            
            # Ensure factor isn't too small (causes issues in loading updates)
            new_factor_nk = jnp.maximum(new_factor_nk, 1e-6)
            
            # Check for NaN
            new_factor_nk = jnp.where(jnp.isnan(new_factor_nk), factor_nk, new_factor_nk)
            
            # Update residual
            hc_n += Jc_n * factor_nk * count_loading_k
            hc_n -= Jc_n * new_factor_nk * count_loading_k
            
            return hc_n, new_factor_nk
            
        hc_n, factor_n = lax.scan(_update_one_coord, hc_n, (factor_n, params.count_loadings.T))
        return hc_n, factor_n
    
    h_countsT, factorsT = vmap(_update_one_column)(
        quad_approx.h_counts.T, 
        quad_approx.J_counts.T, 
        params.factors.T
    )
    
    h_counts = h_countsT.T
    factors = factorsT.T
    
    # Normalize factors but ensure they don't become too small
    scale = jnp.maximum(factors.sum(axis=1), 1e-6)  # Prevent too small scales
    factors = factors / scale[:, None]
    factors = jnp.maximum(factors, 1e-6)  # Ensure factors stay above minimum
    count_loadings = params.count_loadings * scale

    
    params = dataclasses.replace(params, 
                                factors=factors, 
                                count_loadings=count_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    
    return quad_approx, params


def update_row_effect(quad_approx, params):
    """Update row effects - JAX compatible"""
    def _update_one_row(h_m, J_m, row_effect_m):
        # Use original code pattern - no if statements
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m) + 1e-8
        new_row_effect_m = num / den
        
        # Update residual (like original)
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m
    
    h_counts, count_row_effects = vmap(_update_one_row)(quad_approx.h_counts, quad_approx.J_counts, params.count_row_effects)
    
    params = dataclasses.replace(params, count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_column_effect(quad_approx, params):
    """Update column effects - JAX compatible"""
    def _update_one_column(h_n, J_n, col_effect_n):
        # Use original code pattern - no if statements
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n) + 1e-8
        new_col_effect_n = num / den
        
        # Update residual (like original)
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n
    
    h_countsT, count_col_effects = vmap(_update_one_column)(quad_approx.h_counts.T, quad_approx.J_counts.T, params.count_col_effects)
    h_counts = h_countsT.T
    
    # Identifiability constraint (like original)
    mean_col_effect = jnp.mean(count_col_effects)
    count_col_effects -= mean_col_effect
    count_row_effects = params.count_row_effects + mean_col_effect
    
    params = dataclasses.replace(params, 
                                count_col_effects=count_col_effects,
                                count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def scale_data_for_factorization(counts, scaling_method="standardize"):
    """Scale data for better numerical stability and convergence.
    
    Args:
        counts: Array of adjusted count data (can be negative)
        scaling_method: Method for scaling
            - "standardize": (data - mean) / std
            - "robust": (data - median) / MAD  
            - "minmax": (data - min) / (max - min)
            - "none": no scaling
            
    Returns:
        tuple: (scaled_counts, scale_params)
    """
    if scaling_method == "none":
        return counts, {"method": "none"}
    
    elif scaling_method == "standardize":
        mean_val = jnp.mean(counts)
        std_val = jnp.std(counts)
        scaled_counts = (counts - mean_val) / (std_val + 1e-8)
        scale_params = {"method": "standardize", "mean": mean_val, "std": std_val}
        
    elif scaling_method == "robust":
        median_val = jnp.median(counts)
        mad_val = jnp.median(jnp.abs(counts - median_val))
        scaled_counts = (counts - median_val) / (1.4826 * mad_val + 1e-8)  # MAD to std conversion
        scale_params = {"method": "robust", "median": median_val, "mad": mad_val}
        
    elif scaling_method == "minmax":
        min_val = jnp.min(counts)
        max_val = jnp.max(counts)
        scaled_counts = (counts - min_val) / (max_val - min_val + 1e-8)
        scale_params = {"method": "minmax", "min": min_val, "max": max_val}
        
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    print(f"📏 Data scaling applied: {scaling_method}")
    print(f"   Original range: [{float(jnp.min(counts)):.3f}, {float(jnp.max(counts)):.3f}]")
    print(f"   Scaled range: [{float(jnp.min(scaled_counts)):.3f}, {float(jnp.max(scaled_counts)):.3f}]")
    
    return scaled_counts, scale_params


def unscale_params(params, scale_params):
    """Transform parameters back to original data scale."""
    if scale_params["method"] == "none":
        return params
    
    # The scaling affects the magnitude but not the relative patterns
    # For Semi-NMF: scaled_data ≈ row_effects + col_effects + loadings @ factors
    # To get back to original scale: original_data ≈ scale_factor * scaled_predictions + shift
    
    if scale_params["method"] == "standardize":
        scale_factor = scale_params["std"]
        shift = scale_params["mean"]
    elif scale_params["method"] == "robust":
        scale_factor = 1.4826 * scale_params["mad"]
        shift = scale_params["median"]
    elif scale_params["method"] == "minmax":
        scale_factor = scale_params["max"] - scale_params["min"]
        shift = scale_params["min"]
    
    # Adjust loadings and effects to account for scaling
    unscaled_loadings = params.count_loadings * scale_factor
    unscaled_row_effects = params.count_row_effects * scale_factor + shift
    unscaled_col_effects = params.count_col_effects * scale_factor
    
    return dataclasses.replace(params,
                             count_loadings=unscaled_loadings,
                             count_row_effects=unscaled_row_effects,
                             count_col_effects=unscaled_col_effects)

def update_factors(quad_approx, params):
    """Update factors with non-negativity constraint - JAX SAFE VERSION"""
    def _update_one_column(hc_n, Jc_n, factor_n):
        def _update_one_coord(hc_n, args):
            factor_nk, count_loading_k = args
            
            # Coordinate descent update (exactly like original code)
            numerator = jnp.einsum('m,m->', count_loading_k, (hc_n + Jc_n * factor_nk * count_loading_k))
            denominator = jnp.einsum('m,m,m->', Jc_n, count_loading_k, count_loading_k)
            
            # Apply non-negativity constraint - JAX safe, no if statements
            new_factor_nk = jnp.maximum(numerator, 0.0) / (denominator + 1e-6)
            
            # Ensure factor isn't too small
            new_factor_nk = jnp.maximum(new_factor_nk, 1e-6)
            
            # JAX-safe NaN protection
            new_factor_nk = jnp.where(jnp.isnan(new_factor_nk), factor_nk, new_factor_nk)
            
            # Update residual
            hc_n += Jc_n * factor_nk * count_loading_k
            hc_n -= Jc_n * new_factor_nk * count_loading_k
            
            return hc_n, new_factor_nk
            
        hc_n, factor_n = lax.scan(_update_one_coord, hc_n, (factor_n, params.count_loadings.T))
        return hc_n, factor_n
    
    h_countsT, factorsT = vmap(_update_one_column)(
        quad_approx.h_counts.T, 
        quad_approx.J_counts.T, 
        params.factors.T
    )
    
    h_counts = h_countsT.T
    factors = factorsT.T
    
    # Normalize factors
    scale = jnp.maximum(factors.sum(axis=1), 1e-6)
    factors = factors / scale[:, None]
    factors = jnp.maximum(factors, 1e-6)
    count_loadings = params.count_loadings * scale
    
    params = dataclasses.replace(params, 
                                factors=factors, 
                                count_loadings=count_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    
    return quad_approx, params
def update_row_effect(quad_approx, params):
    """Update row effects - JAX compatible"""
    def _update_one_row(h_m, J_m, row_effect_m):
        # Use original code pattern - no if statements
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m) + 1e-8
        new_row_effect_m = num / den
        
        # Update residual (like original)
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m
    
    h_counts, count_row_effects = vmap(_update_one_row)(quad_approx.h_counts, quad_approx.J_counts, params.count_row_effects)
    
    params = dataclasses.replace(params, count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def update_column_effect(quad_approx, params):
    """Update column effects - JAX compatible"""
    def _update_one_column(h_n, J_n, col_effect_n):
        # Use original code pattern - no if statements
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n) + 1e-8
        new_col_effect_n = num / den
        
        # Update residual (like original)
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n
    
    h_countsT, count_col_effects = vmap(_update_one_column)(quad_approx.h_counts.T, quad_approx.J_counts.T, params.count_col_effects)
    h_counts = h_countsT.T
    
    # Identifiability constraint (like original)
    mean_col_effect = jnp.mean(count_col_effects)
    count_col_effects -= mean_col_effect
    count_row_effects = params.count_row_effects + mean_col_effect
    
    params = dataclasses.replace(params, 
                                count_col_effects=count_col_effects,
                                count_row_effects=count_row_effects)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    return quad_approx, params


def backtracking_line_search(counts, mask, params, new_params, 
                           sparsity_penalty, elastic_net_frac, data_variance=1.0, 
                           alpha=0.5, beta=0.5, max_iters=30):
    """Backtracking line search with improved numerical stability"""
    
    # Compute descent direction
    descent_direction = tree_add(new_params, params, -1.0)
    
    # Check if this is actually a descent direction
    dg = grad_normal_loss(params, counts, mask, data_variance)
    dg_direc = tree_dot(dg, descent_direction)
    
    # Use JAX's conditional instead of Python's if statement
    def descent_case():
        baseline = normal_loss(params, counts, mask, data_variance)
        baseline += penalty(params, sparsity_penalty, elastic_net_frac)
        
        def cond_fun(state):
            stepsize, itr = state
            too_small = stepsize < 1e-10
            
            new_params_try = tree_add(params, descent_direction, stepsize)
            new_loss = normal_loss(new_params_try, counts, mask, data_variance)
            new_loss += penalty(new_params_try, sparsity_penalty, elastic_net_frac)
            
            # Armijo condition
            bound = baseline + alpha * stepsize * dg_direc
            needs_reduction = new_loss > bound
            
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
    
    return lax.cond(dg_direc < 0, descent_case, no_descent_case)


def estimate_adjusted_counts_variance(counts, max_factors=20):
    """
    Estimate variance for adjusted counts data (can be negative) - IMPROVED
    """
    print(f"Estimating variance for adjusted counts data...")
    
    # Input validation
    if jnp.any(jnp.isnan(counts)) or jnp.any(jnp.isinf(counts)):
        raise ValueError("Input counts contain NaN or infinite values!")
    
    print(f"Data range: [{float(jnp.min(counts)):.3f}, {float(jnp.max(counts)):.3f}]")
    
    # Method 1: Remove row/column effects first
    row_effects = jnp.mean(counts, axis=1, keepdims=True)
    col_centered = counts - row_effects
    col_effects = jnp.mean(col_centered, axis=0, keepdims=True)
    residuals = col_centered - col_effects
    
    # Method 2: Use MAD (Median Absolute Deviation) for robustness
    mad = jnp.median(jnp.abs(residuals - jnp.median(residuals)))
    robust_std = 1.4826 * mad  # MAD to std conversion
    robust_var = robust_std ** 2
    
    # Method 3: Simple variance as fallback
    simple_var = float(jnp.var(residuals))
    
    # Method 4: Remove low-rank structure using SVD (if data is reasonable size)
    final_var = robust_var
    if counts.size < 100000:  # Only do SVD for smaller matrices
        try:
            U, S, VT = jnp.linalg.svd(residuals, full_matrices=False)
            k = min(max_factors, min(residuals.shape[0], residuals.shape[1]) - 1)
            
            if k > 0:
                # Keep only the noise (remove signal)
                signal_reconstruction = (U[:, :k] * S[:k]) @ VT[:k]
                noise_residuals = residuals - signal_reconstruction
                noise_var = float(jnp.var(noise_residuals))
                
                # Use the more conservative estimate but not too small
                final_var = max(noise_var, robust_var * 0.1)
                
                print(f"Noise variance (SVD): {noise_var:.6f}")
                
        except Exception as e:
            print(f"SVD variance estimation failed ({e}), using robust estimate")
            final_var = robust_var
    
    # Ensure variance is reasonable
    final_var = max(float(final_var), 0.01)  # Minimum variance
    final_var = min(final_var, simple_var * 10)  # Maximum variance
    
    print(f"Robust variance (MAD): {robust_var:.6f}")
    print(f"Simple variance: {simple_var:.6f}")
    print(f"Final variance: {final_var:.6f}")
    
    return final_var


def diagnose_nan_issues(params, counts, mask, data_variance):
    """Diagnose what might be causing NaN values"""
    print("\n🔬 DIAGNOSING NaN ISSUES:")
    
    # Check parameters
    has_nan_factors = jnp.any(jnp.isnan(params.factors))
    has_nan_loadings = jnp.any(jnp.isnan(params.count_loadings))
    has_nan_row = jnp.any(jnp.isnan(params.count_row_effects))
    has_nan_col = jnp.any(jnp.isnan(params.count_col_effects))
    
    print(f"   Factors have NaN: {has_nan_factors}")
    print(f"   Loadings have NaN: {has_nan_loadings}")
    print(f"   Row effects have NaN: {has_nan_row}")
    print(f"   Col effects have NaN: {has_nan_col}")
    
    # Check parameter magnitudes
    factor_min = float(jnp.min(params.factors))
    factor_max = float(jnp.max(params.factors))
    loading_min = float(jnp.min(jnp.abs(params.count_loadings)))
    loading_max = float(jnp.max(jnp.abs(params.count_loadings)))
    
    print(f"   Factor range: [{factor_min:.8f}, {factor_max:.8f}]")
    print(f"   Loading range: [{loading_min:.8f}, {loading_max:.8f}]")
    
    if factor_min < 1e-10:
        print("   ⚠️  Factors are extremely small - this causes division by zero!")
    if loading_max > 1e6:
        print("   ⚠️  Loadings are extremely large - numerical overflow!")
    
    # Check data variance
    print(f"   Data variance: {data_variance:.6f}")
    if data_variance < 1e-6:
        print("   ⚠️  Data variance is extremely small!")
    if data_variance > 1e6:
        print("   ⚠️  Data variance is extremely large!")
    
    # Check mask
    mask_coverage = float(jnp.mean(mask))
    print(f"   Mask coverage: {mask_coverage:.2%}")
    
    # Check quadratic approximation
    try:
        quad_approx = compute_quadratic_approx(counts, mask, params, data_variance)
        J_has_nan = jnp.any(jnp.isnan(quad_approx.J_counts))
        h_has_nan = jnp.any(jnp.isnan(quad_approx.h_counts))
        print(f"   Quadratic J has NaN: {J_has_nan}")
        print(f"   Quadratic h has NaN: {h_has_nan}")
        
        J_min = float(jnp.min(quad_approx.J_counts))
        J_max = float(jnp.max(quad_approx.J_counts))
        print(f"   Quadratic J range: [{J_min:.8f}, {J_max:.8f}]")
        
        if J_min < 1e-10:
            print("   ⚠️  Quadratic curvature is extremely small!")
            
    except Exception as e:
        print(f"   ❌ Quadratic approximation failed: {e}")


def _step(params, counts, mask, sparsity_penalty, elastic_net_frac, num_coord_ascent_iters, data_variance, debug=False):
    """Single optimization step with debugging and NaN checks"""
    if debug:
        print(f"  📊 Starting step - Factors: [{float(jnp.min(params.factors)):.6f}, {float(jnp.max(params.factors)):.6f}]")
        print(f"  📊 Starting step - Loadings: [{float(jnp.min(params.count_loadings)):.6f}, {float(jnp.max(params.count_loadings)):.6f}]")
    
    # Early NaN check
    if jnp.any(jnp.isnan(params.factors)) or jnp.any(jnp.isnan(params.count_loadings)):
        print("❌ NaN detected at start of step!")
        diagnose_nan_issues(params, counts, mask, data_variance)
        return params, jnp.nan, jnp.nan, {
            'factor_change': jnp.nan, 'loading_change': jnp.nan,
            'row_change': jnp.nan, 'col_change': jnp.nan,
            'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
        }
    
    # Store initial params for change tracking
    old_factors = params.factors
    old_loadings = params.count_loadings
    old_row_effects = params.count_row_effects
    old_col_effects = params.count_col_effects
    
    # Compute initial loss for this step
    initial_step_loss = compute_loss(counts, mask, params, sparsity_penalty, elastic_net_frac, data_variance)
    
    if jnp.isnan(initial_step_loss):
        print("❌ NaN in initial loss computation!")
        diagnose_nan_issues(params, counts, mask, data_variance)
        return params, jnp.nan, jnp.nan, {
            'factor_change': jnp.nan, 'loading_change': jnp.nan,
            'row_change': jnp.nan, 'col_change': jnp.nan,
            'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
        }
    
    try:
        quad_approx = compute_quadratic_approx(counts, mask, params, data_variance)
        
        if jnp.any(jnp.isnan(quad_approx.J_counts)) or jnp.any(jnp.isnan(quad_approx.h_counts)):
            print("❌ NaN in quadratic approximation!")
            diagnose_nan_issues(params, counts, mask, data_variance)
            return params, jnp.nan, jnp.nan, {
                'factor_change': jnp.nan, 'loading_change': jnp.nan,
                'row_change': jnp.nan, 'col_change': jnp.nan,
                'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
            }
        
    except Exception as e:
        print(f"❌ Exception in quadratic approximation: {e}")
        return params, jnp.nan, jnp.nan, {
            'factor_change': jnp.nan, 'loading_change': jnp.nan,
            'row_change': jnp.nan, 'col_change': jnp.nan,
            'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
        }
    
    for coord_iter in range(num_coord_ascent_iters):
        if debug:
            print(f"    🔄 Coordinate iteration {coord_iter + 1}")
        
        # Store params before each update
        pre_loadings = params.count_loadings
        pre_factors = params.factors
        pre_row_effects = params.count_row_effects
        pre_col_effects = params.count_col_effects
        
        # Update loadings (can be any sign)
        try:
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            if debug:
                loading_change = jnp.linalg.norm(params.count_loadings - pre_loadings)
                print(f"      📈 Loadings change norm: {float(loading_change):.8f}")
        except Exception as e:
            print(f"❌ Exception in loading update: {e}")
            return params, jnp.nan, jnp.nan, {
                'factor_change': jnp.nan, 'loading_change': jnp.nan,
                'row_change': jnp.nan, 'col_change': jnp.nan,
                'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
            }
        
        # Update factors (must be non-negative)
        try:
            quad_approx, params = update_factors(quad_approx, params)
            if debug:
                factor_change = jnp.linalg.norm(params.factors - pre_factors)
                print(f"      📊 Factors change norm: {float(factor_change):.8f}")
        except Exception as e:
            print(f"❌ Exception in factor update: {e}")
            return params, jnp.nan, jnp.nan, {
                'factor_change': jnp.nan, 'loading_change': jnp.nan,
                'row_change': jnp.nan, 'col_change': jnp.nan,
                'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
            }
        
        # Update row and column effects
        try:
            quad_approx, params = update_row_effect(quad_approx, params)
            if debug:
                row_change = jnp.linalg.norm(params.count_row_effects - pre_row_effects)
                print(f"      📍 Row effects change norm: {float(row_change):.8f}")
                
            quad_approx, params = update_column_effect(quad_approx, params)
            if debug:
                col_change = jnp.linalg.norm(params.count_col_effects - pre_col_effects)
                print(f"      📍 Col effects change norm: {float(col_change):.8f}")
        except Exception as e:
            print(f"❌ Exception in effect updates: {e}")
            return params, jnp.nan, jnp.nan, {
                'factor_change': jnp.nan, 'loading_change': jnp.nan,
                'row_change': jnp.nan, 'col_change': jnp.nan,
                'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
            }
    
    # Compute total parameter changes
    total_factor_change = jnp.linalg.norm(params.factors - old_factors)
    total_loading_change = jnp.linalg.norm(params.count_loadings - old_loadings)
    total_row_change = jnp.linalg.norm(params.count_row_effects - old_row_effects)
    total_col_change = jnp.linalg.norm(params.count_col_effects - old_col_effects)
    
    # Compute loss and held-out likelihood
    try:
        final_step_loss = compute_loss(counts, mask, params, sparsity_penalty, elastic_net_frac, data_variance)
        hll = heldout_loglike(counts, mask, params, data_variance)
    except Exception as e:
        print(f"❌ Exception in loss computation: {e}")
        return params, jnp.nan, jnp.nan, {
            'factor_change': jnp.nan, 'loading_change': jnp.nan,
            'row_change': jnp.nan, 'col_change': jnp.nan,
            'data_loss': jnp.nan, 'penalty_loss': jnp.nan, 'step_improvement': jnp.nan
        }
    
    # Check if step actually improved loss
    step_improvement = initial_step_loss - final_step_loss
    
    # Compute loss components separately for debugging
    data_loss = normal_loss(params, counts, mask, data_variance) / counts.size
    penalty_loss = penalty(params, sparsity_penalty, elastic_net_frac) / counts.size
    
    if debug:
        print(f"  ✅ Step complete:")
        print(f"     Total factor change: {float(total_factor_change):.8f}")
        print(f"     Total loading change: {float(total_loading_change):.8f}")
        print(f"     Total row effects change: {float(total_row_change):.8f}")
        print(f"     Total col effects change: {float(total_col_change):.8f}")
        print(f"     Step loss improvement: {float(step_improvement):.8f}")
        print(f"     Data loss: {float(data_loss):.8f}")
        print(f"     Penalty loss: {float(penalty_loss):.8f}")
        print(f"     Total loss: {float(final_step_loss):.8f}")
        
        if step_improvement < 0:
            print(f"     ⚠️  WARNING: Step made loss worse by {float(-step_improvement):.8f}")
    
    return params, final_step_loss, hll, {
        'factor_change': float(total_factor_change),
        'loading_change': float(total_loading_change),
        'row_change': float(total_row_change),
        'col_change': float(total_col_change),
        'data_loss': float(data_loss),
        'penalty_loss': float(penalty_loss),
        'step_improvement': float(step_improvement)
    }

def initialize_simple_random(counts, num_factors):
    """Simple random initialization that works - FIXED VERSION"""
    num_mice, num_voxels = counts.shape
    key = jr.PRNGKey(42)
    
    # Initialize factors to reasonable positive values (DON'T over-normalize)
    factors = jr.uniform(key, (num_factors, num_voxels), minval=0.01, maxval=0.1)
    # LIGHT normalization only - don't divide by huge sums
    factors = factors / jnp.mean(factors, axis=1, keepdims=True)  # Normalize by mean, not sum
    factors = jnp.maximum(factors, 1e-3)  # Ensure minimum value
    
    # Initialize loadings to larger values to compensate
    count_loadings = jr.normal(jr.fold_in(key, 1), (num_mice, num_factors)) * 1.0  # Larger scale
    
    # Initialize effects to reasonable values
    count_row_effects = jnp.mean(counts, axis=1)
    count_col_effects = jnp.mean(counts - count_row_effects[:, None], axis=0)
    
    print(f"Random init - Factors: [{float(jnp.min(factors)):.6f}, {float(jnp.max(factors)):.6f}]")
    print(f"Random init - Loadings: [{float(jnp.min(count_loadings)):.6f}, {float(jnp.max(count_loadings)):.6f}]")
    
    return AdjustedCountsSemiNMFParams(
        factors=factors,
        count_loadings=count_loadings,
        count_row_effects=count_row_effects,
        count_col_effects=count_col_effects
    )


def fit_adjusted_counts_seminmf(counts, initial_params, mask=None, num_iters=100,
                               sparsity_penalty=0.0, elastic_net_frac=0.5, num_coord_ascent_iters=10,
                               tolerance=1e-5, data_variance=None, debug=False, 
                               scaling_method="standardize"):
    """Fit the Semi-NMF model for adjusted counts data with optional scaling.
    
    Args:
        counts: Array of adjusted count data (can be negative)
        initial_params: Initial parameters for the model
        mask: Optional mask for the data
        num_iters: Number of iterations to run
        sparsity_penalty: Penalty for sparsity
        elastic_net_frac: Fraction of elastic net penalty
        num_coord_ascent_iters: Number of coordinate ascent iterations
        tolerance: Convergence tolerance
        data_variance: Optional data variance
        debug: Whether to print detailed debugging information
        scaling_method: How to scale data ("standardize", "robust", "minmax", "none")
        
    Returns:
        tuple: (params, losses, heldout_loglikes, debug_info, scale_params)
    """
    
    # Scale the data for better numerical stability
    scaled_counts, scale_params = scale_data_for_factorization(counts, scaling_method)
    
    # Initialize parameters
    params = initial_params
    if data_variance is None:
        data_variance = estimate_adjusted_counts_variance(scaled_counts)
    
    if mask is None:
        mask = jnp.ones_like(scaled_counts, dtype=bool)
    
    # Initialize lists to track progress
    losses = []
    heldout_loglikes = []
    debug_info = []
    
    # Compute initial loss and components
    initial_loss = compute_loss(scaled_counts, mask, params, sparsity_penalty, elastic_net_frac, data_variance)
    initial_data_loss = normal_loss(params, scaled_counts, mask, data_variance) / scaled_counts.size
    initial_penalty_loss = penalty(params, sparsity_penalty, elastic_net_frac) / scaled_counts.size
    initial_hll = heldout_loglike(scaled_counts, mask, params, data_variance)
    
    losses.append(initial_loss)
    heldout_loglikes.append(initial_hll)
    
    print(f"🚀 Starting optimization with {num_iters} iterations")
    print(f"📏 Using {scaling_method} scaling")
    print(f"📊 Initial statistics:")
    print(f"   Factors: [{float(jnp.min(params.factors)):.6f}, {float(jnp.max(params.factors)):.6f}], mean={float(jnp.mean(params.factors)):.6f}")
    print(f"   Loadings: [{float(jnp.min(params.count_loadings)):.6f}, {float(jnp.max(params.count_loadings)):.6f}], mean={float(jnp.mean(params.count_loadings)):.6f}")
    print(f"   Row effects: [{float(jnp.min(params.count_row_effects)):.6f}, {float(jnp.max(params.count_row_effects)):.6f}], mean={float(jnp.mean(params.count_row_effects)):.6f}")
    print(f"   Col effects: [{float(jnp.min(params.count_col_effects)):.6f}, {float(jnp.max(params.count_col_effects)):.6f}], mean={float(jnp.mean(params.count_col_effects)):.6f}")
    print(f"💰 Initial loss breakdown:")
    print(f"   Data loss: {initial_data_loss:.8f}")
    print(f"   Penalty loss: {initial_penalty_loss:.8f}")
    print(f"   Total loss: {initial_loss:.8f}")
    print(f"   Held-out log-likelihood: {initial_hll:.8f}")
    print(f"📈 Data variance: {data_variance:.6f}")
    print(f"🎯 Sparsity penalty: {sparsity_penalty:.6f}")
    print(f"=" * 80)
    
    # Check for potential issues
    if jnp.any(jnp.isnan(params.factors)) or jnp.any(jnp.isnan(params.count_loadings)):
        print("❌ WARNING: NaN detected in initial parameters!")
        return params, jnp.array(losses), jnp.array(heldout_loglikes), debug_info, scale_params
    
    # Main optimization loop
    for itr in range(num_iters):
        if debug or (itr + 1) <= 3:  # Always debug first 3 iterations
            print(f"\n🔄 ITERATION {itr+1}")
            detailed_debug = True
        else:
            detailed_debug = False
        
        # Take a step
        params, loss, hll, step_debug = _step(params, scaled_counts, mask, sparsity_penalty, 
                                            elastic_net_frac, num_coord_ascent_iters, 
                                            data_variance, debug=detailed_debug)
        
        losses.append(loss)
        heldout_loglikes.append(hll)
        debug_info.append(step_debug)
        
        # Check for NaN
        if jnp.any(jnp.isnan(params.factors)) or jnp.any(jnp.isnan(params.count_loadings)):
            print(f"❌ NaN detected at iteration {itr+1}! Stopping optimization.")
            break
        
        # Check for convergence
        relative_change = abs(loss - losses[-2]) / (abs(losses[-2]) + 1e-10)
        loss_improved = loss < losses[-2]
        
        # Print progress
        if detailed_debug or (itr + 1) % 5 == 0 or relative_change < tolerance:
            improvement = "📈" if loss_improved else "📉"
            print(f"{improvement} Iteration {itr+1}:")
            print(f"   Loss: {loss:.8f} (change: {loss - losses[-2]:+.8f})")
            print(f"   HLL: {hll:.8f}")
            print(f"   Relative change: {relative_change:.8f}")
            print(f"   Parameter changes: F={step_debug['factor_change']:.8f}, L={step_debug['loading_change']:.8f}")
            
            # Check if parameters are getting too large (might indicate instability)
            max_loading = float(jnp.max(jnp.abs(params.count_loadings)))
            max_factor = float(jnp.max(params.factors))
            if max_loading > 1000 or max_factor > 1000:
                print(f"⚠️  WARNING: Large parameter values (L_max={max_loading:.1f}, F_max={max_factor:.1f})")
            
            # Print current parameter stats
            print(f"   Current factors: [{float(jnp.min(params.factors)):.6f}, {float(jnp.max(params.factors)):.6f}]")
            print(f"   Current loadings: [{float(jnp.min(params.count_loadings)):.6f}, {float(jnp.max(params.count_loadings)):.6f}]")
        
        # Additional convergence checks
        if relative_change < tolerance:
            print(f"✅ Converged after {itr+1} iterations (relative change < {tolerance})")
            break
            
        # Check if we're not making progress
        if itr > 10:
            recent_losses = losses[-5:]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-10 for i in range(1, len(recent_losses))):
                print(f"⚠️  Stopping: Loss hasn't changed in 5 iterations")
                break
    
    # Transform parameters back to original scale
    unscaled_params = unscale_params(params, scale_params)
    
    print(f"\n🏁 Optimization complete!")
    print(f"📊 Final statistics (scaled data):")
    print(f"   Final loss: {losses[-1]:.8f}")
    print(f"   Final HLL: {heldout_loglikes[-1]:.8f}")
    print(f"   Total loss change: {losses[-1] - losses[0]:+.8f}")
    print(f"   Final factors range: [{float(jnp.min(params.factors)):.6f}, {float(jnp.max(params.factors)):.6f}]")
    print(f"   Final loadings range: [{float(jnp.min(params.count_loadings)):.6f}, {float(jnp.max(params.count_loadings)):.6f}]")
    print(f"📏 Unscaled parameters:")
    print(f"   Unscaled loadings range: [{float(jnp.min(unscaled_params.count_loadings)):.6f}, {float(jnp.max(unscaled_params.count_loadings)):.6f}]")
    
    return unscaled_params, jnp.array(losses), jnp.array(heldout_loglikes), debug_info, scale_params


def diagnose_optimization_issues(counts, params, mask, sparsity_penalty, data_variance):
    """Diagnose potential optimization issues"""
    print("\n🔍 DIAGNOSING OPTIMIZATION ISSUES:")
    
    # Check gradient norms
    grad_params = grad_normal_loss(params, counts, mask, data_variance)
    
    factor_grad_norm = float(jnp.linalg.norm(grad_params.factors))
    loading_grad_norm = float(jnp.linalg.norm(grad_params.count_loadings))
    row_grad_norm = float(jnp.linalg.norm(grad_params.count_row_effects))
    col_grad_norm = float(jnp.linalg.norm(grad_params.count_col_effects))
    
    print(f"   Factor gradient norm: {factor_grad_norm:.8f}")
    print(f"   Loading gradient norm: {loading_grad_norm:.8f}")
    print(f"   Row effects gradient norm: {row_grad_norm:.8f}")
    print(f"   Col effects gradient norm: {col_grad_norm:.8f}")
    
    # Check if gradients are too small
    if factor_grad_norm < 1e-8:
        print("   ⚠️  Factor gradients are very small - might be stuck at optimum or saddle point")
    if loading_grad_norm < 1e-8:
        print("   ⚠️  Loading gradients are very small - might be stuck at optimum or saddle point")
    
    # Check quadratic approximation
    quad_approx = compute_quadratic_approx(counts, mask, params, data_variance)
    J_min, J_max = float(jnp.min(quad_approx.J_counts)), float(jnp.max(quad_approx.J_counts))
    h_min, h_max = float(jnp.min(quad_approx.h_counts)), float(jnp.max(quad_approx.h_counts))
    
    print(f"   Quadratic J range: [{J_min:.8f}, {J_max:.8f}]")
    print(f"   Quadratic h range: [{h_min:.8f}, {h_max:.8f}]")
    
    if J_max < 1e-8:
        print("   ⚠️  Quadratic approximation coefficients are very small")
    
    # Check data properties
    pred = (params.count_row_effects[:, None] + 
            params.count_col_effects + 
            jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    
    residuals = counts - pred
    mse = float(jnp.mean(residuals**2))
    mae = float(jnp.mean(jnp.abs(residuals)))
    
    print(f"   Current MSE: {mse:.8f}")
    print(f"   Current MAE: {mae:.8f}")
    print(f"   Data variance used: {data_variance:.8f}")
    
    if mse < data_variance * 0.1:
        print("   ✅ Model fits data well (MSE << data variance)")
    elif mse > data_variance * 10:
        print("   ⚠️  Model fits data poorly (MSE >> data variance)")
    
    # Check mask coverage
    if mask is not None:
        mask_coverage = float(jnp.mean(mask))
        print(f"   Mask coverage: {mask_coverage:.2%}")
        if mask_coverage < 0.1:
            print("   ⚠️  Very little data available for training (high mask sparsity)")


def check_adjusted_counts_requirements(counts, mask=None):
    """Check if the adjusted counts data meets requirements for factorization.
    
    Args:
        counts: Array of adjusted count data (can be negative)
        mask: Optional mask for held-out data
        
    Returns:
        bool: True if data meets requirements, False otherwise
    """
    print("\n🔍 Checking adjusted counts data requirements:")
    
    # Check for NaN and Inf values
    has_nan = jnp.isnan(counts).any()
    has_inf = jnp.isinf(counts).any()
    print(f"   Contains NaN values: {has_nan}")
    print(f"   Contains Inf values: {has_inf}")
    
    # Check value ranges (can be negative for adjusted counts)
    min_val = float(jnp.min(counts))
    max_val = float(jnp.max(counts))
    mean_val = float(jnp.mean(counts))
    std_val = float(jnp.std(counts))
    print(f"   Value range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"   Mean: {mean_val:.2f}")
    print(f"   Std: {std_val:.2f}")
    
    # Check proportion of negative values
    neg_prop = float(jnp.mean(counts < 0))
    print(f"   Proportion negative values: {neg_prop:.2%}")
    
    # Check mask if provided
    if mask is not None:
        mask_sparsity = float(jnp.mean(~mask))
        print(f"   Mask sparsity: {mask_sparsity:.2%}")
    
    # Check for constant rows/columns
    row_std = jnp.std(counts, axis=1)
    col_std = jnp.std(counts, axis=0)
    has_const_rows = (row_std == 0).any()
    has_const_cols = (col_std == 0).any()
    print(f"   Contains constant rows: {has_const_rows}")
    print(f"   Contains constant columns: {has_const_cols}")
    
    # Additional checks for optimization (avoid expensive condition number for large matrices)
    matrix_size = counts.shape[0] * counts.shape[1]
    if matrix_size < 10000:  # Only compute condition number for small matrices
        try:
            cond_num = float(jnp.linalg.cond(counts))
            print(f"   Data condition number estimate: {cond_num:.2f}")
            if cond_num > 1e12:
                print("   ⚠️  Data matrix is poorly conditioned")
        except Exception as e:
            print(f"   Could not compute condition number: {e}")
    else:
        print(f"   Matrix too large ({matrix_size} elements) for condition number computation")
    
    # Return True if basic checks pass
    return not (has_nan or has_inf)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example usage for adjusted counts data
    import jax.numpy as jnp
    import jax.random as jr
    
    # Create some example adjusted counts data (can be negative)
    key = jr.PRNGKey(42)
    num_mice, num_voxels = 50, 100
    num_factors = 5
    
    # Simulate some adjusted counts (raw_counts - bg_counts)
    raw_counts = jr.poisson(key, 10.0, (num_mice, num_voxels))
    bg_counts = jr.poisson(jr.fold_in(key, 1), 8.0, (num_mice, num_voxels))
    adjusted_counts = raw_counts.astype(float) - bg_counts.astype(float)
    
    print("Example: Adjusted Counts Semi-NMF")
    print("=" * 40)
    
    # Check data requirements
    if check_adjusted_counts_requirements(adjusted_counts):
        print("✓ Data passes requirements check")
        
        # Initialize parameters
        print("\nInitializing parameters...")
        initial_params = initialize_adjusted_counts_svd(adjusted_counts, num_factors)
        
        # Estimate variance
        print("\nEstimating data variance...")
        data_variance = estimate_adjusted_counts_variance(adjusted_counts)
        
        # Fit model with scaling
        print("\nFitting Semi-NMF model...")
        params, losses, hlls, debug_info, scale_params = fit_adjusted_counts_seminmf(
            adjusted_counts, 
            initial_params, 
            num_iters=20,
            sparsity_penalty=0.1,
            data_variance=data_variance,
            debug=True,  # Enable detailed debugging
            scaling_method="standardize"  # Try standardization scaling
        )
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Final loss: {losses[-1]:.8f}")
        print(f"   Loss improvement: {losses[0] - losses[-1]:+.8f}")
        print(f"   Final factors range: [{float(jnp.min(params.factors)):.6f}, {float(jnp.max(params.factors)):.6f}]")
        print(f"   Final loadings range: [{float(jnp.min(params.count_loadings)):.6f}, {float(jnp.max(params.count_loadings)):.6f}]")
        print(f"   Data scaling used: {scale_params['method']}")
        
        # Analyze convergence
        print(f"\n📈 CONVERGENCE ANALYSIS:")
        optimization_working = True
        
        if len(debug_info) > 0:
            final_changes = debug_info[-1]
            print(f"   Final factor change: {final_changes['factor_change']:.8f}")
            print(f"   Final loading change: {final_changes['loading_change']:.8f}")
            print(f"   Final data loss: {final_changes['data_loss']:.8f}")
            print(f"   Final penalty loss: {final_changes['penalty_loss']:.8f}")
            print(f"   Final step improvement: {final_changes['step_improvement']:.8f}")
            
            # Check if optimization is working
            total_factor_changes = [d['factor_change'] for d in debug_info]
            total_loading_changes = [d['loading_change'] for d in debug_info]
            step_improvements = [d['step_improvement'] for d in debug_info]
            
            if max(total_factor_changes) < 1e-8:
                print("   ⚠️  WARNING: Factors barely changed during optimization!")
                optimization_working = False
            if max(total_loading_changes) < 1e-8:
                print("   ⚠️  WARNING: Loadings barely changed during optimization!")
                optimization_working = False
            
            if sum(step_improvements) < 0:
                print("   ⚠️  WARNING: Most steps made loss worse!")
                optimization_working = False
            
            if losses[-1] >= losses[0] * 0.99:
                print("   ⚠️  WARNING: Loss did not improve significantly!")
                optimization_working = False
            else:
                print("   ✅ Loss improved during optimization")
                
            # Check for parameter explosion (without clipping, this is important to monitor)
            max_loading = float(jnp.max(jnp.abs(params.count_loadings)))
            max_factor = float(jnp.max(params.factors))
            if max_loading > 1000:
                print(f"   ⚠️  WARNING: Large loadings detected ({max_loading:.1f}) - consider different scaling!")
                optimization_working = False
            if max_factor > 1000:
                print(f"   ⚠️  WARNING: Large factors detected ({max_factor:.1f}) - consider different scaling!")
                optimization_working = False
        
        # If optimization isn't working well, run diagnostics
        if not optimization_working:
            print("\n🔬 Running detailed diagnostics...")
            diagnose_optimization_issues(adjusted_counts, params, None, 0.1, data_variance)
            
            # Suggest different scaling methods
            print("\n💡 SUGGESTIONS:")
            print("   Try different scaling methods:")
            print("   - scaling_method='robust' (less sensitive to outliers)")
            print("   - scaling_method='minmax' (scales to [0,1])")
            print("   - scaling_method='none' (no scaling)")
            print("   - Lower sparsity_penalty")
            print("   - Check data for extreme outliers")
        
    else:
        print("❌ Data failed requirements check")