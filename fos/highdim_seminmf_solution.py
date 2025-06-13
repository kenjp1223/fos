# ===== HIGH-DIMENSIONAL SEMI-NMF SOLUTION =====
# For neuroimaging data with millions of voxels

import jax.numpy as jnp
import jax.random as jr
import dataclasses
from jax import vmap, lax
import adjusted_counts_seminmf as seminmf
from importlib import reload

# Reload module to get your updates
reload(seminmf)

# ===== STEP 1: ADD THESE FUNCTIONS TO YOUR NAMESPACE =====

def initialize_sparse_seminmf(counts, num_factors, sparsity_percent=1.0):
    """
    Initialize Semi-NMF with sparse factors for high-dimensional data.
    
    Args:
        counts: Data matrix (mice × voxels)
        num_factors: Number of factors
        sparsity_percent: Percentage of voxels to be active in each factor (default 1%)
    """
    num_mice, num_voxels = counts.shape
    print(f"\n🧠 Initializing sparse Semi-NMF for {num_voxels:,} voxels")
    print(f"   Target sparsity: {sparsity_percent:.1f}% active voxels per factor")
    
    # Compute row/column effects
    row_effects = jnp.mean(counts, axis=1)
    col_centered = counts - row_effects[:, None]
    col_effects = jnp.mean(col_centered, axis=0)
    residuals = col_centered - col_effects
    
    # Initialize sparse factors
    key = jr.PRNGKey(42)
    factors = []
    count_loadings = []
    
    # Number of active voxels per factor
    num_active = max(100, int(num_voxels * sparsity_percent / 100))
    print(f"   {num_active:,} active voxels per factor")
    
    for k in range(num_factors):
        subkey = jr.fold_in(key, k)
        
        # Strategy 1: Random sparse initialization
        factor = jnp.zeros(num_voxels)
        
        # Select random voxels to be active
        active_idx = jr.choice(subkey, num_voxels, shape=(num_active,), replace=False)
        
        # Initialize active voxels with reasonable values
        active_vals = jr.uniform(jr.fold_in(subkey, 1), shape=(num_active,), 
                                minval=0.5, maxval=2.0)
        factor = factor.at[active_idx].set(active_vals)
        
        # Don't normalize! Keep natural scale
        factors.append(factor)
        
        # Initialize loadings
        loading = jr.normal(jr.fold_in(subkey, 2), (num_mice,)) * 0.1
        count_loadings.append(loading)
    
    factors = jnp.stack(factors)
    count_loadings = jnp.column_stack(count_loadings)
    
    # Print statistics
    print(f"\n📊 Initialization statistics:")
    print(f"   Factors shape: {factors.shape}")
    print(f"   Non-zero elements per factor: {[int(jnp.sum(f > 0)) for f in factors[:5]]}...")
    print(f"   Factor value range: [{float(jnp.min(factors[factors > 0])):.4f}, {float(jnp.max(factors)):.4f}]")
    print(f"   Factor mean (non-zero): {float(jnp.mean(factors[factors > 0])):.4f}")
    print(f"   Loadings range: [{float(jnp.min(count_loadings)):.4f}, {float(jnp.max(count_loadings)):.4f}]")
    
    return seminmf.AdjustedCountsSemiNMFParams(
        factors=factors,
        count_loadings=count_loadings,
        count_row_effects=row_effects,
        count_col_effects=col_effects
    )


def update_factors_no_norm(quad_approx, params):
    """
    Update factors WITHOUT normalization - better for high-dimensional data.
    This replaces the standard update_factors function.
    """
    def _update_one_column(hc_n, Jc_n, factor_n):
        def _update_one_coord(hc_n, args):
            factor_nk, count_loading_k = args
            
            # Coordinate descent update
            numerator = jnp.einsum('m,m->', count_loading_k, (hc_n + Jc_n * factor_nk * count_loading_k))
            denominator = jnp.einsum('m,m,m->', Jc_n, count_loading_k, count_loading_k)
            
            # Apply non-negativity constraint
            denominator = jnp.maximum(denominator, 1e-8)
            new_factor_nk = jnp.maximum(numerator, 0.0) / denominator
            
            # Allow true zeros (don't force minimum)
            # This maintains sparsity
            
            # JAX-safe NaN protection
            new_factor_nk = jnp.where(jnp.isnan(new_factor_nk), factor_nk, new_factor_nk)
            
            # Clip extreme values
            new_factor_nk = jnp.clip(new_factor_nk, 0.0, 100.0)
            
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
    
    # NO NORMALIZATION!
    # Just ensure non-negativity and clip extremes
    factors = jnp.maximum(factors, 0.0)
    factors = jnp.minimum(factors, 100.0)  # Prevent explosion
    
    params = dataclasses.replace(params, factors=factors)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts)
    
    return quad_approx, params


# ===== STEP 2: MONKEY-PATCH THE UPDATE FUNCTION =====
# Replace the normalization-based update with the no-norm version
seminmf.update_factors = update_factors_no_norm

# ===== STEP 3: RUN YOUR FACTORIZATION =====

# Check your data dimensions
num_mice, num_voxels = counts.shape
print(f"\n🔍 Data check:")
print(f"   Shape: {num_mice} mice × {num_voxels:,} voxels")
print(f"   Data range: [{float(jnp.min(counts)):.2f}, {float(jnp.max(counts)):.2f}]")
print(f"   Data sparsity: {float(jnp.mean(counts == 0)):.1%} zeros")

# Initialize with sparse factors
print(f"\n🚀 Initializing {best_num_factors} sparse factors...")
initial_params = initialize_sparse_seminmf(counts, best_num_factors, sparsity_percent=1.0)

# Create mask for evaluation
mask = jr.uniform(jr.PRNGKey(42), counts.shape) > 0.1  # 10% held out

# Fit the model
print(f"\n🔧 Fitting Semi-NMF model...")
params, losses, heldout_loglikes, debug_info, scale_params = \
    seminmf.fit_adjusted_counts_seminmf(
        counts,
        initial_params,
        mask=mask,
        sparsity_penalty=best_sparsity_penalty,
        elastic_net_frac=0.5,  # Mixed is more stable
        num_iters=500,
        num_coord_ascent_iters=10,  # Important: use at least 5-10
        tolerance=1e-5,
        data_variance=estimated_data_variance,
        debug=False,
        scaling_method="none"  # Don't scale for high-dim data
    )

# ===== STEP 4: ANALYZE RESULTS =====
print("\n" + "="*60)
print("RESULTS ANALYSIS")
print("="*60)

print(f"\n📈 Convergence:")
print(f"   Initial loss: {losses[0]:.4f}")
print(f"   Final loss: {losses[-1]:.4f}")
print(f"   Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
print(f"   Initial HLL: {heldout_loglikes[0]:.4f}")
print(f"   Final HLL: {heldout_loglikes[-1]:.4f}")
print(f"   HLL improvement: {heldout_loglikes[-1] - heldout_loglikes[0]:.4f}")

print(f"\n📊 Final parameters:")
print(f"   Factors range: [{float(jnp.min(params.factors[params.factors > 0])):.6f}, {float(jnp.max(params.factors)):.6f}]")
print(f"   Factors sparsity: {float(jnp.mean(params.factors == 0)):.1%} zeros")
print(f"   Loadings range: [{float(jnp.min(params.count_loadings)):.4f}, {float(jnp.max(params.count_loadings)):.4f}]")

# Check factor sparsity patterns
print(f"\n🔍 Factor sparsity analysis:")
for i in range(min(5, best_num_factors)):
    n_active = int(jnp.sum(params.factors[i] > 0.01))
    max_val = float(jnp.max(params.factors[i]))
    print(f"   Factor {i}: {n_active:,} active voxels (max value: {max_val:.3f})")

# ===== OPTIONAL: DIMENSIONALITY REDUCTION FIRST =====
print("\n" + "="*60)
print("💡 SUGGESTIONS FOR VERY HIGH-DIMENSIONAL DATA")
print("="*60)

print("""
If you're still having issues with 24M voxels, consider:

1. **Spatial downsampling**: Reduce resolution first
   ```python
   # Example: Average every 2x2x2 voxel block
   downsampled_counts = downsample_3d(counts, factor=2)
   ```

2. **Mask to brain regions only**: Remove background voxels
   ```python
   brain_mask = counts.var(axis=0) > threshold
   masked_counts = counts[:, brain_mask]
   ```

3. **Use subset for initialization**: Initialize on subset, then expand
   ```python
   subset_idx = jr.choice(key, num_voxels, shape=(100000,))
   subset_counts = counts[:, subset_idx]
   # Initialize on subset, then expand factors
   ```

4. **Increase sparsity**: Use even sparser factors (0.1% active)
   ```python
   initial_params = initialize_sparse_seminmf(counts, best_num_factors, sparsity_percent=0.1)
   ```

5. **Consider using randomized SVD** (if you have sklearn):
   ```python
   from sklearn.decomposition import TruncatedSVD
   svd = TruncatedSVD(n_components=50)
   reduced_data = svd.fit_transform(counts.T).T
   ```
""")

# ===== SAVE RESULTS =====
print(f"\n💾 Saving results...")
results = {
    'params': params,
    'losses': losses,
    'heldout_loglikes': heldout_loglikes,
    'scale_params': scale_params,
    'sparsity': float(jnp.mean(params.factors == 0)),
    'num_factors': best_num_factors,
    'num_voxels': num_voxels
}

# Save as needed
# jnp.savez('seminmf_results.npz', **results)
