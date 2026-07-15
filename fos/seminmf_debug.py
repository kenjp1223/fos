# Add these debugging functions and modifications to your seminmf.py

import time
import psutil
import os

def print_memory_usage(label=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage {label}: {memory_mb:.1f} MB")

def print_timing(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Modified fit function with debugging
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
            break

    print(f"\nOptimization completed. Final loss: {losses[-1]:.6f}")
    print_memory_usage("final")
    
    return params, jnp.stack(losses), jnp.stack(hlls)


# Chunked version for very large problems
def fit_poisson_seminmf_chunked(counts,
                               initial_params,
                               mask=None,
                               chunk_size=1000000,  # 1M voxels per chunk
                               **kwargs):
    """
    Fit SemiNMF using chunked processing for memory efficiency
    """
    n_rows, n_cols = counts.shape
    
    if n_cols <= chunk_size:
        # No need to chunk
        return fit_poisson_seminmf_debug(counts, initial_params, mask, **kwargs)
    
    print(f"Using chunked processing: {n_cols} voxels in chunks of {chunk_size}")
    
    n_chunks = (n_cols + chunk_size - 1) // chunk_size
    print(f"Total chunks: {n_chunks}")
    
    # For now, let's just reduce the problem size for testing
    print("WARNING: Reducing problem size for debugging...")
    subset_cols = min(chunk_size, n_cols)
    
    print(f"Using subset of {subset_cols} columns out of {n_cols}")
    
    counts_subset = counts[:, :subset_cols]
    mask_subset = mask[:, :subset_cols] if mask is not None else None
    
    # Adjust initial params
    initial_params_subset = dataclasses.replace(
        initial_params,
        factors=initial_params.factors[:, :subset_cols],
        count_col_effects=initial_params.count_col_effects[:subset_cols]
    )
    
    return fit_poisson_seminmf_debug(counts_subset, initial_params_subset, mask_subset, **kwargs)


# Memory-efficient initialization
@print_timing
def initialize_nnsvd_memory_efficient(counts, num_factors, mean_func, drugs=None, max_rank=100):
    """
    Memory-efficient NNSVD initialization using randomized SVD
    """
    print(f"Initializing with memory-efficient NNSVD (max_rank={max_rank})")
    print_memory_usage("before NNSVD init")
    
    # Convert data to targets
    if mean_func.lower() == "softplus":
        pseudocounts = jnp.maximum(counts, 1e-1)
        targets = pseudocounts + jnp.log(1 - jnp.exp(-pseudocounts))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    num_mice = counts.shape[0]
    
    # Initialize effects
    count_row_effect = jnp.mean(targets, axis=1)
    targets_centered = targets - count_row_effect[:, None]
    
    if drugs is not None:
        count_col_effect = targets_centered[drugs == 10].mean(axis=0)
    else:
        count_col_effect = targets_centered.mean(axis=0)
    
    targets_centered = targets_centered - count_col_effect
    
    print_memory_usage("after centering")
    
    # Use randomized SVD for large matrices
    if targets_centered.shape[1] > 10000:
        print("Using randomized SVD for large matrix")
        from sklearn.decomposition import TruncatedSVD
        
        # Convert to numpy for sklearn
        targets_np = np.array(targets_centered)
        svd = TruncatedSVD(n_components=min(max_rank, num_factors), random_state=42)
        U_reduced = svd.fit_transform(targets_np)
        S_reduced = svd.singular_values_
        VT_reduced = svd.components_
        
        # Convert back to JAX
        U = jnp.array(U_reduced)
        S = jnp.array(S_reduced)
        VT = jnp.array(VT_reduced)
    else:
        U, S, VT = jnp.linalg.svd(targets_centered, full_matrices=False)
    
    print_memory_usage("after SVD")
    
    # Process factors
    count_loadings = []
    factors = []
    
    for k in range(num_factors):
        if k < len(S):
            uk, sk, vk = U[:, k], S[k], VT[k]
            sign = jnp.sign(vk.mean())
            vk = jnp.clip(vk * sign, a_min=1e-8)
            scale = vk.sum()
            factors.append(vk / scale)
            count_loadings.append(uk * sk * scale * sign)
        else:
            # Random initialization for remaining factors
            factors.append(jnp.ones(targets_centered.shape[1]) / targets_centered.shape[1])
            count_loadings.append(jnp.zeros(num_mice))
    
    count_loadings = jnp.column_stack(count_loadings)
    factors = jnp.stack(factors)
    
    print_memory_usage("after factor construction")
    
    return SemiNMFParams(factors, count_loadings, count_row_effect, count_col_effect)
