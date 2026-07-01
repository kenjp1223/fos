"""
JAX-Accelerated Sparse Matrix Factorization
Much faster version using JAX JIT compilation and vectorization
"""

import jax.numpy as jnp
import jax
from jax import random, jit, grad, vmap
import numpy as np
from functools import partial
import itertools

class JAXSparseMatrixFactorization:
    """
    JAX-accelerated sparse matrix factorization with JIT compilation
    """
    
    def __init__(self, n_factors, sparsity_penalty=0.1, elastic_net_frac=0.5, 
                 max_num_iters=100, learning_rate=0.01, random_state=42):
        self.n_factors = n_factors
        self.sparsity_penalty = sparsity_penalty
        self.elastic_net_frac = elastic_net_frac
        self.max_num_iters = max_num_iters
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Regularization weights
        self.l1_reg = sparsity_penalty * elastic_net_frac
        self.l2_reg = sparsity_penalty * (1 - elastic_net_frac)
        
        # JAX random key
        self.key = random.PRNGKey(random_state)
    
    def _initialize_factors(self, data):
        """Initialize factors using JAX random"""
        n_obs, n_voxels = data.shape
        key1, key2 = random.split(self.key)
        
        # Scale based on data variance
        scale = jnp.sqrt(jnp.abs(data).mean() / self.n_factors)
        
        loadings = random.normal(key1, (n_obs, self.n_factors)) * scale
        factors = random.normal(key2, (self.n_factors, n_voxels)) * scale
        
        return loadings, factors
    
    @partial(jit, static_argnums=(0,))
    def _compute_loss_and_grads(self, loadings, factors, data, train_mask):
        """
        Compute loss and gradients - JIT compiled for speed
        """
        # Reconstruction
        reconstruction = loadings @ factors
        
        # Masked reconstruction error
        if train_mask is not None:
            error = (data - reconstruction) * train_mask
            mse_loss = jnp.sum(error ** 2) / jnp.sum(train_mask)
        else:
            error = data - reconstruction
            mse_loss = jnp.mean(error ** 2)
        
        # Regularization terms
        l1_penalty = self.l1_reg * (jnp.sum(jnp.abs(loadings)) + jnp.sum(jnp.abs(factors)))
        l2_penalty = self.l2_reg * (jnp.sum(loadings ** 2) + jnp.sum(factors ** 2))
        
        total_loss = mse_loss + l1_penalty + l2_penalty
        
        return total_loss, mse_loss
    
    @partial(jit, static_argnums=(0,))
    def _soft_threshold(self, x, threshold):
        """Vectorized soft thresholding"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)
    
    @partial(jit, static_argnums=(0,))
    def _update_step(self, loadings, factors, data, train_mask):
        """
        Single optimization step - JIT compiled
        Uses proximal gradient descent instead of coordinate ascent
        """
        # Compute gradients
        def loss_fn(loadings, factors):
            reconstruction = loadings @ factors
            if train_mask is not None:
                error = (data - reconstruction) * train_mask
                mse_loss = jnp.sum(error ** 2) / jnp.sum(train_mask)
            else:
                error = data - reconstruction
                mse_loss = jnp.mean(error ** 2)
            
            l2_penalty = self.l2_reg * (jnp.sum(loadings ** 2) + jnp.sum(factors ** 2))
            return mse_loss + l2_penalty
        
        # Compute gradients w.r.t. loadings and factors
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_loadings, grad_factors = grad_fn(loadings, factors)
        
        # Gradient descent updates
        new_loadings = loadings - self.learning_rate * grad_loadings
        new_factors = factors - self.learning_rate * grad_factors
        
        # Apply soft thresholding for L1 sparsity
        l1_threshold = self.learning_rate * self.l1_reg
        new_loadings = self._soft_threshold(new_loadings, l1_threshold)
        new_factors = self._soft_threshold(new_factors, l1_threshold)
        
        return new_loadings, new_factors
    
    @partial(jit, static_argnums=(0,))
    def _compute_held_out_loglike(self, loadings, factors, data, held_out_mask):
        """Compute held-out log-likelihood - JIT compiled"""
        reconstruction = loadings @ factors
        
        # Extract held-out data
        held_out_data = jnp.where(held_out_mask, data, 0)
        held_out_pred = jnp.where(held_out_mask, reconstruction, 0)
        
        # Count held-out points
        n_held_out = jnp.sum(held_out_mask)
        
        # Approximate log-likelihood
        # For positive values: Poisson-like
        # For negative values: Gaussian-like
        positive_mask = held_out_data >= 0
        negative_mask = held_out_data < 0
        
        # Positive values (Poisson approximation)
        pos_pred = jnp.maximum(held_out_pred, 0.1)
        pos_loglike = jnp.where(
            positive_mask & held_out_mask,
            held_out_data * jnp.log(pos_pred) - pos_pred,
            0
        )
        
        # Negative values (Gaussian approximation)
        neg_loglike = jnp.where(
            negative_mask & held_out_mask,
            -0.5 * (held_out_data - held_out_pred) ** 2 / jnp.maximum(jnp.abs(held_out_pred), 1.0),
            0
        )
        
        total_loglike = jnp.sum(pos_loglike + neg_loglike) / jnp.maximum(n_held_out, 1)
        return total_loglike
    
    def fit(self, data, mask=None, held_out_mask=None, verbose=False):
        """
        Fit the model using JAX-accelerated optimization
        """
        # Convert to JAX arrays
        data = jnp.array(data)
        
        # Create training mask
        if held_out_mask is not None:
            held_out_mask = jnp.array(held_out_mask)
            train_mask = ~held_out_mask
            if mask is not None:
                train_mask = train_mask & jnp.array(mask)
        else:
            train_mask = jnp.array(mask) if mask is not None else None
        
        # Initialize factors
        loadings, factors = self._initialize_factors(data)
        
        losses = []
        held_out_loglikes = []
        
        for iteration in range(self.max_num_iters):
            # Update parameters
            loadings, factors = self._update_step(loadings, factors, data, train_mask)
            
            # Compute losses
            total_loss, mse_loss = self._compute_loss_and_grads(loadings, factors, data, train_mask)
            losses.append(float(total_loss))
            
            # Compute held-out log-likelihood
            if held_out_mask is not None:
                held_out_loglike = self._compute_held_out_loglike(loadings, factors, data, held_out_mask)
                held_out_loglikes.append(float(held_out_loglike))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.6f}, MSE = {mse_loss:.6f}")
                if held_out_mask is not None:
                    print(f"  Held-out log-likelihood = {held_out_loglike:.6f}")
        
        # Store results (convert back to numpy for compatibility)
        self.loadings = np.array(loadings)
        self.factors = np.array(factors)
        self.training_losses_ = losses
        self.held_out_loglikes_ = held_out_loglikes if held_out_mask is not None else None
        
        return self
    
    def transform(self, data=None):
        """Get loadings"""
        return self.loadings if data is None else np.linalg.lstsq(self.factors.T, data.T, rcond=None)[0].T
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data"""
        if loadings is None:
            loadings = self.loadings
        return loadings @ self.factors
    
    def get_sparsity_metrics(self):
        """Compute sparsity metrics"""
        loading_sparsity = np.mean(np.abs(self.loadings) < 1e-6)
        factor_sparsity = np.mean(np.abs(self.factors) < 1e-6)
        
        return {
            'loading_sparsity': loading_sparsity,
            'factor_sparsity': factor_sparsity,
            'total_params': self.loadings.size + self.factors.size,
            'sparse_params': np.sum(np.abs(self.loadings) < 1e-6) + np.sum(np.abs(self.factors) < 1e-6)
        }


def jax_hyperparameter_search(data, held_out_mask, 
                              sparsity_values=[0.01, 0.1, 1.0],
                              n_factors_values=[5, 10, 15, 20],
                              sparsity_penalty_values=[0.05, 0.1, 0.2],
                              learning_rates=[0.001, 0.01, 0.1],
                              elastic_net_frac=0.5,
                              max_num_iters=50,
                              random_state=42,
                              verbose=True):
    """
    JAX-accelerated hyperparameter search
    """
    results = []
    best_loglike = -np.inf
    best_params = None
    best_model = None
    
    # Create parameter grid
    param_combinations = list(itertools.product(
        sparsity_values, n_factors_values, sparsity_penalty_values, learning_rates
    ))
    
    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations with JAX acceleration...")
    
    for i, (sparsity, n_factors, sparsity_penalty, lr) in enumerate(param_combinations):
        if verbose:
            print(f"\n[{i+1}/{len(param_combinations)}] Testing: "
                  f"n_factors={n_factors}, sparsity_penalty={sparsity_penalty:.3f}, lr={lr}")
        
        try:
            # Initialize JAX model
            model = JAXSparseMatrixFactorization(
                n_factors=n_factors,
                sparsity_penalty=sparsity_penalty,
                elastic_net_frac=elastic_net_frac,
                max_num_iters=max_num_iters,
                learning_rate=lr,
                random_state=random_state + i  # Different seed for each run
            )
            
            # Fit model
            model.fit(data, held_out_mask=held_out_mask, verbose=False)
            
            # Get final metrics
            final_loss = model.training_losses_[-1]
            final_held_out_loglike = model.held_out_loglikes_[-1]
            
            # Compute sparsity metrics
            sparsity_metrics = model.get_sparsity_metrics()
            
            result = {
                'sparsity': sparsity,
                'n_factors': n_factors,
                'sparsity_penalty': sparsity_penalty,
                'learning_rate': lr,
                'final_loss': final_loss,
                'held_out_loglike': final_held_out_loglike,
                'loading_sparsity': sparsity_metrics['loading_sparsity'],
                'factor_sparsity': sparsity_metrics['factor_sparsity'],
                'model': model
            }
            
            results.append(result)
            
            if verbose:
                print(f"  Loss: {final_loss:.6f}, Held-out LogLike: {final_held_out_loglike:.6f}")
                print(f"  Sparsity: L={sparsity_metrics['loading_sparsity']:.3f}, F={sparsity_metrics['factor_sparsity']:.3f}")
            
            # Track best model
            if final_held_out_loglike > best_loglike:
                best_loglike = final_held_out_loglike
                best_params = {
                    'sparsity': sparsity,
                    'n_factors': n_factors, 
                    'sparsity_penalty': sparsity_penalty,
                    'learning_rate': lr
                }
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"  Failed: {str(e)}")
            continue
    
    if verbose:
        print(f"\nBest parameters: {best_params}")
        print(f"Best held-out log-likelihood: {best_loglike:.6f}")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_loglike': best_loglike,
        'all_results': results
    }


# Utility function to check if JAX can use GPU
def check_jax_devices():
    """Check available JAX devices"""
    devices = jax.devices()
    print("Available JAX devices:")
    for device in devices:
        print(f"  {device}")
    
    if any('gpu' in str(device).lower() for device in devices):
        print("🚀 GPU acceleration available!")
    else:
        print("💻 Using CPU (still much faster with JIT)")
    
    return devices


# Example usage comparison
def compare_implementations(data, held_out_mask, n_factors=8, max_iters=20):
    """Compare JAX vs NumPy implementations"""
    import time
    
    print("=== Implementation Comparison ===")
    
    # JAX version
    print("Testing JAX implementation...")
    start_time = time.time()
    jax_model = JAXSparseMatrixFactorization(
        n_factors=n_factors,
        max_num_iters=max_iters,
        learning_rate=0.01
    )
    jax_model.fit(data, held_out_mask=held_out_mask, verbose=False)
    jax_time = time.time() - start_time
    
    # Original NumPy version (from your original code)
    print("Testing NumPy implementation...")
    start_time = time.time()
    from sparse_neural_factorization import SparseMatrixFactorization
    numpy_model = SparseMatrixFactorization(
        n_factors=n_factors,
        max_num_iters=max_iters,
        mean_func='random'
    )
    numpy_model.fit(data, held_out_mask=held_out_mask, verbose=False)
    numpy_time = time.time() - start_time
    
    print(f"\nSpeed Comparison:")
    print(f"JAX time: {jax_time:.2f} seconds")
    print(f"NumPy time: {numpy_time:.2f} seconds")
    print(f"Speedup: {numpy_time/jax_time:.1f}x faster with JAX")
    
    return jax_model, numpy_model