"""
Sparse Matrix Factorization for Neural Data
Author: Claude
Date: 2025

Optimized for background-subtracted neural voxel data with hyperparameter search
and cross-validation using held-out voxels.
"""

import numpy as np
import itertools
from scipy.stats import poisson
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SparseMatrixFactorization:
    """
    Sparse matrix factorization for background-subtracted neural data
    using coordinate ascent with elastic net regularization.
    """
    
    def __init__(self, n_factors, sparsity_penalty=0.1, elastic_net_frac=0.5, 
                 max_num_iters=100, num_coord_ascent_iters=5, 
                 mean_func='svd', random_state=42):
        """
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        sparsity_penalty : float
            L1 regularization strength for sparsity
        elastic_net_frac : float
            Balance between L1 and L2 regularization (0=L2 only, 1=L1 only)
        max_num_iters : int
            Maximum number of alternating minimization iterations
        num_coord_ascent_iters : int
            Number of coordinate ascent steps per factor update
        mean_func : str
            Initialization method ('svd', 'random', 'nmf')
        """
        self.n_factors = n_factors
        self.sparsity_penalty = sparsity_penalty
        self.elastic_net_frac = elastic_net_frac
        self.max_num_iters = max_num_iters
        self.num_coord_ascent_iters = num_coord_ascent_iters
        self.mean_func = mean_func
        self.random_state = random_state
        
        # Computed regularization weights
        self.l1_reg = sparsity_penalty * elastic_net_frac
        self.l2_reg = sparsity_penalty * (1 - elastic_net_frac)
        
    def _initialize_factors(self, data):
        """Initialize loadings and factors matrices"""
        np.random.seed(self.random_state)
        n_obs, n_voxels = data.shape
        
        if self.mean_func == 'svd':
            # Use SVD for initialization
            U, s, Vt = np.linalg.svd(data, full_matrices=False)
            k = min(self.n_factors, len(s))
            self.loadings = U[:, :k] @ np.diag(np.sqrt(s[:k]))
            self.factors = np.diag(np.sqrt(s[:k])) @ Vt[:k, :]
            
            # Pad with random if needed
            if k < self.n_factors:
                extra_load = np.random.randn(n_obs, self.n_factors - k) * 0.1
                extra_fact = np.random.randn(self.n_factors - k, n_voxels) * 0.1
                self.loadings = np.hstack([self.loadings, extra_load])
                self.factors = np.vstack([self.factors, extra_fact])
                
        elif self.mean_func == 'random':
            # Random initialization
            scale = np.sqrt(np.abs(data).mean() / self.n_factors)
            self.loadings = np.random.randn(n_obs, self.n_factors) * scale
            self.factors = np.random.randn(self.n_factors, n_voxels) * scale
            
        elif self.mean_func == 'nmf':
            # Non-negative initialization (useful for count data origins)
            scale = np.sqrt(np.abs(data).mean() / self.n_factors)
            self.loadings = np.abs(np.random.randn(n_obs, self.n_factors)) * scale
            self.factors = np.abs(np.random.randn(self.n_factors, n_voxels)) * scale
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding for L1 regularization"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _update_loadings(self, data, mask=None):
        """Update loadings matrix using coordinate ascent"""
        n_obs, n_voxels = data.shape
        
        for _ in range(self.num_coord_ascent_iters):
            for i in range(n_obs):
                for k in range(self.n_factors):
                    # Compute residual without current factor
                    residual = data[i, :] - self.loadings[i, :] @ self.factors
                    residual += self.loadings[i, k] * self.factors[k, :]
                    
                    # Apply mask if provided
                    if mask is not None:
                        residual = residual * mask[i, :]
                        factor_masked = self.factors[k, :] * mask[i, :]
                        denominator = np.sum(factor_masked ** 2) + self.l2_reg
                    else:
                        factor_masked = self.factors[k, :]
                        denominator = np.sum(self.factors[k, :] ** 2) + self.l2_reg
                    
                    if denominator > 1e-10:
                        numerator = np.sum(residual * factor_masked)
                        # Coordinate ascent update with soft thresholding
                        self.loadings[i, k] = self._soft_threshold(
                            numerator / denominator, self.l1_reg / denominator
                        )
    
    def _update_factors(self, data, mask=None):
        """Update factors matrix using coordinate ascent"""
        n_obs, n_voxels = data.shape
        
        for _ in range(self.num_coord_ascent_iters):
            for k in range(self.n_factors):
                for j in range(n_voxels):
                    # Compute residual without current factor
                    residual = data[:, j] - self.loadings @ self.factors[:, j]
                    residual += self.loadings[:, k] * self.factors[k, j]
                    
                    # Apply mask if provided
                    if mask is not None:
                        residual = residual * mask[:, j]
                        loading_masked = self.loadings[:, k] * mask[:, j]
                        denominator = np.sum(loading_masked ** 2) + self.l2_reg
                    else:
                        loading_masked = self.loadings[:, k]
                        denominator = np.sum(self.loadings[:, k] ** 2) + self.l2_reg
                    
                    if denominator > 1e-10:
                        numerator = np.sum(residual * loading_masked)
                        # Coordinate ascent update with soft thresholding
                        self.factors[k, j] = self._soft_threshold(
                            numerator / denominator, self.l1_reg / denominator
                        )
    
    def _compute_loss(self, data, mask=None):
        """Compute reconstruction loss with regularization"""
        reconstruction = self.loadings @ self.factors
        
        if mask is not None:
            mse_loss = np.sum(((data - reconstruction) * mask) ** 2)
            n_observed = np.sum(mask)
        else:
            mse_loss = np.sum((data - reconstruction) ** 2)
            n_observed = data.size
        
        # Add regularization terms
        l1_penalty = self.l1_reg * (np.sum(np.abs(self.loadings)) + np.sum(np.abs(self.factors)))
        l2_penalty = self.l2_reg * (np.sum(self.loadings ** 2) + np.sum(self.factors ** 2))
        
        total_loss = mse_loss / n_observed + l1_penalty + l2_penalty
        return total_loss, mse_loss / n_observed
    
    def _compute_held_out_loglike(self, data, held_out_mask):
        """Compute log-likelihood on held-out data"""
        reconstruction = self.loadings @ self.factors
        
        # Only compute on held-out voxels
        held_out_data = data[held_out_mask]
        held_out_pred = reconstruction[held_out_mask]
        
        # For background-subtracted Poisson data, we approximate the log-likelihood
        # assuming the reconstructed values represent the true underlying rates
        # We add a small offset to handle negative predictions
        held_out_pred_positive = np.maximum(held_out_pred, 0.1)
        
        # Approximate log-likelihood (handling negative observed values)
        # For negative observed values (due to background subtraction), 
        # we use a Gaussian approximation instead
        loglike = 0
        for obs, pred in zip(held_out_data, held_out_pred_positive):
            if obs >= 0:
                # Use Poisson log-likelihood for non-negative observations
                loglike += poisson.logpmf(int(max(0, obs)), pred)
            else:
                # Use Gaussian approximation for negative observations
                # (background subtraction artifact)
                loglike += -0.5 * ((obs - pred) ** 2) / max(pred, 1.0)
        
        return loglike / len(held_out_data)  # Average log-likelihood
    
    def fit(self, data, mask=None, held_out_mask=None, verbose=False):
        """
        Fit the matrix factorization model
        
        Parameters:
        -----------
        data : array-like, shape (n_obs, n_voxels)
            Input data matrix
        mask : array-like, shape (n_obs, n_voxels), optional
            Mask for training data (1 = use, 0 = ignore)
        held_out_mask : array-like, shape (n_obs, n_voxels), optional
            Mask for held-out validation data (True = held out)
        """
        data = np.array(data)
        self._initialize_factors(data)
        
        # Training mask (inverse of held-out mask)
        if held_out_mask is not None:
            train_mask = ~held_out_mask
            if mask is not None:
                train_mask = train_mask & mask
        else:
            train_mask = mask
        
        losses = []
        held_out_loglikes = []
        
        for iteration in range(self.max_num_iters):
            # Update loadings and factors alternately
            self._update_loadings(data, train_mask)
            self._update_factors(data, train_mask)
            
            # Compute losses
            total_loss, mse_loss = self._compute_loss(data, train_mask)
            losses.append(total_loss)
            
            # Compute held-out log-likelihood if validation data provided
            if held_out_mask is not None:
                held_out_loglike = self._compute_held_out_loglike(data, held_out_mask)
                held_out_loglikes.append(held_out_loglike)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.6f}, MSE = {mse_loss:.6f}")
                if held_out_mask is not None:
                    print(f"  Held-out log-likelihood = {held_out_loglike:.6f}")
        
        self.training_losses_ = losses
        self.held_out_loglikes_ = held_out_loglikes if held_out_mask is not None else None
        
        return self
    
    def transform(self, data=None):
        """Get the loadings (transform data to factor space)"""
        if data is None:
            return self.loadings
        else:
            # For new data, solve for loadings given factors
            # This is a simplified version - in practice you'd want to optimize
            return np.linalg.lstsq(self.factors.T, data.T, rcond=None)[0].T
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data from loadings"""
        if loadings is None:
            loadings = self.loadings
        return loadings @ self.factors
    
    def get_sparsity_metrics(self):
        """Compute sparsity metrics for the fitted model"""
        loading_sparsity = np.mean(np.abs(self.loadings) < 1e-6)
        factor_sparsity = np.mean(np.abs(self.factors) < 1e-6)
        
        return {
            'loading_sparsity': loading_sparsity,
            'factor_sparsity': factor_sparsity,
            'total_params': self.loadings.size + self.factors.size,
            'sparse_params': np.sum(np.abs(self.loadings) < 1e-6) + np.sum(np.abs(self.factors) < 1e-6)
        }


def hyperparameter_search(data, held_out_mask, 
                         sparsity_values=[0.01, 0.1, 1.0],
                         n_factors_values=[5, 10, 15, 20],
                         sparsity_penalty_values=[0.05, 0.1, 0.2],
                         elastic_net_frac=0.5,
                         max_num_iters=50,
                         num_coord_ascent_iters=3,
                         mean_func='svd',
                         random_state=42,
                         verbose=True):
    """
    Hyperparameter search for sparse matrix factorization
    
    Parameters:
    -----------
    data : array-like, shape (n_obs, n_voxels)
        Input neural data
    held_out_mask : array-like, shape (n_obs, n_voxels)
        Boolean mask for held-out validation data
    sparsity_values : list
        L1 regularization strengths to try
    n_factors_values : list
        Numbers of factors to try
    sparsity_penalty_values : list  
        Overall sparsity penalty strengths to try
    
    Returns:
    --------
    dict : Results dictionary with best parameters and all results
    """
    
    results = []
    best_loglike = -np.inf
    best_params = None
    best_model = None
    
    # Create parameter grid
    param_combinations = list(itertools.product(
        sparsity_values, n_factors_values, sparsity_penalty_values
    ))
    
    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, (sparsity, n_factors, sparsity_penalty) in enumerate(param_combinations):
        if verbose:
            print(f"\n[{i+1}/{len(param_combinations)}] Testing: "
                  f"sparsity={sparsity}, n_factors={n_factors}, "
                  f"sparsity_penalty={sparsity_penalty}")
        
        try:
            # Initialize model
            model = SparseMatrixFactorization(
                n_factors=n_factors,
                sparsity_penalty=sparsity_penalty,
                elastic_net_frac=elastic_net_frac,
                max_num_iters=max_num_iters,
                num_coord_ascent_iters=num_coord_ascent_iters,
                mean_func=mean_func,
                random_state=random_state
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
                'final_loss': final_loss,
                'held_out_loglike': final_held_out_loglike,
                'loading_sparsity': sparsity_metrics['loading_sparsity'],
                'factor_sparsity': sparsity_metrics['factor_sparsity'],
                'model': model
            }
            
            results.append(result)
            
            if verbose:
                print(f"  Loss: {final_loss:.6f}, Held-out LogLike: {final_held_out_loglike:.6f}")
                print(f"  Loading sparsity: {sparsity_metrics['loading_sparsity']:.3f}, Factor sparsity: {sparsity_metrics['factor_sparsity']:.3f}")
            
            # Track best model
            if final_held_out_loglike > best_loglike:
                best_loglike = final_held_out_loglike
                best_params = {
                    'sparsity': sparsity,
                    'n_factors': n_factors, 
                    'sparsity_penalty': sparsity_penalty
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


def create_held_out_mask(shape, holdout_fraction=0.1, random_state=42):
    """
    Create a random held-out mask for cross-validation
    
    Parameters:
    -----------
    shape : tuple
        Shape of the data (n_obs, n_voxels)
    holdout_fraction : float
        Fraction of data to hold out
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray : Boolean mask where True = held out
    """
    np.random.seed(random_state)
    return np.random.random(shape) < holdout_fraction


# Utility functions for analysis
def plot_training_progress(model, figsize=(12, 4)):
    """Plot training loss and held-out likelihood"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Training loss
    ax1.plot(model.training_losses_)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    
    # Held-out likelihood
    if model.held_out_loglikes_ is not None:
        ax2.plot(model.held_out_loglikes_)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Held-out Log-likelihood')
        ax2.set_title('Validation Progress')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_factor_patterns(model, n_factors_to_show=4, figsize=(15, 8)):
    """Plot spatial patterns of the learned factors"""
    import matplotlib.pyplot as plt
    
    n_factors_to_show = min(n_factors_to_show, model.n_factors)
    fig, axes = plt.subplots(2, n_factors_to_show//2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_factors_to_show):
        axes[i].plot(model.factors[i, :])
        axes[i].set_title(f'Factor {i+1}')
        axes[i].set_xlabel('Voxel Index')
        axes[i].set_ylabel('Factor Weight')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_sparsity(model):
    """Analyze and print sparsity statistics"""
    metrics = model.get_sparsity_metrics()
    
    print("=== Sparsity Analysis ===")
    print(f"Loading sparsity: {metrics['loading_sparsity']:.1%}")
    print(f"Factor sparsity: {metrics['factor_sparsity']:.1%}")
    print(f"Total parameters: {metrics['total_params']:,}")
    print(f"Near-zero parameters: {metrics['sparse_params']:,}")
    print(f"Effective parameters: {metrics['total_params'] - metrics['sparse_params']:,}")
    
    return metrics