"""
Non-negative Matrix Factorization with Coordinate Descent
- Non-negative factors (handles negative data)
- Single data type (no dual modeling)
- Coordinate descent optimization
- Baseline row/column effects
- NNSVD initialization option
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

class CoordinateDescentNMF(nn.Module):
    """
    Non-negative Matrix Factorization with coordinate descent optimization
    Handles background-subtracted neural data with baseline effects
    """
    
    def __init__(self, n_factors, sparsity_penalty=0.1, elastic_net_frac=0.5,
                 max_num_iters=50, num_coord_ascent_iters=5, 
                 device='auto', random_state=42):
        super().__init__()
        
        self.n_factors = n_factors
        self.sparsity_penalty = sparsity_penalty
        self.elastic_net_frac = elastic_net_frac
        self.max_num_iters = max_num_iters
        self.num_coord_ascent_iters = num_coord_ascent_iters
        self.random_state = random_state
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        
        # Will be initialized in fit()
        self.factors = None
        self.loadings = None  
        self.row_effects = None
        self.col_effects = None
        
        # Training history
        self.training_losses_ = None
        self.held_out_loglikes_ = None
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding for L1 regularization"""
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)
    
    def _initialize_parameters(self, data_shape, init_method='nnsvd', data=None):
        """Initialize factors, loadings, and effects"""
        n_obs, n_voxels = data_shape
        
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
        
        if init_method == 'nnsvd' and data is not None:
            self._initialize_nnsvd(data)
        elif init_method == 'random':
            self._initialize_random(data_shape)
        else:
            self._initialize_random(data_shape)
    
    def _initialize_random(self, data_shape):
        """Random initialization"""
        n_obs, n_voxels = data_shape
        
        # Non-negative factors, normalized
        self.factors = torch.rand(self.n_factors, n_voxels, device=self.device)
        self.factors = self.factors / (self.factors.sum(dim=1, keepdim=True) + 1e-8)
        
        # Loadings (can be negative for background-subtracted data)
        self.loadings = torch.randn(n_obs, self.n_factors, device=self.device) * 0.1
        
        # Baseline effects
        self.row_effects = torch.zeros(n_obs, device=self.device)
        self.col_effects = torch.zeros(n_voxels, device=self.device)
    
    def _initialize_nnsvd(self, data):
        """Non-negative SVD initialization"""
        print("Initializing with NNSVD...")
        
        with torch.no_grad():
            # Convert to numpy for SVD
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
            else:
                data_np = np.array(data)
            
            n_obs, n_voxels = data_np.shape
            
            # Initialize row effects (mean of each observation)
            self.row_effects = torch.FloatTensor(data_np.mean(axis=1)).to(self.device)
            data_centered = data_np - self.row_effects.cpu().numpy()[:, None]
            
            # Initialize column effects (mean across observations)
            self.col_effects = torch.FloatTensor(data_centered.mean(axis=0)).to(self.device)
            data_residual = data_centered - self.col_effects.cpu().numpy()
            
            # SVD of residual
            try:
                U, S, Vt = np.linalg.svd(data_residual, full_matrices=False)
            except np.linalg.LinAlgError:
                print("SVD failed, using random initialization")
                self._initialize_random((n_obs, n_voxels))
                return
            
            # Extract factors and loadings from SVD
            factors_list = []
            loadings_list = []
            
            for k in range(min(self.n_factors, len(S))):
                # Get k-th component
                uk = U[:, k]
                sk = S[k]
                vk = Vt[k, :]
                
                # Make factors non-negative by flipping sign if needed
                if vk.mean() < 0:
                    vk = -vk
                    uk = -uk
                
                # Ensure factors are non-negative and normalize
                vk_pos = np.maximum(vk, 1e-8)
                vk_norm = vk_pos / (vk_pos.sum() + 1e-8)
                
                factors_list.append(vk_norm)
                loadings_list.append(uk * sk * vk_pos.sum())
            
            # Pad if we need more factors
            while len(factors_list) < self.n_factors:
                factors_list.append(np.ones(n_voxels) / n_voxels)
                loadings_list.append(np.zeros(n_obs))
            
            # Convert to tensors
            self.factors = torch.FloatTensor(np.array(factors_list)).to(self.device)
            self.loadings = torch.FloatTensor(np.array(loadings_list).T).to(self.device)
            
            print(f"NNSVD initialization complete: {self.n_factors} factors")
    
    def _forward(self):
        """Forward pass: compute reconstruction"""
        return (self.row_effects.unsqueeze(1) + 
                self.col_effects.unsqueeze(0) + 
                torch.mm(self.loadings, self.factors))
    
    def _compute_loss(self, data, train_mask=None):
        """Compute reconstruction loss + regularization"""
        reconstruction = self._forward()
        
        # Reconstruction loss (MSE for background-subtracted data)
        if train_mask is not None:
            error = (data - reconstruction) * train_mask.float()
            mse_loss = torch.sum(error ** 2) / torch.sum(train_mask.float())
        else:
            error = data - reconstruction
            mse_loss = torch.mean(error ** 2)
        
        # Elastic net regularization on loadings
        l1_penalty = self.elastic_net_frac * self.sparsity_penalty * torch.sum(torch.abs(self.loadings))
        l2_penalty = 0.5 * (1 - self.elastic_net_frac) * self.sparsity_penalty * torch.sum(self.loadings ** 2)
        
        total_loss = mse_loss + l1_penalty + l2_penalty
        return total_loss, mse_loss.item()
    
    def _compute_held_out_loglike(self, data, held_out_mask):
        """Compute held-out log-likelihood"""
        if held_out_mask is None:
            return 0.0
        
        with torch.no_grad():
            reconstruction = self._forward()
            
            # Extract held-out data
            held_out_data = data[held_out_mask]
            held_out_pred = reconstruction[held_out_mask]
            
            if len(held_out_data) == 0:
                return 0.0
            
            # Gaussian log-likelihood for background-subtracted data
            residual = held_out_data - held_out_pred
            # Use empirical variance for more robust likelihood
            variance = torch.var(residual) + 1e-8
            log_likelihood = -0.5 * torch.sum(residual ** 2) / variance
            log_likelihood -= 0.5 * len(held_out_data) * torch.log(2 * np.pi * variance)
            
            return log_likelihood.item() / len(held_out_data)
    
    def _update_loadings(self, data, train_mask=None):
        """Coordinate descent update for loadings"""
        n_obs, n_voxels = data.shape
        
        # Compute residual without loadings contribution
        baseline = self.row_effects.unsqueeze(1) + self.col_effects.unsqueeze(0)
        
        for _ in range(self.num_coord_ascent_iters):
            for i in range(n_obs):
                for k in range(self.n_factors):
                    # Current residual for observation i
                    current_recon = baseline[i] + torch.mm(self.loadings[i:i+1], self.factors)[0]
                    residual = data[i] - current_recon + self.loadings[i, k] * self.factors[k]
                    
                    # Apply mask if provided
                    if train_mask is not None:
                        residual = residual * train_mask[i].float()
                        factor_masked = self.factors[k] * train_mask[i].float()
                        denominator = torch.sum(factor_masked ** 2) + (1 - self.elastic_net_frac) * self.sparsity_penalty
                    else:
                        factor_masked = self.factors[k]
                        denominator = torch.sum(self.factors[k] ** 2) + (1 - self.elastic_net_frac) * self.sparsity_penalty
                    
                    if denominator > 1e-10:
                        numerator = torch.sum(residual * factor_masked)
                        # Soft thresholding for L1 regularization
                        threshold = self.elastic_net_frac * self.sparsity_penalty
                        self.loadings[i, k] = self._soft_threshold(numerator / denominator, threshold / denominator)
    
    def _update_factors(self, data, train_mask=None):
        """Coordinate descent update for factors (enforce non-negativity)"""
        n_obs, n_voxels = data.shape
        
        # Compute baseline
        baseline = self.row_effects.unsqueeze(1) + self.col_effects.unsqueeze(0)
        
        for _ in range(self.num_coord_ascent_iters):
            for k in range(self.n_factors):
                for j in range(n_voxels):
                    # Current residual for voxel j
                    current_recon = baseline[:, j] + torch.mv(self.loadings, self.factors[:, j])
                    residual = data[:, j] - current_recon + self.loadings[:, k] * self.factors[k, j]
                    
                    # Apply mask if provided
                    if train_mask is not None:
                        residual = residual * train_mask[:, j].float()
                        loading_masked = self.loadings[:, k] * train_mask[:, j].float()
                        denominator = torch.sum(loading_masked ** 2)
                    else:
                        loading_masked = self.loadings[:, k]
                        denominator = torch.sum(self.loadings[:, k] ** 2)
                    
                    if denominator > 1e-10:
                        numerator = torch.sum(residual * loading_masked)
                        # Non-negative constraint for factors
                        self.factors[k, j] = torch.clamp(numerator / denominator, min=1e-8)
            
            # Normalize factors after each sweep
            factor_sums = self.factors.sum(dim=1, keepdim=True) + 1e-8
            self.factors = self.factors / factor_sums
            self.loadings = self.loadings * factor_sums.T
    
    def _update_row_effects(self, data, train_mask=None):
        """Update row effects (baseline for each observation)"""
        n_obs, n_voxels = data.shape
        
        for i in range(n_obs):
            # Compute residual without row effect
            factor_contrib = torch.mm(self.loadings[i:i+1], self.factors)[0]
            residual = data[i] - self.col_effects - factor_contrib + self.row_effects[i]
            
            if train_mask is not None:
                residual = residual * train_mask[i].float()
                denominator = torch.sum(train_mask[i].float())
            else:
                denominator = n_voxels
            
            if denominator > 0:
                self.row_effects[i] = torch.sum(residual) / denominator
    
    def _update_col_effects(self, data, train_mask=None):
        """Update column effects (baseline for each voxel)"""
        n_obs, n_voxels = data.shape
        
        for j in range(n_voxels):
            # Compute residual without column effect
            factor_contrib = torch.mv(self.loadings, self.factors[:, j])
            residual = data[:, j] - self.row_effects - factor_contrib + self.col_effects[j]
            
            if train_mask is not None:
                residual = residual * train_mask[:, j].float()
                denominator = torch.sum(train_mask[:, j].float())
            else:
                denominator = n_obs
            
            if denominator > 0:
                self.col_effects[j] = torch.sum(residual) / denominator
        
        # Center column effects (remove mean)
        col_mean = self.col_effects.mean()
        self.col_effects = self.col_effects - col_mean
        self.row_effects = self.row_effects + col_mean
    
    def fit(self, data, mask=None, held_out_mask=None, init_method='nnsvd', verbose=False):
        """
        Fit the coordinate descent NMF model
        
        Parameters:
        -----------
        data : array-like, shape (n_obs, n_voxels)
            Input neural data matrix (can contain negative values)
        mask : array-like, shape (n_obs, n_voxels), optional
            Training mask (True = use for training)
        held_out_mask : array-like, shape (n_obs, n_voxels), optional
            Held-out validation mask (True = held out for validation)
        init_method : str
            Initialization method ('nnsvd' or 'random')
        verbose : bool
            Whether to print training progress
        """
        
        # Convert to tensors
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        # Process masks
        if held_out_mask is not None:
            if isinstance(held_out_mask, np.ndarray):
                held_out_mask = torch.BoolTensor(held_out_mask).to(self.device)
            else:
                held_out_mask = held_out_mask.to(self.device)
            
            train_mask = ~held_out_mask
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.BoolTensor(mask).to(self.device)
                train_mask = train_mask & mask
        else:
            train_mask = torch.BoolTensor(mask).to(self.device) if mask is not None else None
        
        # Initialize parameters
        self._initialize_parameters(data.shape, init_method, data)
        
        # Training metrics
        losses = []
        held_out_loglikes = []
        
        if verbose:
            print(f"Training Coordinate Descent NMF:")
            print(f"  Data: {data.shape[0]}×{data.shape[1]} → {self.n_factors} factors")
            print(f"  Initialization: {init_method}")
            print(f"  Device: {self.device}")
            
            # Adjust iterations for large datasets
            if data.numel() > 5e6:  # > 5M elements
                if self.num_coord_ascent_iters > 2:
                    print(f"  Large dataset detected - reducing coord ascent iters from {self.num_coord_ascent_iters} to 2")
                    self.num_coord_ascent_iters = 2
                if self.max_num_iters > 20:
                    print(f"  Large dataset detected - reducing max iters from {self.max_num_iters} to 20")
                    self.max_num_iters = 20
        
        # Training loop
        for iteration in range(self.max_num_iters):
            iter_start = time.time()
            
            if verbose:
                print(f"Iter {iteration:3d}: ", end="", flush=True)
            
            # Coordinate descent updates with progress
            if verbose:
                print("Loadings...", end="", flush=True)
            self._update_loadings(data, train_mask)
            
            if verbose:
                print("Row effects...", end="", flush=True)
            self._update_row_effects(data, train_mask)
            
            if verbose:
                print("Factors...", end="", flush=True)
            self._update_factors(data, train_mask)
            
            if verbose:
                print("Col effects...", end="", flush=True)
            self._update_col_effects(data, train_mask)
            
            # Compute metrics
            total_loss, mse_loss = self._compute_loss(data, train_mask)
            losses.append(total_loss.item())
            
            if held_out_mask is not None:
                held_out_loglike = self._compute_held_out_loglike(data, held_out_mask)
                held_out_loglikes.append(held_out_loglike)
            
            iter_time = time.time() - iter_start
            
            if verbose:
                print(f" Loss = {total_loss.item():.6f}, MSE = {mse_loss:.6f}, Time = {iter_time:.1f}s")
                if held_out_mask is not None:
                    print(f"          Held-out LogLike = {held_out_loglike:.6f}")
                    
                # Show memory usage for large datasets
                if torch.cuda.is_available() and data.numel() > 1e6:
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    print(f"          GPU Memory: {mem_used:.1f} GB")
        
        # Store training history
        self.training_losses_ = losses
        self.held_out_loglikes_ = held_out_loglikes if held_out_mask is not None else None
        
        if verbose:
            print(f"Training complete!")
            print(f"Final loss: {losses[-1]:.6f}")
            self._print_factor_stats()
        
        return self
    
    def _print_factor_stats(self):
        """Print statistics about learned factors"""
        with torch.no_grad():
            print(f"\nFactor Statistics:")
            for k in range(self.n_factors):
                factor_norm = torch.norm(self.factors[k])
                factor_max = torch.max(self.factors[k])
                factor_min = torch.min(self.factors[k])
                loading_norm = torch.norm(self.loadings[:, k])
                print(f"  Factor {k+1}: norm={factor_norm:.3f}, range=[{factor_min:.3f}, {factor_max:.3f}], loading_norm={loading_norm:.3f}")
    
    def transform(self, data=None):
        """Get loadings"""
        with torch.no_grad():
            return self.loadings.cpu().numpy()
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data from loadings"""
        with torch.no_grad():
            if loadings is None:
                reconstruction = self._forward()
            else:
                if isinstance(loadings, np.ndarray):
                    loadings = torch.FloatTensor(loadings).to(self.device)
                reconstruction = (self.row_effects.unsqueeze(1) + 
                                self.col_effects.unsqueeze(0) + 
                                torch.mm(loadings, self.factors))
            
            return reconstruction.cpu().numpy()
    
    def get_components(self):
        """Get all learned components"""
        with torch.no_grad():
            return {
                'factors': self.factors.cpu().numpy(),        # (n_factors, n_voxels) - non-negative
                'loadings': self.loadings.cpu().numpy(),      # (n_obs, n_factors) - can be negative
                'row_effects': self.row_effects.cpu().numpy(), # (n_obs,) - baseline per observation
                'col_effects': self.col_effects.cpu().numpy(), # (n_voxels,) - baseline per voxel
            }
    
    def get_sparsity_metrics(self):
        """Compute sparsity metrics"""
        with torch.no_grad():
            loadings_np = self.loadings.cpu().numpy()
            factors_np = self.factors.cpu().numpy()
            
            loading_sparsity = np.mean(np.abs(loadings_np) < 1e-6)
            # Factors are non-negative, so check for small values
            factor_sparsity = np.mean(factors_np < 1e-6)
            
            return {
                'loading_sparsity': loading_sparsity,
                'factor_sparsity': factor_sparsity,
                'total_params': loadings_np.size + factors_np.size,
                'sparse_params': np.sum(np.abs(loadings_np) < 1e-6) + np.sum(factors_np < 1e-6),
                'non_negative_factors': True,
                'factors_min': factors_np.min(),
                'factors_max': factors_np.max()
            }


# Hyperparameter search for coordinate descent NMF
def coordinate_descent_hyperparameter_search(data, held_out_mask,
                                           n_factors_values=[5, 8, 12, 15],
                                           sparsity_penalty_values=[0.01, 0.1, 0.5],
                                           elastic_net_frac_values=[0.0, 0.5, 1.0],
                                           init_method='nnsvd',
                                           max_num_iters=30,
                                           verbose=True):
    """Hyperparameter search for coordinate descent NMF"""
    
    import itertools
    
    results = []
    best_loglike = -np.inf
    best_params = None
    best_model = None
    
    param_combinations = list(itertools.product(
        n_factors_values, sparsity_penalty_values, elastic_net_frac_values
    ))
    
    if verbose:
        print(f"🔍 Testing {len(param_combinations)} combinations with coordinate descent NMF...")
    
    for i, (n_factors, sparsity_penalty, elastic_net_frac) in enumerate(param_combinations):
        if verbose:
            print(f"\n[{i+1:2d}/{len(param_combinations)}] n_factors={n_factors}, penalty={sparsity_penalty:.2f}, elastic_net={elastic_net_frac:.1f}")
        
        try:
            model = CoordinateDescentNMF(
                n_factors=n_factors,
                sparsity_penalty=sparsity_penalty,
                elastic_net_frac=elastic_net_frac,
                max_num_iters=max_num_iters,
                device='auto'
            )
            
            start_time = time.time()
            model.fit(data, held_out_mask=held_out_mask, init_method=init_method, verbose=False)
            fit_time = time.time() - start_time
            
            final_loss = model.training_losses_[-1]
            final_held_out_loglike = model.held_out_loglikes_[-1]
            sparsity_metrics = model.get_sparsity_metrics()
            
            result = {
                'n_factors': n_factors,
                'sparsity_penalty': sparsity_penalty,
                'elastic_net_frac': elastic_net_frac,
                'final_loss': final_loss,
                'held_out_loglike': final_held_out_loglike,
                'loading_sparsity': sparsity_metrics['loading_sparsity'],
                'factor_sparsity': sparsity_metrics['factor_sparsity'],
                'fit_time': fit_time,
                'model': model
            }
            
            results.append(result)
            
            if verbose:
                print(f"    Loss: {final_loss:.6f}, LogLike: {final_held_out_loglike:.6f}")
                print(f"    Sparsity: L={sparsity_metrics['loading_sparsity']:.2%}, F={sparsity_metrics['factor_sparsity']:.2%}")
                print(f"    Time: {fit_time:.1f}s")
            
            if final_held_out_loglike > best_loglike:
                best_loglike = final_held_out_loglike
                best_params = {
                    'n_factors': n_factors,
                    'sparsity_penalty': sparsity_penalty,
                    'elastic_net_frac': elastic_net_frac
                }
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"    ❌ Failed: {str(e)}")
            continue
    
    if verbose:
        print(f"\n🏆 Best parameters: {best_params}")
        print(f"🏆 Best held-out log-likelihood: {best_loglike:.6f}")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_loglike': best_loglike,
        'all_results': results
    }


if __name__ == "__main__":
    print("Coordinate Descent NMF with Non-negative Factors")
    print("- Non-negative factors (handles negative data)")
    print("- Coordinate descent optimization") 
    print("- Baseline row/column effects")
    print("- NNSVD initialization")
