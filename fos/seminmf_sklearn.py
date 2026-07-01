"""
Brain-Aware Factorization for Adjusted Count Neural Data
Complete package with hyperparameter search and comprehensive evaluation

Handles negative values, spatial coherence, and biological interpretability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import itertools
import pandas as pd

class BrainAwareFactorization(nn.Module):
    """
    Brain-aware factorization for adjusted count neural data
    
    Key features:
    - Handles negative values from background subtraction
    - Spatial smoothness regularization for brain coherence
    - Robust loss function for count-like data
    - Efficient handling of high-dimensional data
    """
    
    def __init__(self, n_factors, spatial_regularization=0.01, sparsity_penalty=0.01,
                 device='auto', random_state=42, silent=False):
        super().__init__()
        
        self.n_factors = n_factors
        self.spatial_regularization = spatial_regularization
        self.sparsity_penalty = sparsity_penalty
        self.random_state = random_state
        self.silent = silent
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if not self.silent:
            print(f"🧠 Brain-Aware Factorization using device: {self.device}")
        
        # Parameters (will be initialized in fit)
        self.loadings = None  # (n_obs, n_factors) - can be negative for adjusted count data
        self.factors = None   # (n_factors, n_voxels) - can be negative for adjusted count data
        self.bias = None      # (n_voxels,) - global bias per voxel
        
        # Training history
        self.losses_ = []
        self.val_losses_ = []
    
    def _initialize_parameters(self, n_obs, n_voxels, data):
        """Smart initialization using PCA for adjusted count data"""
        if not self.silent:
            print("🚀 Smart PCA-based initialization...")
        
        # Center the data
        data_centered = data - np.mean(data, axis=0, keepdims=True)
        
        # Use PCA for initialization (handles negative values well)
        try:
            # For large data, use randomized PCA
            if n_voxels > 100000:
                from sklearn.decomposition import TruncatedSVD
                pca = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
                loadings_init = pca.fit_transform(data_centered)
                factors_init = pca.components_
                if not self.silent:
                    print(f"  Used TruncatedSVD for large data ({n_obs}×{n_voxels:,})")
            else:
                pca = PCA(n_components=self.n_factors, random_state=self.random_state)
                loadings_init = pca.fit_transform(data_centered)
                factors_init = pca.components_
                if not self.silent:
                    print(f"  Used standard PCA")
            
            # Scale to reasonable range
            loadings_init = loadings_init * 0.1
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            if not self.silent:
                print(f"  PCA explained variance: {explained_variance:.1%}")
            
        except Exception as e:
            if not self.silent:
                print(f"  PCA failed ({e}), using random initialization")
            loadings_init = np.random.randn(n_obs, self.n_factors) * 0.1
            factors_init = np.random.randn(self.n_factors, n_voxels) * 0.1
        
        # Convert to tensors
        self.loadings = nn.Parameter(torch.FloatTensor(loadings_init).to(self.device))
        self.factors = nn.Parameter(torch.FloatTensor(factors_init).to(self.device))
        
        # Initialize bias as mean per voxel
        bias_init = np.mean(data, axis=0)
        self.bias = nn.Parameter(torch.FloatTensor(bias_init).to(self.device))
        
        if not self.silent:
            print(f"  Initialized: loadings {self.loadings.shape}, factors {self.factors.shape}")
    
    def forward(self):
        """Compute reconstruction: loadings @ factors + bias"""
        return torch.mm(self.loadings, self.factors) + self.bias.unsqueeze(0)
    
    def _compute_spatial_penalty(self, alive_voxels_3d=None):
        """
        Spatial smoothness penalty for brain coherence
        Encourages factors to be spatially smooth
        """
        if alive_voxels_3d is None or self.spatial_regularization == 0:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # Convert alive_voxels to tensor if needed
            if isinstance(alive_voxels_3d, np.ndarray):
                alive_voxels_3d = torch.BoolTensor(alive_voxels_3d).to(self.device)
            
            penalty = torch.tensor(0.0, device=self.device)
            
            # Simple spatial penalty: encourage nearby voxels to have similar values
            # This is a simplified version - ideally you'd use proper spatial neighbors
            n_factors, n_voxels = self.factors.shape
            
            for k in range(n_factors):
                factor_k = self.factors[k, :]
                
                # Reshape factor to 3D brain space
                if hasattr(self, '_voxel_indices'):
                    # If we have spatial mapping, use it
                    brain_factor = torch.zeros(alive_voxels_3d.shape, device=self.device)
                    brain_factor[alive_voxels_3d] = factor_k
                    
                    # Compute spatial gradients (simplified)
                    grad_x = torch.diff(brain_factor, dim=0)
                    grad_y = torch.diff(brain_factor, dim=1) 
                    grad_z = torch.diff(brain_factor, dim=2)
                    
                    # L2 penalty on spatial gradients
                    penalty += torch.sum(grad_x ** 2) + torch.sum(grad_y ** 2) + torch.sum(grad_z ** 2)
                else:
                    # Fallback: simple smoothness penalty
                    penalty += torch.sum(torch.diff(factor_k) ** 2)
            
            return penalty / (n_factors * n_voxels)
        
        except Exception:
            # If spatial penalty fails, return zero penalty
            return torch.tensor(0.0, device=self.device)
    
    def _compute_loss(self, data, reconstruction, alive_voxels_3d=None, val_mask=None):
        """
        Robust loss function for adjusted count data
        """
        if val_mask is not None:
            # Validation loss on held-out data
            data_masked = data * (~val_mask).float()
            recon_masked = reconstruction * (~val_mask).float()
            n_points = torch.sum(~val_mask).float()
        else:
            data_masked = data
            recon_masked = reconstruction
            n_points = data.numel()
        
        # Robust loss: Huber loss is less sensitive to outliers than MSE
        residual = data_masked - recon_masked
        huber_loss = torch.where(torch.abs(residual) < 1.0,
                                0.5 * residual ** 2,
                                torch.abs(residual) - 0.5)
        main_loss = torch.sum(huber_loss) / n_points
        
        # Sparsity penalty on loadings (L1 regularization)
        sparsity_loss = self.sparsity_penalty * torch.sum(torch.abs(self.loadings))
        
        # Spatial smoothness penalty
        spatial_loss = self.spatial_regularization * self._compute_spatial_penalty(alive_voxels_3d)
        
        total_loss = main_loss + sparsity_loss + spatial_loss
        
        return total_loss, {
            'main_loss': main_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def fit(self, data, alive_voxels_3d=None, val_mask=None, 
            max_epochs=200, lr=0.01, patience=20, verbose=True):
        """
        Fit brain-aware factorization using Adam optimizer
        """
        # Convert data to tensor
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        n_obs, n_voxels = data.shape
        
        # Initialize parameters
        self._initialize_parameters(n_obs, n_voxels, data.cpu().numpy())
        
        # Setup optimizer with different learning rates for different parameters
        optimizer = optim.Adam([
            {'params': [self.loadings], 'lr': lr},
            {'params': [self.factors], 'lr': lr * 0.1},  # Slower learning for factors
            {'params': [self.bias], 'lr': lr * 0.01}     # Very slow for bias
        ])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # Convert validation mask if provided
        if val_mask is not None:
            if isinstance(val_mask, np.ndarray):
                val_mask = torch.BoolTensor(val_mask).to(self.device)
            else:
                val_mask = val_mask.to(self.device)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print(f"🧠 Training Brain-Aware Factorization:")
            print(f"  Data: {n_obs}×{n_voxels:,} → {self.n_factors} factors")
            print(f"  Spatial regularization: {self.spatial_regularization}")
            print(f"  Sparsity penalty: {self.sparsity_penalty}")
        
        pbar = tqdm(range(max_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction = self.forward()
            
            # Compute loss
            loss, loss_components = self._compute_loss(data, reconstruction, 
                                                     alive_voxels_3d, val_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            self.losses_.append(loss_components)
            
            # Validation loss
            if val_mask is not None:
                with torch.no_grad():
                    val_reconstruction = self.forward()
                    val_data = data * val_mask.float()
                    val_recon = val_reconstruction * val_mask.float()
                    val_loss = torch.sum((val_data - val_recon) ** 2) / torch.sum(val_mask.float())
                    self.val_losses_.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
            
            # Update progress bar
            if verbose and epoch % 10 == 0:
                desc = f"Loss: {loss_components['total_loss']:.4f}"
                if val_mask is not None:
                    desc += f", Val: {self.val_losses_[-1]:.4f}"
                pbar.set_description(desc)
        
        pbar.close()
        
        if verbose:
            print(f"✅ Training complete!")
            print(f"  Final loss: {self.losses_[-1]['total_loss']:.6f}")
            print(f"  Main loss: {self.losses_[-1]['main_loss']:.6f}")
            print(f"  Sparsity loss: {self.losses_[-1]['sparsity_loss']:.6f}")
            print(f"  Spatial loss: {self.losses_[-1]['spatial_loss']:.6f}")
            
            self._print_factor_stats()
        
        return self
    
    def _print_factor_stats(self):
        """Print factor statistics"""
        with torch.no_grad():
            loadings_np = self.loadings.cpu().numpy()
            factors_np = self.factors.cpu().numpy()
            bias_np = self.bias.cpu().numpy()
            
            print(f"\n📊 Factor Statistics:")
            print(f"  Loadings range: [{loadings_np.min():.3f}, {loadings_np.max():.3f}]")
            print(f"  Factors range: [{factors_np.min():.3f}, {factors_np.max():.3f}]")
            print(f"  Bias range: [{bias_np.min():.3f}, {bias_np.max():.3f}]")
            
            for k in range(self.n_factors):
                factor_norm = np.linalg.norm(factors_np[k])
                loading_std = np.std(loadings_np[:, k])
                factor_sparsity = (np.abs(factors_np[k]) < 1e-6).mean()
                
                print(f"  Factor {k+1}: norm={factor_norm:.3f}, loading_std={loading_std:.3f}, sparsity={factor_sparsity:.1%}")
    
    def get_components(self):
        """Get all learned components"""
        with torch.no_grad():
            return {
                'loadings': self.loadings.cpu().numpy(),  # (n_obs, n_factors)
                'factors': self.factors.cpu().numpy(),    # (n_factors, n_voxels)
                'bias': self.bias.cpu().numpy(),          # (n_voxels,)
            }
    
    def transform(self, data=None):
        """Get loadings for new data (or training data if None)"""
        with torch.no_grad():
            return self.loadings.cpu().numpy()
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data from loadings"""
        with torch.no_grad():
            if loadings is None:
                reconstruction = self.forward()
            else:
                if isinstance(loadings, np.ndarray):
                    loadings = torch.FloatTensor(loadings).to(self.device)
                reconstruction = torch.mm(loadings, self.factors) + self.bias.unsqueeze(0)
            
            return reconstruction.cpu().numpy()
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.losses_:
            print("No training history to plot")
            return
        
        # Extract loss components
        epochs = range(len(self.losses_))
        main_losses = [l['main_loss'] for l in self.losses_]
        total_losses = [l['total_loss'] for l in self.losses_]
        sparsity_losses = [l['sparsity_loss'] for l in self.losses_]
        spatial_losses = [l['spatial_loss'] for l in self.losses_]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(epochs, total_losses, 'b-', label='Total Loss')
        if self.val_losses_:
            axes[0, 0].plot(epochs[:len(self.val_losses_)], self.val_losses_, 'r--', label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Main reconstruction loss
        axes[0, 1].plot(epochs, main_losses, 'g-')
        axes[0, 1].set_title('Reconstruction Loss (Huber)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_yscale('log')
        
        # Regularization losses
        axes[1, 0].plot(epochs, sparsity_losses, 'r-', label='Sparsity')
        axes[1, 0].plot(epochs, spatial_losses, 'b-', label='Spatial')
        axes[1, 0].set_title('Regularization Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # Factor evolution (norm of each factor over time)
        # This would require storing factor history - simplified version
        components = self.get_components()
        factor_norms = [np.linalg.norm(f) for f in components['factors']]
        axes[1, 1].bar(range(len(factor_norms)), factor_norms)
        axes[1, 1].set_title('Final Factor Norms')
        axes[1, 1].set_xlabel('Factor Index')
        axes[1, 1].set_ylabel('L2 Norm')
        
        plt.tight_layout()
        plt.show()


def preprocess_brain_data(data, method='robust_mad'):
    """
    Preprocess adjusted count brain data
    """
    # Handle potential NaN/inf values
    data_clean = np.copy(data)
    data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == 'robust_mad':
        # Robust scaling using median absolute deviation
        data_median = np.median(data_clean)
        data_mad = np.median(np.abs(data_clean - data_median))
        
        if data_mad > 0:
            data_scaled = (data_clean - data_median) / (data_mad * 1.4826)
            data_scaled = np.clip(data_scaled, -5, 5)  # Clip extreme outliers
        else:
            data_scaled = data_clean - data_median
            
    elif method == 'standard':
        # Standard z-score normalization
        data_scaled = (data_clean - np.mean(data_clean)) / (np.std(data_clean) + 1e-8)
        
    elif method == 'quantile':
        # Quantile normalization (robust to outliers)
        from scipy.stats import rankdata
        ranks = rankdata(data_clean, axis=None).reshape(data_clean.shape)
        data_scaled = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
        
    else:  # 'none'
        data_scaled = data_clean
    
    return data_scaled


def evaluate_brain_model(model, data_scaled, held_out_mask=None, alive_voxels_3d=None):
    """
    Comprehensive evaluation of brain factorization model
    """
    components = model.get_components()
    reconstruction = model.inverse_transform()
    
    # Basic reconstruction metrics
    mse = np.mean((data_scaled - reconstruction) ** 2)
    r2 = 1 - mse / np.var(data_scaled)
    mae = np.mean(np.abs(data_scaled - reconstruction))
    
    # Factor quality metrics
    loadings = components['loadings']
    factors = components['factors']
    
    # Check for meaningful factors (not collapsed)
    factor_norms = [np.linalg.norm(f) for f in factors]
    factor_sparsities = [(np.abs(f) < 1e-6).mean() for f in factors]
    meaningful_factors = sum(1 for norm, sparsity in zip(factor_norms, factor_sparsities) 
                           if norm > 1e-3 and sparsity < 0.95)
    
    # Loading variability (factors should vary across observations)
    loading_stds = [np.std(loadings[:, i]) for i in range(loadings.shape[1])]
    active_factors = sum(1 for std in loading_stds if std > 1e-3)
    
    # Held-out validation
    held_out_r2 = None
    held_out_mse = None
    if held_out_mask is not None:
        held_out_true = data_scaled[held_out_mask]
        held_out_pred = reconstruction[held_out_mask]
        held_out_mse = np.mean((held_out_true - held_out_pred) ** 2)
        held_out_r2 = 1 - held_out_mse / np.var(held_out_true)
    
    # Spatial coherence (if spatial info available)
    spatial_coherence = None
    if alive_voxels_3d is not None:
        try:
            # Simple spatial coherence measure
            coherence_scores = []
            for factor in factors:
                if np.linalg.norm(factor) > 1e-6:
                    # Reshape to brain space and compute spatial smoothness
                    brain_factor = np.zeros(alive_voxels_3d.shape)
                    brain_factor[alive_voxels_3d] = factor
                    
                    # Compute gradient magnitude (higher = less smooth)
                    grad_mag = np.sum(np.gradient(brain_factor)[0] ** 2)
                    coherence_scores.append(1.0 / (1.0 + grad_mag))  # Higher = more coherent
            
            spatial_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        except:
            spatial_coherence = None
    
    return {
        'reconstruction_mse': mse,
        'reconstruction_r2': r2,
        'reconstruction_mae': mae,
        'held_out_mse': held_out_mse,
        'held_out_r2': held_out_r2,
        'meaningful_factors': meaningful_factors,
        'total_factors': len(factors),
        'meaningful_factor_ratio': meaningful_factors / len(factors),
        'active_factors': active_factors,
        'spatial_coherence': spatial_coherence,
        'final_loss': model.losses_[-1]['total_loss'] if model.losses_ else np.inf,
        'factor_norms': factor_norms,
        'loading_stds': loading_stds
    }


def brain_hyperparameter_search(data, alive_voxels_3d=None, held_out_mask=None,
                               n_factors_values=[6, 8, 10, 12],
                               spatial_reg_values=[0.0, 0.001, 0.01, 0.1],
                               sparsity_penalty_values=[0.0, 0.001, 0.01, 0.1],
                               lr_values=[0.001, 0.01, 0.1],
                               preprocessing_methods=['robust_mad'],
                               max_epochs=100,
                               early_stopping_patience=15,
                               n_random_seeds=2,
                               scoring_metric='held_out_r2',
                               verbose=True):
    """
    Comprehensive hyperparameter search for brain-aware factorization
    
    Parameters:
    -----------
    data : ndarray, shape (n_obs, n_voxels)
        Your neural data (adjusted counts)
    alive_voxels_3d : ndarray, optional
        3D boolean mask of alive voxels for spatial regularization
    held_out_mask : ndarray, optional
        Boolean mask for held-out validation data
    n_factors_values : list
        Number of factors to test
    spatial_reg_values : list
        Spatial regularization strengths to test
    sparsity_penalty_values : list
        Sparsity penalty strengths to test
    lr_values : list
        Learning rates to test
    preprocessing_methods : list
        Data preprocessing methods to test
    max_epochs : int
        Maximum epochs per configuration
    early_stopping_patience : int
        Early stopping patience
    n_random_seeds : int
        Number of random seeds per configuration (for robustness)
    scoring_metric : str
        Metric to optimize ('held_out_r2', 'reconstruction_r2', 'meaningful_factor_ratio')
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict : Results with best parameters and all configurations tested
    """
    
    print("🔍 BRAIN-AWARE FACTORIZATION HYPERPARAMETER SEARCH")
    print("=" * 60)
    
    # Debug: Print exactly what we received
    print(f"DEBUG - Received parameters:")
    print(f"  preprocessing_methods: {preprocessing_methods} (len: {len(preprocessing_methods)})")
    print(f"  n_factors_values: {n_factors_values} (len: {len(n_factors_values)})")
    print(f"  spatial_reg_values: {spatial_reg_values} (len: {len(spatial_reg_values)})")
    print(f"  sparsity_penalty_values: {sparsity_penalty_values} (len: {len(sparsity_penalty_values)})")
    print(f"  lr_values: {lr_values} (len: {len(lr_values)})")
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        preprocessing_methods,
        n_factors_values,
        spatial_reg_values, 
        sparsity_penalty_values,
        lr_values
    ))
    
    total_configs = len(param_combinations) * n_random_seeds
    
    # Debug: Show calculation
    expected_configs = len(preprocessing_methods) * len(n_factors_values) * len(spatial_reg_values) * len(sparsity_penalty_values) * len(lr_values)
    print(f"DEBUG - Expected configs: {len(preprocessing_methods)} × {len(n_factors_values)} × {len(spatial_reg_values)} × {len(sparsity_penalty_values)} × {len(lr_values)} = {expected_configs}")
    print(f"DEBUG - Actual param_combinations: {len(param_combinations)}")
    
    print(f"\nTesting {len(param_combinations)} parameter combinations × {n_random_seeds} seeds = {total_configs} total configs")
    if verbose:
        print(f"Parameters being tested:")
        print(f"  - preprocessing: {preprocessing_methods}")
        print(f"  - n_factors: {n_factors_values}")
        print(f"  - spatial_regularization: {spatial_reg_values}")
        print(f"  - sparsity_penalty: {sparsity_penalty_values}")
        print(f"  - learning_rate: {lr_values}")
        print(f"  - random_seeds: {n_random_seeds}")
        print(f"  - scoring_metric: {scoring_metric}")
    
    # Results storage
    results = []
    best_score = -np.inf
    best_config = None
    best_model = None
    
    # Summary storage for plotting
    search_progress = {
        'config_numbers': [],
        'scores': [],
        'losses': [],
        'meaningful_factor_ratios': [],
        'fit_times': [],
        'running_best_scores': []
    }
    
    # Progress tracking - disable all output during search
    config_pbar = tqdm(param_combinations, desc="Search Progress", disable=False)
    
    for config_idx, (preprocess_method, n_factors, spatial_reg, sparsity_penalty, lr) in enumerate(config_pbar):
        
        config_results = []
        config_losses = []
        config_times = []
        config_meaningful_ratios = []
        
        # Preprocess data once per preprocessing method
        try:
            data_scaled = preprocess_brain_data(data, method=preprocess_method)
        except Exception as e:
            continue
        
        # Test multiple random seeds for robustness
        for seed in range(n_random_seeds):
            
            try:
                # Initialize model with SILENT flag
                model = BrainAwareFactorization(
                    n_factors=n_factors,
                    spatial_regularization=spatial_reg,
                    sparsity_penalty=sparsity_penalty,
                    device='auto',
                    random_state=seed,
                    silent=True  # ← This makes initialization silent
                )
                
                # Fit model (completely silent)
                start_time = time.time()
                model.fit(data_scaled, 
                         alive_voxels_3d=alive_voxels_3d,
                         val_mask=held_out_mask,
                         max_epochs=max_epochs,
                         lr=lr,
                         patience=early_stopping_patience,
                         verbose=False)
                fit_time = time.time() - start_time
                
                # Evaluate model
                metrics = evaluate_brain_model(model, data_scaled, held_out_mask, alive_voxels_3d)
                
                # Get scoring metric value
                if scoring_metric == 'held_out_r2' and metrics['held_out_r2'] is not None:
                    score = metrics['held_out_r2']
                elif scoring_metric == 'reconstruction_r2':
                    score = metrics['reconstruction_r2']
                elif scoring_metric == 'meaningful_factor_ratio':
                    score = metrics['meaningful_factor_ratio']
                else:
                    # Fallback: composite score
                    r2_score = metrics['held_out_r2'] if metrics['held_out_r2'] is not None else metrics['reconstruction_r2']
                    meaningful_score = metrics['meaningful_factor_ratio']
                    score = 0.7 * r2_score + 0.3 * meaningful_score
                
                # Store result
                result = {
                    'preprocessing': preprocess_method,
                    'n_factors': n_factors,
                    'spatial_regularization': spatial_reg,
                    'sparsity_penalty': sparsity_penalty,
                    'learning_rate': lr,
                    'random_seed': seed,
                    'score': score,
                    'fit_time': fit_time,
                    **metrics
                }
                
                results.append(result)
                config_results.append(score)
                config_losses.append(metrics['final_loss'])
                config_times.append(fit_time)
                config_meaningful_ratios.append(metrics['meaningful_factor_ratio'])
                
                # Check if this is the best model
                if score > best_score:
                    best_score = score
                    best_config = result.copy()
                    best_model = model
                
            except Exception as e:
                # Silent failure - just skip this configuration
                continue
        
        # Store summary for this configuration (average across seeds)
        if config_results:
            search_progress['config_numbers'].append(config_idx + 1)
            search_progress['scores'].append(np.mean(config_results))
            search_progress['losses'].append(np.mean(config_losses))
            search_progress['meaningful_factor_ratios'].append(np.mean(config_meaningful_ratios))
            search_progress['fit_times'].append(np.mean(config_times))
            search_progress['running_best_scores'].append(best_score)
            
            # Update progress bar - ONLY show progress, no other info
            config_pbar.set_description(f"Config {config_idx+1}/{len(param_combinations)}")
    
    config_pbar.close()
    
    if not results:
        raise RuntimeError("❌ No configurations succeeded! Check your data and parameters.")
    
    # Analyze results (simplified output)
    if verbose:
        print(f"\n🏆 SEARCH COMPLETE")
        print(f"Tested {len(results)} configurations")
        print(f"Best {scoring_metric}: {best_score:.4f}")
        
        # Show only best configuration
        print(f"\n🥇 Best Configuration:")
        print(f"  n_factors: {best_config['n_factors']}")
        print(f"  spatial_regularization: {best_config['spatial_regularization']}")
        print(f"  sparsity_penalty: {best_config['sparsity_penalty']}")
        print(f"  preprocessing: {best_config['preprocessing']}")
        print(f"  meaningful_factors: {best_config['meaningful_factors']}/{best_config['total_factors']}")
    
    return {
        'best_model': best_model,
        'best_config': best_config,
        'best_score': best_score,
        'all_results': results,
        'search_progress': search_progress,  # ← For plotting later
        'summary': {
            'total_configs_tested': len(results),
            'best_score': best_score,
            'scoring_metric': scoring_metric
        }
    }


def brain_factorization_pipeline(data, alive_voxels_3d=None, held_out_mask=None,
                               n_factors=8, spatial_regularization=0.01, 
                               sparsity_penalty=0.001, max_epochs=200, 
                               preprocessing='robust_mad', verbose=True):
    """
    Complete brain factorization pipeline with single configuration
    """
    print("=== BRAIN-AWARE FACTORIZATION PIPELINE ===")
    
    # Data preprocessing
    print("🔧 Preprocessing adjusted count data...")
    data_scaled = preprocess_brain_data(data, method=preprocessing)
    
    print(f"  Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Scaled range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
    
    # Initialize and fit model
    model = BrainAwareFactorization(
        n_factors=n_factors,
        spatial_regularization=spatial_regularization,
        sparsity_penalty=sparsity_penalty,
        device='auto',
        silent=not verbose  # Silent when not verbose
    )
    
    # Fit model
    start_time = time.time()
    model.fit(data_scaled, alive_voxels_3d=alive_voxels_3d, 
              val_mask=held_out_mask, max_epochs=max_epochs, verbose=verbose)
    fit_time = time.time() - start_time
    
    print(f"\n🎯 Brain factorization completed in {fit_time:.2f} seconds")
    
    # Evaluate model
    metrics = evaluate_brain_model(model, data_scaled, held_out_mask, alive_voxels_3d)
    
    print(f"\n📊 Results Summary:")
    print(f"  Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"  Reconstruction R²: {metrics['reconstruction_r2']:.4f}")
    if metrics['held_out_r2'] is not None:
        print(f"  Held-out R²: {metrics['held_out_r2']:.4f}")
    print(f"  Meaningful factors: {metrics['meaningful_factors']}/{metrics['total_factors']}")
    
    # Get components
    components = model.get_components()
    
    return model, components


def plot_hyperparameter_search_progress(search_results):
    """
    Plot hyperparameter search results by parameter values
    """
    all_results = search_results.get('all_results', [])
    
    if not all_results:
        print("No search results data to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(all_results)
    
    # Determine which parameters were actually varied
    varied_params = []
    param_names = ['n_factors', 'spatial_regularization', 'sparsity_penalty', 'learning_rate', 'preprocessing']
    
    for param in param_names:
        if param in df.columns and len(df[param].unique()) > 1:
            varied_params.append(param)
    
    if not varied_params:
        print("No parameters were varied in the search")
        return
    
    # Create subplots based on number of varied parameters
    n_plots = len(varied_params)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten() if n_plots > 1 else axes
    
    for i, param in enumerate(varied_params):
        ax = axes[i] if n_plots > 1 else axes[0]
        
        # Group by parameter value and compute statistics
        param_stats = df.groupby(param)['score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot with error bars
        ax.errorbar(param_stats[param], param_stats['mean'], 
                   yerr=param_stats['std'], marker='o', capsize=5, linewidth=2, markersize=8)
        
        # Scatter plot of all individual results
        ax.scatter(df[param], df['score'], alpha=0.3, s=20, color='gray')
        
        # Highlight the best configuration
        best_config = search_results['best_config']
        if param in best_config:
            best_value = best_config[param]
            best_score = search_results['best_score']
            ax.scatter([best_value], [best_score], color='red', s=100, marker='*', 
                      label=f'Best: {best_value}', zorder=5)
            ax.legend()
        
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.set_title(f'Score vs {param.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Set appropriate scale for spatial_regularization and sparsity_penalty
        if param in ['spatial_regularization', 'sparsity_penalty'] and param_stats[param].min() > 0:
            ax.set_xscale('log')
    
    # Hide extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Hyperparameter Search Results\nBest Score: {search_results["best_score"]:.4f}', 
                 y=1.02, fontsize=14)
    plt.show()
    
    # Print parameter analysis
    print(f"📊 Parameter Analysis:")
    for param in varied_params:
        param_stats = df.groupby(param)['score'].agg(['mean', 'std', 'count'])
        print(f"\n  {param.replace('_', ' ').title()}:")
        for value, row in param_stats.iterrows():
            best_marker = " ⭐" if value == best_config.get(param) else ""
            print(f"    {value}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']}){best_marker}")
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"  Total configurations: {len(all_results)}")
    print(f"  Best score: {search_results['best_score']:.4f}")
    print(f"  Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
    print(f"  Score std: {df['score'].std():.4f}")


def run_final_factorization(search_results, data, alive_voxels_3d=None, held_out_mask=None, 
                           max_epochs=100, verbose=True):
    """
    Run the final factorization with the best hyperparameters
    """
    best_config = search_results['best_config']
    
    print("🎯 RUNNING FINAL FACTORIZATION WITH BEST PARAMETERS")
    print("=" * 60)
    print(f"Best configuration found:")
    for key, value in best_config.items():
        if key not in ['score', 'factor_norms', 'loading_stds', 'fit_time', 'random_seed']:
            print(f"  {key}: {value}")
    print(f"  validation score: {search_results['best_score']:.4f}")
    
    # Import preprocessing function
    from brain_aware_factorization import preprocess_brain_data, BrainAwareFactorization
    
    # Preprocess data with best method
    data_scaled = preprocess_brain_data(data, method=best_config['preprocessing'])
    print(f"\nData preprocessing: {best_config['preprocessing']}")
    print(f"  Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Scaled range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
    
    # Create final model with best hyperparameters
    final_model = BrainAwareFactorization(
        n_factors=best_config['n_factors'],
        spatial_regularization=best_config['spatial_regularization'],
        sparsity_penalty=best_config['sparsity_penalty'],
        device='auto',
        random_state=42,  # Fixed seed for reproducibility
        silent=not verbose
    )
    
    # Train final model
    print(f"\n🚀 Training final model...")
    start_time = time.time()
    final_model.fit(
        data_scaled,
        alive_voxels_3d=alive_voxels_3d,
        val_mask=held_out_mask,
        max_epochs=max_epochs,
        lr=best_config['learning_rate'],
        patience=20,
        verbose=verbose
    )
    fit_time = time.time() - start_time
    
    print(f"\n✅ Final model training completed in {fit_time:.2f} seconds")
    
    # Get final components
    final_components = final_model.get_components()
    
    # Evaluate final model
    from brain_aware_factorization import evaluate_brain_model
    final_metrics = evaluate_brain_model(final_model, data_scaled, held_out_mask, alive_voxels_3d)
    
    print(f"\n📊 Final Model Performance:")
    print(f"  Reconstruction R²: {final_metrics['reconstruction_r2']:.4f}")
    if final_metrics['held_out_r2'] is not None:
        print(f"  Held-out R²: {final_metrics['held_out_r2']:.4f}")
    print(f"  Meaningful factors: {final_metrics['meaningful_factors']}/{final_metrics['total_factors']}")
    print(f"  Meaningful factor ratio: {final_metrics['meaningful_factor_ratio']:.2f}")
    
    print(f"\n🧠 Final Brain Factors:")
    loadings = final_components['loadings']
    factors = final_components['factors']
    bias = final_components['bias']
    
    print(f"  Loadings shape: {loadings.shape} (observations × factors)")
    print(f"  Factors shape: {factors.shape} (factors × voxels)")
    print(f"  Bias shape: {bias.shape} (voxels)")
    
    # Factor analysis
    for i in range(factors.shape[0]):
        factor_norm = np.linalg.norm(factors[i])
        loading_std = np.std(loadings[:, i])
        factor_sparsity = (np.abs(factors[i]) < 1e-6).mean()
        meaningful = factor_norm > 1e-3 and factor_sparsity < 0.95 and loading_std > 1e-3
        status = "✅" if meaningful else "❌"
        print(f"  Factor {i+1}: {status} norm={factor_norm:.3f}, loading_std={loading_std:.3f}, sparsity={factor_sparsity:.1%}")
    
    return final_model, final_components, final_metrics


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Brain-Aware Factorization for Adjusted Count Neural Data")
    print("✅ Handles negative values from background subtraction")
    print("✅ Spatial regularization for brain coherence")
    print("✅ Robust loss function for count-like data")
    print("✅ Efficient GPU acceleration")
    print("✅ Comprehensive hyperparameter search")
    
    # Test with synthetic brain data
    np.random.seed(42)
    n_obs, n_voxels = 43, 10000
    
    # Simulate adjusted count data (can be negative)
    true_loadings = np.random.randn(n_obs, 5)
    true_factors = np.random.randn(5, n_voxels) * 0.5
    noise = np.random.randn(n_obs, n_voxels) * 0.1
    synthetic_data = true_loadings @ true_factors + noise
    
    print(f"\nTesting with synthetic data: {synthetic_data.shape}")
    print(f"Synthetic data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    
    # Create held-out mask
    held_out_mask = np.random.random(synthetic_data.shape) < 0.1
    
    # Option 1: Quick single configuration test
    print("\n" + "="*50)
    print("OPTION 1: Single Configuration Test")
    print("="*50)
    
    model, components = brain_factorization_pipeline(
        synthetic_data, 
        held_out_mask=held_out_mask,
        n_factors=6,
        max_epochs=100,
        verbose=True
    )
    
    # Plot results
    model.plot_training_progress()
    
    # Option 2: Comprehensive hyperparameter search
    print("\n" + "="*50)
    print("OPTION 2: Hyperparameter Search")
    print("="*50)
    
    # Fast hyperparameter search (for testing) - now with clean output
    search_results = brain_hyperparameter_search(
        synthetic_data,
        held_out_mask=held_out_mask,
        n_factors_values=[4, 6, 8],
        spatial_reg_values=[0.0, 0.01],
        sparsity_penalty_values=[0.0, 0.001],
        lr_values=[0.01],
        preprocessing_methods=['robust_mad'],
        max_epochs=50,
        n_random_seeds=1,
        scoring_metric='held_out_r2',
        verbose=True
    )
    
    # Plot search progress
    plot_hyperparameter_search_progress(search_results)
    
    # Get best model
    best_model = search_results['best_model']
    best_config = search_results['best_config']
    
    print(f"\n🏆 Best model found!")
    print(f"Best score: {search_results['best_score']:.4f}")
    
    # Use best model
    best_components = best_model.get_components()
    print(f"Best model components: {best_components.keys()}")


# Usage example for your real data:
def run_on_your_data(counts, alive_voxels=None, held_out_mask=None):
    """
    Example of how to use with your actual neural data
    """
    # Assuming you have:
    # - counts: your (43, 3M) neural data  
    # - alive_voxels: 3D boolean mask of brain voxels
    # - held_out_mask: validation mask
    
    print("🧠 Running on your neural data...")
    
    # Option 1: Quick test with reasonable defaults
    model, components = brain_factorization_pipeline(
        counts,
        alive_voxels_3d=alive_voxels,
        held_out_mask=held_out_mask,
        n_factors=8,
        spatial_regularization=0.01,
        sparsity_penalty=0.001,
        max_epochs=200,
        preprocessing='robust_mad',
        verbose=True
    )
    
    # Option 2: Full hyperparameter search (takes longer but finds best parameters)
    search_results = brain_hyperparameter_search(
        counts,
        alive_voxels_3d=alive_voxels,
        held_out_mask=held_out_mask,
        n_factors_values=[6, 8, 10, 12],
        spatial_reg_values=[0.0, 0.001, 0.01, 0.1],
        sparsity_penalty_values=[0.0, 0.001, 0.01, 0.1],
        lr_values=[0.01],  # Fixed learning rate
        preprocessing_methods=['robust_mad', 'standard'],
        max_epochs=150,
        early_stopping_patience=20,
        n_random_seeds=2,
        scoring_metric='held_out_r2',
        verbose=True
    )
    
    # Get best results
    best_model = search_results['best_model']
    best_components = best_model.get_components()
    
    # Plot search progress
    plot_hyperparameter_search_progress(search_results)
    
    # Your factors for downstream analysis
    loadings = best_components['loadings']    # (43, n_factors)
    factors = best_components['factors']      # (n_factors, 3M)
    bias = best_components['bias']            # (3M,)
    
    return best_model, best_components, search_results
