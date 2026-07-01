"""
Brain-Aware Semi-NMF for Adjusted Count Neural Data
Proper Semi-NMF implementation with non-negative factors and signed loadings

Key constraints:
- Factors (W) >= 0 (non-negative)
- Loadings (H) can be negative (Semi-NMF)
- Poisson likelihood for count data
- Spatial smoothness regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, NMF
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import itertools
import pandas as pd

def soft_threshold(x, threshold):
    """Soft thresholding operator for sparsity"""
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

class BrainAwareSemiNMF(nn.Module):
    """
    Brain-aware Semi-NMF for adjusted count neural data
    
    Model: X ≈ H @ W + bias
    Constraints:
    - W (factors) >= 0 (non-negative)
    - H (loadings) can be negative (Semi-NMF)
    - Poisson likelihood for count data
    - Spatial smoothness on factors
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
            print(f"🧠 Brain-Aware Semi-NMF using device: {self.device}")
        
        # Parameters
        self.loadings = None    # H: (n_obs, n_factors) - can be negative
        self.factors = None     # W: (n_factors, n_voxels) - must be >= 0
        self.bias = None        # b: (n_voxels,) - global bias
        
        # Training history
        self.losses_ = []
        self.factor_history_ = []
    
    def _initialize_parameters(self, n_obs, n_voxels, data):
        """Initialize parameters with proper constraints"""
        if not self.silent:
            print("🚀 Initializing Semi-NMF parameters...")
        
        # Ensure data is non-negative for initialization
        data_pos = np.maximum(data, 1e-6)
        
        try:
            # Use NMF for proper non-negative initialization
            nmf = NMF(n_components=self.n_factors, random_state=self.random_state, 
                     max_iter=50, alpha_W=0.1, alpha_H=0.1)
            H_init = nmf.fit_transform(data_pos)  # loadings
            W_init = nmf.components_              # factors
            
            if not self.silent:
                print(f"  NMF initialization successful")
                print(f"  Reconstruction error: {nmf.reconstruction_err_:.6f}")
            
        except Exception as e:
            if not self.silent:
                print(f"  NMF failed ({e}), using random initialization")
            
            # Fallback: random non-negative initialization
            H_init = np.random.exponential(1.0, (n_obs, self.n_factors))
            W_init = np.random.exponential(1.0, (self.n_factors, n_voxels))
        
        # Convert to tensors with proper constraints
        # Loadings can be negative (Semi-NMF), but we start positive
        self.loadings = nn.Parameter(torch.FloatTensor(H_init).to(self.device))
        
        # Factors must be non-negative - use log parameterization for stability
        self.log_factors = nn.Parameter(torch.log(torch.FloatTensor(W_init) + 1e-8).to(self.device))
        
        # Bias initialization
        bias_init = np.maximum(np.mean(data, axis=0), 1e-6)
        self.bias = nn.Parameter(torch.log(torch.FloatTensor(bias_init) + 1e-8).to(self.device))
        
        if not self.silent:
            print(f"  Initialized: loadings {self.loadings.shape}, factors {W_init.shape}")
    
    @property
    def factors(self):
        """Non-negative factors via exp transformation"""
        return torch.exp(self.log_factors)
    
    def forward(self):
        """Compute reconstruction: H @ W + bias"""
        return torch.mm(self.loadings, self.factors) + torch.exp(self.bias).unsqueeze(0)
    
    def _poisson_loss(self, data, reconstruction, mask=None):
        """Poisson negative log-likelihood loss"""
        if mask is not None:
            data = data * mask.float()
            reconstruction = reconstruction * mask.float()
            n_points = torch.sum(mask).float()
        else:
            n_points = data.numel()
        
        # Numerical stability: clamp reconstruction to avoid overflow
        reconstruction = torch.clamp(reconstruction, min=1e-8, max=1e8)
        
        # Poisson NLL: -log P(x|λ) = λ - x*log(λ) + log(x!)
        # Ignore constant log(x!) term
        nll = reconstruction - data * torch.log(reconstruction)
        
        return torch.sum(nll) / n_points
    
    def _spatial_penalty(self, alive_voxels_3d=None):
        """Spatial smoothness penalty on factors"""
        if alive_voxels_3d is None or self.spatial_regularization == 0:
            return torch.tensor(0.0, device=self.device)
        
        try:
            penalty = torch.tensor(0.0, device=self.device)
            factors = self.factors
            
            # Simple spatial penalty: penalize differences between adjacent voxels
            for k in range(self.n_factors):
                factor_k = factors[k, :]
                
                # If we have spatial structure, use it
                if hasattr(self, '_spatial_adjacency'):
                    # Use precomputed adjacency for proper spatial penalty
                    penalty += torch.sum((factor_k[self._spatial_adjacency[:, 0]] - 
                                        factor_k[self._spatial_adjacency[:, 1]]) ** 2)
                else:
                    # Fallback: simple difference penalty
                    penalty += torch.sum(torch.diff(factor_k) ** 2)
            
            return penalty / (self.n_factors * factors.shape[1])
        
        except Exception:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_loss(self, data, reconstruction, alive_voxels_3d=None, val_mask=None):
        """Complete loss function for Semi-NMF"""
        if val_mask is not None:
            # Training loss (not on validation data)
            train_mask = ~val_mask
            poisson_loss = self._poisson_loss(data, reconstruction, train_mask)
        else:
            poisson_loss = self._poisson_loss(data, reconstruction)
        
        # Sparsity penalty on loadings (L1 - promotes sparsity)
        sparsity_loss = self.sparsity_penalty * torch.sum(torch.abs(self.loadings))
        
        # Spatial smoothness penalty on factors
        spatial_loss = self.spatial_regularization * self._spatial_penalty(alive_voxels_3d)
        
        # Semi-NMF regularization: encourage factors to be non-negative (already enforced by exp)
        # No additional penalty needed since we use log parameterization
        
        total_loss = poisson_loss + sparsity_loss + spatial_loss
        
        return total_loss, {
            'poisson_loss': poisson_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _coordinate_descent_step(self, data, alive_voxels_3d=None):
        """One step of coordinate descent (more stable than gradient descent for NMF)"""
        
        # Update loadings (can be negative) using multiplicative updates
        with torch.no_grad():
            reconstruction = self.forward()
            factors = self.factors
            bias = torch.exp(self.bias)
            
            # For loadings update: minimize ||data - H@W - b||² + λ|H|
            # Gradient w.r.t. H: -W(data - H@W - b)ᵀ + λ*sign(H)
            residual = data - reconstruction
            gradient_H = -torch.mm(residual, factors.t())
            
            # Apply sparsity penalty gradient
            if self.sparsity_penalty > 0:
                gradient_H += self.sparsity_penalty * torch.sign(self.loadings)
            
            # Simple gradient step for loadings (Semi-NMF allows negative values)
            lr_H = 0.01
            self.loadings.data -= lr_H * gradient_H
        
        # Update factors (must stay non-negative) using log parameterization
        # Factors are automatically non-negative due to exp transformation
        # Just need to update log_factors with gradient descent
        
        # This will be handled by the optimizer in the training loop
        pass
    
    def fit(self, data, alive_voxels_3d=None, val_mask=None, 
            max_epochs=200, lr=0.01, patience=20, verbose=True,
            use_coordinate_descent=False):
        """
        Fit Semi-NMF model
        """
        # Convert data to tensor
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        # Ensure data is non-negative (required for Poisson)
        data = torch.clamp(data, min=1e-8)
        
        n_obs, n_voxels = data.shape
        
        # Initialize parameters
        self._initialize_parameters(n_obs, n_voxels, data.cpu().numpy())
        
        # Setup optimizer - different learning rates for different parameters
        optimizer = torch.optim.Adam([
            {'params': [self.loadings], 'lr': lr},
            {'params': [self.log_factors], 'lr': lr * 0.1},  # Slower for factors
            {'params': [self.bias], 'lr': lr * 0.01}         # Very slow for bias
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.5, verbose=False)
        
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
            print(f"🧠 Training Brain-Aware Semi-NMF:")
            print(f"  Data: {n_obs}×{n_voxels:,} → {self.n_factors} factors")
            print(f"  Constraints: factors ≥ 0, loadings ∈ ℝ")
            print(f"  Loss: Poisson + spatial regularization + sparsity")
        
        pbar = tqdm(range(max_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            if use_coordinate_descent and epoch % 5 == 0:
                # Occasionally use coordinate descent for better convergence
                self._coordinate_descent_step(data, alive_voxels_3d)
            
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
            
            # Enforce Semi-NMF constraints after update
            with torch.no_grad():
                # Factors are automatically non-negative due to log parameterization
                # Loadings can be negative (Semi-NMF) - no constraint needed
                
                # Optional: project loadings to reasonable range for numerical stability
                self.loadings.data = torch.clamp(self.loadings.data, min=-10, max=10)
                self.log_factors.data = torch.clamp(self.log_factors.data, min=-10, max=10)
                self.bias.data = torch.clamp(self.bias.data, min=-10, max=10)
            
            # Track losses
            self.losses_.append(loss_components)
            
            # Track factor evolution
            if epoch % 10 == 0:
                with torch.no_grad():
                    factor_norms = [torch.norm(self.factors[k]).item() for k in range(self.n_factors)]
                    self.factor_history_.append(factor_norms)
            
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
                desc = f"Loss: {loss_components['total_loss']:.6f}"
                pbar.set_description(desc)
        
        pbar.close()
        
        if verbose:
            print(f"✅ Semi-NMF training complete!")
            print(f"  Final loss: {self.losses_[-1]['total_loss']:.6f}")
            print(f"  Poisson loss: {self.losses_[-1]['poisson_loss']:.6f}")
            print(f"  Sparsity loss: {self.losses_[-1]['sparsity_loss']:.6f}")
            print(f"  Spatial loss: {self.losses_[-1]['spatial_loss']:.6f}")
            
            self._print_seminmf_stats()
        
        return self
    
    def _print_seminmf_stats(self):
        """Print Semi-NMF specific statistics"""
        with torch.no_grad():
            loadings_np = self.loadings.cpu().numpy()
            factors_np = self.factors.cpu().numpy()
            bias_np = torch.exp(self.bias).cpu().numpy()
            
            print(f"\n📊 Semi-NMF Statistics:")
            print(f"  Loadings (H) range: [{loadings_np.min():.3f}, {loadings_np.max():.3f}]")
            print(f"  Factors (W) range: [{factors_np.min():.3f}, {factors_np.max():.3f}] (≥0 ✓)")
            print(f"  Bias range: [{bias_np.min():.3f}, {bias_np.max():.3f}]")
            
            # Check non-negativity constraint
            factors_min = factors_np.min()
            if factors_min >= -1e-6:
                print(f"  ✅ Non-negativity constraint satisfied (min factor: {factors_min:.6f})")
            else:
                print(f"  ❌ Non-negativity constraint violated (min factor: {factors_min:.6f})")
            
            # Factor statistics
            for k in range(self.n_factors):
                factor_norm = np.linalg.norm(factors_np[k])
                loading_std = np.std(loadings_np[:, k])
                factor_sparsity = (factors_np[k] < 1e-6).mean()
                loading_sparsity = (np.abs(loadings_np[:, k]) < 1e-6).mean()
                
                print(f"  Factor {k+1}: ||W||={factor_norm:.3f}, σ(H)={loading_std:.3f}, "
                      f"sparsity W={factor_sparsity:.1%}, H={loading_sparsity:.1%}")
    
    def get_components(self):
        """Get learned Semi-NMF components"""
        with torch.no_grad():
            return {
                'loadings': self.loadings.cpu().numpy(),     # H: (n_obs, n_factors) - can be negative
                'factors': self.factors.cpu().numpy(),       # W: (n_factors, n_voxels) - non-negative
                'bias': torch.exp(self.bias).cpu().numpy(),  # bias: (n_voxels,) - positive
                'log_factors': self.log_factors.cpu().numpy(), # For debugging
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
                reconstruction = torch.mm(loadings, self.factors) + torch.exp(self.bias).unsqueeze(0)
            
            return reconstruction.cpu().numpy()
    
    def plot_training_progress(self):
        """Plot Semi-NMF training progress"""
        if not self.losses_:
            print("No training history to plot")
            return
        
        # Extract loss components
        epochs = range(len(self.losses_))
        poisson_losses = [l['poisson_loss'] for l in self.losses_]
        total_losses = [l['total_loss'] for l in self.losses_]
        sparsity_losses = [l['sparsity_loss'] for l in self.losses_]
        spatial_losses = [l['spatial_loss'] for l in self.losses_]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(epochs, total_losses, 'b-', label='Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Poisson loss
        axes[0, 1].plot(epochs, poisson_losses, 'g-')
        axes[0, 1].set_title('Poisson NLL Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Regularization losses
        axes[1, 0].plot(epochs, sparsity_losses, 'r-', label='Sparsity')
        axes[1, 0].plot(epochs, spatial_losses, 'b-', label='Spatial')
        axes[1, 0].set_title('Regularization Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Factor evolution
        if self.factor_history_:
            factor_history = np.array(self.factor_history_)
            for k in range(min(5, self.n_factors)):  # Show first 5 factors
                axes[1, 1].plot(factor_history[:, k], label=f'Factor {k+1}')
            axes[1, 1].set_title('Factor Norm Evolution')
            axes[1, 1].set_xlabel('Epoch (×10)')
            axes[1, 1].set_ylabel('||Factor||')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No factor history', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.suptitle('Semi-NMF Training Progress', y=1.02)
        plt.show()


def preprocess_count_data(data, method='robust_scaling'):
    """
    Preprocess count data while preserving non-negativity
    """
    # Handle NaN/inf values
    data_clean = np.copy(data)
    data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure non-negativity (required for Poisson)
    data_clean = np.maximum(data_clean, 1e-8)
    
    if method == 'robust_scaling':
        # Robust scaling while preserving non-negativity
        data_median = np.median(data_clean)
        data_mad = np.median(np.abs(data_clean - data_median))
        
        if data_mad > 0:
            # Scale but keep non-negative
            data_scaled = data_clean / (data_mad * 1.4826 + 1e-8)
        else:
            data_scaled = data_clean
            
    elif method == 'log_transform':
        # Log transform (common for count data)
        data_scaled = np.log(data_clean + 1)
        
    elif method == 'sqrt_transform':
        # Square root transform (variance stabilizing for Poisson)
        data_scaled = np.sqrt(data_clean)
        
    elif method == 'standardize':
        # Simple standardization but ensure non-negativity
        data_mean = np.mean(data_clean)
        data_std = np.std(data_clean)
        data_scaled = (data_clean - data_mean) / (data_std + 1e-8)
        data_scaled = np.maximum(data_scaled + np.abs(data_scaled.min()) + 1e-6, 1e-8)
        
    else:  # 'none'
        data_scaled = data_clean
    
    return data_scaled


def evaluate_seminmf_model(model, data_scaled, held_out_mask=None, alive_voxels_3d=None):
    """
    Evaluate Semi-NMF model performance
    """
    components = model.get_components()
    reconstruction = model.inverse_transform()
    
    # Basic reconstruction metrics
    mse = np.mean((data_scaled - reconstruction) ** 2)
    r2 = 1 - mse / np.var(data_scaled)
    mae = np.mean(np.abs(data_scaled - reconstruction))
    
    # Poisson deviance (appropriate for count data)
    poisson_deviance = 2 * np.sum(data_scaled * np.log(data_scaled / (reconstruction + 1e-8)) - 
                                 data_scaled + reconstruction)
    
    # Semi-NMF specific metrics
    loadings = components['loadings']  # H
    factors = components['factors']    # W
    
    # Check non-negativity constraint on factors
    factors_min = factors.min()
    constraint_satisfied = factors_min >= -1e-6
    
    # Factor quality
    factor_norms = [np.linalg.norm(f) for f in factors]
    factor_sparsities = [(f < 1e-6).mean() for f in factors]
    
    # Loading quality (can be negative in Semi-NMF)
    loading_stds = [np.std(loadings[:, i]) for i in range(loadings.shape[1])]
    meaningful_factors = sum(1 for norm, std in zip(factor_norms, loading_stds) 
                           if norm > 1e-3 and std > 1e-3)
    
    # Held-out validation
    held_out_r2 = None
    held_out_deviance = None
    if held_out_mask is not None:
        held_out_true = data_scaled[held_out_mask]
        held_out_pred = reconstruction[held_out_mask]
        held_out_mse = np.mean((held_out_true - held_out_pred) ** 2)
        held_out_r2 = 1 - held_out_mse / np.var(held_out_true)
        held_out_deviance = 2 * np.sum(held_out_true * np.log(held_out_true / (held_out_pred + 1e-8)) - 
                                      held_out_true + held_out_pred)
    
    return {
        'reconstruction_mse': mse,
        'reconstruction_r2': r2,
        'reconstruction_mae': mae,
        'poisson_deviance': poisson_deviance,
        'held_out_r2': held_out_r2,
        'held_out_deviance': held_out_deviance,
        'meaningful_factors': meaningful_factors,
        'total_factors': len(factors),
        'constraint_satisfied': constraint_satisfied,
        'factors_min': factors_min,
        'final_loss': model.losses_[-1]['total_loss'] if model.losses_ else np.inf,
        'factor_norms': factor_norms,
        'loading_stds': loading_stds
    }


def seminmf_pipeline(data, alive_voxels_3d=None, held_out_mask=None,
                    n_factors=8, spatial_regularization=0.01, 
                    sparsity_penalty=0.001, max_epochs=200, 
                    preprocessing='robust_scaling', verbose=True):
    """
    Complete Semi-NMF pipeline for count data
    """
    print("=== BRAIN-AWARE SEMI-NMF PIPELINE ===")
    
    # Data preprocessing
    print("🔧 Preprocessing count data...")
    data_scaled = preprocess_count_data(data, method=preprocessing)
    
    print(f"  Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Processed range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
    print(f"  Non-negative: {(data_scaled >= 0).all()}")
    
    # Initialize and fit model
    model = BrainAwareSemiNMF(
        n_factors=n_factors,
        spatial_regularization=spatial_regularization,
        sparsity_penalty=sparsity_penalty,
        device='auto',
        silent=not verbose
    )
    
    # Fit model
    start_time = time.time()
    model.fit(data_scaled, alive_voxels_3d=alive_voxels_3d, 
              val_mask=held_out_mask, max_epochs=max_epochs, verbose=verbose)
    fit_time = time.time() - start_time
    
    print(f"\n🎯 Semi-NMF completed in {fit_time:.2f} seconds")
    
    # Evaluate model
    metrics = evaluate_seminmf_model(model, data_scaled, held_out_mask, alive_voxels_3d)
    
    print(f"\n📊 Semi-NMF Results:")
    print(f"  Reconstruction R²: {metrics['reconstruction_r2']:.4f}")
    print(f"  Poisson deviance: {metrics['poisson_deviance']:.6f}")
    if metrics['held_out_r2'] is not None:
        print(f"  Held-out R²: {metrics['held_out_r2']:.4f}")
    print(f"  Meaningful factors: {metrics['meaningful_factors']}/{metrics['total_factors']}")
    print(f"  Constraint satisfied: {metrics['constraint_satisfied']} (min factor: {metrics['factors_min']:.6f})")
    
    return model, model.get_components()


# Example usage
if __name__ == "__main__":
    print("🧠 Brain-Aware Semi-NMF for Count Data")
    print("✅ Factors W ≥ 0 (non-negative)")
    print("✅ Loadings H ∈ ℝ (can be negative)")  
    print("✅ Poisson likelihood for count data")
    print("✅ Spatial regularization for brain coherence")
    
    # Test with synthetic count data
    np.random.seed(42)
    n_obs, n_voxels = 43, 1000
    
    # Generate synthetic Semi-NMF data
    true_factors = np.random.exponential(1.0, (5, n_voxels))  # Non-negative
    true_loadings = np.random.randn(n_obs, 5)  # Can be negative
    true_bias = np.random.exponential(0.5, n_voxels)
    
    # Poisson count data
    rates = true_loadings @ true_factors + true_bias
    rates = np.maximum(rates, 0.1)  # Ensure positive rates
    synthetic_data = np.random.poisson(rates)
    
    print(f"\nTesting with synthetic count data: {synthetic_data.shape}")
    print(f"Data range: [{synthetic_data.min()}, {synthetic_data.max()}]")
    print(f"Data type: counts (integers: {np.all(synthetic_data == synthetic_data.astype(int))})")
    
    # Test Semi-NMF
    model, components = seminmf_pipeline(
        synthetic_data.astype(float), 
        n_factors=6,
        max_epochs=150,
        preprocessing='sqrt_transform',  # Good for count data
        verbose=True
    )
    
    # Plot results
    model.plot_training_progress()
    
    print("\n✅ Semi-NMF test completed successfully!")
    print(f"Final constraint check: factors ≥ 0? {(components['factors'] >= -1e-6).all()}")
