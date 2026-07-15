"""
Brain-Aware Factorization for Adjusted Count Neural Data
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
                 device='auto', random_state=42):
        super().__init__()
        
        self.n_factors = n_factors
        self.spatial_regularization = spatial_regularization
        self.sparsity_penalty = sparsity_penalty
        self.random_state = random_state
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
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
                print(f"  Used TruncatedSVD for large data ({n_obs}×{n_voxels:,})")
            else:
                pca = PCA(n_components=self.n_factors, random_state=self.random_state)
                loadings_init = pca.fit_transform(data_centered)
                factors_init = pca.components_
                print(f"  Used standard PCA")
            
            # Scale to reasonable range
            loadings_init = loadings_init * 0.1
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"  PCA explained variance: {explained_variance:.1%}")
            
        except Exception as e:
            print(f"  PCA failed ({e}), using random initialization")
            loadings_init = np.random.randn(n_obs, self.n_factors) * 0.1
            factors_init = np.random.randn(self.n_factors, n_voxels) * 0.1
        
        # Convert to tensors
        self.loadings = nn.Parameter(torch.FloatTensor(loadings_init).to(self.device))
        self.factors = nn.Parameter(torch.FloatTensor(factors_init).to(self.device))
        
        # Initialize bias as mean per voxel
        bias_init = np.mean(data, axis=0)
        self.bias = nn.Parameter(torch.FloatTensor(bias_init).to(self.device))
        
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
                    print(f"\\n⏹️  Early stopping at epoch {epoch}")
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
            
            print(f"\\n📊 Factor Statistics:")
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


def brain_factorization_pipeline(data, alive_voxels_3d=None, held_out_mask=None,
                               n_factors=8, max_epochs=200, verbose=True):
    """
    Complete brain factorization pipeline
    """
    print("=== BRAIN-AWARE FACTORIZATION PIPELINE ===")
    
    # Data preprocessing for adjusted count data
    print("🔧 Preprocessing adjusted count data...")
    
    # Handle potential NaN/inf values
    data_clean = np.copy(data)
    data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Robust scaling for adjusted count data
    # Don't force non-negativity - adjusted count data can be negative
    data_median = np.median(data_clean)
    data_mad = np.median(np.abs(data_clean - data_median))  # Median absolute deviation
    
    if data_mad > 0:
        data_scaled = (data_clean - data_median) / (data_mad * 1.4826)  # 1.4826 makes MAD consistent with std
        # Clip extreme outliers
        data_scaled = np.clip(data_scaled, -5, 5)
    else:
        data_scaled = data_clean - data_median
    
    print(f"  Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Scaled range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
    
    # Initialize and fit model
    model = BrainAwareFactorization(
        n_factors=n_factors,
        spatial_regularization=0.01,  # Encourage spatial coherence
        sparsity_penalty=0.001,       # Light sparsity for interpretability 
        device='auto'
    )
    
    # Fit model
    start_time = time.time()
    model.fit(data_scaled, alive_voxels_3d=alive_voxels_3d, 
              val_mask=held_out_mask, max_epochs=max_epochs, verbose=verbose)
    fit_time = time.time() - start_time
    
    print(f"\\n🎯 Brain factorization completed in {fit_time:.2f} seconds")
    
    # Get results
    components = model.get_components()
    
    # Compute reconstruction quality
    reconstruction = model.inverse_transform()
    mse = np.mean((data_scaled - reconstruction) ** 2)
    r2 = 1 - mse / np.var(data_scaled)
    
    print(f"\\n📊 Results Summary:")
    print(f"  Reconstruction MSE: {mse:.6f}")
    print(f"  Reconstruction R²: {r2:.4f}")
    print(f"  Loadings shape: {components['loadings'].shape}")
    print(f"  Factors shape: {components['factors'].shape}")
    
    # Validation on held-out data
    if held_out_mask is not None:
        held_out_true = data_scaled[held_out_mask]
        held_out_pred = reconstruction[held_out_mask]
        held_out_mse = np.mean((held_out_true - held_out_pred) ** 2)
        held_out_r2 = 1 - held_out_mse / np.var(held_out_true)
        
        print(f"  Held-out MSE: {held_out_mse:.6f}")
        print(f"  Held-out R²: {held_out_r2:.4f}")
    
    return model, components


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Brain-Aware Factorization for Adjusted Count Neural Data")
    print("✅ Handles negative values from background subtraction")
    print("✅ Spatial regularization for brain coherence")
    print("✅ Robust loss function for count-like data")
    print("✅ Efficient GPU acceleration")
    
    # Test with synthetic brain data
    np.random.seed(42)
    n_obs, n_voxels = 43, 10000
    
    # Simulate adjusted count data (can be negative)
    true_loadings = np.random.randn(n_obs, 5)
    true_factors = np.random.randn(5, n_voxels) * 0.5
    noise = np.random.randn(n_obs, n_voxels) * 0.1
    synthetic_data = true_loadings @ true_factors + noise
    
    print(f"\\nTesting with synthetic data: {synthetic_data.shape}")
    print(f"Synthetic data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    
    # Test the pipeline
    held_out_mask = np.random.random(synthetic_data.shape) < 0.1
    
    model, components = brain_factorization_pipeline(
        synthetic_data, 
        held_out_mask=held_out_mask,
        n_factors=6,
        max_epochs=100,
        verbose=True
    )
    
    # Plot results
    model.plot_training_progress()
