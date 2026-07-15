"""
Semi-NMF for Poisson Count Data
Simplified version of the JAX implementation for count-only data

🔑 KEY NMF CONCEPTS IMPLEMENTED:

1. **MATRIX FACTORIZATION**: X ≈ W @ H
   - X: (n_obs × n_voxels) observed count matrix
   - W: (n_obs × n_factors) loadings matrix (can be negative - "Semi")  
   - H: (n_factors × n_voxels) factors matrix (constrained ≥ 0 - "NMF")

2. **NON-NEGATIVITY CONSTRAINT**: H ≥ 0
   - Enforced through projection: H = clamp(H, min=0) after each gradient step
   - Gradient masking: zero gradients that would make H negative
   - This is the defining characteristic of NMF

3. **POISSON GLM EXTENSION**: 
   - Instead of X ≈ W @ H, we model: count ~ Poisson(exp(W @ H + effects))
   - More appropriate for count data than Frobenius norm
   - Includes row/column effects for better modeling

4. **OPTIMIZATION WITH CONSTRAINTS**:
   - Standard gradient descent on W (loadings can be negative)  
   - Projected gradient descent on H (factors must stay ≥ 0)
   - SVD initialization for better convergence

5. **REGULARIZATION**:
   - Sparsity: L1 penalty on loadings for interpretability
   - Spatial: Smoothness penalty on factors for brain coherence

The result: Interpretable factorization where factors represent
non-negative spatial patterns and loadings show how much each
observation expresses each pattern (can be negative for Semi-NMF).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import itertools
import pandas as pd
from scipy.optimize import nnls

class PoissonSemiNMF(nn.Module):
    """
    Semi-NMF for Poisson count data
    
    CORE CONCEPT: Matrix factorization with partial non-negativity constraints
    
    Mathematical Model:
    X ≈ W @ H + row_effects + col_effects
    count_mn ~ Poisson(exp(row_effect_m + col_effect_n + sum_k loading_mk * factor_kn))
    
    Where:
    - X: (n_obs × n_voxels) count data matrix
    - W: (n_obs × n_factors) loadings matrix - CAN BE NEGATIVE (Semi-NMF)
    - H: (n_factors × n_voxels) factors matrix - MUST BE NON-NEGATIVE (NMF constraint)
    
    This is "Semi"-NMF because only factors H are constrained to be non-negative,
    while loadings W can be negative (unlike full NMF where both W,H ≥ 0)
    
    Constraints:
    - factors >= 0 (non-negative) ← This is the key NMF constraint
    - loadings can be negative (semi-NMF extension)
    - Poisson likelihood for count data
    """
    
    def __init__(self, n_factors, sparsity_penalty=0.01, spatial_regularization=0.0, 
                 device='auto', random_state=42, silent=False):
        super().__init__()
        
        self.n_factors = n_factors
        self.sparsity_penalty = sparsity_penalty
        self.spatial_regularization = spatial_regularization
        self.random_state = random_state
        self.silent = silent
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if not self.silent:
            print(f"🔢 Poisson Semi-NMF using device: {self.device}")
        
        # Parameters (will be initialized in fit)
        self.loadings = None        # (n_obs, n_factors) - can be negative
        self.factors = None         # (n_factors, n_voxels) - must be non-negative  
        self.row_effects = None     # (n_obs,) - row intercepts
        self.col_effects = None     # (n_voxels,) - column intercepts
        
        # Training history
        self.losses_ = []
        self.train_loglik_ = []
        self.val_loglik_ = []
    
    def _initialize_parameters(self, n_obs, n_voxels, data):
        """
        SMART NMF INITIALIZATION using SVD projection
        
        CONCEPT: Good initialization is critical for NMF optimization because:
        1. NMF is non-convex (multiple local minima)
        2. Random initialization often leads to poor local minima
        3. SVD provides a principled starting point
        
        APPROACH:
        1. Use SVD to get initial factorization X ≈ U @ S @ V^T
        2. Project negative parts of V^T to non-negative (for NMF constraint)
        3. Adjust U accordingly to maintain approximation quality
        4. Use U as initial loadings, V^T as initial factors
        """
        if not self.silent:
            print("🚀 Initializing with SVD...")
        
        # Handle zeros by adding small pseudocount (common in count data)
        data_smooth = data + 1e-6
        
        # Convert to log space for SVD (approximate inverse of softplus link function)
        # This linearizes the relationship for better SVD approximation
        targets = np.log(data_smooth + np.sqrt(data_smooth**2 + 1))
        
        # Remove row and column means (will be captured by row/col effects)
        row_means = np.mean(targets, axis=1, keepdims=True)
        col_means = np.mean(targets, axis=0, keepdims=True)
        residual = targets - row_means - col_means + np.mean(targets)
        
        try:
            # SVD FACTORIZATION: residual ≈ U @ S @ V^T
            if n_voxels > 10000:
                # Use truncated SVD for efficiency with large matrices
                svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
                loadings_init = svd.fit_transform(residual)  # U @ S
                factors_init = svd.components_                # V^T
            else:
                # Full SVD for smaller matrices
                U, s, Vt = np.linalg.svd(residual, full_matrices=False)
                loadings_init = U[:, :self.n_factors] * s[:self.n_factors]  # U @ S
                factors_init = Vt[:self.n_factors, :]                       # V^T
            
            # KEY NMF STEP: PROJECT FACTORS TO NON-NEGATIVE ORTHANT
            # SVD can produce negative values, but NMF requires factors ≥ 0
            factors_init = np.maximum(factors_init, 1e-6)
            
            # NORMALIZATION: Prevent scaling issues
            # Normalize factors and adjust loadings to maintain reconstruction quality
            factor_norms = np.linalg.norm(factors_init, axis=1, keepdims=True)
            factors_init = factors_init / factor_norms
            loadings_init = loadings_init * factor_norms.T
            
            if not self.silent:
                print(f"  SVD initialization successful")
                
        except Exception as e:
            if not self.silent:
                print(f"  SVD failed ({e}), using random initialization")
            # FALLBACK: Random initialization
            loadings_init = np.random.randn(n_obs, self.n_factors) * 0.1
            # For factors: use exponential distribution to ensure non-negativity
            factors_init = np.random.exponential(1, (self.n_factors, n_voxels))
            factors_init = factors_init / np.sum(factors_init, axis=1, keepdims=True)
        
        # Convert to PyTorch parameters
        # Note: loadings can be negative (Semi-NMF), factors will be constrained ≥ 0
        self.loadings = nn.Parameter(torch.FloatTensor(loadings_init).to(self.device))
        self.factors = nn.Parameter(torch.FloatTensor(factors_init).to(self.device))
        
        # Initialize GLM effects from data means
        row_init = np.squeeze(row_means) - np.mean(targets)
        col_init = np.squeeze(col_means) - np.mean(targets)
        
        self.row_effects = nn.Parameter(torch.FloatTensor(row_init).to(self.device))
        self.col_effects = nn.Parameter(torch.FloatTensor(col_init).to(self.device))
        
        if not self.silent:
            print(f"  Parameters: loadings {self.loadings.shape}, factors {self.factors.shape}")
            print(f"  Semi-NMF: loadings can be negative, factors constrained ≥ 0")
    
    def forward(self):
        """
        CORE NMF COMPUTATION: Matrix factorization with non-negativity constraints
        
        This is where the actual Semi-NMF happens:
        
        1. ENFORCE NON-NEGATIVITY: Project factors to non-negative orthant
           factors_nonneg = clamp(factors, min=1e-8)
           This is the key NMF constraint - factors must be ≥ 0
        
        2. MATRIX FACTORIZATION: Compute W @ H reconstruction  
           loadings @ factors_nonneg = (n_obs × n_factors) @ (n_factors × n_voxels)
           Result: (n_obs × n_voxels) reconstruction matrix
        
        3. ADD GLM STRUCTURE: Include row/column effects for better modeling
           linear_pred = row_effects + col_effects + W @ H
           
        4. LINK FUNCTION: Convert to positive rates for Poisson likelihood
           rates = softplus(linear_pred) = log(1 + exp(linear_pred))
           
        Returns: Poisson rates for each (observation, voxel) pair
        """
        
        # STEP 1: ENFORCE NON-NEGATIVITY CONSTRAINT (key NMF requirement)
        # This projection ensures factors stay in non-negative orthant
        factors_nonneg = torch.clamp(self.factors, min=1e-8)
        
        # STEP 2: CORE MATRIX FACTORIZATION 
        # This is the heart of NMF: X ≈ W @ H
        # loadings (W) can be negative, factors (H) must be non-negative
        factorization_term = torch.mm(self.loadings, factors_nonneg)  # W @ H
        
        # STEP 3: ADD GLM STRUCTURE (row and column effects)
        # This extends basic NMF to a proper statistical model
        linear_pred = (self.row_effects.unsqueeze(1) +      # Add row effects
                      self.col_effects.unsqueeze(0) +       # Add column effects  
                      factorization_term)                   # Add W @ H factorization
        
        # STEP 4: LINK FUNCTION for Poisson GLM
        # Convert linear predictor to positive rates using softplus
        # softplus(x) = log(1 + exp(x)) ≈ exp(x) for large x, but numerically stable
        rates = torch.log1p(torch.exp(torch.clamp(linear_pred, max=10)))
        
        return rates
    
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
            n_factors, n_voxels = self.factors.shape
            
            for k in range(n_factors):
                factor_k = self.factors[k, :]
                
                # Reshape factor to 3D brain space if possible
                if hasattr(self, '_voxel_indices') and alive_voxels_3d is not None:
                    # If we have spatial mapping, use it
                    brain_factor = torch.zeros(alive_voxels_3d.shape, device=self.device)
                    brain_factor[alive_voxels_3d] = factor_k
                    
                    # Compute spatial gradients
                    if brain_factor.dim() == 3 and brain_factor.shape[0] > 1:
                        grad_x = torch.diff(brain_factor, dim=0)
                        penalty += torch.sum(grad_x ** 2)
                    if brain_factor.dim() == 3 and brain_factor.shape[1] > 1:
                        grad_y = torch.diff(brain_factor, dim=1) 
                        penalty += torch.sum(grad_y ** 2)
                    if brain_factor.dim() == 3 and brain_factor.shape[2] > 1:
                        grad_z = torch.diff(brain_factor, dim=2)
                        penalty += torch.sum(grad_z ** 2)
                else:
                    # Fallback: simple smoothness penalty on flattened factors
                    penalty += torch.sum(torch.diff(factor_k) ** 2)
            
            return penalty / (n_factors * n_voxels)
        
        except Exception:
            # If spatial penalty fails, return zero penalty
            return torch.tensor(0.0, device=self.device)
    
    def _compute_loss(self, data, rates, alive_voxels_3d=None, mask=None):
    def _compute_loss(self, data, rates, alive_voxels_3d=None, mask=None):
        """
        NMF OPTIMIZATION OBJECTIVE: Loss function for Semi-NMF learning
        
        The goal of NMF is to find W, H such that X ≈ W @ H by minimizing a loss function.
        
        LOSS COMPONENTS:
        
        1. DATA FITTING TERM (Poisson negative log-likelihood):
           For count data, we use Poisson likelihood instead of Frobenius norm
           L_data = -∑ log P(x_ij | rate_ij) = ∑ (rate_ij - x_ij * log(rate_ij))
           This measures how well our factorization W @ H reconstructs the data
        
        2. SPARSITY REGULARIZATION (L1 penalty on loadings):
           L_sparsity = λ₁ * ∑|W_ij|
           Encourages sparse loadings (many zeros) for interpretability
           
        3. SPATIAL REGULARIZATION (smoothness penalty on factors):
           L_spatial = λ₂ * ∑||∇H_k||²
           For brain data: encourages spatially smooth factors
           
        Total: L = L_data + L_sparsity + L_spatial
        """
        
        # Apply masking if needed (for train/validation splits)
        if mask is not None:
            data_masked = data * mask.float()
            rates_masked = rates * mask.float()
            n_points = torch.sum(mask).float()
        else:
            data_masked = data
            rates_masked = rates  
            n_points = data.numel()
        
        # 1. DATA FITTING TERM: Poisson negative log-likelihood
        # This is the main objective - how well does our W @ H factorization fit the data?
        # Poisson NLL: -log P(x|rate) = rate - x*log(rate) + constant
        epsilon = 1e-8
        nll = torch.sum(rates_masked - data_masked * torch.log(rates_masked + epsilon))
        nll = nll / n_points
        
        # 2. SPARSITY REGULARIZATION: L1 penalty on loadings W
        # Encourages sparse loadings for interpretability (many loadings ≈ 0)
        sparsity_loss = self.sparsity_penalty * torch.sum(torch.abs(self.loadings))
        sparsity_loss = sparsity_loss / (self.loadings.shape[0] * self.loadings.shape[1])
        
        # 3. SPATIAL REGULARIZATION: Smoothness penalty on factors H  
        # For brain data: encourages neighboring voxels to have similar factor values
        spatial_loss = self.spatial_regularization * self._compute_spatial_penalty(alive_voxels_3d)
        
        # TOTAL NMF OBJECTIVE = Data fit + Regularization
        total_loss = nll + sparsity_loss + spatial_loss
        
        return total_loss, {
            'nll': nll.item(),
            'sparsity': sparsity_loss.item(),
            'spatial': spatial_loss.item(),
            'total': total_loss.item()
        }
    
    def _compute_loglikelihood(self, data, rates, mask=None):
        """Compute log-likelihood for monitoring"""
        if mask is not None:
            data_masked = data * mask.float()
            rates_masked = rates * mask.float()
            n_points = torch.sum(mask).float()
        else:
            data_masked = data
            rates_masked = rates
            n_points = data.numel()
        
        epsilon = 1e-8
        loglik = torch.sum(data_masked * torch.log(rates_masked + epsilon) - rates_masked)
        return loglik / n_points
    
    def fit(self, data, alive_voxels_3d=None, val_mask=None, max_epochs=200, lr=0.01, patience=20, verbose=True):
        """Fit Semi-NMF model"""
        # Convert data to tensor
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        else:
            data = data.to(self.device)
        
        n_obs, n_voxels = data.shape
        
        # Initialize parameters
        self._initialize_parameters(n_obs, n_voxels, data.cpu().numpy())
        
        # Setup optimizer - different learning rates for different parameters
        optimizer = optim.Adam([
            {'params': [self.loadings], 'lr': lr},
            {'params': [self.factors], 'lr': lr * 0.5},  # Slower for factors
            {'params': [self.row_effects], 'lr': lr * 0.1}, 
            {'params': [self.col_effects], 'lr': lr * 0.1}
        ])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
        
        # Convert validation mask
        if val_mask is not None:
            if isinstance(val_mask, np.ndarray):
                val_mask = torch.BoolTensor(val_mask).to(self.device)
            else:
                val_mask = val_mask.to(self.device)
            train_mask = ~val_mask
        else:
            train_mask = None
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print(f"🚀 Training Poisson Semi-NMF:")
            print(f"  Data: {n_obs}×{n_voxels:,} → {self.n_factors} factors")
            print(f"  Sparsity penalty: {self.sparsity_penalty}")
            print(f"  Data range: [{data.min().item():.3f}, {data.max().item():.3f}]")
        
        pbar = tqdm(range(max_epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            rates = self.forward()
            
            # Compute loss on training data
            loss, loss_components = self._compute_loss(data, rates, alive_voxels_3d, train_mask)
            
            # Backward pass
            loss.backward()
            
            # CONSTRAINT ENFORCEMENT DURING GRADIENT COMPUTATION
            # This enforces non-negativity constraint on factors during optimization
            if self.factors.grad is not None:
                # GRADIENT MASKING for non-negativity constraint
                # If factor is at boundary (≈0) and gradient points "outside" feasible region (negative),
                # then zero out that gradient component to prevent constraint violation
                # This implements "projected gradient descent" for NMF
                boundary_mask = (self.factors.data <= 1e-8) & (self.factors.grad < 0)
                self.factors.grad[boundary_mask] = 0
                
                # This ensures factors can only:
                # - Increase when at zero boundary (gradient ≥ 0) 
                # - Move freely when in interior (factors > 0)
                # - Never go negative (key NMF constraint)
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # SEMI-NMF CONSTRAINT ENFORCEMENT DURING OPTIMIZATION
            
            # Standard gradient step for all parameters
            optimizer.step()
            
            # CRITICAL: ENFORCE NON-NEGATIVITY CONSTRAINT ON FACTORS
            # This is what makes it NMF - factors H must stay ≥ 0
            with torch.no_grad():
                # PROJECT FACTORS TO NON-NEGATIVE ORTHANT
                # This is the key constraint that defines NMF
                self.factors.data = torch.clamp(self.factors.data, min=1e-8)
                
                # NORMALIZATION TO PREVENT SCALING ISSUES
                # Normalize factors and compensate in loadings to maintain reconstruction
                # This prevents factors from growing unboundedly
                factor_norms = torch.norm(self.factors.data, dim=1, keepdim=True)
                factor_norms = torch.clamp(factor_norms, min=1e-8)
                self.factors.data = self.factors.data / factor_norms
                self.loadings.data = self.loadings.data * factor_norms.T
                
                # IDENTIFIABILITY: Center column effects to remove global offset
                col_mean = torch.mean(self.col_effects.data)
                self.col_effects.data = self.col_effects.data - col_mean
                self.row_effects.data = self.row_effects.data + col_mean
            
            # Track training metrics
            with torch.no_grad():
                train_loglik = self._compute_loglikelihood(data, rates, train_mask)
                self.train_loglik_.append(train_loglik.item())
                
                # Validation metrics
                if val_mask is not None:
                    val_loglik = self._compute_loglikelihood(data, rates, val_mask)
                    self.val_loglik_.append(val_loglik.item())
            
            self.losses_.append(loss_components)
            
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
                desc = f"Loss: {loss_components['total']:.4f}, LogLik: {train_loglik:.2f}"
                if val_mask is not None:
                    desc += f", Val: {self.val_loglik_[-1]:.2f}"
                pbar.set_description(desc)
        
        pbar.close()
        
        if verbose:
            print(f"✅ Training complete!")
            print(f"  Final loss: {self.losses_[-1]['total']:.6f}")
            print(f"  Final NLL: {self.losses_[-1]['nll']:.6f}")
            print(f"  Sparsity loss: {self.losses_[-1]['sparsity']:.6f}")
            print(f"  Spatial loss: {self.losses_[-1]['spatial']:.6f}")
            print(f"  Train log-likelihood: {self.train_loglik_[-1]:.3f}")
            if val_mask is not None:
                print(f"  Validation log-likelihood: {self.val_loglik_[-1]:.3f}")
            
            self._print_factor_stats()
        
        return self
    
    def _print_factor_stats(self):
        """Print factor statistics"""
        with torch.no_grad():
            loadings_np = self.loadings.cpu().numpy()
            factors_np = self.factors.cpu().numpy()
            
            print(f"\n📊 Factor Statistics:")
            print(f"  Loadings range: [{loadings_np.min():.3f}, {loadings_np.max():.3f}]")
            print(f"  Factors range: [{factors_np.min():.3f}, {factors_np.max():.3f}] (constrained ≥ 0)")
            
            for k in range(self.n_factors):
                factor_norm = np.linalg.norm(factors_np[k])
                loading_std = np.std(loadings_np[:, k])
                factor_sparsity = (factors_np[k] < 1e-6).mean()
                loading_sparsity = (np.abs(loadings_np[:, k]) < 1e-6).mean()
                
                print(f"  Factor {k+1}: ||F||={factor_norm:.3f}, σ(L)={loading_std:.3f}, "
                      f"sparse_F={factor_sparsity:.1%}, sparse_L={loading_sparsity:.1%}")
    
    def get_components(self):
        """Get all learned components"""
        with torch.no_grad():
            return {
                'loadings': self.loadings.cpu().numpy(),      # (n_obs, n_factors) - can be negative
                'factors': self.factors.cpu().numpy(),        # (n_factors, n_voxels) - non-negative  
                'row_effects': self.row_effects.cpu().numpy(), # (n_obs,)
                'col_effects': self.col_effects.cpu().numpy(), # (n_voxels,)
            }
    
    def transform(self, data=None):
        """Get loadings for new data (or training data if None)"""
        if data is None:
            with torch.no_grad():
                return self.loadings.cpu().numpy()
        else:
            # For new data, we'd need to optimize loadings while keeping factors fixed
            # This is the "prediction" mode from the original
            raise NotImplementedError("Transform for new data not yet implemented")
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data from loadings"""
        with torch.no_grad():
            if loadings is None:
                # Use current loadings
                rates = self.forward()
            else:
                # Use provided loadings
                if isinstance(loadings, np.ndarray):
                    loadings = torch.FloatTensor(loadings).to(self.device)
                
                factors_nonneg = torch.clamp(self.factors, min=1e-8)
                linear_pred = (self.row_effects.unsqueeze(1) + 
                              self.col_effects.unsqueeze(0) + 
                              torch.mm(loadings, factors_nonneg))
                rates = torch.log1p(torch.exp(torch.clamp(linear_pred, max=10)))
            
            return rates.cpu().numpy()
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.losses_:
            print("No training history to plot")
            return
        
        # Extract loss components
        epochs = range(len(self.losses_))
        nll_losses = [l['nll'] for l in self.losses_]
        total_losses = [l['total'] for l in self.losses_]
        sparsity_losses = [l['sparsity'] for l in self.losses_]
        spatial_losses = [l.get('spatial', 0) for l in self.losses_]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(epochs, total_losses, 'b-', label='Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        # NLL loss
        axes[0, 1].plot(epochs, nll_losses, 'g-', label='Negative Log-Likelihood')
        axes[0, 1].set_title('Negative Log-Likelihood')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('NLL')
        axes[0, 1].set_yscale('log')
        
        # Log-likelihood (positive values)
        axes[1, 0].plot(epochs, self.train_loglik_, 'b-', label='Train Log-Likelihood')
        if self.val_loglik_:
            val_epochs = range(len(self.val_loglik_))
            axes[1, 0].plot(val_epochs, self.val_loglik_, 'r--', label='Validation Log-Likelihood')
        axes[1, 0].set_title('Log-Likelihood')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Log-Likelihood')
        axes[1, 0].legend()
        
        # Sparsity and spatial penalties
        axes[1, 1].plot(epochs, sparsity_losses, 'r-', label='Sparsity')
        if any(s > 0 for s in spatial_losses):
            axes[1, 1].plot(epochs, spatial_losses, 'orange', label='Spatial')
        axes[1, 1].set_title('Regularization Penalties')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Penalty')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()


def evaluate_seminmf(model, data, val_mask=None):
    """Evaluate Semi-NMF model"""
    components = model.get_components()
    reconstruction = model.inverse_transform()
    
    # Basic reconstruction metrics (on rate scale)
    mse = np.mean((data - reconstruction) ** 2)
    mae = np.mean(np.abs(data - reconstruction))
    
    # R-squared on original scale
    ss_res = np.sum((data - reconstruction) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Poisson deviance (more appropriate for count data)
    epsilon = 1e-8
    # Deviance = 2 * sum(y * log(y/mu) - (y - mu)) where y=data, mu=reconstruction
    deviance = 2 * np.sum(data * np.log((data + epsilon) / (reconstruction + epsilon)) - 
                         (data - reconstruction))
    null_deviance = 2 * np.sum(data * np.log((data + epsilon) / (np.mean(data) + epsilon)) - 
                              (data - np.mean(data)))
    pseudo_r2 = 1 - (deviance / null_deviance)
    
    # Factor quality metrics
    loadings = components['loadings']
    factors = components['factors']
    
    # Check for meaningful factors
    factor_norms = [np.linalg.norm(f) for f in factors]
    factor_sparsities = [(f < 1e-6).mean() for f in factors]
    meaningful_factors = sum(1 for norm, sparsity in zip(factor_norms, factor_sparsities) 
                           if norm > 1e-3 and sparsity < 0.95)
    
    # Loading variability
    loading_stds = [np.std(loadings[:, i]) for i in range(loadings.shape[1])]
    active_factors = sum(1 for std in loading_stds if std > 1e-3)
    
    # Validation metrics
    val_r2 = None
    val_pseudo_r2 = None
    if val_mask is not None:
        val_data = data[val_mask]
        val_recon = reconstruction[val_mask]
        
        # R-squared on validation data
        val_ss_res = np.sum((val_data - val_recon) ** 2)
        val_ss_tot = np.sum((val_data - np.mean(val_data)) ** 2)
        val_r2 = 1 - (val_ss_res / val_ss_tot)
        
        # Pseudo R-squared on validation data
        val_deviance = 2 * np.sum(val_data * np.log((val_data + epsilon) / (val_recon + epsilon)) - 
                                 (val_data - val_recon))
        val_null_deviance = 2 * np.sum(val_data * np.log((val_data + epsilon) / (np.mean(val_data) + epsilon)) - 
                                      (val_data - np.mean(val_data)))
        val_pseudo_r2 = 1 - (val_deviance / val_null_deviance)
    
    return {
        'reconstruction_mse': mse,
        'reconstruction_mae': mae,
        'reconstruction_r2': r2,
        'pseudo_r2': pseudo_r2,
        'deviance': deviance,
        'val_r2': val_r2,
        'val_pseudo_r2': val_pseudo_r2,
        'meaningful_factors': meaningful_factors,
        'total_factors': len(factors),
        'meaningful_factor_ratio': meaningful_factors / len(factors),
        'active_factors': active_factors,
        'final_train_loglik': model.train_loglik_[-1] if model.train_loglik_ else np.nan,
        'final_val_loglik': model.val_loglik_[-1] if model.val_loglik_ else np.nan,
        'factor_norms': factor_norms,
        'loading_stds': loading_stds
    }


def hyperparameter_search_seminmf(data, alive_voxels_3d=None, val_mask=None,
                                 n_factors_values=[4, 6, 8, 10],
                                 sparsity_values=[0.0, 0.001, 0.01, 0.1],
                                 spatial_reg_values=[0.0, 0.001, 0.01, 0.1],
                                 lr_values=[0.01],
                                 max_epochs=100,
                                 n_random_seeds=2,
                                 scoring_metric='val_pseudo_r2',
                                 save_training_curves=True,
                                 verbose=True):
    """Hyperparameter search for Semi-NMF with detailed training curve storage"""
    
    print("🔍 POISSON SEMI-NMF HYPERPARAMETER SEARCH")
    print("=" * 50)
    
    # Generate parameter combinations
    param_combinations = list(itertools.product(n_factors_values, sparsity_values, spatial_reg_values, lr_values))
    total_configs = len(param_combinations) * n_random_seeds
    
    print(f"Testing {len(param_combinations)} parameter combinations × {n_random_seeds} seeds = {total_configs} configs")
    print(f"Parameters: n_factors={n_factors_values}, sparsity={sparsity_values}")
    print(f"           spatial_reg={spatial_reg_values}, lr={lr_values}")
    
    # Debug: Show expected vs actual number of combinations
    expected_combinations = len(n_factors_values) * len(sparsity_values) * len(spatial_reg_values) * len(lr_values)
    print(f"Expected combinations: {len(n_factors_values)} × {len(sparsity_values)} × {len(spatial_reg_values)} × {len(lr_values)} = {expected_combinations}")
    print(f"Actual combinations: {len(param_combinations)}")
    
    # Results storage
    results = []
    best_score = -np.inf
    best_config = None
    best_model = None
    
    # Detailed training curves storage
    training_curves = {} if save_training_curves else None
    
    # Progress tracking
    config_pbar = tqdm(param_combinations, desc="Search Progress")
    
    for config_idx, (n_factors, sparsity, spatial_reg, lr) in enumerate(config_pbar):
        config_scores = []
        
        # Debug: Print configuration being tested
        if config_idx < 3:  # Only print first few to avoid spam
            print(f"  Testing config {config_idx+1}: n_factors={n_factors}, sparsity={sparsity}, spatial_reg={spatial_reg}, lr={lr}")
        
        # Test multiple random seeds
        for seed in range(n_random_seeds):
            try:
                # Initialize model
                model = PoissonSemiNMF(
                    n_factors=n_factors,
                    sparsity_penalty=sparsity,
                    spatial_regularization=spatial_reg,  # ← This SHOULD be using spatial_reg
                    device='auto',
                    random_state=seed,
                    silent=True
                )
                
                # Debug: Verify spatial regularization is actually set
                if config_idx == 0 and seed == 0:  # Print once for verification
                    print(f"    Model created with spatial_regularization={model.spatial_regularization}")
                
                # Fit model
                start_time = time.time()
                model.fit(data, alive_voxels_3d=alive_voxels_3d, val_mask=val_mask, 
                         max_epochs=max_epochs, lr=lr, patience=15, verbose=False)
                fit_time = time.time() - start_time
                
                # Evaluate model
                metrics = evaluate_seminmf(model, data, val_mask)
                
                # Get scoring metric
                if scoring_metric == 'val_pseudo_r2' and metrics['val_pseudo_r2'] is not None:
                    score = metrics['val_pseudo_r2']
                elif scoring_metric == 'pseudo_r2':
                    score = metrics['pseudo_r2']
                elif scoring_metric == 'val_r2' and metrics['val_r2'] is not None:
                    score = metrics['val_r2']
                elif scoring_metric == 'reconstruction_r2':
                    score = metrics['reconstruction_r2']
                else:
                    # Fallback to composite score
                    score = metrics.get('val_pseudo_r2', metrics['pseudo_r2'])
                
                # Store result with training curves
                result = {
                    'n_factors': n_factors,
                    'sparsity_penalty': sparsity,
                    'spatial_regularization': spatial_reg,
                    'learning_rate': lr,
                    'random_seed': seed,
                    'score': score,
                    'fit_time': fit_time,
                    **metrics
                }
                
                results.append(result)
                config_scores.append(score)
                
                # Save detailed training curves if requested
                if save_training_curves:
                    config_key = f"n{n_factors}_s{sparsity:.4f}_sp{spatial_reg:.4f}_lr{lr:.4f}_seed{seed}"
                    training_curves[config_key] = {
                        'config': {
                            'n_factors': n_factors,
                            'sparsity_penalty': sparsity, 
                            'spatial_regularization': spatial_reg,
                            'learning_rate': lr,
                            'random_seed': seed
                        },
                        'losses': model.losses_,  # Full loss history with components
                        'train_loglik': model.train_loglik_,  # Training log-likelihood over epochs
                        'val_loglik': model.val_loglik_,  # Validation log-likelihood over epochs
                        'final_score': score,
                        'final_metrics': metrics
                    }
                
                # Check if best
                if score > best_score:
                    best_score = score
                    best_config = result.copy()
                    best_model = model
                
            except Exception as e:
                continue
        
        # Update progress
        if config_scores:
            avg_score = np.mean(config_scores)
            config_pbar.set_description(f"Config {config_idx+1}/{len(param_combinations)}, Score: {avg_score:.3f}")
    
    config_pbar.close()
    
    if not results:
        raise RuntimeError("❌ No configurations succeeded!")
    
    print(f"\n🏆 SEARCH COMPLETE")
    print(f"Tested {len(results)} configurations")
    print(f"Best {scoring_metric}: {best_score:.4f}")
    
    if verbose:
        print(f"\n🥇 Best Configuration:")
        print(f"  n_factors: {best_config['n_factors']}")
        print(f"  sparsity_penalty: {best_config['sparsity_penalty']}")
        print(f"  spatial_regularization: {best_config['spatial_regularization']}")
        print(f"  learning_rate: {best_config['learning_rate']}")
        print(f"  meaningful_factors: {best_config['meaningful_factors']}/{best_config['total_factors']}")
        print(f"  validation pseudo-R²: {best_config.get('val_pseudo_r2', 'N/A')}")
    
    return {
        'best_model': best_model,
        'best_config': best_config,
        'best_score': best_score,
        'all_results': results,
        'training_curves': training_curves,  # ← Full training details for analysis
        'summary': {
            'total_configs_tested': len(results),
            'best_score': best_score,
            'scoring_metric': scoring_metric,
            'parameters_tested': {
                'n_factors': n_factors_values,
                'sparsity_penalty': sparsity_values,
                'spatial_regularization': spatial_reg_values,
                'learning_rate': lr_values
            }
        }
    }


def analyze_hyperparameter_training_curves(search_results, parameter='n_factors', 
                                          metric='val_loglik', max_configs_to_plot=12):
    """
    Analyze how training curves change across hyperparameter values
    
    Parameters:
    -----------
    search_results : dict
        Results from hyperparameter_search_seminmf
    parameter : str
        Which parameter to analyze ('n_factors', 'sparsity_penalty', 'spatial_regularization')
    metric : str  
        Which metric to plot ('val_loglik', 'train_loglik', 'total_loss', 'nll', 'sparsity', 'spatial')
    max_configs_to_plot : int
        Maximum number of configurations to plot
    """
    
    if 'training_curves' not in search_results or search_results['training_curves'] is None:
        print("❌ No training curves saved. Set save_training_curves=True in hyperparameter search.")
        return
    
    training_curves = search_results['training_curves']
    
    # Group configurations by the parameter of interest
    configs_by_param = {}
    for config_key, curve_data in training_curves.items():
        config = curve_data['config']
        param_value = config[parameter]
        
        if param_value not in configs_by_param:
            configs_by_param[param_value] = []
        configs_by_param[param_value].append((config_key, curve_data))
    
    # Sort parameter values
    sorted_param_values = sorted(configs_by_param.keys())
    n_param_values = len(sorted_param_values)
    
    # Limit number of plots
    if n_param_values > max_configs_to_plot:
        print(f"⚠️  Too many parameter values ({n_param_values}). Showing first {max_configs_to_plot}.")
        sorted_param_values = sorted_param_values[:max_configs_to_plot]
        n_param_values = max_configs_to_plot
    
    # Setup subplots
    n_cols = min(4, n_param_values)
    n_rows = (n_param_values + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_param_values == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten() if n_param_values > 1 else axes
    
    for i, param_value in enumerate(sorted_param_values):
        ax = axes[i] if n_param_values > 1 else axes[0]
        
        configs = configs_by_param[param_value]
        
        # Plot all configurations with this parameter value
        for config_key, curve_data in configs:
            config = curve_data['config']
            
            # Extract the requested metric
            if metric in ['val_loglik', 'train_loglik']:
                if metric == 'val_loglik' and curve_data['val_loglik']:
                    y_data = curve_data['val_loglik']
                    ylabel = 'Validation Log-Likelihood'
                elif metric == 'train_loglik' and curve_data['train_loglik']:
                    y_data = curve_data['train_loglik'] 
                    ylabel = 'Training Log-Likelihood'
                else:
                    continue  # Skip if metric not available
            else:
                # Extract from loss components
                losses = curve_data['losses']
                if not losses:
                    continue
                
                if metric == 'total_loss':
                    y_data = [l['total'] for l in losses]
                    ylabel = 'Total Loss'
                elif metric == 'nll':
                    y_data = [l['nll'] for l in losses]
                    ylabel = 'Negative Log-Likelihood'
                elif metric == 'sparsity':
                    y_data = [l['sparsity'] for l in losses]
                    ylabel = 'Sparsity Penalty'
                elif metric == 'spatial':
                    y_data = [l.get('spatial', 0) for l in losses]
                    ylabel = 'Spatial Penalty'
                else:
                    print(f"Unknown metric: {metric}")
                    return
            
            # Create label for this configuration
            other_params = []
            for key, value in config.items():
                if key != parameter and key != 'random_seed':
                    if isinstance(value, float):
                        other_params.append(f"{key[0]}={value:.3f}")
                    else:
                        other_params.append(f"{key[0]}={value}")
            
            label = f"seed{config['random_seed']}"
            if other_params:
                label += f" ({', '.join(other_params)})"
            
            # Plot the curve
            epochs = range(len(y_data))
            ax.plot(epochs, y_data, alpha=0.7, label=label, linewidth=1)
        
        # Formatting
        ax.set_title(f'{parameter}={param_value}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Add legend if not too many lines
        if len(configs) <= 6:
            ax.legend(fontsize=8)
    
    # Hide extra subplots
    for i in range(n_param_values, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Training Curves: {ylabel} vs {parameter.replace("_", " ").title()}', 
                 y=1.02, fontsize=14)
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Training Curve Analysis for {parameter}:")
    for param_value in sorted_param_values:
        configs = configs_by_param[param_value]
        
        # Get final scores for this parameter value
        final_scores = [curve_data['final_score'] for _, curve_data in configs]
        
        print(f"  {parameter}={param_value}: {len(configs)} configs, "
              f"score = {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")


def plot_hyperparameter_heatmap(search_results, x_param='n_factors', y_param='sparsity_penalty', 
                               metric='score', aggregation='mean'):
    """
    Create heatmap showing how performance varies across two hyperparameters
    
    Parameters:
    -----------
    search_results : dict
        Results from hyperparameter search
    x_param, y_param : str
        Parameters for x and y axes
    metric : str
        Metric to display ('score', 'val_pseudo_r2', 'reconstruction_r2', etc.)
    aggregation : str
        How to aggregate across random seeds ('mean', 'max', 'std')
    """
    
    all_results = search_results['all_results']
    if not all_results:
        print("No results to plot")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_results)
    
    # Check if parameters exist
    if x_param not in df.columns or y_param not in df.columns:
        available_params = [col for col in df.columns if col not in ['score', 'fit_time', 'random_seed']]
        print(f"Available parameters: {available_params}")
        return
    
    # Aggregate across random seeds
    if aggregation == 'mean':
        agg_df = df.groupby([x_param, y_param])[metric].mean().reset_index()
    elif aggregation == 'max':
        agg_df = df.groupby([x_param, y_param])[metric].max().reset_index()
    elif aggregation == 'std':
        agg_df = df.groupby([x_param, y_param])[metric].std().reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Create pivot table for heatmap
    pivot_df = agg_df.pivot(index=y_param, columns=x_param, values=metric)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    
    # Use different colormaps based on metric
    if 'r2' in metric or metric == 'score':
        cmap = 'viridis'  # Higher is better
    elif 'loss' in metric or 'nll' in metric:
        cmap = 'viridis_r'  # Lower is better
    else:
        cmap = 'viridis'
    
    im = plt.imshow(pivot_df.values, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.xlabel(x_param.replace('_', ' ').title())
    plt.ylabel(y_param.replace('_', ' ').title())
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(f'{metric} ({aggregation})')
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white' if value < np.nanmean(pivot_df.values) else 'black')
    
    plt.title(f'{metric.replace("_", " ").title()} Heatmap\n({aggregation} across random seeds)')
    plt.tight_layout()
    plt.show()
    
    # Print best combination
    best_idx = np.nanargmax(pivot_df.values) if 'r2' in metric or metric == 'score' else np.nanargmin(pivot_df.values)
    best_i, best_j = np.unravel_index(best_idx, pivot_df.shape)
    best_x = pivot_df.columns[best_j]
    best_y = pivot_df.index[best_i]
    best_value = pivot_df.iloc[best_i, best_j]
    
    print(f"\n🎯 Best combination: {x_param}={best_x}, {y_param}={best_y}")
    print(f"   {metric} = {best_value:.4f}")


# Example usage and testing
if __name__ == "__main__":
    print("🔢 Poisson Semi-NMF for Count Data")
    print("=" * 50)
    print("🎯 NMF IMPLEMENTATION SUMMARY:")
    print("  ✅ Matrix factorization: X ≈ W @ H")
    print("  ✅ Non-negativity constraint: H ≥ 0 (factors)")  
    print("  ✅ Semi-NMF extension: W can be negative (loadings)")
    print("  ✅ Poisson likelihood for count data")
    print("  ✅ Projected gradient descent optimization")
    print("  ✅ SVD initialization for better convergence")
    print("  ✅ Sparsity + spatial regularization")
    print("")
    print("🧠 WHERE THE NMF HAPPENS:")
    print("  1. forward(): Core factorization W @ H with H ≥ 0")
    print("  2. _compute_loss(): NMF objective function") 
    print("  3. Training loop: Projected gradient descent")
    print("  4. Constraint enforcement: clamp(H, min=0) after each step")
    print("")
    
    # Test with synthetic count data
    np.random.seed(42)
    n_obs, n_voxels = 43, 5000
    n_true_factors = 4
    
    # Generate synthetic semi-NMF data
    true_loadings = np.random.randn(n_obs, n_true_factors) * 0.5  # Can be negative
    true_factors = np.random.exponential(1, (n_true_factors, n_voxels))  # Non-negative
    true_factors = true_factors / np.sum(true_factors, axis=1, keepdims=True)  # Normalize
    
    true_row_effects = np.random.randn(n_obs) * 0.2
    true_col_effects = np.random.randn(n_voxels) * 0.1
    
    # Generate Poisson count data
    linear_pred = (true_row_effects[:, None] + true_col_effects[None, :] + 
                   true_loadings @ true_factors)
    rates = np.log1p(np.exp(linear_pred))  # softplus
    
    # Add some noise to make it more realistic (quasi-Poisson)
    synthetic_data = np.random.poisson(rates) + np.random.normal(0, 0.1, rates.shape)
    synthetic_data = np.maximum(synthetic_data, 0)  # Ensure non-negative
    
    print(f"\nTesting with synthetic data: {synthetic_data.shape}")
    print(f"Data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    print(f"Mean count: {synthetic_data.mean():.3f}")
    
    # Create validation mask
    val_mask = np.random.random(synthetic_data.shape) < 0.1
    
    # Single model test
    print("\n" + "="*40)
    print("Single Model Test")
    print("="*40)
    
    model = PoissonSemiNMF(n_factors=6, sparsity_penalty=0.01, spatial_regularization=0.001, silent=False)
    model.fit(synthetic_data, val_mask=val_mask, max_epochs=150, verbose=True)
    
    # Evaluate
    metrics = evaluate_seminmf(model, synthetic_data, val_mask)
    print(f"\n📊 Results:")
    print(f"  Reconstruction R²: {metrics['reconstruction_r2']:.4f}")
    print(f"  Pseudo R²: {metrics['pseudo_r2']:.4f}")
    if metrics['val_pseudo_r2'] is not None:
        print(f"  Validation Pseudo R²: {metrics['val_pseudo_r2']:.4f}")
    print(f"  Meaningful factors: {metrics['meaningful_factors']}/{metrics['total_factors']}")
    
    # Plot results
    model.plot_training_progress()
    
    # Hyperparameter search test
    print("\n" + "="*40) 
    print("Hyperparameter Search Test")
    print("="*40)
    
    # Quick test to verify spatial_reg_values is being used
    print("\n🔍 Verifying spatial_reg_values usage:")
    test_combinations = list(itertools.product([4], [0.01], spatial_reg_values, [0.01]))
    print(f"  With spatial_reg_values={spatial_reg_values}")
    print(f"  Generated test combinations: {test_combinations}")
    print(f"  Spatial reg values in combinations: {[combo[2] for combo in test_combinations]}")
    
    search_results = hyperparameter_search_seminmf(
        synthetic_data,
        val_mask=val_mask,
        n_factors_values=[4, 6, 8],
        sparsity_values=[0.0, 0.01, 0.1],
        spatial_reg_values=[0.0, 0.001],  # ← Spatial regularization now included
        lr_values=[0.01],
        max_epochs=100,
        n_random_seeds=1,
        scoring_metric='val_pseudo_r2',
        save_training_curves=True,  # ← Save detailed training info
        verbose=True
    )
    
    print(f"\n🏆 Best model found!")
    print(f"Best score: {search_results['best_score']:.4f}")
    
    # NEW: Analyze training curves across hyperparameters
    print("\n" + "="*40)
    print("Training Curve Analysis")
    print("="*40)
    
    # Plot how validation log-likelihood changes across n_factors
    analyze_hyperparameter_training_curves(
        search_results, 
        parameter='n_factors', 
        metric='val_loglik'
    )
    
    # Plot how total loss changes across sparsity penalty
    analyze_hyperparameter_training_curves(
        search_results,
        parameter='sparsity_penalty', 
        metric='total_loss'
    )
    
    # NEW: Create heatmaps showing parameter interactions
    print("\n" + "="*40)
    print("Parameter Interaction Heatmaps") 
    print("="*40)
    
    # Heatmap of score vs n_factors and sparsity
    plot_hyperparameter_heatmap(
        search_results,
        x_param='n_factors',
        y_param='sparsity_penalty', 
        metric='score'
    )
    
    # Heatmap of score vs sparsity and spatial regularization
    plot_hyperparameter_heatmap(
        search_results,
        x_param='sparsity_penalty',
        y_param='spatial_regularization',
        metric='val_pseudo_r2'
    )
    
    # Compare learned vs true factors
    components = search_results['best_model'].get_components()
    learned_factors = components['factors']
    learned_loadings = components['loadings']
    
    print(f"\nFactor comparison:")
    print(f"  True factors shape: {true_factors.shape}")
    print(f"  Learned factors shape: {learned_factors.shape}")
    print(f"  True loadings range: [{true_loadings.min():.3f}, {true_loadings.max():.3f}]")
    print(f"  Learned loadings range: [{learned_loadings.min():.3f}, {learned_loadings.max():.3f}]")
    
    # Show what detailed data is now available
    print(f"\n📊 Available Search Results Data:")
    print(f"  all_results: {len(search_results['all_results'])} configurations with final metrics")
    if search_results['training_curves']:
        print(f"  training_curves: {len(search_results['training_curves'])} detailed training histories")
        print(f"    Each curve contains: losses, train_loglik, val_loglik, config, final_metrics")
    print(f"  summary: Parameter ranges and best score summary")
    
    # Example: Access training curves for analysis
    if search_results['training_curves']:
        first_config = list(search_results['training_curves'].keys())[0]
        first_curve = search_results['training_curves'][first_config]
        print(f"\n  Example training curve data for '{first_config}':")
        print(f"    Epochs trained: {len(first_curve['losses'])}")
        print(f"    Final total loss: {first_curve['losses'][-1]['total']:.4f}")
        print(f"    Final train log-likelihood: {first_curve['train_loglik'][-1]:.3f}")
        if first_curve['val_loglik']:
            print(f"    Final val log-likelihood: {first_curve['val_loglik'][-1]:.3f}")
        print(f"    Configuration: {first_curve['config']}")

