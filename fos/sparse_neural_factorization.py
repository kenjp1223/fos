"""
PyTorch Sparse Matrix Factorization for Neural Data
Optimized for Windows + NVIDIA GPU (RTX 3060)

GPU-accelerated sparse matrix factorization for background-subtracted 
neural voxel data with hyperparameter search and cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import time
import warnings
from scipy.stats import poisson
warnings.filterwarnings('ignore')

class SparseMatrixFactorization(nn.Module):
    """
    PyTorch-based sparse matrix factorization for neural data
    Supports both CPU and GPU acceleration
    """
    
    def __init__(self, n_factors, sparsity_penalty=0.1, elastic_net_frac=0.5, 
                 max_num_iters=100, learning_rate=0.01, device='auto', random_state=42):
        """
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        sparsity_penalty : float
            Overall regularization strength for sparsity
        elastic_net_frac : float
            Balance between L1 and L2 regularization (0=L2 only, 1=L1 only)
        max_num_iters : int
            Maximum number of optimization iterations
        learning_rate : float
            Learning rate for Adam optimizer
        device : str or torch.device
            Device to use ('auto', 'cpu', 'cuda', or specific device)
        random_state : int
            Random seed for reproducibility
        """
        super().__init__()
        
        self.n_factors = n_factors
        self.sparsity_penalty = sparsity_penalty
        self.elastic_net_frac = elastic_net_frac
        self.max_num_iters = max_num_iters
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Print device info
        if self.device.type == 'cuda' and torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"💻 Using CPU")
        
        # Regularization weights
        self.l1_reg = sparsity_penalty * elastic_net_frac
        self.l2_reg = sparsity_penalty * (1 - elastic_net_frac)
        
        # Set random seed
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        # Parameters (will be initialized in fit())
        self.loadings = None
        self.factors = None
        
    def _initialize_factors(self, data_shape):
        """Initialize loadings and factors using Xavier initialization"""
        n_obs, n_voxels = data_shape
        
        # Xavier/Glorot initialization
        loading_scale = np.sqrt(2.0 / (n_obs + self.n_factors))
        factor_scale = np.sqrt(2.0 / (self.n_factors + n_voxels))
        
        self.loadings = nn.Parameter(
            torch.randn(n_obs, self.n_factors, device=self.device) * loading_scale
        )
        self.factors = nn.Parameter(
            torch.randn(self.n_factors, n_voxels, device=self.device) * factor_scale
        )
        
    def forward(self):
        """Forward pass - compute reconstruction"""
        return torch.mm(self.loadings, self.factors)
    
    def _compute_reconstruction_loss(self, data, train_mask=None):
        """Compute reconstruction loss (MSE)"""
        reconstruction = self.forward()
        
        if train_mask is not None:
            # Only compute loss on training data
            error = (data - reconstruction) * train_mask.float()
            mse_loss = torch.sum(error ** 2) / torch.sum(train_mask.float())
        else:
            error = data - reconstruction
            mse_loss = torch.mean(error ** 2)
        
        return mse_loss
    
    def _compute_regularization_loss(self):
        """Compute L1 + L2 regularization"""
        l1_penalty = self.l1_reg * (torch.sum(torch.abs(self.loadings)) + 
                                   torch.sum(torch.abs(self.factors)))
        
        l2_penalty = self.l2_reg * (torch.sum(self.loadings ** 2) + 
                                   torch.sum(self.factors ** 2))
        
        return l1_penalty + l2_penalty
    
    def compute_loss(self, data, train_mask=None):
        """Compute total loss (reconstruction + regularization)"""
        mse_loss = self._compute_reconstruction_loss(data, train_mask)
        reg_loss = self._compute_regularization_loss()
        total_loss = mse_loss + reg_loss
        
        return total_loss, mse_loss.item()
    
    def _compute_held_out_loglike(self, data, held_out_mask):
        """Compute log-likelihood on held-out data"""
        with torch.no_grad():
            reconstruction = self.forward()
            
            # Extract held-out data points
            held_out_data = data[held_out_mask]
            held_out_pred = reconstruction[held_out_mask]
            
            if len(held_out_data) == 0:
                return 0.0
            
            # Separate positive and negative observations
            positive_mask = held_out_data >= 0
            negative_mask = held_out_data < 0
            
            total_loglike = 0.0
            
            # Positive observations: Poisson-like log-likelihood
            if torch.sum(positive_mask) > 0:
                pos_data = held_out_data[positive_mask]
                pos_pred = torch.clamp(held_out_pred[positive_mask], min=0.1)
                pos_loglike = torch.sum(pos_data * torch.log(pos_pred) - pos_pred)
                total_loglike += pos_loglike
            
            # Negative observations: Gaussian-like log-likelihood
            if torch.sum(negative_mask) > 0:
                neg_data = held_out_data[negative_mask]
                neg_pred = held_out_pred[negative_mask]
                neg_var = torch.clamp(torch.abs(neg_pred), min=1.0)
                neg_loglike = torch.sum(-0.5 * (neg_data - neg_pred) ** 2 / neg_var)
                total_loglike += neg_loglike
            
            # Average log-likelihood
            avg_loglike = total_loglike / len(held_out_data)
            return avg_loglike.item()
    
    def fit(self, data, mask=None, held_out_mask=None, verbose=False):
        """
        Fit the sparse matrix factorization model
        
        Parameters:
        -----------
        data : array-like, shape (n_obs, n_voxels)
            Input neural data matrix
        mask : array-like, shape (n_obs, n_voxels), optional
            Training mask (True = use for training)
        held_out_mask : array-like, shape (n_obs, n_voxels), optional
            Held-out validation mask (True = held out for validation)
        verbose : bool
            Whether to print training progress
        """
        
        # Convert data to PyTorch tensor
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
            
            # Training mask is inverse of held-out mask
            train_mask = ~held_out_mask
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.BoolTensor(mask).to(self.device)
                train_mask = train_mask & mask
        else:
            train_mask = torch.BoolTensor(mask).to(self.device) if mask is not None else None
        
        # Initialize parameters
        self._initialize_factors(data.shape)
        
        # Move to device
        self.to(self.device)
        
        # Setup optimizer (Adam works well for sparse problems)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Training metrics
        losses = []
        held_out_loglikes = []
        
        if verbose:
            print(f"Training sparse factorization: {data.shape[0]}×{data.shape[1]} → {data.shape[0]}×{self.n_factors} + {self.n_factors}×{data.shape[1]}")
            print(f"Device: {self.device}, Iterations: {self.max_num_iters}")
        
        # Training loop
        for iteration in range(self.max_num_iters):
            optimizer.zero_grad()
            
            # Forward pass and compute loss
            total_loss, mse_loss = self.compute_loss(data, train_mask)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Store metrics
            losses.append(total_loss.item())
            
            # Compute held-out log-likelihood
            if held_out_mask is not None:
                held_out_loglike = self._compute_held_out_loglike(data, held_out_mask)
                held_out_loglikes.append(held_out_loglike)
            
            # Print progress
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: Loss = {total_loss.item():.6f}, MSE = {mse_loss:.6f}")
                if held_out_mask is not None:
                    print(f"                 Held-out LogLike = {held_out_loglike:.6f}")
        
        # Store training history
        self.training_losses_ = losses
        self.held_out_loglikes_ = held_out_loglikes if held_out_mask is not None else None
        
        # Final sparsity info
        if verbose:
            sparsity = self.get_sparsity_metrics()
            print(f"Final sparsity: Loadings {sparsity['loading_sparsity']:.1%}, Factors {sparsity['factor_sparsity']:.1%}")
        
        return self
    
    def transform(self, data=None):
        """Get loadings (transform data to factor space)"""
        with torch.no_grad():
            if data is None:
                return self.loadings.cpu().numpy()
            else:
                # For new data, solve for optimal loadings
                if isinstance(data, np.ndarray):
                    data = torch.FloatTensor(data).to(self.device)
                
                # Solve least squares: data = loadings @ factors
                # loadings = data @ factors.T @ (factors @ factors.T)^(-1)
                factors_T = self.factors.T
                gram_matrix = torch.mm(self.factors, factors_T)
                
                # Add small regularization for numerical stability
                reg_eye = torch.eye(gram_matrix.shape[0], device=self.device) * 1e-6
                gram_inv = torch.inverse(gram_matrix + reg_eye)
                
                new_loadings = torch.mm(torch.mm(data, factors_T), gram_inv)
                return new_loadings.cpu().numpy()
    
    def inverse_transform(self, loadings=None):
        """Reconstruct data from loadings"""
        with torch.no_grad():
            if loadings is None:
                reconstruction = self.forward()
            else:
                if isinstance(loadings, np.ndarray):
                    loadings = torch.FloatTensor(loadings).to(self.device)
                reconstruction = torch.mm(loadings, self.factors)
            
            return reconstruction.cpu().numpy()
    
    def get_sparsity_metrics(self):
        """Compute sparsity metrics"""
        with torch.no_grad():
            loadings_np = self.loadings.cpu().numpy()
            factors_np = self.factors.cpu().numpy()
            
            # Count near-zero elements
            loading_sparsity = np.mean(np.abs(loadings_np) < 1e-6)
            factor_sparsity = np.mean(np.abs(factors_np) < 1e-6)
            
            return {
                'loading_sparsity': loading_sparsity,
                'factor_sparsity': factor_sparsity,
                'total_params': loadings_np.size + factors_np.size,
                'sparse_params': np.sum(np.abs(loadings_np) < 1e-6) + np.sum(np.abs(factors_np) < 1e-6),
                'compression_ratio': 1 - (loadings_np.size + factors_np.size) / (loadings_np.shape[0] * factors_np.shape[1])
            }


def hyperparameter_search(data, held_out_mask, 
                         sparsity_values=[0.01, 0.1, 1.0],
                         n_factors_values=[5, 10, 15, 20],
                         sparsity_penalty_values=[0.05, 0.1, 0.2],
                         learning_rates=[0.001, 0.01, 0.1],
                         elastic_net_frac=0.5,
                         max_num_iters=50,
                         device='auto',
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
        L1 regularization strengths to try (deprecated, use sparsity_penalty_values)
    n_factors_values : list
        Numbers of factors to try
    sparsity_penalty_values : list  
        Overall sparsity penalty strengths to try
    learning_rates : list
        Learning rates to try
    
    Returns:
    --------
    dict : Results dictionary with best parameters and all results
    """
    
    results = []
    best_loglike = -np.inf
    best_params = None
    best_model = None
    
    # Create parameter grid (use sparsity_penalty_values, ignore old sparsity_values)
    param_combinations = list(itertools.product(
        n_factors_values, sparsity_penalty_values, learning_rates
    ))
    
    if verbose:
        print(f"🔍 Testing {len(param_combinations)} parameter combinations with PyTorch...")
        if torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU")
    
    start_time = time.time()
    
    for i, (n_factors, sparsity_penalty, lr) in enumerate(param_combinations):
        if verbose:
            print(f"\n[{i+1:2d}/{len(param_combinations)}] n_factors={n_factors}, penalty={sparsity_penalty:.3f}, lr={lr}")
        
        try:
            # Initialize model
            model = SparseMatrixFactorization(
                n_factors=n_factors,
                sparsity_penalty=sparsity_penalty,
                elastic_net_frac=elastic_net_frac,
                max_num_iters=max_num_iters,
                learning_rate=lr,
                device=device,
                random_state=random_state + i  # Different seed for each run
            )
            
            # Fit model
            fit_start = time.time()
            model.fit(data, held_out_mask=held_out_mask, verbose=False)
            fit_time = time.time() - fit_start
            
            # Get final metrics
            final_loss = model.training_losses_[-1]
            final_held_out_loglike = model.held_out_loglikes_[-1]
            
            # Compute sparsity metrics
            sparsity_metrics = model.get_sparsity_metrics()
            
            result = {
                'n_factors': n_factors,
                'sparsity_penalty': sparsity_penalty,
                'learning_rate': lr,
                'final_loss': final_loss,
                'held_out_loglike': final_held_out_loglike,
                'loading_sparsity': sparsity_metrics['loading_sparsity'],
                'factor_sparsity': sparsity_metrics['factor_sparsity'],
                'compression_ratio': sparsity_metrics['compression_ratio'],
                'fit_time': fit_time,
                'model': model
            }
            
            results.append(result)
            
            if verbose:
                print(f"    Loss: {final_loss:.6f}, LogLike: {final_held_out_loglike:.6f}")
                print(f"    Sparsity: L={sparsity_metrics['loading_sparsity']:.2%}, F={sparsity_metrics['factor_sparsity']:.2%}")
                print(f"    Time: {fit_time:.1f}s")
            
            # Track best model
            if final_held_out_loglike > best_loglike:
                best_loglike = final_held_out_loglike
                best_params = {
                    'n_factors': n_factors, 
                    'sparsity_penalty': sparsity_penalty,
                    'learning_rate': lr
                }
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"    ❌ Failed: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 Best parameters: {best_params}")
        print(f"🏆 Best held-out log-likelihood: {best_loglike:.6f}")
        print(f"⏱️  Total search time: {total_time:.1f}s")
        if best_model:
            best_sparsity = best_model.get_sparsity_metrics()
            print(f"📊 Best model compression: {best_sparsity['compression_ratio']:.1%}")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_loglike': best_loglike,
        'all_results': results,
        'search_time': total_time
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


def check_gpu_setup():
    """Check PyTorch GPU setup"""
    print("=== PyTorch GPU Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_props.name}")
            print(f"    Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            print(f"    Compute capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Quick performance test
        print("\n🧪 GPU Performance Test:")
        start = time.time()
        x = torch.randn(2000, 2000, device='cuda')
        y = torch.mm(x, x.T)
        result = torch.sum(y).item()
        gpu_time = time.time() - start
        
        print(f"  Matrix multiplication (2000×2000): {gpu_time:.3f}s")
        print(f"  Result: {result:.2e}")
        print("🚀 GPU ready for neural factorization!")
        
        return True
    else:
        print("❌ No GPU available - will use CPU")
        print("💡 To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return False


# Utility functions for analysis and visualization
def plot_training_progress(model, figsize=(12, 4)):
    """Plot training loss and held-out likelihood"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Training loss
        axes[0].plot(model.training_losses_)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Progress')
        axes[0].grid(True)
        axes[0].set_yscale('log')
        
        # Held-out likelihood
        if model.held_out_loglikes_ is not None:
            axes[1].plot(model.held_out_loglikes_)
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Held-out Log-likelihood')
            axes[1].set_title('Validation Progress')
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No validation data', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")


def plot_factor_patterns(model, n_factors_to_show=4, figsize=(15, 8), plot_type='line', z_idx=100, alive_voxels=None):
    """Plot spatial patterns of the learned factors"""
    try:
        import matplotlib.pyplot as plt
        
        factors = model.factors.detach().cpu().numpy()
        n_factors_to_show = min(n_factors_to_show, factors.shape[0])
        
        if plot_type == 'coronal' and alive_voxels is not None:
            # Coronal brain slice visualization
            fig, axes = plt.subplots(2, (n_factors_to_show + 1) // 2, figsize=figsize)
            if n_factors_to_show == 1:
                axes = [axes]
            axes = axes.flatten()
            
            for i in range(n_factors_to_show):
                # Reconstruct 3D factor from flattened alive voxels
                factor_3d = np.nan * np.zeros(alive_voxels.shape)
                factor_3d[alive_voxels] = factors[i, :]
                
                # Plot coronal slice
                im = axes[i].imshow(factor_3d[z_idx, :, :])
                axes[i].set_title(f'Factor {i+1} (Slice {z_idx})')
                
                # Add colorbar and sparsity info
                plt.colorbar(im, ax=axes[i])
                sparsity = np.mean(np.abs(factors[i, :]) < 1e-6)
                axes[i].text(0.02, 0.98, f'Sparsity: {sparsity:.1%}', 
                            transform=axes[i].transAxes, va='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Original line plot visualization
            fig, axes = plt.subplots(2, (n_factors_to_show + 1) // 2, figsize=figsize)
            if n_factors_to_show == 1:
                axes = [axes]
            axes = axes.flatten()
            
            for i in range(n_factors_to_show):
                axes[i].plot(factors[i, :])
                axes[i].set_title(f'Factor {i+1}')
                axes[i].set_xlabel('Voxel Index')
                axes[i].set_ylabel('Factor Weight')
                axes[i].grid(True)
                
                # Add sparsity info
                sparsity = np.mean(np.abs(factors[i, :]) < 1e-6)
                axes[i].text(0.02, 0.98, f'Sparsity: {sparsity:.1%}', 
                            transform=axes[i].transAxes, va='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_factors_to_show, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")


def analyze_sparsity(model):
    """Analyze and print sparsity statistics"""
    metrics = model.get_sparsity_metrics()
    
    print("=== Sparsity Analysis ===")
    print(f"Loading sparsity: {metrics['loading_sparsity']:.1%}")
    print(f"Factor sparsity: {metrics['factor_sparsity']:.1%}")
    print(f"Total parameters: {metrics['total_params']:,}")
    print(f"Near-zero parameters: {metrics['sparse_params']:,}")
    print(f"Effective parameters: {metrics['total_params'] - metrics['sparse_params']:,}")
    print(f"Compression ratio: {metrics['compression_ratio']:.1%}")
    
    return metrics


if __name__ == "__main__":
    print("PyTorch Sparse Neural Factorization")
    print("====================================")
    check_gpu_setup()
