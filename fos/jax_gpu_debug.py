"""
JAX GPU Debugging Script
Run this to diagnose JAX GPU detection issues
"""

def diagnose_jax_gpu():
    """Comprehensive JAX GPU diagnostic"""
    print("=== JAX GPU DIAGNOSTIC ===\n")
    
    # Step 1: Check if JAX is installed
    try:
        import jax
        import jax.numpy as jnp
        print(f"✓ JAX installed: version {jax.__version__}")
    except ImportError as e:
        print(f"❌ JAX not installed: {e}")
        return
    
    # Step 2: Check JAX backend
    try:
        import jaxlib
        print(f"✓ JAXlib installed: version {jaxlib.__version__}")
    except ImportError:
        print("❌ JAXlib not installed")
        return
    
    # Step 3: Check available devices
    print(f"\n--- JAX Devices ---")
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
        print(f"    Platform: {device.platform}")
        print(f"    Device kind: {device.device_kind}")
    
    # Step 4: Check default backend
    print(f"\nDefault backend: {jax.default_backend()}")
    
    # Step 5: Check if CUDA is available
    print(f"\n--- CUDA Check ---")
    try:
        # Try to create array on GPU
        if any('gpu' in str(d).lower() for d in devices):
            print("✓ GPU devices detected")
            
            # Test GPU computation
            try:
                gpu_device = jax.devices('gpu')[0]
                x = jnp.array([1, 2, 3, 4, 5])
                with jax.default_device(gpu_device):
                    y = x * 2
                    result = y.sum()
                print(f"✓ GPU computation successful: {result}")
                
            except Exception as e:
                print(f"❌ GPU computation failed: {e}")
        else:
            print("❌ No GPU devices found")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # Step 6: Environment checks
    print(f"\n--- Environment Check ---")
    
    # Check NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ NVIDIA driver working")
            # Extract CUDA version from nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"  NVIDIA CUDA Version: {cuda_version}")
                    break
        else:
            print("❌ nvidia-smi failed")
    except Exception as e:
        print(f"❌ nvidia-smi check failed: {e}")
    
    # Check CUDA toolkit
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"✓ NVCC installed: {line.strip()}")
                    break
        else:
            print("❌ nvcc not found (CUDA toolkit may not be installed)")
    except Exception as e:
        print(f"❌ nvcc check failed: {e}")
    
    # Step 7: Check environment variables
    print(f"\n--- Environment Variables ---")
    import os
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Step 8: JAX configuration
    print(f"\n--- JAX Configuration ---")
    print(f"  XLA bridge: {jax.lib.xla_bridge.get_backend().platform}")
    
    try:
        from jax.lib import xla_bridge
        print(f"  XLA bridge backend: {xla_bridge.get_backend()}")
    except Exception as e:
        print(f"  XLA bridge error: {e}")


def get_installation_commands():
    """Get the right installation commands based on system"""
    print("\n=== INSTALLATION COMMANDS ===\n")
    
    print("Option 1: Install JAX with CUDA support (recommended)")
    print("pip uninstall jax jaxlib")  # Remove existing
    print("pip install --upgrade pip")
    print("pip install --upgrade \"jax[cuda12_pip]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    print("\nOption 2: Install specific CUDA version")
    print("# For CUDA 11.8:")
    print("pip install --upgrade \"jax[cuda11_pip]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    print("\nOption 3: Use conda-forge (alternative)")
    print("conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia")
    
    print("\nOption 4: Manual installation with specific versions")
    print("# Check your CUDA version first with nvidia-smi")
    print("# Then install matching JAX version")
    
    print("\n=== TROUBLESHOOTING STEPS ===\n")
    print("1. Check NVIDIA driver: nvidia-smi")
    print("2. Check CUDA toolkit: nvcc --version") 
    print("3. Restart Python kernel after installation")
    print("4. Set environment variable: export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda")
    print("5. If still failing, try CPU-only version first:")
    print("   pip install --upgrade jax jaxlib")


def quick_gpu_test():
    """Quick test to verify GPU is working"""
    print("\n=== QUICK GPU TEST ===\n")
    
    try:
        import jax
        import jax.numpy as jnp
        import time
        
        # Test data
        size = 5000
        x = jnp.array(jnp.random.normal(0, 1, (size, size)))
        
        # CPU test
        print("Testing CPU performance...")
        start = time.time()
        with jax.default_device(jax.devices('cpu')[0]):
            result_cpu = jnp.dot(x, x).block_until_ready()
        cpu_time = time.time() - start
        print(f"CPU time: {cpu_time:.3f} seconds")
        
        # GPU test
        if any('gpu' in str(d).lower() for d in jax.devices()):
            print("Testing GPU performance...")
            start = time.time()
            with jax.default_device(jax.devices('gpu')[0]):
                result_gpu = jnp.dot(x, x).block_until_ready()
            gpu_time = time.time() - start
            print(f"GPU time: {gpu_time:.3f} seconds")
            print(f"Speedup: {cpu_time/gpu_time:.1f}x")
            
            # Verify results match
            if jnp.allclose(result_cpu, result_gpu, atol=1e-5):
                print("✓ CPU and GPU results match")
            else:
                print("❌ CPU and GPU results differ")
        else:
            print("❌ No GPU available for testing")
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")


if __name__ == "__main__":
    diagnose_jax_gpu()
    get_installation_commands()
    quick_gpu_test()
