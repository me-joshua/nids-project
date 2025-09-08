#!/usr/bin/env python3
"""
CUDA Setup Configuration for NIDS Project
==========================================

This module configures CUDA library paths for TensorFlow GPU acceleration
across the entire project. Import this module at the beginning of any script
that needs GPU acceleration.

Usage:
    import cuda_setup  # This will automatically configure CUDA paths
    import tensorflow as tf  # TensorFlow will now find CUDA libraries
"""

import os
import site
import sys

# Flag to prevent multiple configurations
_cuda_configured = False

def configure_cuda_libraries():
    """Configure CUDA library paths for TensorFlow GPU acceleration"""
    global _cuda_configured
    
    if _cuda_configured:
        return
    
    print("🔧 Configuring CUDA libraries for GPU acceleration...")
    
    # Add CUDA library paths for TensorFlow 2.10.1
    cuda_lib_paths = []
    
    for path in site.getsitepackages():
        nvidia_path = os.path.join(path, 'nvidia')
        if os.path.exists(nvidia_path):
            # Add all CUDA library bin directories
            for cuda_lib in os.listdir(nvidia_path):
                if cuda_lib.startswith('cu'):  # cuda_runtime, cublas, cudnn, etc.
                    bin_path = os.path.join(nvidia_path, cuda_lib, 'bin')
                    if os.path.exists(bin_path):
                        cuda_lib_paths.append(bin_path)
                    
                    # Also check for lib directories
                    lib_path = os.path.join(nvidia_path, cuda_lib, 'lib')
                    if os.path.exists(lib_path):
                        cuda_lib_paths.append(lib_path)
    
    if cuda_lib_paths:
        # Add CUDA library directories to PATH at the beginning
        current_path = os.environ.get('PATH', '')
        new_path = ';'.join(cuda_lib_paths) + ';' + current_path
        os.environ['PATH'] = new_path
        
        print(f"✅ Added {len(cuda_lib_paths)} CUDA library paths to system PATH")
        print("📁 CUDA library directories:")
        for i, path in enumerate(cuda_lib_paths):
            if i < 5:  # Show first 5 paths
                print(f"   • {path}")
            elif i == 5:
                print(f"   • ... and {len(cuda_lib_paths) - 5} more")
                break
        
        # Set environment variables for TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        # Remove any CPU-only restrictions
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        print("🚀 CUDA configuration complete! GPU acceleration should now be available.")
    else:
        print("⚠️  No CUDA library paths found in site-packages")
        print("   Make sure nvidia-* packages are installed:")
        print("   pip install nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11")
    
    _cuda_configured = True

def check_gpu_availability():
    """Check if GPU is available after CUDA configuration"""
    try:
        import tensorflow as tf
        
        print("\n🔍 GPU Availability Check:")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   CUDA built support: {tf.test.is_built_with_cuda()}")
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"   GPU devices found: {len(physical_devices)} ⚡")
            for i, device in enumerate(physical_devices):
                print(f"      GPU {i}: {device.name}")
            
            # Configure GPU memory growth to avoid OOM errors
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("   GPU memory growth configured ✅")
            except RuntimeError as e:
                print(f"   GPU memory growth setup failed: {e}")
            
            return True
        else:
            print("   No GPU devices detected ❌")
            logical_devices = tf.config.list_logical_devices()
            print(f"   Available devices: {[device.name for device in logical_devices]}")
            return False
            
    except ImportError:
        print("   TensorFlow not available for GPU check")
        return False

# Automatically configure CUDA when this module is imported
configure_cuda_libraries()

# Export functions for manual use
__all__ = ['configure_cuda_libraries', 'check_gpu_availability']
