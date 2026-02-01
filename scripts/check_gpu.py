import torch
print("=" * 60)
print("GPU CONFIGURATION CHECK")
print("=" * 60)
print(f"✓ PyTorch Version: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ Current Device: cuda:{torch.cuda.current_device()}")
else:
    print("⚠️  CUDA Not Available - Running on CPU")
    print("   This is normal for Python 3.13 as CUDA wheels aren't ready yet")
print("=" * 60)
