import torch

if torch.cuda.is_available():
    # 1. Select the current device (typically device 0)
    device = torch.device("cuda")
    
    # 2. Get the compute capability as a (major, minor) tuple
    major, minor = torch.cuda.get_device_capability(device) 
    
    print(f"GPU Compute Capability: {major}.{minor}")

    # 3. Check for Tensor Core support
    if major >= 7:
        print("✅ This GPU (Compute Capability 7.0+) has Tensor Cores and fully supports AMP acceleration for maximum speed.")
    else:
        print("⚠️ This GPU does not have Tensor Cores (pre-Volta architecture). AMP may still reduce memory use, but the speedup will be minimal.")
else:
    print("❌ CUDA not available. Cannot check GPU capability.")