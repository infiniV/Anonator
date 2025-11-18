import torch
import sys

print("=" * 80)
print("GPU DETECTION TEST")
print("=" * 80)
print()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    print("Testing GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU computation successful!")
else:
    print("CUDA is NOT available")
    print("Reasons:")
    print("- PyTorch may not be built with CUDA support")
    print("- NVIDIA drivers may not be installed")
    print("- GPU may not be detected")

print()
print("=" * 80)
print("Now testing ONNX Runtime GPU availability...")
print("=" * 80)
print()

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print()
        print("SUCCESS: CUDAExecutionProvider is available!")
        print("ONNX Runtime can use GPU acceleration!")
    else:
        print()
        print("WARNING: CUDAExecutionProvider NOT available")
        print("ONNX Runtime will use CPU only")
except Exception as e:
    print(f"Error checking ONNX Runtime: {e}")

print()
print("=" * 80)
