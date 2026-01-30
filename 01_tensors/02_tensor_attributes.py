"""
Module 1.2: Tensor Attributes
=============================
Learn about the key properties of tensors: shape, dtype, and device.

Run this file: python 02_tensor_attributes.py
"""

import torch

print("=" * 50)
print("TENSOR ATTRIBUTES")
print("=" * 50)

# ============================================
# 1. Shape - The dimensions of a tensor
# ============================================
print("\n1. SHAPE")
print("-" * 30)

# Create a sample tensor
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

print(f"Tensor:\n{tensor}")
print(f"\nShape: {tensor.shape}")        # torch.Size([3, 4])
print(f"Size: {tensor.size()}")          # Same as shape
print(f"Number of dimensions: {tensor.ndim}")  # 2
print(f"Total elements: {tensor.numel()}")     # 12

# Understanding shape
print("\nUnderstanding shape [3, 4]:")
print(f"  - 3 rows (first dimension)")
print(f"  - 4 columns (second dimension)")

# Different dimensional tensors
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.randn(2, 3, 4)

print(f"\nScalar shape: {scalar.shape}")       # []
print(f"Vector shape: {vector.shape}")         # [3]
print(f"Matrix shape: {matrix.shape}")         # [2, 2]
print(f"3D tensor shape: {tensor_3d.shape}")   # [2, 3, 4]

# ============================================
# 2. Data Type (dtype)
# ============================================
print("\n2. DATA TYPE (dtype)")
print("-" * 30)

# Default dtypes
int_tensor = torch.tensor([1, 2, 3])
float_tensor = torch.tensor([1.0, 2.0, 3.0])

print(f"Integer list -> dtype: {int_tensor.dtype}")    # torch.int64
print(f"Float list -> dtype: {float_tensor.dtype}")    # torch.float32

# Specifying dtype explicitly
tensor_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor_float64 = torch.tensor([1, 2, 3], dtype=torch.float64)
tensor_int32 = torch.tensor([1, 2, 3], dtype=torch.int32)
tensor_bool = torch.tensor([1, 0, 1], dtype=torch.bool)

print(f"\nFloat32: {tensor_float32} | dtype: {tensor_float32.dtype}")
print(f"Float64: {tensor_float64} | dtype: {tensor_float64.dtype}")
print(f"Int32: {tensor_int32} | dtype: {tensor_int32.dtype}")
print(f"Bool: {tensor_bool} | dtype: {tensor_bool.dtype}")

# Converting dtype
original = torch.tensor([1, 2, 3])
converted = original.float()  # Convert to float32
print(f"\nOriginal dtype: {original.dtype}")
print(f"Converted dtype: {converted.dtype}")

# Other conversion methods
print("\nConversion methods:")
print(f"  .float()  -> {original.float().dtype}")
print(f"  .double() -> {original.double().dtype}")
print(f"  .int()    -> {original.float().int().dtype}")
print(f"  .long()   -> {original.long().dtype}")
print(f"  .bool()   -> {original.bool().dtype}")

# Using .to() for conversion
tensor_to = original.to(torch.float16)
print(f"  .to(torch.float16) -> {tensor_to.dtype}")

# ============================================
# 3. Device - Where the tensor lives
# ============================================
print("\n3. DEVICE (CPU vs GPU)")
print("-" * 30)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Create tensor on CPU (default)
cpu_tensor = torch.tensor([1, 2, 3])
print(f"\nCPU tensor device: {cpu_tensor.device}")

# Move to GPU if available
if cuda_available:
    # Method 1: .to('cuda')
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"GPU tensor device: {gpu_tensor.device}")
    
    # Method 2: .cuda()
    gpu_tensor2 = cpu_tensor.cuda()
    print(f"GPU tensor device (method 2): {gpu_tensor2.device}")
    
    # Create directly on GPU
    direct_gpu = torch.tensor([1, 2, 3], device='cuda')
    print(f"Direct GPU tensor: {direct_gpu.device}")
    
    # Move back to CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"Back to CPU: {back_to_cpu.device}")
else:
    print("GPU not available - tensors will stay on CPU")
    print("This is fine for learning! GPU is optional.")

# ============================================
# 4. Combining attributes
# ============================================
print("\n4. COMBINING ATTRIBUTES")
print("-" * 30)

# You can specify multiple attributes at once
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor = torch.zeros(3, 4, dtype=torch.float32, device=device)

print(f"Created tensor with:")
print(f"  Shape: {tensor.shape}")
print(f"  Dtype: {tensor.dtype}")
print(f"  Device: {tensor.device}")

# Copy attributes from another tensor
source = torch.randn(2, 2, dtype=torch.float64)
target = torch.zeros(3, 3, dtype=source.dtype, device=source.device)
print(f"\nCopied dtype ({target.dtype}) and device ({target.device}) from source")

# ============================================
# 5. Important dtype notes for deep learning
# ============================================
print("\n5. DTYPE IN DEEP LEARNING")
print("-" * 30)

print("""
Common dtypes in deep learning:

torch.float32 (default for neural networks)
  - Standard precision for training
  - Good balance of speed and accuracy
  
torch.float16 (half precision)
  - Faster training on modern GPUs
  - Uses less memory
  - May lose some precision

torch.float64 (double precision)
  - Highest precision
  - Rarely needed in deep learning
  - Slower and uses more memory

torch.long (int64)
  - Used for class labels
  - Used for indices

IMPORTANT: Neural network weights are usually float32!
""")

# ============================================
# Summary
# ============================================
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
THREE KEY ATTRIBUTES:

1. SHAPE - Dimensions of the tensor
   tensor.shape or tensor.size()
   
2. DTYPE - Data type of elements
   tensor.dtype
   Common: float32, float64, int64, bool
   
3. DEVICE - Where tensor is stored
   tensor.device
   Values: 'cpu' or 'cuda:0' (GPU)

CHECKING ATTRIBUTES:
  print(tensor.shape)
  print(tensor.dtype)
  print(tensor.device)

CHANGING ATTRIBUTES:
  tensor.float()        # Change dtype
  tensor.to('cuda')     # Change device
  tensor.to(dtype=torch.float32, device='cuda')  # Both
""")

print("\nNext: Run 03_basic_operations.py to learn tensor math!")
