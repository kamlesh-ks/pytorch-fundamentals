"""
Module 2.2: Reshaping Tensors
=============================
Learn how to change tensor dimensions - crucial for neural networks!
"""

import torch

print("=" * 50)
print("RESHAPING TENSORS")
print("=" * 50)

# ============================================
# 1. reshape() - The most common operation
# ============================================
print("\n1. RESHAPE()")
print("-" * 30)

# Create a 1D tensor
original = torch.arange(12)
print(f"Original (shape {original.shape}): {original}")

# Reshape to different dimensions
reshaped_3x4 = original.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{reshaped_3x4}")

reshaped_4x3 = original.reshape(4, 3)
print(f"\nReshaped to 4x3:\n{reshaped_4x3}")

reshaped_2x2x3 = original.reshape(2, 2, 3)
print(f"\nReshaped to 2x2x3:\n{reshaped_2x2x3}")

# Using -1 to auto-calculate one dimension
auto_rows = original.reshape(-1, 4)  # -1 means "figure it out"
print(f"\nReshape (-1, 4) -> {auto_rows.shape}:\n{auto_rows}")

auto_cols = original.reshape(3, -1)
print(f"\nReshape (3, -1) -> {auto_cols.shape}:\n{auto_cols}")

# ============================================
# 2. view() - Similar to reshape
# ============================================
print("\n2. VIEW()")
print("-" * 30)

original = torch.arange(12)

# view() works like reshape() but requires contiguous memory
viewed = original.view(3, 4)
print(f"View (3, 4):\n{viewed}")

# view() shares memory with original!
viewed[0, 0] = 999
print(f"\nAfter modifying view, original: {original}")

# Reset
original = torch.arange(12)

print("""
Note: view() and reshape() are similar, but:
- view() requires contiguous memory, shares storage
- reshape() may copy data if needed
- For beginners, reshape() is safer to use
""")

# ============================================
# 3. flatten() and ravel() - Make 1D
# ============================================
print("\n3. FLATTEN AND RAVEL")
print("-" * 30)

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(f"Matrix:\n{matrix}")

# Flatten to 1D
flattened = matrix.flatten()
print(f"\nFlattened: {flattened}")

# Flatten specific dimensions
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print(f"\n3D tensor shape: {tensor_3d.shape}")

flat_all = tensor_3d.flatten()
print(f"Flatten all: shape {flat_all.shape}")

flat_partial = tensor_3d.flatten(start_dim=1)  # Keep first dim
print(f"Flatten from dim 1: shape {flat_partial.shape}")

# ============================================
# 4. squeeze() and unsqueeze() - Add/remove dimensions
# ============================================
print("\n4. SQUEEZE AND UNSQUEEZE")
print("-" * 30)

# squeeze() removes dimensions of size 1
x = torch.zeros(1, 3, 1, 4, 1)
print(f"Original shape: {x.shape}")
print(f"Squeezed shape: {x.squeeze().shape}")

# Remove specific dimension
print(f"Squeeze dim 0: {x.squeeze(0).shape}")
print(f"Squeeze dim 2: {x.squeeze(2).shape}")

# unsqueeze() adds a dimension of size 1
y = torch.tensor([1, 2, 3])
print(f"\nOriginal y: {y}, shape: {y.shape}")

print(f"Unsqueeze dim 0: {y.unsqueeze(0)}, shape: {y.unsqueeze(0).shape}")
print(f"Unsqueeze dim 1: {y.unsqueeze(1)}, shape: {y.unsqueeze(1).shape}")

# Practical use: adding batch dimension
single_image = torch.randn(3, 224, 224)  # [channels, height, width]
batched = single_image.unsqueeze(0)      # [batch, channels, height, width]
print(f"\nSingle image: {single_image.shape}")
print(f"With batch dim: {batched.shape}")

# ============================================
# 5. transpose() and permute() - Swap dimensions
# ============================================
print("\n5. TRANSPOSE AND PERMUTE")
print("-" * 30)

# transpose() swaps two dimensions
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(f"Original (shape {matrix.shape}):\n{matrix}")

transposed = matrix.transpose(0, 1)  # Swap rows and columns
print(f"\nTransposed (shape {transposed.shape}):\n{transposed}")

# .T is shorthand for 2D transpose
print(f"\nUsing .T:\n{matrix.T}")

# permute() rearranges all dimensions
tensor_3d = torch.randn(2, 3, 4)
print(f"\n3D tensor shape: {tensor_3d.shape}")

permuted = tensor_3d.permute(2, 0, 1)  # Rearrange to (4, 2, 3)
print(f"Permuted (2,0,1) shape: {permuted.shape}")

# Common use: image format conversion
# PyTorch uses: [batch, channels, height, width]
# Some libraries use: [batch, height, width, channels]
pytorch_format = torch.randn(1, 3, 224, 224)  # [B, C, H, W]
other_format = pytorch_format.permute(0, 2, 3, 1)  # [B, H, W, C]
print(f"\nPyTorch format: {pytorch_format.shape}")
print(f"Other format: {other_format.shape}")

# ============================================
# 6. expand() and repeat() - Duplicate data
# ============================================
print("\n6. EXPAND AND REPEAT")
print("-" * 30)

x = torch.tensor([[1], [2], [3]])  # Shape: (3, 1)
print(f"Original (shape {x.shape}):\n{x}")

# expand() broadcasts without copying (memory efficient)
expanded = x.expand(3, 4)
print(f"\nExpanded to (3, 4):\n{expanded}")

# repeat() actually copies the data
repeated = x.repeat(1, 4)  # Repeat 1 time in dim 0, 4 times in dim 1
print(f"\nRepeated (1, 4):\n{repeated}")

# ============================================
# 7. Common reshaping patterns
# ============================================
print("\n7. COMMON PATTERNS IN DEEP LEARNING")
print("-" * 30)

print("""
PATTERN 1: Flattening for fully connected layer
  Input: [batch, channels, height, width]
  Output: [batch, features]
  Code: x.flatten(start_dim=1) or x.view(batch_size, -1)

PATTERN 2: Adding batch dimension
  Input: [channels, height, width]
  Output: [1, channels, height, width]
  Code: x.unsqueeze(0)

PATTERN 3: Removing batch dimension
  Input: [1, channels, height, width]
  Output: [channels, height, width]
  Code: x.squeeze(0)

PATTERN 4: Channel manipulation
  [batch, height, width, channels] -> [batch, channels, height, width]
  Code: x.permute(0, 3, 1, 2)
""")

# Example: Flattening for fully connected layer
batch_size = 4
feature_maps = torch.randn(batch_size, 32, 7, 7)  # After conv layers
print(f"Feature maps shape: {feature_maps.shape}")

flattened = feature_maps.flatten(start_dim=1)
print(f"Flattened for FC: {flattened.shape}")
print(f"Each sample has {flattened.shape[1]} features")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
RESHAPE OPERATIONS:
  .reshape(shape)    - Change shape (may copy)
  .view(shape)       - Change shape (no copy, needs contiguous)
  .flatten()         - Make 1D
  .squeeze()         - Remove size-1 dimensions
  .unsqueeze(dim)    - Add size-1 dimension
  .transpose(d1, d2) - Swap two dimensions
  .permute(dims)     - Reorder all dimensions
  .expand(size)      - Broadcast (no copy)
  .repeat(times)     - Duplicate (copies)

KEY TIPS:
  - Use -1 in reshape to auto-calculate one dimension
  - Total elements must remain the same after reshape
  - flatten(start_dim=1) is common before FC layers
  - unsqueeze(0) adds batch dimension
""")

print("\nNext: Run 03_combining_tensors.py!")
