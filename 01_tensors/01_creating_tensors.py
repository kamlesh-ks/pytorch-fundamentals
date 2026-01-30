"""
Module 1.1: Creating Tensors
============================
Learn different ways to create tensors in PyTorch.

Run this file: python 01_creating_tensors.py
"""

import torch
import numpy as np

print("=" * 50)
print("CREATING TENSORS IN PYTORCH")
print("=" * 50)

# ============================================
# 1. Creating tensors from Python lists
# ============================================
print("\n1. FROM PYTHON LISTS")
print("-" * 30)

# From a simple list (creates 1D tensor / vector)
data_1d = [1, 2, 3, 4, 5]
tensor_1d = torch.tensor(data_1d)
print(f"1D Tensor: {tensor_1d}")
print(f"Shape: {tensor_1d.shape}")

# From nested lists (creates 2D tensor / matrix)
data_2d = [[1, 2, 3], 
           [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print(f"\n2D Tensor:\n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}")  # (rows, columns)

# From deeper nested lists (creates 3D tensor)
data_3d = [[[1, 2], [3, 4]], 
           [[5, 6], [7, 8]]]
tensor_3d = torch.tensor(data_3d)
print(f"\n3D Tensor:\n{tensor_3d}")
print(f"Shape: {tensor_3d.shape}")

# ============================================
# 2. Creating tensors with specific values
# ============================================
print("\n2. TENSORS WITH SPECIFIC VALUES")
print("-" * 30)

# All zeros
zeros = torch.zeros(3, 4)  # 3 rows, 4 columns
print(f"Zeros (3x4):\n{zeros}")

# All ones
ones = torch.ones(2, 3)
print(f"\nOnes (2x3):\n{ones}")

# Filled with a specific value
filled = torch.full((2, 2), 7.0)  # 2x2 filled with 7
print(f"\nFilled with 7:\n{filled}")

# Identity matrix (1s on diagonal)
eye = torch.eye(3)
print(f"\nIdentity matrix:\n{eye}")

# ============================================
# 3. Creating tensors with random values
# ============================================
print("\n3. RANDOM TENSORS")
print("-" * 30)

# Random values between 0 and 1 (uniform distribution)
rand_uniform = torch.rand(2, 3)
print(f"Random uniform [0,1):\n{rand_uniform}")

# Random values from normal distribution (mean=0, std=1)
rand_normal = torch.randn(2, 3)
print(f"\nRandom normal:\n{rand_normal}")

# Random integers
rand_int = torch.randint(low=0, high=10, size=(2, 3))
print(f"\nRandom integers [0,10):\n{rand_int}")

# ============================================
# 4. Creating tensors with sequences
# ============================================
print("\n4. SEQUENCE TENSORS")
print("-" * 30)

# Range of values (like Python's range)
range_tensor = torch.arange(0, 10, 2)  # start, end, step
print(f"Arange (0 to 10, step 2): {range_tensor}")

# Linearly spaced values
linspace_tensor = torch.linspace(0, 1, 5)  # 5 values from 0 to 1
print(f"Linspace (0 to 1, 5 points): {linspace_tensor}")

# ============================================
# 5. Creating tensors from NumPy arrays
# ============================================
print("\n5. FROM NUMPY ARRAYS")
print("-" * 30)

numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array:\n{numpy_array}")

# Method 1: torch.tensor() - creates a copy
tensor_copy = torch.tensor(numpy_array)
print(f"\nTensor (copy): {tensor_copy}")

# Method 2: torch.from_numpy() - shares memory (changes affect both!)
tensor_shared = torch.from_numpy(numpy_array)
print(f"Tensor (shared): {tensor_shared}")

# ============================================
# 6. Creating tensors with same shape as another
# ============================================
print("\n6. TENSORS WITH SAME SHAPE")
print("-" * 30)

original = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"Original shape: {original.shape}")

# Create zeros with same shape
zeros_like = torch.zeros_like(original)
print(f"Zeros like original:\n{zeros_like}")

# Create ones with same shape
ones_like = torch.ones_like(original)
print(f"\nOnes like original:\n{ones_like}")

# Create random with same shape
rand_like = torch.rand_like(original, dtype=torch.float32)
print(f"\nRandom like original:\n{rand_like}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY: TENSOR CREATION METHODS")
print("=" * 50)
print("""
torch.tensor(data)      - From Python list/NumPy array
torch.zeros(size)       - All zeros
torch.ones(size)        - All ones
torch.full(size, value) - Filled with value
torch.eye(n)            - Identity matrix
torch.rand(size)        - Random [0, 1)
torch.randn(size)       - Random normal
torch.randint(...)      - Random integers
torch.arange(...)       - Range sequence
torch.linspace(...)     - Linear sequence
torch.from_numpy(arr)   - From NumPy (shared memory)
torch.*_like(tensor)    - Same shape as another tensor
""")

print("\nNext: Run 02_tensor_attributes.py to learn about tensor properties!")
