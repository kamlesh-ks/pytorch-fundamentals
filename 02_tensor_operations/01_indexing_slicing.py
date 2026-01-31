"""
Module 2.1: Indexing and Slicing
================================
Learn how to access specific elements and subsets of tensors.

This is VERY similar to NumPy
"""

import torch

print("=" * 50)
print("INDEXING AND SLICING")
print("=" * 50)

# ============================================
# 1. Basic indexing (1D tensors)
# ============================================
print("\n1. BASIC INDEXING (1D)")
print("-" * 30)

vector = torch.tensor([10, 20, 30, 40, 50])
print(f"Vector: {vector}")

# Access single elements
print(f"\nFirst element (index 0): {vector[0]}")
print(f"Third element (index 2): {vector[2]}")
print(f"Last element (index -1): {vector[-1]}")
print(f"Second to last (index -2): {vector[-2]}")

# Slicing [start:end] - end is exclusive!
print(f"\nFirst three [0:3]: {vector[0:3]}")
print(f"From index 2 to end [2:]: {vector[2:]}")
print(f"Up to index 3 [:3]: {vector[:3]}")
print(f"Every other element [::2]: {vector[::2]}")
print(f"Reversed: {torch.flip(vector, dims=[0])}")

# ============================================
# 2. Indexing 2D tensors (matrices)
# ============================================
print("\n2. INDEXING 2D TENSORS")
print("-" * 30)

matrix = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
print(f"Matrix (shape {matrix.shape}):\n{matrix}")

# Single element: [row, column]
print(f"\nElement at [0, 0]: {matrix[0, 0]}")
print(f"Element at [1, 2]: {matrix[1, 2]}")
print(f"Element at [2, -1]: {matrix[2, -1]}")

# Entire row or column
print(f"\nFirst row [0, :]: {matrix[0, :]}")
print(f"First column [:, 0]: {matrix[:, 0]}")
print(f"Last column [:, -1]: {matrix[:, -1]}")

# Submatrix
print(f"\nTop-left 2x2 [:2, :2]:\n{matrix[:2, :2]}")
print(f"\nBottom-right 2x2 [1:, 2:]:\n{matrix[1:, 2:]}")

# ============================================
# 3. Indexing 3D tensors
# ============================================
print("\n3. INDEXING 3D TENSORS")
print("-" * 30)

# Think of 3D as: [batch, rows, columns] or [channels, height, width]
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print(f"3D tensor (shape {tensor_3d.shape}):\n{tensor_3d}")

print(f"\nFirst 'slice' [0, :, :]:\n{tensor_3d[0, :, :]}")
print(f"\nSecond 'slice' [1, :, :]:\n{tensor_3d[1, :, :]}")
print(f"\nElement [1, 2, 3]: {tensor_3d[1, 2, 3]}")

# ============================================
# 4. Boolean indexing (filtering)
# ============================================
print("\n4. BOOLEAN INDEXING")
print("-" * 30)

data = torch.tensor([1, -2, 3, -4, 5, -6])
print(f"Data: {data}")

# Create boolean mask
mask = data > 0
print(f"Mask (data > 0): {mask}")

# Apply mask to filter
positives = data[mask]
print(f"Positive values: {positives}")

# Can do in one step
print(f"Values > 2: {data[data > 2]}")
print(f"Values != 3: {data[data != 3]}")

# 2D example
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"\nMatrix:\n{matrix}")
print(f"Values > 3: {matrix[matrix > 3]}")  # Returns 1D tensor!

# ============================================
# 5. Fancy indexing (index arrays)
# ============================================
print("\n5. FANCY INDEXING")
print("-" * 30)

data = torch.tensor([10, 20, 30, 40, 50])
print(f"Data: {data}")

# Index with a list of indices
indices = torch.tensor([0, 2, 4])
print(f"Indices [0, 2, 4]: {data[indices]}")

# Select specific rows from matrix
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])
print(f"\nMatrix:\n{matrix}")

row_indices = torch.tensor([0, 2, 3])
print(f"Rows [0, 2, 3]:\n{matrix[row_indices]}")

# ============================================
# 6. Modifying with indexing
# ============================================
print("\n6. MODIFYING TENSORS WITH INDEXING")
print("-" * 30)

x = torch.zeros(5)
print(f"Original: {x}")

# Modify single element
x[0] = 1
print(f"After x[0] = 1: {x}")

# Modify slice
x[2:4] = 5
print(f"After x[2:4] = 5: {x}")

# Modify with boolean mask
x[x == 5] = 9
print(f"After x[x == 5] = 9: {x}")

# 2D modification
matrix = torch.zeros(3, 3)
matrix[1, :] = 1  # Set entire row
matrix[:, 2] = 2  # Set entire column
print(f"\nModified matrix:\n{matrix}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
INDEXING SYNTAX:
  tensor[index]           - Single element
  tensor[start:end]       - Slice (end exclusive)
  tensor[start:end:step]  - Slice with step
  tensor[::step]          - Every nth element
  tensor[::-1]            - Reversed

2D INDEXING:
  tensor[row, col]        - Single element
  tensor[row, :]          - Entire row
  tensor[:, col]          - Entire column
  tensor[r1:r2, c1:c2]    - Submatrix

BOOLEAN INDEXING:
  tensor[tensor > value]  - Filter by condition
  
FANCY INDEXING:
  tensor[indices]         - Select specific positions

REMEMBER:
  - Indexing starts at 0
  - Negative indices count from end
  - Slices are [start:end) - end is exclusive!
""")

print("\nNext: Run 02_reshaping.py!")
