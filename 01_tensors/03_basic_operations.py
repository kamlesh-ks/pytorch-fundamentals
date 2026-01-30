"""
Module 1.3: Basic Tensor Operations
===================================
Learn mathematical operations with tensors.

Run this file: python 03_basic_operations.py
"""

import torch

print("=" * 50)
print("BASIC TENSOR OPERATIONS")
print("=" * 50)

# ============================================
# 1. Element-wise arithmetic
# ============================================
print("\n1. ELEMENT-WISE ARITHMETIC")
print("-" * 30)

a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
b = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

print(f"a = {a}")
print(f"b = {b}")

# Addition
print(f"\na + b = {a + b}")
print(f"torch.add(a, b) = {torch.add(a, b)}")

# Subtraction
print(f"\na - b = {a - b}")
print(f"torch.sub(a, b) = {torch.sub(a, b)}")

# Multiplication (element-wise, NOT matrix multiplication)
print(f"\na * b = {a * b}")
print(f"torch.mul(a, b) = {torch.mul(a, b)}")

# Division
print(f"\na / b = {a / b}")
print(f"torch.div(a, b) = {torch.div(a, b)}")

# Power
print(f"\na ** 2 = {a ** 2}")
print(f"torch.pow(a, 2) = {torch.pow(a, 2)}")

# ============================================
# 2. Scalar operations
# ============================================
print("\n2. SCALAR OPERATIONS")
print("-" * 30)

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(f"x = {x}")

print(f"\nx + 10 = {x + 10}")
print(f"x * 2 = {x * 2}")
print(f"x / 2 = {x / 2}")
print(f"x - 1 = {x - 1}")

# ============================================
# 3. Matrix operations
# ============================================
print("\n3. MATRIX OPERATIONS")
print("-" * 30)

# Matrix multiplication
A = torch.tensor([[1, 2], 
                  [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], 
                  [7, 8]], dtype=torch.float32)

print(f"Matrix A:\n{A}")
print(f"\nMatrix B:\n{B}")

# Matrix multiplication (3 ways - all equivalent)
print(f"\nMatrix multiplication (A @ B):\n{A @ B}")
print(f"\ntorch.matmul(A, B):\n{torch.matmul(A, B)}")
print(f"\ntorch.mm(A, B):\n{torch.mm(A, B)}")

# Element-wise multiplication (different from matrix multiplication!)
print(f"\nElement-wise (A * B):\n{A * B}")

# Transpose
print(f"\nTranspose of A:\n{A.T}")
print(f"torch.transpose(A, 0, 1):\n{torch.transpose(A, 0, 1)}")

# ============================================
# 4. Reduction operations
# ============================================
print("\n4. REDUCTION OPERATIONS")
print("-" * 30)

data = torch.tensor([[1, 2, 3],
                     [4, 5, 6]], dtype=torch.float32)
print(f"Data:\n{data}")

# Sum
print(f"\nSum (all): {data.sum()}")
print(f"Sum (dim=0, columns): {data.sum(dim=0)}")
print(f"Sum (dim=1, rows): {data.sum(dim=1)}")

# Mean
print(f"\nMean (all): {data.mean()}")
print(f"Mean (dim=0): {data.mean(dim=0)}")
print(f"Mean (dim=1): {data.mean(dim=1)}")

# Max and Min
print(f"\nMax (all): {data.max()}")
print(f"Min (all): {data.min()}")

# Argmax (index of maximum)
print(f"\nArgmax (all): {data.argmax()}")  # Flattened index
print(f"Argmax (dim=0): {data.argmax(dim=0)}")  # Per column
print(f"Argmax (dim=1): {data.argmax(dim=1)}")  # Per row

# Standard deviation and variance
print(f"\nStd: {data.std()}")
print(f"Var: {data.var()}")

# ============================================
# 5. Comparison operations
# ============================================
print("\n5. COMPARISON OPERATIONS")
print("-" * 30)

x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([5, 4, 3, 2, 1])

print(f"x = {x}")
print(f"y = {y}")

print(f"\nx > 2: {x > 2}")
print(f"x == y: {x == y}")
print(f"x >= y: {x >= y}")

# Element-wise max/min
print(f"\nElement-wise max: {torch.maximum(x, y)}")
print(f"Element-wise min: {torch.minimum(x, y)}")

# ============================================
# 6. Common mathematical functions
# ============================================
print("\n6. MATHEMATICAL FUNCTIONS")
print("-" * 30)

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
print(f"x = {x}")

print(f"\nabs(x - 2): {torch.abs(x - 2)}")
print(f"sqrt(x): {torch.sqrt(x)}")
print(f"exp(x): {torch.exp(x)}")
print(f"log(x + 1): {torch.log(x + 1)}")  # +1 to avoid log(0)

# Trigonometric
angles = torch.tensor([0.0, 3.14159/2, 3.14159])
print(f"\nsin({angles}): {torch.sin(angles)}")
print(f"cos({angles}): {torch.cos(angles)}")

# Clamp (limit values to a range)
values = torch.tensor([-2, -1, 0, 1, 2, 3, 4])
print(f"\nOriginal: {values}")
print(f"Clamp [0, 2]: {torch.clamp(values, min=0, max=2)}")

# ============================================
# 7. In-place operations
# ============================================
print("\n7. IN-PLACE OPERATIONS")
print("-" * 30)

# Operations ending with _ modify the tensor directly
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Original x: {x}")

x.add_(10)  # In-place addition
print(f"After x.add_(10): {x}")

x.mul_(2)  # In-place multiplication
print(f"After x.mul_(2): {x}")

print("""
Note: In-place operations save memory but can cause
issues with autograd. Use with caution during training!
""")

# ============================================
# 8. Understanding dimensions
# ============================================
print("\n8. UNDERSTANDING DIMENSIONS (dim parameter)")
print("-" * 30)

data = torch.tensor([[1, 2, 3],
                     [4, 5, 6]])
print(f"Data (shape {data.shape}):\n{data}")

print("""
dim=0 means "along rows" (vertical direction)
  - Collapse rows, keep columns
  - Result has same number of columns
  
dim=1 means "along columns" (horizontal direction)
  - Collapse columns, keep rows
  - Result has same number of rows
""")

print(f"sum(dim=0): {data.sum(dim=0)} <- sum each column")
print(f"sum(dim=1): {data.sum(dim=1)} <- sum each row")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
ELEMENT-WISE: +, -, *, /, **
  - Works on tensors of same shape
  - Or broadcasts smaller tensors

MATRIX MULTIPLY: @ or torch.matmul()
  - Different from element-wise *

REDUCTIONS: sum(), mean(), max(), min()
  - Use dim parameter to reduce along specific axis

COMPARISONS: >, <, ==, >=, <=
  - Return boolean tensors

IN-PLACE: add_(), mul_(), etc.
  - Modify tensor directly
  - Save memory but use carefully

KEY INSIGHT:
  Most operations are element-wise by default!
  Matrix multiplication needs @ or matmul()
""")

