"""
Module 2.3: Combining Tensors
=============================
Learn how to combine multiple tensors together.
"""

import torch

print("=" * 50)
print("COMBINING TENSORS")
print("=" * 50)

# ============================================
# 1. torch.cat() - Concatenate along existing dimension
# ============================================
print("\n1. CONCATENATION (torch.cat)")
print("-" * 30)

a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[5, 6],
                  [7, 8]])

print(f"Tensor a:\n{a}")
print(f"\nTensor b:\n{b}")

# Concatenate along dimension 0 (rows)
cat_dim0 = torch.cat([a, b], dim=0)
print(f"\nConcatenate dim=0 (stack vertically):\n{cat_dim0}")
print(f"Shape: {cat_dim0.shape}")

# Concatenate along dimension 1 (columns)
cat_dim1 = torch.cat([a, b], dim=1)
print(f"\nConcatenate dim=1 (stack horizontally):\n{cat_dim1}")
print(f"Shape: {cat_dim1.shape}")

# Can concatenate more than 2 tensors
c = torch.tensor([[9, 10],
                  [11, 12]])
cat_three = torch.cat([a, b, c], dim=0)
print(f"\nConcatenate three tensors:\n{cat_three}")

# ============================================
# 2. torch.stack() - Stack along NEW dimension
# ============================================
print("\n2. STACKING (torch.stack)")
print("-" * 30)

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.tensor([7, 8, 9])

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")

# Stack creates a NEW dimension
stacked_0 = torch.stack([x, y, z], dim=0)
print(f"\nStack dim=0:\n{stacked_0}")
print(f"Shape: {stacked_0.shape}")  # (3, 3)

stacked_1 = torch.stack([x, y, z], dim=1)
print(f"\nStack dim=1:\n{stacked_1}")
print(f"Shape: {stacked_1.shape}")  # (3, 3)

# The difference with cat:
print(f"""
DIFFERENCE BETWEEN CAT AND STACK:
- cat: joins along EXISTING dimension (no new dimension)
- stack: creates a NEW dimension

Example with 1D tensors [1,2,3] and [4,5,6]:
- cat(dim=0)   -> [1,2,3,4,5,6]      shape: (6,)
- stack(dim=0) -> [[1,2,3],[4,5,6]]  shape: (2, 3)
""")

# ============================================
# 3. Practical example: Creating batches
# ============================================
print("\n3. PRACTICAL: CREATING BATCHES")
print("-" * 30)

# Simulating 4 individual images (grayscale, 28x28)
image1 = torch.randn(1, 28, 28)  # [channels, height, width]
image2 = torch.randn(1, 28, 28)
image3 = torch.randn(1, 28, 28)
image4 = torch.randn(1, 28, 28)

# Stack to create a batch
batch = torch.stack([image1, image2, image3, image4], dim=0)
print(f"Single image shape: {image1.shape}")
print(f"Batch shape: {batch.shape}")  # [batch, channels, height, width]

# ============================================
# 4. torch.chunk() and torch.split() - Divide tensors
# ============================================
print("\n4. SPLITTING TENSORS")
print("-" * 30)

# Create a tensor to split
data = torch.arange(12).reshape(4, 3)
print(f"Original data:\n{data}")

# chunk: split into equal parts
chunks = torch.chunk(data, chunks=2, dim=0)
print(f"\nChunk into 2 parts (dim=0):")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}:\n{chunk}")

# split: split by specific sizes
splits = torch.split(data, split_size_or_sections=[1, 3], dim=0)
print(f"\nSplit [1, 3] (dim=0):")
for i, split in enumerate(splits):
    print(f"  Split {i}:\n{split}")

# ============================================
# 5. torch.hstack, vstack, dstack
# ============================================
print("\n5. CONVENIENCE FUNCTIONS")
print("-" * 30)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# hstack: horizontal stack (along columns)
print(f"a: {a}")
print(f"b: {b}")
print(f"hstack: {torch.hstack([a, b])}")

# vstack: vertical stack (along rows)
print(f"vstack:\n{torch.vstack([a, b])}")

# For 2D tensors
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[5, 6], [7, 8]])

print(f"\nMatrix m1:\n{m1}")
print(f"Matrix m2:\n{m2}")
print(f"\nhstack:\n{torch.hstack([m1, m2])}")
print(f"\nvstack:\n{torch.vstack([m1, m2])}")

# ============================================
# 6. Broadcasting review
# ============================================
print("\n6. BROADCASTING (AUTOMATIC SIZE MATCHING)")
print("-" * 30)

# Broadcasting automatically expands dimensions
a = torch.tensor([[1], [2], [3]])  # Shape: (3, 1)
b = torch.tensor([10, 20, 30, 40])  # Shape: (4,)

print(f"a (shape {a.shape}):\n{a}")
print(f"b (shape {b.shape}): {b}")

# PyTorch broadcasts automatically!
result = a + b  # (3, 1) + (4,) -> (3, 4)
print(f"\na + b (shape {result.shape}):\n{result}")

print("""
Broadcasting rules:
1. Align shapes from the right
2. Dimensions must be equal or one of them must be 1
3. Missing dimensions are treated as 1

Example: (3, 1) + (4,)
  Step 1: Align    -> (3, 1) and (1, 4)
  Step 2: Expand   -> (3, 4) and (3, 4)
  Step 3: Add element-wise
""")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
COMBINING:
  torch.cat([a, b], dim)   - Join along existing dimension
  torch.stack([a, b], dim) - Join along NEW dimension
  torch.hstack([a, b])     - Horizontal stack
  torch.vstack([a, b])     - Vertical stack

SPLITTING:
  torch.chunk(x, n, dim)   - Split into n equal parts
  torch.split(x, sizes, dim) - Split by specific sizes

BROADCASTING:
  - Automatic size matching for operations
  - Shapes aligned from right
  - Size-1 dims are expanded
  
KEY INSIGHT:
  Use STACK to create batches from individual samples
  Use CAT to combine batches into larger batches
""")

print("\nNow try the exercises!")
