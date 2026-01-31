"""
Module 3.1: Gradient Basics
===========================
Understand what gradients are and how PyTorch computes them.
"""

import torch

print("=" * 50)
print("GRADIENT BASICS")
print("=" * 50)

# ============================================
# 1. What is a gradient? (Simple math refresher)
# ============================================
print("\n1. WHAT IS A GRADIENT?")
print("-" * 30)

print("""
A gradient (derivative) tells us:
"How much does the OUTPUT change when I change the INPUT?"

Example: y = x²
  - When x = 3, y = 9
  - Gradient dy/dx = 2x = 6
  - This means: if x increases by 1, y increases by about 6
  
WHY THIS MATTERS FOR NEURAL NETWORKS:
  - We want to minimize error (loss)
  - Gradients tell us which direction to adjust weights
  - We move weights in the direction that reduces error
""")

# ============================================
# 2. Creating tensors that track gradients
# ============================================
print("\n2. TRACKING GRADIENTS")
print("-" * 30)

# Regular tensor - no gradient tracking
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Regular tensor: {x}")
print(f"requires_grad: {x.requires_grad}")

# Tensor WITH gradient tracking
x_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"\nTensor with gradients: {x_grad}")
print(f"requires_grad: {x_grad.requires_grad}")

# Or enable it later
x = torch.tensor([1.0, 2.0, 3.0])
x.requires_grad = True
print(f"\nEnabled later: requires_grad = {x.requires_grad}")

# ============================================
# 3. Computing gradients with backward()
# ============================================
print("\n3. COMPUTING GRADIENTS")
print("-" * 30)

# Simple example: y = x²
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x²

print(f"x = {x}")
print(f"y = x² = {y}")

# Compute gradient
y.backward()  # Computes dy/dx

print(f"\nGradient dy/dx at x=3: {x.grad}")
print("(dy/dx = 2x = 2*3 = 6 ✓)")

# ============================================
# 4. Gradients with vectors
# ============================================
print("\n4. VECTOR GRADIENTS")
print("-" * 30)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()  # Need a scalar for backward()

print(f"x = {x}")
print(f"y = x² = {y}")
print(f"z = sum(y) = {z}")

z.backward()
print(f"\nGradient dz/dx: {x.grad}")
print("(dz/dx = 2x = [2, 4, 6] ✓)")

# ============================================
# 5. More complex example
# ============================================
print("\n5. COMPLEX COMPUTATION")
print("-" * 30)

# Forward pass
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# y = w*x + b (like a simple neural network!)
y = w * x + b

print(f"x = {x.item()}")
print(f"w = {w.item()}")
print(f"b = {b.item()}")
print(f"y = w*x + b = {y.item()}")

# Backward pass
y.backward()

print(f"\nGradients:")
print(f"dy/dx = {x.grad}")  # Should be w = 3
print(f"dy/dw = {w.grad}")  # Should be x = 2
print(f"dy/db = {b.grad}")  # Should be 1

print("""
Why these gradients?
  y = w*x + b
  dy/dw = x = 2  (w's effect on y)
  dy/dx = w = 3  (x's effect on y)
  dy/db = 1      (b's direct effect on y)
""")

# ============================================
# 6. Gradient accumulation (important!)
# ============================================
print("\n6. GRADIENT ACCUMULATION")
print("-" * 30)

x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x * 3
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second computation - gradients ACCUMULATE!
y2 = x * 3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # 6, not 3!

print("""
WARNING: Gradients accumulate by default!
This is useful for some algorithms, but usually you want:

x.grad.zero_()  # Reset gradients before each backward

In training loops, use:
optimizer.zero_grad()  # Resets all parameter gradients
""")

# ============================================
# 7. Detaching from gradient computation
# ============================================
print("\n7. DETACHING TENSORS")
print("-" * 30)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

# Detach: create a tensor that doesn't track gradients
y_detached = y.detach()
print(f"y requires_grad: {y.requires_grad}")
print(f"y_detached requires_grad: {y_detached.requires_grad}")

# torch.no_grad(): disable gradient tracking in a block
with torch.no_grad():
    z = x * 2
    print(f"\nInside no_grad, z requires_grad: {z.requires_grad}")

print("""
When to use no_grad():
  - During inference (prediction), not training
  - Saves memory and computation
  - Use for validation/test phases
""")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
KEY CONCEPTS:

1. GRADIENTS tell us how to adjust parameters
   - They point toward increasing values
   - We go OPPOSITE direction to minimize loss

2. requires_grad=True tracks operations
   - Creates a computational graph
   - Enables gradient computation

3. .backward() computes gradients
   - Fills in the .grad attribute
   - Must be called on a scalar

4. GRADIENTS ACCUMULATE
   - Zero them with .zero_grad() before each step
   - Or use optimizer.zero_grad()

5. no_grad() disables tracking
   - Use during inference
   - Saves memory

THE LEARNING PROCESS:
  Forward:  input → prediction → loss
  Backward: loss → gradients → update weights
""")

print("\nNext: Run 02_computational_graph.py!")
