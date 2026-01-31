"""
Module 3.2: Computational Graph
===============================
Understand how PyTorch tracks operations to compute gradients.
"""

import torch

print("=" * 50)
print("COMPUTATIONAL GRAPH")
print("=" * 50)

# ============================================
# 1. What is a computational graph?
# ============================================
print("\n1. WHAT IS A COMPUTATIONAL GRAPH?")
print("-" * 30)

print("""
PyTorch builds a graph of all operations performed on tensors
with requires_grad=True. This graph is used to compute gradients.

Example: y = (a + b) * c

         [y]          <- Output
          |
         [*]          <- Multiplication
        /   \\
     [+]     [c]      <- Addition and c
    /   \\
  [a]   [b]           <- Inputs

PyTorch records this graph and uses it to compute gradients
via the chain rule of calculus.
""")

# ============================================
# 2. Visualizing the graph
# ============================================
print("\n2. SEEING THE GRAPH")
print("-" * 30)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(4.0, requires_grad=True)

# Build computation
temp = a + b
y = temp * c

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"temp = a + b = {temp}")
print(f"y = temp * c = {y}")

# Each tensor knows its history
print(f"\ny.grad_fn: {y.grad_fn}")  # Multiplication
print(f"temp.grad_fn: {temp.grad_fn}")  # Addition

# Leaf tensors have no grad_fn
print(f"\na.grad_fn: {a.grad_fn}")  # None (leaf)
print(f"a.is_leaf: {a.is_leaf}")

# ============================================
# 3. Chain rule in action
# ============================================
print("\n3. CHAIN RULE")
print("-" * 30)

print("""
The chain rule says:
  dy/da = (dy/d_temp) * (d_temp/da)

For y = (a + b) * c:
  temp = a + b
  y = temp * c
  
  dy/d_temp = c = 4
  d_temp/da = 1
  
  dy/da = c * 1 = 4
  dy/db = c * 1 = 4
  dy/dc = temp = 5
""")

y.backward()

print(f"dy/da = {a.grad} (expected: c = 4)")
print(f"dy/db = {b.grad} (expected: c = 4)")
print(f"dy/dc = {c.grad} (expected: a+b = 5)")

# ============================================
# 4. Dynamic graphs (PyTorch's superpower)
# ============================================
print("\n4. DYNAMIC COMPUTATION GRAPHS")
print("-" * 30)

print("""
PyTorch uses DYNAMIC computational graphs:
- Graph is built fresh for each forward pass
- Can change based on input data
- Allows for loops, conditionals, etc.

This is different from TensorFlow 1.x (static graphs).
""")

def dynamic_computation(x, do_square=True):
    """Graph changes based on the flag!"""
    if do_square:
        return x ** 2
    else:
        return x * 3

x = torch.tensor(2.0, requires_grad=True)

# First call - squaring
y1 = dynamic_computation(x, do_square=True)
y1.backward()
print(f"With squaring: dy/dx = {x.grad}")  # 2*2 = 4

# Reset gradient
x.grad.zero_()

# Second call - tripling
y2 = dynamic_computation(x, do_square=False)
y2.backward()
print(f"With tripling: dy/dx = {x.grad}")  # 3

# ============================================
# 5. Graph is consumed after backward
# ============================================
print("\n5. GRAPH RETENTION")
print("-" * 30)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

print(f"Before backward: y.grad_fn = {y.grad_fn}")

y.backward()

print(f"After backward: Gradients computed")
print(f"x.grad = {x.grad}")

# The graph is freed after backward()
# Trying to backward again will fail!
print("""
Note: By default, the graph is freed after backward().
To keep it (rare), use: y.backward(retain_graph=True)
""")

# ============================================
# 6. Neural network context
# ============================================
print("\n6. HOW THIS APPLIES TO NEURAL NETWORKS")
print("-" * 30)

# Simulating a tiny neural network
# Input -> Linear -> ReLU -> Output

# "Weights" (learnable parameters)
w1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
b1 = torch.tensor([0.1, 0.1], requires_grad=True)

# Input (not learnable)
x = torch.tensor([1.0, 2.0])

# Forward pass
linear_out = x @ w1 + b1  # Linear layer
relu_out = torch.relu(linear_out)  # ReLU activation
loss = relu_out.sum()  # Simple "loss"

print(f"Input x: {x}")
print(f"After linear: {linear_out}")
print(f"After ReLU: {relu_out}")
print(f"Loss: {loss}")

# Backward pass
loss.backward()

print(f"\nGradients computed!")
print(f"w1.grad:\n{w1.grad}")
print(f"b1.grad: {b1.grad}")

print("""
This is EXACTLY what happens in neural network training:
1. Forward: data flows through layers
2. Loss: compute how wrong we are
3. Backward: gradients flow back through layers
4. Update: adjust weights using gradients
""")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
COMPUTATIONAL GRAPH:
  - Records all operations on requires_grad=True tensors
  - Used to compute gradients via chain rule
  - Built dynamically during forward pass

KEY ATTRIBUTES:
  tensor.grad_fn      - Function that created this tensor
  tensor.is_leaf      - Is this a leaf node (no grad_fn)?
  tensor.grad         - Computed gradient (after backward)

DYNAMIC GRAPHS:
  - Graph can change each forward pass
  - Supports Python control flow (if, for, while)
  - Graph is freed after backward() by default

THE FLOW:
  Forward:  Build graph, compute output
  Backward: Traverse graph, compute gradients
  Update:   Use gradients to update weights
""")

print("\nNext: Run 03_practical_example.py!")
