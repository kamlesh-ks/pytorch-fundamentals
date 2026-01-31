"""
Module 3.3: Practical Gradient Example
======================================
Let's use gradients to solve a real problem: linear regression!
"""

import torch

print("=" * 50)
print("PRACTICAL: LINEAR REGRESSION WITH GRADIENTS")
print("=" * 50)

# ============================================
# The Problem
# ============================================
print("\n THE PROBLEM")
print("-" * 30)

print("""
We have data points that follow: y = 2x + 1 (with some noise)
We want to LEARN the parameters (w=2, b=1) from the data.

This is the simplest machine learning problem!
""")

# Generate some data: y = 2x + 1 + noise
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(0, 10, 20)  # 20 points from 0 to 10
y_true = 2 * X + 1 + torch.randn(20) * 0.5  # True relationship + noise

print(f"X (first 5): {X[:5]}")
print(f"y (first 5): {y_true[:5]}")

# ============================================
# Initialize Parameters
# ============================================
print("\n INITIALIZE PARAMETERS")
print("-" * 30)

# Start with random guesses
w = torch.tensor(0.0, requires_grad=True)  # Weight (should learn ~2)
b = torch.tensor(0.0, requires_grad=True)  # Bias (should learn ~1)

print(f"Initial w = {w.item():.4f} (should be ~2)")
print(f"Initial b = {b.item():.4f} (should be ~1)")

# ============================================
# Training Loop
# ============================================
print("\n TRAINING")
print("-" * 30)

learning_rate = 0.01
n_epochs = 100

print("Epoch | Loss     | w      | b")
print("-" * 40)

for epoch in range(n_epochs):
    # 1. FORWARD PASS: Make predictions
    y_pred = w * X + b
    
    # 2. COMPUTE LOSS: Mean Squared Error
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 3. BACKWARD PASS: Compute gradients
    loss.backward()
    
    # 4. UPDATE PARAMETERS (gradient descent)
    # Note: We use torch.no_grad() because we don't want to track
    # the update operation in the computational graph
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 5. ZERO GRADIENTS for next iteration
    w.grad.zero_()
    b.grad.zero_()
    
    # Print progress
    if epoch % 10 == 0:
        print(f"{epoch:5d} | {loss.item():.4f} | {w.item():.4f} | {b.item():.4f}")

print(f"\n Final: w = {w.item():.4f}, b = {b.item():.4f}")
print(f" True:  w = 2.0000, b = 1.0000")

# ============================================
# Understanding What Happened
# ============================================
print("\n WHAT HAPPENED?")
print("-" * 30)

print("""
Each iteration:

1. FORWARD: Predicted y = w*x + b
   
2. LOSS: Measured error with MSE = mean((predicted - actual)²)
   
3. BACKWARD: PyTorch computed gradients:
   - d(loss)/dw: How much w affects the loss
   - d(loss)/db: How much b affects the loss
   
4. UPDATE: Moved parameters in opposite direction of gradient
   - w = w - learning_rate * gradient
   - If gradient is positive, w decreases
   - If gradient is negative, w increases
   - This REDUCES the loss!
   
5. ZERO GRAD: Cleared gradients for next iteration
   (Remember: gradients accumulate by default)

This is GRADIENT DESCENT - the foundation of all deep learning!
""")

# ============================================
# Visualizing the gradient descent
# ============================================
print("\n GRADIENT DESCENT INTUITION")
print("-" * 30)

print("""
Imagine you're blindfolded on a hilly terrain.
You want to reach the lowest point (minimum loss).

The GRADIENT tells you:
- Which direction is uphill
- How steep the slope is

To go DOWNHILL, you step in the OPPOSITE direction:
  new_position = old_position - step_size * gradient

The LEARNING RATE controls step size:
- Too small: Slow progress
- Too large: Might overshoot the minimum
- Just right: Efficient descent
""")

# ============================================
# This is exactly what PyTorch does!
# ============================================
print("\n IN REAL PYTORCH CODE")
print("-" * 30)

print("""
In real PyTorch code, you'll use:

import torch.optim as optim

# Create optimizer (handles updates for you)
optimizer = optim.SGD([w, b], lr=0.01)

for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y_true)
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update parameters

The optimizer does steps 4-5 automatically!
""")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
TRAINING LOOP PATTERN:

1. Forward pass:  y_pred = model(x)
2. Compute loss:  loss = loss_function(y_pred, y_true)
3. Zero grads:    optimizer.zero_grad()
4. Backward pass: loss.backward()
5. Update params: optimizer.step()

KEY INSIGHTS:

- Gradients point toward INCREASING loss
- We move OPPOSITE direction to DECREASE loss
- Learning rate controls step size
- This is GRADIENT DESCENT

YOU NOW UNDERSTAND HOW NEURAL NETWORKS LEARN!
""")

print("\nCongratulations! You've completed Module 3!")
print("Next: Move to Module 4 to build actual neural networks!")
