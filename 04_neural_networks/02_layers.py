"""
Module 4.2: Common Neural Network Layers
========================================
Learn about the building blocks of neural networks.
"""

import torch
import torch.nn as nn

print("=" * 50)
print("COMMON NEURAL NETWORK LAYERS")
print("=" * 50)

# ============================================
# 1. Linear (Fully Connected) Layer
# ============================================
print("\n1. LINEAR LAYER (nn.Linear)")
print("-" * 30)

# Linear transformation: y = x @ W.T + b
linear = nn.Linear(in_features=4, out_features=3)

print(f"Linear(4, 3)")
print(f"  Weight shape: {linear.weight.shape}")  # (out, in)
print(f"  Bias shape: {linear.bias.shape}")      # (out,)

x = torch.randn(2, 4)  # Batch of 2, each with 4 features
output = linear(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")  # (2, 3)

# Without bias
linear_no_bias = nn.Linear(4, 3, bias=False)
print(f"\nLinear without bias - has bias: {linear_no_bias.bias is not None}")

# ============================================
# 2. Activation Functions
# ============================================
print("\n2. ACTIVATION FUNCTIONS")
print("-" * 30)

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input x: {x}")

# ReLU: max(0, x) - Most common!
relu = nn.ReLU()
print(f"\nReLU(x): {relu(x)}")
print("  Negative values become 0")

# Sigmoid: 1/(1+e^-x) - Output between 0 and 1
sigmoid = nn.Sigmoid()
print(f"\nSigmoid(x): {sigmoid(x)}")
print("  Used for binary classification")

# Tanh: Output between -1 and 1
tanh = nn.Tanh()
print(f"\nTanh(x): {tanh(x)}")
print("  Like sigmoid but centered at 0")

# Softmax: Converts to probability distribution
softmax = nn.Softmax(dim=0)
logits = torch.tensor([2.0, 1.0, 0.1])
print(f"\nLogits: {logits}")
print(f"Softmax(logits): {softmax(logits)}")
print(f"Sum: {softmax(logits).sum()}")  # Always sums to 1
print("  Used for multi-class classification")

# LeakyReLU: Allows small negative values
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
print(f"\nLeakyReLU(x): {leaky_relu(x)}")
print("  Allows small gradient for negative values")

print("""
WHEN TO USE WHICH:
  ReLU      - Default choice for hidden layers
  Sigmoid   - Binary classification output
  Softmax   - Multi-class classification output
  Tanh      - When you need output in [-1, 1]
""")

# ============================================
# 3. Dropout (Regularization)
# ============================================
print("\n3. DROPOUT")
print("-" * 30)

dropout = nn.Dropout(p=0.5)  # 50% of neurons dropped

x = torch.ones(10)
print(f"Input: {x}")

# In training mode - dropout active
dropout.train()
print(f"Training mode output: {dropout(x)}")
print("  Some values are 0, others are scaled up")

# In eval mode - dropout disabled
dropout.eval()
print(f"Eval mode output: {dropout(x)}")
print("  All values pass through")

print("""
DROPOUT PURPOSE:
  - Prevents overfitting
  - Randomly "drops" neurons during training
  - Forces network to be more robust
  - ALWAYS disable during evaluation!
""")

# ============================================
# 4. Batch Normalization
# ============================================
print("\n4. BATCH NORMALIZATION")
print("-" * 30)

# BatchNorm normalizes across the batch
batch_norm = nn.BatchNorm1d(num_features=3)

# Input: (batch_size, features)
x = torch.randn(4, 3) * 10 + 5  # Mean ~5, std ~10
print(f"Input (mean, std): {x.mean():.2f}, {x.std():.2f}")

batch_norm.train()
output = batch_norm(x)
print(f"After BatchNorm: mean={output.mean():.4f}, std={output.std():.4f}")
print("  Normalizes to approximately mean=0, std=1")

print("""
BATCH NORMALIZATION PURPOSE:
  - Stabilizes training
  - Allows higher learning rates
  - Reduces internal covariate shift
  - Has learnable scale and shift parameters
""")

# ============================================
# 5. Embedding Layer
# ============================================
print("\n5. EMBEDDING LAYER")
print("-" * 30)

# Converts integer indices to dense vectors
# Great for words, categories, etc.
embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)

# Input: indices (like word IDs)
indices = torch.tensor([1, 2, 5, 0])
print(f"Input indices: {indices}")

vectors = embedding(indices)
print(f"Output vectors shape: {vectors.shape}")
print(f"Each index becomes a {embedding.embedding_dim}-dimensional vector")

print("""
EMBEDDING PURPOSE:
  - Convert categorical data to dense vectors
  - Used for words, user IDs, product IDs, etc.
  - Vectors are learned during training
""")

# ============================================
# 6. Sequential - Easy way to build networks
# ============================================
print("\n6. nn.Sequential")
print("-" * 30)

# Instead of writing a class, use Sequential
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

print("Model structure:")
print(model)

x = torch.randn(5, 10)  # Batch of 5
model.eval()  # Disable dropout for consistent output
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Access specific layers
print(f"\nFirst layer: {model[0]}")
print(f"First layer weight shape: {model[0].weight.shape}")

# ============================================
# 7. Common patterns
# ============================================
print("\n7. COMMON LAYER PATTERNS")
print("-" * 30)

print("""
CLASSIFICATION NETWORK:
  Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear -> Softmax
  
REGRESSION NETWORK:
  Linear -> ReLU -> Linear -> ReLU -> Linear (no activation at end)
  
WITH BATCH NORM:
  Linear -> BatchNorm -> ReLU -> Dropout -> ...

TYPICAL HIDDEN LAYER:
  Linear -> BatchNorm -> ReLU -> Dropout
  (Order can vary, this is one common pattern)
""")

# Example classification network
classifier = nn.Sequential(
    nn.Linear(784, 256),    # Input layer
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),    # Hidden layer
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),     # Output: 10 classes
    # Note: Often Softmax is in the loss function, not here
)

print("Example classifier for MNIST:")
print(classifier)

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
ESSENTIAL LAYERS:

nn.Linear(in, out)     - Fully connected layer
nn.ReLU()              - Activation (most common)
nn.Sigmoid()           - Activation (binary output)
nn.Softmax(dim)        - Activation (multi-class)
nn.Dropout(p)          - Regularization
nn.BatchNorm1d(n)      - Normalization
nn.Embedding(n, dim)   - For categorical data

BUILDING NETWORKS:
  - Use nn.Sequential for simple architectures
  - Use custom nn.Module for complex ones
  - Remember: model.train() and model.eval()

TIPS:
  - Start with ReLU, it usually works well
  - Add Dropout to prevent overfitting
  - BatchNorm can speed up training
  - Check layer shapes match: out of one = in of next
""")

print("\nNext: Run 03_building_networks.py for complete examples!")
