"""
Module 4.1: Understanding nn.Module
===================================
The foundation of all PyTorch neural networks.
"""

import torch
import torch.nn as nn

print("=" * 50)
print("UNDERSTANDING nn.Module")
print("=" * 50)

# ============================================
# 1. What is nn.Module?
# ============================================
print("\n1. WHAT IS nn.Module?")
print("-" * 30)

print("""
nn.Module is the base class for ALL neural network components in PyTorch.

It provides:
- Parameter management (automatic tracking of weights)
- GPU/CPU movement (.to(device))
- Saving and loading models
- Training/evaluation mode switching
- And much more!

EVERY neural network you build will inherit from nn.Module.
""")

# ============================================
# 2. Your first nn.Module
# ============================================
print("\n2. YOUR FIRST nn.Module")
print("-" * 30)

class SimpleLinear(nn.Module):
    """A simple linear transformation: y = x @ W + b"""
    
    def __init__(self, in_features, out_features):
        # Always call parent's __init__ first!
        super().__init__()
        
        # Create learnable parameters
        # nn.Parameter tells PyTorch these should be trained
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        # Define how input flows through the layer
        return x @ self.weight + self.bias

# Create an instance
model = SimpleLinear(in_features=3, out_features=2)

# Test it
x = torch.tensor([1.0, 2.0, 3.0])
output = model(x)  # This calls forward() automatically!

print(f"Input: {x}")
print(f"Output: {output}")
print(f"Output shape: {output.shape}")

# ============================================
# 3. Inspecting parameters
# ============================================
print("\n3. INSPECTING PARAMETERS")
print("-" * 30)

print("All parameters in the model:")
for name, param in model.named_parameters():
    print(f"  {name}: shape {param.shape}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Parameters have gradients tracked
print(f"\nWeight requires_grad: {model.weight.requires_grad}")

# ============================================
# 4. Using built-in layers (preferred!)
# ============================================
print("\n4. BUILT-IN LAYERS")
print("-" * 30)

# PyTorch provides pre-built layers - use these!
linear_layer = nn.Linear(in_features=3, out_features=2)

print(f"Built-in Linear layer:")
print(f"  Weight shape: {linear_layer.weight.shape}")
print(f"  Bias shape: {linear_layer.bias.shape}")

# Same functionality, but optimized
x = torch.tensor([1.0, 2.0, 3.0])
output = linear_layer(x)
print(f"\nInput: {x}")
print(f"Output: {output}")

# ============================================
# 5. Building a network with multiple layers
# ============================================
print("\n5. MULTI-LAYER NETWORK")
print("-" * 30)

class TwoLayerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Data flows: input -> layer1 -> ReLU -> layer2 -> output
        x = self.layer1(x)
        x = torch.relu(x)  # Activation function
        x = self.layer2(x)
        return x

# Create network: 10 inputs -> 5 hidden -> 2 outputs
network = TwoLayerNetwork(10, 5, 2)

# Test with random input
x = torch.randn(10)
output = network(x)

print(f"Network structure:")
print(network)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

print("\nAll parameters:")
for name, param in network.named_parameters():
    print(f"  {name}: {param.shape}")

# ============================================
# 6. Training vs Evaluation mode
# ============================================
print("\n6. TRAINING VS EVALUATION MODE")
print("-" * 30)

print(f"Default mode - training: {network.training}")

# Switch to evaluation mode
network.eval()
print(f"After .eval() - training: {network.training}")

# Switch back to training mode
network.train()
print(f"After .train() - training: {network.training}")

print("""
Why does mode matter?
- Some layers behave differently during training vs evaluation
- Dropout: active during training, disabled during evaluation
- BatchNorm: uses batch stats in training, running stats in eval

ALWAYS use:
- model.train() before training
- model.eval() before validation/testing
""")

# ============================================
# 7. Moving to GPU
# ============================================
print("\n7. DEVICE MANAGEMENT")
print("-" * 30)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Move entire model to device
network = network.to(device)
print(f"Model moved to: {next(network.parameters()).device}")

# Input must also be on the same device!
x = torch.randn(10).to(device)
output = network(x)
print(f"Output device: {output.device}")

print("""
IMPORTANT: Model and data must be on the same device!
Common pattern:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = MyModel().to(device)
  x = x.to(device)
""")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
nn.Module KEY POINTS:

1. ALWAYS inherit from nn.Module
   class MyNet(nn.Module):

2. ALWAYS call super().__init__()
   def __init__(self):
       super().__init__()

3. Define layers in __init__
   self.layer = nn.Linear(10, 5)

4. Define forward pass in forward()
   def forward(self, x):
       return self.layer(x)

5. Call model directly (not model.forward())
   output = model(x)  # Correct
   output = model.forward(x)  # Works but not recommended

6. Use .train() and .eval() appropriately

7. Use .to(device) for GPU/CPU management

PARAMETERS ARE TRACKED AUTOMATICALLY
when you use nn.Module and nn.Parameter!
""")

print("\nNext: Run 02_layers.py to learn about different layer types!")
