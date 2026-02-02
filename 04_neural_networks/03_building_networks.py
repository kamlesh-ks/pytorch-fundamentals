"""
Module 4.3: Building Complete Networks
======================================
Put it all together to build practical neural networks.
"""

import torch
import torch.nn as nn

print("=" * 50)
print("BUILDING COMPLETE NEURAL NETWORKS")
print("=" * 50)

# ============================================
# 1. Simple Classifier Network
# ============================================
print("\n1. SIMPLE CLASSIFIER (e.g., for MNIST)")
print("-" * 30)

class SimpleClassifier(nn.Module):
    """
    A simple feedforward network for classification.
    Input: Flattened image (e.g., 28x28 = 784 pixels)
    Output: Class probabilities (e.g., 10 digits)
    """
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super().__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine into sequential
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input if needed (for images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

# Create and test
model = SimpleClassifier()
print(model)

# Test with fake batch of images
fake_images = torch.randn(32, 1, 28, 28)  # 32 grayscale 28x28 images
output = model(fake_images)
print(f"\nInput shape: {fake_images.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (logits for first image): {output[0]}")

# ============================================
# 2. Regression Network
# ============================================
print("\n2. REGRESSION NETWORK (e.g., House Prices)")
print("-" * 30)

class RegressionNetwork(nn.Module):
    """
    Network for predicting continuous values.
    No activation on output layer!
    """
    
    def __init__(self, input_features, hidden_sizes=[64, 32]):
        super().__init__()
        
        layers = []
        prev_size = input_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Output: single value, NO activation!
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # Remove last dim

# Test
model = RegressionNetwork(input_features=13)  # 13 house features
print(model)

fake_house_data = torch.randn(16, 13)  # 16 houses, 13 features each
predicted_prices = model(fake_house_data)
print(f"\nInput shape: {fake_house_data.shape}")
print(f"Output shape: {predicted_prices.shape}")

# ============================================
# 3. Network with Skip Connections
# ============================================
print("\n3. NETWORK WITH SKIP CONNECTIONS")
print("-" * 30)

class ResidualBlock(nn.Module):
    """
    A residual block: output = ReLU(x + F(x))
    Skip connections help with training deep networks.
    """
    
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.BatchNorm1d(features)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Skip connection: add input to output
        return self.relu(x + self.block(x))


class ResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, num_classes):
        super().__init__()
        
        # Initial projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Stack of residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

model = ResidualNetwork(784, 256, num_blocks=3, num_classes=10)
print(f"Residual Network with {sum(p.numel() for p in model.parameters())} parameters")

# Test
x = torch.randn(8, 784)
output = model(x)
print(f"Input: {x.shape} -> Output: {output.shape}")

# ============================================
# 4. Network with Custom Forward Logic
# ============================================
print("\n4. CUSTOM FORWARD LOGIC")
print("-" * 30)

class MultiHeadNetwork(nn.Module):
    """
    Network with multiple output heads.
    Useful for multi-task learning.
    """
    
    def __init__(self, input_size):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Separate heads for different tasks
        self.classification_head = nn.Linear(64, 10)  # 10 classes
        self.regression_head = nn.Linear(64, 1)       # 1 value
    
    def forward(self, x, task='both'):
        # Shared processing
        shared_features = self.shared(x)
        
        if task == 'classification':
            return self.classification_head(shared_features)
        elif task == 'regression':
            return self.regression_head(shared_features)
        else:  # 'both'
            return {
                'classification': self.classification_head(shared_features),
                'regression': self.regression_head(shared_features)
            }

model = MultiHeadNetwork(784)
x = torch.randn(4, 784)

# Different ways to use it
class_output = model(x, task='classification')
reg_output = model(x, task='regression')
both_output = model(x, task='both')

print(f"Classification output: {class_output.shape}")
print(f"Regression output: {reg_output.shape}")
print(f"Both outputs: classification={both_output['classification'].shape}, "
      f"regression={both_output['regression'].shape}")

# ============================================
# 5. Saving and Loading Models
# ============================================
print("\n5. SAVING AND LOADING MODELS")
print("-" * 30)

model = SimpleClassifier()

# Method 1: Save entire model (not recommended for production)
# torch.save(model, 'model.pth')
# loaded = torch.load('model.pth')

# Method 2: Save state dict (recommended!)
# torch.save(model.state_dict(), 'model_weights.pth')
# model.load_state_dict(torch.load('model_weights.pth'))

print("""
SAVING MODELS:

# Save weights only (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load weights
model = SimpleClassifier()  # Create architecture first
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode

# Save with optimizer state (for resuming training)
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pth')
""")

# ============================================
# 6. Model Summary Helper
# ============================================
print("\n6. MODEL SUMMARY")
print("-" * 30)

def model_summary(model):
    """Print a summary of the model."""
    print(f"{'Layer':<30} {'Output Shape':<20} {'Params':>10}")
    print("-" * 62)
    
    total_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        print(f"{name:<30} {str(param.shape):<20} {params:>10,}")
    
    print("-" * 62)
    print(f"{'Total Parameters:':<50} {total_params:>10,}")
    print(f"{'Trainable Parameters:':<50} "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):>10,}")

model = SimpleClassifier(784, [256, 128], 10)
model_summary(model)

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
BUILDING NETWORKS:

1. CLASSIFIER: End with Linear(hidden, num_classes)
   - Softmax usually in loss function (CrossEntropyLoss)

2. REGRESSION: End with Linear(hidden, 1) - NO activation!
   - Use MSELoss or L1Loss

3. SKIP CONNECTIONS: output = x + F(x)
   - Help train deeper networks

4. MULTI-TASK: Multiple output heads sharing features

ARCHITECTURE TIPS:

- Start simple, add complexity as needed
- Power of 2 for hidden sizes (64, 128, 256, 512)
- Add Dropout between layers to prevent overfitting
- BatchNorm can stabilize training
- Match input/output sizes of consecutive layers!

DEBUGGING:

- Print shapes at each step
- Start with tiny data to test
- Check gradients are flowing
- Verify output shape matches labels
""")

print("Next: Move to Module 5 to learn how to train them!")
