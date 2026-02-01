# Module 4: Building Neural Networks

## What You'll Learn

Now that you understand tensors and autograd, it's time to build actual neural networks!

- The `nn.Module` class (base of all networks)
- Linear layers (fully connected)
- Activation functions (ReLU, Sigmoid, etc.)
- Building your own network architecture

## The nn.Module Pattern

Every neural network in PyTorch follows this pattern:

```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define how data flows through layers
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

## Files in This Module

1. `01_nn_module.py` - Understanding nn.Module
2. `02_layers.py` - Different layer types
3. `03_building_networks.py` - Complete network examples

