# PyTorch Libraries Overview

This document provides an overview of key PyTorch libraries and their purposes.

## 1. torch

**Purpose**: Core PyTorch library for tensor computations and automatic differentiation.

**Key features**:
- Tensor operations (similar to NumPy, but with GPU support)
- Autograd for automatic differentiation
- Supports both CPU and GPU computations
- Provides the foundation for building and training neural networks

**Example use**:
```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.sum(x)
```

## 2. torch.nn

**Purpose**: Provides building blocks for constructing neural network architectures.

**Key features**:
- Neural network layers (Linear, Conv2d, LSTM, etc.)
- Activation functions (ReLU, Sigmoid, Tanh, etc.)
- Loss functions (MSELoss, CrossEntropyLoss, etc.)
- Utilities for parameter initialization

**Example use**:
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
```

## 3. torch.utils.data

**Purpose**: Provides utilities for data loading and manipulation.

**Key features**:
- Dataset and DataLoader classes for efficient data handling
- Utilities for batching, shuffling, and parallel data loading
- Subset and random split functions for dataset manipulation

**Example use**:
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # ... implementation ...

dataloader = DataLoader(MyDataset(), batch_size=32, shuffle=True)
```

## 4. torch.optim

**Purpose**: Implements various optimization algorithms for training neural networks.

**Key features**:
- Optimizers like SGD, Adam, RMSprop, etc.
- Learning rate scheduling
- Gradient clipping utilities

**Example use**:
```python
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## 5. torchvision

**Purpose**: Provides utilities for computer vision tasks.

**Key features**:
- Popular datasets (MNIST, CIFAR10, ImageNet, etc.)
- Common image transformations and data augmentation techniques
- Pre-trained models for various computer vision tasks
- Vision-specific neural network architectures

**Example use**:
```python
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
```

## 6. torch.cuda

**Purpose**: Enables GPU acceleration for PyTorch operations.

**Key features**:
- GPU memory management
- Utilities for moving tensors and models between CPU and GPU
- Multi-GPU support

**Example use**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
```

This overview covers the main PyTorch libraries we've discussed. Each library plays a crucial role in the PyTorch ecosystem, enabling efficient development and training of deep learning models.
