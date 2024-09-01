# MNIST Neural Network Script: Detailed Analysis

## Imports and Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
```

- `torch`: Main PyTorch library for tensor computations and neural network operations
- `torch.nn`: Contains neural network layers and loss functions
- `torch.utils.data`: Provides utilities for data loading and manipulation
- `torch.optim`: Contains optimization algorithms like SGD
- `torchvision`: Provides datasets and common image transformations
- `numpy`: For numerical computations
- `matplotlib.pyplot`: For creating plots
- `sklearn.metrics`: For computing the confusion matrix
- `seaborn`: For creating enhanced visualizations

## Data Transformation

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

- Creates a composition of transformations
- `ToTensor()`: Converts images to PyTorch tensors
- `Normalize((0.5,), (0.5,))`: Normalizes the tensor with mean 0.5 and std 0.5

## Model Definition

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x
```

- Defines a simple neural network model
- `__init__`: Initializes the layers
  - `nn.Linear(784, 128)`: Fully connected layer with 784 inputs and 128 outputs
  - `nn.ReLU()`: ReLU activation function
  - `nn.Linear(128, 10)`: Output layer with 128 inputs and 10 outputs (one for each digit)
- `forward`: Defines the forward pass of the network
  - `torch.flatten(x, 1)`: Flattens the input tensor, keeping the batch dimension
  - Applies the layers in sequence: fc -> relu -> out

## Training Function

```python
def train_model(model, train_set):
    batch_size = 64
    num_epochs = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader)}")

    print("Training complete")
```

- Sets batch size and number of epochs
- Creates a DataLoader for the training set
- Defines loss function (CrossEntropyLoss) and optimizer (SGD)
- Trains the model for the specified number of epochs:
  - Iterates through batches of data
  - Computes the forward pass, loss, and backward pass
  - Updates model parameters
  - Prints the average loss for each epoch

## Evaluation Function

```python
def evaluate_model(model, test_set):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)

    return average_loss, accuracy
```

- Sets the model to evaluation mode
- Creates a DataLoader for the test set
- Iterates through the test set:
  - Computes model outputs
  - Calculates the number of correct predictions
  - Computes the loss
- Calculates and returns the average loss and accuracy

## Data Manipulation Functions

```python
def include_digits(dataset, included_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] in included_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)

def exclude_digits(dataset, excluded_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)
```

- `include_digits`: Creates a subset of the dataset including only specified digits
- `exclude_digits`: Creates a subset of the dataset excluding specified digits
- Both functions use list comprehensions to filter the dataset based on the digit labels

## Visualization Functions

```python
def plot_distribution(dataset, title):
    labels = [data[1] for data in dataset]
    unique_labels, label_counts = torch.unique(torch.tensor(labels), return_counts=True)

    plt.figure(figsize=(4, 2))

    counts_dict = {
        label.item(): count.item() for label, count in zip(unique_labels, label_counts)
    }

    all_labels = np.arange(10)
    all_label_counts = [counts_dict.get(label, 0) for label in all_labels]

    plt.bar(all_labels, all_label_counts)
    plt.title(title)
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.xticks(all_labels)
    plt.show()

def compute_confusion_matrix(model, testset):
    true_labels = []
    predicted_labels = []

    for image, label in testset:
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        true_labels.append(label)
        predicted_labels.append(predicted.item())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    return cm

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
```

- `plot_distribution`: 
  - Counts the occurrences of each digit in the dataset
  - Creates a bar plot showing the distribution of digits
- `compute_confusion_matrix`:
  - Iterates through the test set, getting model predictions
  - Computes and returns the confusion matrix
- `plot_confusion_matrix`:
  - Creates a heatmap visualization of the confusion matrix using seaborn

This script provides a comprehensive set of tools for training, evaluating, and analyzing a simple neural network on the MNIST dataset, with flexibility for data manipulation and result visualization.
