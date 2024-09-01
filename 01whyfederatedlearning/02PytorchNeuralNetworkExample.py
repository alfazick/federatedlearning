import torch
import torch.nn as nn

# define a simple neural network

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2 = nn.Linear(5,1)
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x
    

# create an instance of the network
model = SimpleNN()
print(model)

# create a random input tensor
input_tensor = torch.randn(1,10)
print(input_tensor)


# Forward pass through network
output = model(input_tensor)# Compute the model output
print("Output shape: ", output.shape)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# Simulate one step of training
target = torch.randn(1,1)
loss = criterion(output,target)# Calculate loss
loss.backward()# Backpropagate the loss
optimizer.step()# Adjust model weights

print("Loss: ", loss.item())
optimizer.zero_grad()   # Clear existing gradients
