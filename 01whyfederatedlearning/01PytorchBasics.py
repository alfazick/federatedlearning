import torch 

# Create a tensor
x = torch.tensor([1,2,3,4,5])
print("Original tensor:",x)

# Perform operation
y = x + 2
print("After adding 2: ",y)

z = x * 2
print("After multiplying by 2:",z)

# Create a 2d Tensor
matrix = torch.tensor([[1,2],[3,4]])
print("2D tensor:\n",matrix)

# Matrix multiplication
result = torch.matmul(matrix,matrix)
print("Matrix multiplication result:\n",result)

# Move tensor to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_gpu = x.to(device)
print("Tensor moved to:", x_gpu.device)