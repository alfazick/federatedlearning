import torch
from torch.utils.data import Dataset, Subset, DataLoader,random_split

# Create a simple custom dataset
class SimpleDataset(Dataset):
    def __init__(self,size):
        self.data = torch.randint(0,10,(size,))
        self.labels = torch.randint(0,2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    

# create an instance of dataset
full_dataset = SimpleDataset(1000)


# Create a subset of the dataset
subset_indices = range(0,500)
subset = Subset(full_dataset,subset_indices)
print("Subset size:", len(subset))

# Randomly split dataset
train_size = 800
val_size = 200
train_dataset,val_dataset = random_split(full_dataset,[train_size,val_size])
print("Train set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))


# Create a dataloaders
train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Iterate through the DataLoader
for batch_data, batch_labels in train_loader:
    print("Batch shape:", batch_data.shape)
    print("Labels shape:", batch_labels.shape)
    break  # Just print the first batch

