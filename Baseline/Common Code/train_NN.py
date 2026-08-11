import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx].copy()).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y



def train_model(model, optimizer, loss_fn, X_train, y_train, epochs=30, batch_size=32):
   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
    #                               torch.tensor(y_train, dtype=torch.long))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    dataset = CustomDataset(X_train, y_train) #x_train.npy file
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    os.makedirs("./model_folder", exist_ok=True)

    for epoch in range(epochs):
        model.train(True)  
        running_loss = 0.0

        
        for inputs, labels in train_loader:
            optimizer.zero_grad()  

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)  
            loss = loss_fn(outputs, labels) 
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model_path = os.path.join("./model_folder", "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model Saved in {model_path}")
    return model