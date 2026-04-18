import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split
import joblib
import os

# Import từ các file local
from model import NMRModel
from dataset import GetObjFeature

def apply_normalization(ds, mean, std):
    for data in ds:
        data.y_norm = (data.y - mean) / std
    return ds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    data_path = '../data/NMRShift.pkl'
    checkpoint_path = '../models/checkpoint.pth'
    best_model_path = '../models/best_model.pth'
    
    if not os.path.exists('../models'): os.makedirs('../models')

    print("Loading dataset...")
    graph_obj = GetObjFeature(data_path)
    dataset = graph_obj.run()

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    torch.manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    all_y_carbon = []
    for data in train_ds:
        carbon_shifts = data.y[data.carbon_mask & ~torch.isnan(data.y)]
        all_y_carbon.append(carbon_shifts)
    all_y_carbon = torch.cat(all_y_carbon)
    Y_MEAN, Y_STD = all_y_carbon.mean().item(), all_y_carbon.item()
    
    train_ds = apply_normalization(train_ds, Y_MEAN, Y_STD)
    val_ds = apply_normalization(val_ds, Y_MEAN, Y_STD)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = NMRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.HuberLoss(delta=0.5)
    mae_metric = nn.L1Loss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_mae = float('inf')
    for epoch in range(300):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data).view(-1)
            mask = ~torch.isnan(data.y_norm) & data.carbon_mask
            if mask.sum() > 0:
                loss = criterion(outputs[mask], data.y_norm[mask])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print(f"Epoch {epoch+1} completed.")

    print('Finished Training')