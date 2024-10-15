import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from GATT import GATT, TabularDataset
from utils import train_batch, valid_batch, calculate_roc, calculate_f1_score

def getData(filename="dataset.tsv", batch_size=32, return_tabular=False):
    Data = pd.read_csv(filename, sep="\t")
    
    # manual split data into Dependent and Independent variables
    LabelNames = Data.columns[56:]
    X, Y = Data.iloc[:, 1:56], Data.loc[:, LabelNames]
    le = LabelEncoder()
    X[X.columns[0]] = le.fit_transform(X[X.columns[0]])

    X_temp, Test_x, Y_temp, Test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
    Train_x, Valid_x, Train_y, Valid_y = train_test_split(X_temp, Y_temp, test_size=0.20, random_state=42)
    if return_tabular:
        return Train_x, Valid_x, Test_x, Train_y, Valid_y, Test_y

    print(f"Train: {Train_x.shape[0]}")
    print(f"Valid: {Valid_x.shape[0]}")
    print(f"Test: {Test_x.shape[0]}")

    # Dataset (Tarbular dataset) random split
    train_dataset = TabularDataset(Train_x, Train_y)
    valid_dataset = TabularDataset(Valid_x, Valid_y)
    test_dataset  = TabularDataset(Test_x, Test_y)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training samples: {len(Train_x)}")
    print(f"Number of validation samples: {len(Valid_x)}")
    print(f"Number of test samples: {len(Test_x)}")
    print("")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    return train_loader, valid_loader, test_loader

def main(args):
    # Load data
    train_loader, valid_loader, test_loader = getData(filename="20231130 TMD 신환 (latest) 정리본_na_omit.tsv", batch_size=1)
    valid_loader = test_loader
    
    # Initialize model with cuda device
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    model = GATT(args.input_dim, args.hidden_dim, args.num_classes)
    model = model.to(device)

    # Loss functions and optimizer
    ssl_lossf = F.smooth_l1_loss  # For self-supervised learning
    cls_lossf = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Optimizer
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8, verbose=False)

    # Training loop
    max_roc_dict_val = {f'label_{i}': 0 for i in range(args.num_classes)}
    max_f1_val = .0
    Final_x_val, Final_y_val = list(range(args.num_classes)), list(range(args.num_classes))

    for epoch in range(args.num_epochs):
        model.train()
        losses = torch.zeros(3)  # [ssl_loss, cls_loss, total_loss]
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            losses += train_batch(model, x, y, mask, ssl_lossf, cls_lossf, optimizer, args.cls_loss_weight)
        
        avg_loss = losses / len(train_loader)
        avg_loss_char = "-".join([f'{float(i):.4f}' for i in avg_loss])
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss_char}")
        
        # Validation
        val_roc, val_f1, val_pred, val_label = calculate_roc(model, valid_loader, device)
        
        if val_f1 > max_f1_val:
            max_f1_val = val_f1
            model.save(os.path.join(args.save_dir, 'best_model.pth'))

        print(f"### Epoch {epoch+1} #########################################")
        print(f"Validation AUROC: {np.array(val_roc).mean():.3f}")
        print(f"F1 Values: {val_f1:.3f}")

        BestROC = list(max_roc_dict_val.values())).mean()
        print(f"Max ROC Values: {BestROC.mean():.3f} - {BestROC.mean():.3f}")
        print(f"Max F1 Values: {max_f1_val:.3f}")
        scheduler.step( np.array(val_roc).mean() )
    print("Training complete!")
    
    # Get Performance Metrics of the Test set from the best model
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    # val_roc, val_f1, val_pred, val_label = calculate_roc(model, valid_loader, device)

    # Get Feature Importance using SHAP
    # shap_values = shap_explainer(model, valid_loader, device)
    # feature_importance = np.abs(shap_values).mean(axis=(0, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with various learning rate schedulers')
    parser.add_argument('--input_dim', type=int, default=55, help='Input dimension')
    parser.add_argument('--num_classes', type=int, default=28, help='Number of classes')
    parser.add_argument('--save_dir', type=str, default='MODEL', help='Directory to save models')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the model')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device index')
    args = parser.parse_args()
    main(args)
