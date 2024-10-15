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
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    import numpy as np
    import xgboost as xgb

    Train_x, Valid_x, Test_x, Train_y, Valid_y, Test_y = getData(return_tabular=True)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=80, random_state=42),
        'LogisticRegression': MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=50)),
        'GradientBoosting': MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)),
        'LGBM': MultiOutputClassifier(lgb.LGBMClassifier(n_estimators=100, random_state=42)),
        'XGBoost': MultiOutputClassifier(xgb.XGBClassifier(n_estimators=100, random_state=42)),
        'CatBoost': MultiOutputClassifier(CatBoostClassifier(n_estimators=100, random_state=42)),
        'SVM': MultiOutputClassifier(SVC(random_state=42, probability=True))
    }

    #model_name, model = 'SVM', MultiOutputClassifier(SVC(random_state=42, probability=True))
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        
        model.fit(Train_x, Train_y)
        
        val_pred = model.predict_proba(Valid_x)
        val_pred = [i[:, 1] for i in val_pred]
        val_pred = np.vstack(val_pred).T

        tst_pred = model.predict_proba(Test_x)
        tst_pred = [i[:, 1] for i in tst_pred]
        tst_pred = np.vstack(tst_pred).T

        Final_x_val, Final_y_val = pd.DataFrame(val_pred), pd.DataFrame(Valid_y)
        Fianl_x_tst, Final_y_tst = pd.DataFrame(tst_pred), pd.DataFrame(Test_y)

        Final_x_val.to_csv('ML_results/' + model_name + '_xval.csv', index=False)
        Final_y_val.to_csv('ML_results/' + model_name + '_yval.csv', index=False)
        Fianl_x_tst.to_csv('ML_results/' + model_name + '_xtst.csv', index=False)
        Final_y_tst.to_csv('ML_results/' + model_name + '_ytst.csv', index=False)

        
        AUROC_mean, AUROC_CI_lower, AUROC_CI_upper = calculate_AUROC(np.array(Test_y), np.array(tst_pred))

        AUROC_mean = np.array([roc_auc_score(Test_y.iloc[:, col], tst_pred[:, col]) for col in range(Test_y.shape[1])]).mean()
        F1 = calculate_f1_score(Test_y, tst_pred)

        results[model_name] = {
            'AUROC': AUROC_mean,
            'AUROC_CI_lower': AUROC_CI_lower,
            'AUROC_CI_upper': AUROC_CI_upper,
            'F1 Score': F1
        }

    # 결과 출력
    for model_name, scores in results.items():
        print(f"\n{model_name}:")
        print(f"  AUROC: {scores['AUROC']:.4f}")
        print(f"  F1 Score: {scores['F1 Score']:.4f}")
        
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
