import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score
import shap


def train_batch(model, x, y, mask, ssl_lossf, cls_lossf, optimizer, cls_loss_weight=0.5):
    # Forward pass
    decoded, classified = model(x, mask)
    
    # Compute losses
    ssl_loss = ssl_lossf(decoded, x)  # Self-supervised learning loss
    cls_loss = cls_lossf(classified, y)  # Classification loss
    
    # Combine losses
    loss = (1-cls_loss_weight) * ssl_loss + cls_loss_weight * cls_loss

    losses = torch.tensor([ssl_loss.item(), cls_loss.item(), loss.item()])
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return losses

def valid_batch(model, x, y, mask, ssl_lossf, cls_lossf, optimizer, cls_loss_weight=0.5):
    # Forward pass
    decoded, classified = model(x, mask)

    # Compute losses
    ssl_loss = ssl_lossf(decoded, x)
    cls_loss = cls_lossf(classified, y)

    # Combine losses
    loss = (1-cls_loss_weight) * ssl_loss + cls_loss_weight * cls_loss
    losses = torch.tensor([ssl_loss.item(), cls_loss.item(), loss.item()])
    
    return losses, decoded, classified

def calculate_roc(model, data_loader, device):
    model.eval()
    val_ys, pred_output = [], []    
    with torch.no_grad():
        for x, y, mask in data_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            _, classified = model(x, mask)
            val_ys.append(y)
            pred_output.append(classified)
    all_y = torch.vstack(val_ys).cpu().numpy()
    all_pred = torch.sigmoid(torch.vstack(pred_output)).cpu().numpy()
    roc_vals = [roc_auc_score(all_y[:, col], all_pred[:, col]) for col in range(all_y.shape[1])]    
    f1 = calculate_f1_score(all_y, all_pred)
    return roc_vals, f1, all_pred, all_y

# F1 Score function
def calculate_f1_score(y_true, y_pred):
    y_pred = (y_pred > 0.5)
    return f1_score(y_true, y_pred, average='macro')

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    
    auroc_mean = np.mean([roc_auc_score(y_true.iloc[:, col], y_pred_proba[:, col]) for col in range(y_true.shape[1])])
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    mcc = np.mean([matthews_corrcoef(y_true.iloc[:, col], y_pred[:, col]) for col in range(y_true.shape[1])])
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'AUROC': auroc_mean,
        'F1 Score (Macro)': f1_macro,
        'F1 Score (Weighted)': f1_weighted,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'MCC': mcc,
        'Cohen\'s Kappa': cohen_kappa
    }

def calculate_AUROC(Test_y, tst_pred, n_bootstraps =1000):
    rng = np.random.RandomState(seed=42)
    AUROCs = []
    CI_lowers = []
    CI_uppers = []

    for col in range(Test_y.shape[1]):
        true_vals = Test_y[:, col]
        pred_vals = tst_pred[:, col]
        auroc = roc_auc_score(true_vals, pred_vals)
        AUROCs.append(auroc)
        
        # Boostrap based CI calcuation
        bootstrap_means = []
        for i in range(n_bootstraps):
            indices = rng.choice(len(true_vals), size=len(true_vals), replace=True)
            if len(np.unique(true_vals[indices])) < 2:
                continue
            sample_auroc = roc_auc_score(true_vals[indices], pred_vals[indices])
            bootstrap_means.append(sample_auroc)
        
        CI_lower = np.percentile(bootstrap_means, 2.5)
        CI_upper = np.percentile(bootstrap_means, 97.5)
        CI_lowers.append(CI_lower)
        CI_uppers.append(CI_upper)
    AUROC_mean = np.array(AUROCs).mean()
    CI_lower_mean = np.array(CI_lowers).mean()
    CI_upper_mean = np.array(CI_uppers).mean()
    return AUROC_mean, CI_lower_mean, CI_upper_mean

def model_prediction(x):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()


# SHAP Explainer
def shap_explainer(model, data_loader, device):
    x_sample, _, _ = next(iter(data_loader))
    x_sample = x_sample.to(device).cpu().numpy()
    model.eval()
    shap_values = []
    explainer = shap.KernelExplainer(model_prediction, x_sample)
    for i, (x, y, mask) in enumerate(data_loader):
        if i * x.size(0) > 100:  # limit data sample size
            break
        x = x.to(device).cpu().numpy()
        shap_values.append(explainer.shap_values(x))
    return np.concatenate(shap_values, axis=0)

# shap_values = shap_explainer(model, valid_loader, device)
# feature_importance = np.abs(shap_values).mean(axis=(0, 2))