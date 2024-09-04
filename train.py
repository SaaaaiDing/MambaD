import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataset import load_data  
from mamba_model import Mamba  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import config  
import random
from thop import profile  

# Set seed for reproducibility
seed = 12345678
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
seq_len = config.seq_len
d_model = config.d_model
state_size = config.state_size
num_classes = config.num_classes
batch_size = config.batch_size
num_epochs = config.num_epochs
learning_rate = config.learning_rate
weight_decay = config.weight_decay
gamma = config.gamma

# File paths
train_path = ''
test_path = ''
save_path = ''

os.makedirs(save_path, exist_ok=True)

# Load data
train_loader, test_loader = load_data(train_path, test_path, batch_size)

# Initialize model
model = Mamba(seq_len, d_model, state_size, num_classes, device).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)

# Calculate Params/M and FLOPs/M
input_example = torch.randn(1, seq_len, d_model).to(device)  # Adjust input shape as per your model
flops, params = profile(model, inputs=(input_example,))
params_m = params / 1e6
flops_m = flops / 1e6
print(f"Params (M): {params_m:.2f}, FLOPs (M): {flops_m:.2f}")

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for features, labels in tqdm(train_loader):
        features, labels = features.to(device), labels.to(device)

        if torch.isnan(features).any() or torch.isnan(labels).any():
            raise ValueError("Input data contains NaN values.")
        if torch.isinf(features).any() or torch.isinf(labels).any():
            raise ValueError("Input data contains infinite values.")

        # Forward pass
        outputs = model(features)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            raise ValueError("Model output contains NaN or infinite values.")

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_precision = precision_score(all_labels, all_predictions, average='macro')
    epoch_recall = recall_score(all_labels, all_predictions, average='macro')
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    epoch_kappa = cohen_kappa_score(all_labels, all_predictions)

    print(f'Loss: {epoch_loss}, Acc: {epoch_acc}, Precision: {epoch_precision}, Recall: {epoch_recall}, F1: {epoch_f1}, Kappa: {epoch_kappa}')

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_kappa

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Calculate statistics
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    avg_loss = total_loss / total
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    kappa = cohen_kappa_score(all_labels, all_predictions)

    print(classification_report(all_labels, all_predictions))
    return avg_loss, accuracy, precision, recall, f1, kappa, all_labels, all_predictions

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.num_classes, yticklabels=config.num_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def main():
    best_test_loss = float('inf')
    best_model_path = None

    log = {
        'epoch': [],
        'lr': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'train_kappa': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'test_kappa': [],
        'params_m': params_m,
        'flops_m': flops_m
    }

    for epoch in range(num_epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1, train_kappa = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_precision, test_recall, test_f1, test_kappa, test_labels, test_preds = evaluate(model, test_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train Kappa: {train_kappa:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
              f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Kappa: {test_kappa:.4f}, '
              f'Params (M): {params_m:.2f}, FLOPs (M): {flops_m:.2f}')

        # Update log
        log['epoch'].append(epoch + 1)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['train_loss'].append(train_loss)
        log['train_accuracy'].append(train_acc)
        log['train_precision'].append(train_precision)
        log['train_recall'].append(train_recall)
        log['train_f1'].append(train_f1)
        log['train_kappa'].append(train_kappa)
        log['test_loss'].append(test_loss)
        log['test_accuracy'].append(test_acc)
        log['test_precision'].append(test_precision)
        log['test_recall'].append(test_recall)
        log['test_f1'].append(test_f1)
        log['test_kappa'].append(test_kappa)

        # Save the best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print('Best model saved!')

        # Save model and log every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            np.save(os.path.join(save_path, f'log_epoch_{epoch+1}.npy'), log)
            print(f'Model and log saved at epoch {epoch+1}.')

        scheduler.step()

    # Save the final log
    np.save(os.path.join(save_path, 'final_log.npy'), log)
    print('Training complete. Final log saved.')

    # Load best model for confusion matrix
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        _, _, _, _, _, _, test_preds = evaluate(model, test_loader, criterion, device)
        plot_confusion_matrix(test_labels, test_preds, save_path)

if __name__ == '__main__':
    main()

