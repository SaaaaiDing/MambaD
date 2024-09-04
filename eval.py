import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import config
from mamba_model import Mamba 
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = config.seq_len
d_model = config.d_model
state_size = config.state_size
num_classes = config.num_classes
batch_size = config.batch_size
num_epochs = config.num_epochs
learning_rate = config.learning_rate
weight_decay = config.weight_decay
gamma = config.gamma

class MDataset(Dataset):
    def __init__(self, data_path, seq_len=, feature_dim=):
        data = np.load(data_path)
        features = data['features']
        scaler = StandardScaler()
        features = scaler.fit_transform(features.reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim)
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

val_path = ''
val_dataset = MDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = Mamba(seq_len, d_model, state_size, num_classes, device).to(device)
model_path = ''
model.load_state_dict(torch.load(model_path))
model.eval()

def validate(model, val_loader, device, save_path):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in val_loader:
            print(f"Features: {features}")
            print(f"Labels: {labels}")
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            print(f"Outputs: {outputs}")
            _, predicted = torch.max(outputs.data, 1)
            print(f"Predicted: {predicted}")
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)
    report = classification_report(all_labels, all_predictions, zero_division=1)

    print("Validation Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(report)

    os.makedirs(save_path, exist_ok=True)
    result_file = os.path.join(save_path, 'validation_results.txt')
    with open(result_file, 'w') as f:
        f.write("Validation Results:\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

eval_save_path = ''
validate(model, val_loader, device, eval_save_path)
