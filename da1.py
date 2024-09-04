import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

# Define Time2Vec and related classes
def t2v(tau, f, out_features, w, b, w0, b0):
    v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    v2 = v2.expand_as(v1)
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hidden_dim)

    def forward(self, x):
        x = self.l1(x)
        return x

# Define the custom dataset class
class MDataset(Dataset):
    def __init__(self, data_path, seq_len=101, feature_dim=22852, hidden_dim=16):
        data = np.load(data_path)
        self.features = torch.tensor(data['features'], dtype=torch.float32).view(-1, seq_len, feature_dim)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Initialize Time2Vec
        self.time2vec = Time2Vec(activation="sin", hidden_dim=self.hidden_dim)
        
        # Create time step information
        self.time_inputs = torch.arange(self.seq_len).repeat(self.features.shape[0], 1).float().unsqueeze(-1)
        
        # Compute time embeddings
        time_embedded = self.time2vec(self.time_inputs)
        print(f"Time embeddings shape: {time_embedded.shape}")
        
        # Ensure the time embedding dimensions are compatible with features
        #expanded_time_embedded = time_embedded.unsqueeze(2).repeat(1, 1, self.feature_dim // (self.hidden_dim * 2), 1).view(self.features.shape[0], self.seq_len, -1)
        expanded_time_embedded = time_embedded.unsqueeze(2).expand(-1, -1, self.feature_dim // self.hidden_dim, -1).contiguous().view(self.features.shape[0], self.seq_len, -1)
        print(f"Expanded time embeddings shape: {expanded_time_embedded.shape}")
        
        # Combine time features with original features
        self.features_time_embedded = torch.cat([self.features, expanded_time_embedded], dim=2)
        print(f"Combined features shape: {self.features_time_embedded.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features_time_embedded[idx], self.labels[idx]

def load_data(train_path, test_path, batch_size=16):
    train_dataset = MDataset(train_path)
    test_dataset = MDataset(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_path = '/storage/home/sqd5856/Desktop/default/mamba/mamba-130m-hf/0631/train/train.npz'
    test_path = '/storage/home/sqd5856/Desktop/default/mamba/mamba-130m-hf/0631/test/test.npz'
    
    train_loader, test_loader = load_data(train_path, test_path)

    def print_sample_data(loader, num_samples=1):
        for i, (features, labels) in enumerate(loader):
            if i >= num_samples:
                break
            print(f"Sample {i} - Features shape: {features.shape}, Labels shape: {labels.shape}")
            print(f"Sample {i} - Features: {features[0]}")
            print(f"Sample {i} - Labels: {labels[0]}")

    print("Training Data Samples:")
    print_sample_data(train_loader)

    print("Test Data Samples:")
    print_sample_data(test_loader)

