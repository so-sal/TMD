import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class TabularDataset(Dataset):
    def __init__(self, X, Y, mask_prob=0.10):
        self.X = torch.FloatTensor(X.values)
        self.Y = torch.FloatTensor(Y.values)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        # Create a mask for input
        mask = torch.bernoulli(torch.full(x.shape, self.mask_prob))
        return x, y, mask

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attention = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1)
        gated_output = self.gate(x).sigmoid() * torch.matmul(attention, v)
        return gated_output + x  # Residual connection

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.gated_attention = GatedAttention(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        x = x + self.gated_attention(x)
        x = self.layer_norm2(x + self.ffn(x))
        return x

class GATT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # Apply masking: replace masked values with -1
        x_unmasked = self.embedding(x)
        for transformer in self.transformer_blocks:
            x_unmasked = transformer(x_unmasked)
        classified = self.classifier(x_unmasked)        

        if mask is not None:
            x_masked = x.clone()
            x_masked[mask == 1] = -1
            
            # Proceed with the rest of the model
            x_embedded = self.embedding(x_masked)
            for transformer in self.transformer_blocks:
                x_embedded = transformer(x_embedded)
            decoded = self.decoder(x_embedded)
            return decoded, classified
        return

