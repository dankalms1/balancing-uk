import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────
# ─────────────── Layer Definitions ──────────────────────────────────
# ────────────────────────────────────────────────────────────────────
class SeasonalAttn(nn.Module):
    """
    Seasonal attention mechanism that applies a learnable attention weight
    to each time step in the input sequence.
    """
    def __init__(self, seq_len=48):
        super().__init__()
        self.attn = nn.Parameter(torch.empty(seq_len))
        nn.init.uniform_(self.attn, -0.01, 0.01)

    def forward(self, x):
        return x * self.attn.view(1, -1, 1) # (B,N,F)


class BiLSTM(nn.Module):
    def __init__(self, num_feats, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_feats,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        out, (_ , _) = self.lstm(x)
        return out


class AttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        e = self.v(torch.tanh(self.W_h(x)))   # (B, N, 1)
        alpha = torch.softmax(e, dim=1)       # (B, N, 1)
        context = torch.sum(alpha * x, dim=1) # (B, 2H)
        return context

# ────────────────────────────────────────────────────────────────────
# ─────────────── Model Definitions ──────────────────────────────────
# ────────────────────────────────────────────────────────────────────
class SA_BiLSTM(nn.Module):
    def __init__(self, num_feats, seq_len=48, hidden_size=64, num_layers=2):
        super().__init__()
        self.seasonal = SeasonalAttn(seq_len)
        self.lstm = BiLSTM(num_feats, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.seasonal(x)       # (B, N, F)
        out = self.lstm(x)         # (B, N, 2H)
        last = out[:, -1, :]       # (B, 2H)
        return self.fc(last).squeeze(-1)  # (B,)

if __name__ == "__main__":
    print("This file only defines model classes. Import it instead.")