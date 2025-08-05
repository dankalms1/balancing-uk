import torch
from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    def __init__(self, X, y, seq_len, horizon=1):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return self.X.shape[0] - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x_seq    = self.X[idx : idx + self.seq_len]
        target_i = idx + self.seq_len - 1 + self.horizon
        y_target = self.y[target_i]
        return x_seq, y_target


class MultiFeedDataset(Dataset):
    """
    Sliding-window dataset producing:
      x_h   (history covariates):   (seq_len, n_hist_feats)
      x_f   (known-future covars):  (feed_len, n_fut_feats)
      y_t   (targets):              (fut_len,)
      mi, wi, si, di (calendar idx): (seq_len + feed_len,)
    """
    def __init__(
        self,
        hist: torch.Tensor,        # (T, n_hist_feats)
        full_fut: torch.Tensor,    # (T, feed_len, n_fut_feats)
        y: torch.Tensor,           # (T,)
        month_idx: torch.Tensor,   # (T,)
        weekday_idx: torch.Tensor, # (T,)
        sp_idx: torch.Tensor,      # (T,)
        dtype_idx: torch.Tensor,   # (T,)
        seq_len: int,
        feed_len: int,
        fut_len: int,
    ):
        assert fut_len <= feed_len, "fut_len must be ≤ feed_len"
        self.hist        = hist.float()
        self.fut         = full_fut.float()
        self.y           = y.float()
        self.month_idx   = month_idx.long()
        self.weekday_idx = weekday_idx.long()
        self.sp_idx      = sp_idx.long()
        self.dtype_idx   = dtype_idx.long()
        self.seq_len     = seq_len
        self.feed_len    = feed_len
        self.fut_len     = fut_len

    def __len__(self) -> int:
        # only allow windows where history, future covariates, and targets are all available
        # i.e., idx + seq_len + feed_len <= T, and idx + seq_len + fut_len <= T
        # since fut_len ≤ feed_len, we bound by feed_len:
        return self.hist.size(0) - self.seq_len - self.feed_len + 1

    def __getitem__(self, idx: int):
        # 1) history window
        x_h = self.hist[idx : idx + self.seq_len]                        # (seq_len, n_hist_feats)

        # 2) known-future covariates
        anchor = idx + self.seq_len - 1
        x_f    = self.fut[anchor, : self.feed_len]                       # (feed_len, n_fut_feats)

        # 3) targets for next fut_len steps
        start_y = idx + self.seq_len
        y_t     = self.y[start_y : start_y + self.fut_len]              # (fut_len,)

        # 4) calendar indices for seq_len + feed_len
        ci_start = idx
        ci_end   = idx + self.seq_len + self.feed_len
        mi = self.month_idx[ci_start : ci_end]                           # (seq_len+feed_len,)
        wi = self.weekday_idx[ci_start : ci_end]
        si = self.sp_idx[ci_start : ci_end]
        di = self.dtype_idx[ci_start : ci_end]

        return x_h, x_f, y_t, mi, wi, si, di