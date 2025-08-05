import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────
# ─────────────── Layer Definitions ──────────────────────────────────
# ────────────────────────────────────────────────────────────────────
class TimeFeatureEmbedding(nn.Module):
    """Embed month, weekday, settlement-period, and day-type indices."""
    def __init__(self):
        super().__init__()
        # define embedding sizes for each calendar feature
        self.emb_month = nn.Embedding(12, 4)  # months → 4-dim
        self.emb_day   = nn.Embedding(7,  3)  # weekdays → 3-dim
        self.emb_sp    = nn.Embedding(50, 6)  # periods → 6-dim
        self.emb_dtype = nn.Embedding(2,  2)  # weekday/weekend → 2-dim

    def forward(self, month_idx, day_idx, sp_idx, dtype_idx):
        # perform each embedding lookup
        e1 = self.emb_month(month_idx)
        e2 = self.emb_day(day_idx)
        e3 = self.emb_sp(sp_idx)
        e4 = self.emb_dtype(dtype_idx)
        # concatenate on last dim → (B, L, sum(embed_dims))
        return torch.cat([e1, e2, e3, e4], dim=-1)


class VariableSelection(nn.Module):
    """Learn feature-wise soft attention weights via a self-projection."""
    def __init__(self, input_dim: int):
        super().__init__()
        # project each feature to a score, same dim as input
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (B, T, D)
        z = self.proj(x)                   # raw scores (B, T, D)
        w = torch.softmax(z, dim=-1)       # weights across features
        return w * x                       # reweighted input


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for variable-length sequences."""
    def __init__(self, input_dim: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # batch_first=True → inputs shaped (B, T, input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            num_layers = num_layers,
            bidirectional=True,
            batch_first=True,
            dropout = dropout if num_layers>1 else 0.0,
        )

    def forward(self, x, hidden=None):
        # returns:
        #   H   (B, T, 2*hidden_size): outputs at all timesteps
        #   (h_n, c_n): final hidden/cell states
        return self.lstm(x, hidden)


class AdditiveAttention(nn.Module):
    """
    Compute additive (Bahdanau) attention over encoder outputs.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        # project encoder & decoder states to common attn_dim
        self.W = nn.Linear(enc_dim, attn_dim, bias=False)
        self.U = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, s_prev, mask=None):
        # 1) encode hidden states
        Hp = self.W(H)                          # (B, T, attn_dim)
        # 2) project decoder state & broadcast
        Sp = self.U(s_prev).unsqueeze(1)        # (B, 1, attn_dim)
        # 3) combine & nonlinearity
        E  = torch.tanh(Hp + Sp)                # (B, T, attn_dim)
        # 4) score → (B, T)
        e  = self.v(E).squeeze(-1)
        # 5) apply mask if provided
        if mask is not None:
            e = e.masked_fill(mask==0, float("-inf"))
        # 6) normalize to weights
        alpha = torch.softmax(e, dim=1)         # (B, T)
        # 7) context vector
        c = (alpha.unsqueeze(-1) * H).sum(dim=1)  # (B, enc_dim)
        return c, alpha


class DualLSTMDecoder(nn.Module):
    """
    Two parallel LSTMCells generating one-step forecasts at each timestep.
    """
    def __init__(self, enc_dim: int, dec_hidden: int):
        super().__init__()
        # LSTMCell for past-context
        self.lstm_h = nn.LSTMCell(enc_dim, dec_hidden)
        # LSTMCell for future-context
        self.lstm_f = nn.LSTMCell(enc_dim, dec_hidden)
        # combine both hidden states to output scalar
        self.ffn    = nn.Linear(2*dec_hidden, 1)

    def forward(self, hist_ctx, fut_ctx):
        B, L, _ = hist_ctx.shape
        # 1) initialize decoder states to zeros
        h_h = hist_ctx.new_zeros(B, self.lstm_h.hidden_size)
        c_h = h_h.clone()
        h_f = h_h.clone()
        c_f = h_h.clone()

        outputs = []
        # 2) for each timestep t in [0…L-1]
        for t in range(L):
            # step both LSTMCells
            h_h, c_h = self.lstm_h(hist_ctx[:, t], (h_h, c_h))
            h_f, c_f = self.lstm_f(fut_ctx[:,  t], (h_f, c_f))
            # combine and project
            comb     = torch.cat([h_h, h_f], dim=-1)
            y_t      = self.ffn(comb).squeeze(-1)
            outputs.append(y_t)

        # 3) stack → (B, L)
        return torch.stack(outputs, dim=1)


# ────────────────────────────────────────────────────────────────────
# ─────────────── Layer Definitions ──────────────────────────────────
# ────────────────────────────────────────────────────────────────────
class BiAttnPointForecaster(nn.Module):
    """
    Full bi-attentional forecaster:
      1) embed time features
      2) select variables
      3) encode with BiLSTM
      4) apply dual additive attention
      5) decode with DualLSTMDecoder
    """
    def __init__(self,
                 num_hist_feats: int,
                 num_fut_feats:  int,
                 time_feat_dim:  int,
                 lstm_hidden:    int,
                 dec_hidden:     int,
                 attn_dim:       int,
                 hist_len:       int,
                 feed_len:       int,
                 fut_len:        int):
        super().__init__()
        self.hist_len = hist_len
        self.feed_len = feed_len
        self.fut_len  = fut_len

        # 1) time feature embedding
        self.time_embed        = TimeFeatureEmbedding()

        # 2) variable selection layers
        self.var_select_past   = VariableSelection(num_hist_feats + time_feat_dim)
        self.var_select_future = VariableSelection(num_fut_feats  + time_feat_dim)

        # 3) BiLSTM encoders
        self.enc_hist = BiLSTMEncoder(num_hist_feats + time_feat_dim, lstm_hidden)
        self.enc_fut  = BiLSTMEncoder(num_fut_feats  + time_feat_dim, lstm_hidden)

        # 4) dual additive attention
        enc_dim = 2 * lstm_hidden
        self.attn_hist = AdditiveAttention(enc_dim, dec_hidden, attn_dim)
        self.attn_fut  = AdditiveAttention(enc_dim, dec_hidden, attn_dim)

        # project final encoder state → decoder init
        self.init_h = nn.Linear(enc_dim, dec_hidden)
        self.init_c = nn.Linear(enc_dim, dec_hidden)

        # 5) decoder
        self.decoder = DualLSTMDecoder(enc_dim, dec_hidden)

    def forward(self, x_hist, x_fut,
                month_idx, weekday_idx, sp_idx, dtype_idx,
                mask_hist=None, mask_fut=None):
        # ─── 1) embed all calendar features ────────────────────────────
        emb = self.time_embed(month_idx, weekday_idx, sp_idx, dtype_idx)
        emb_hist = emb[:, :self.hist_len]
        emb_fut  = emb[:, self.hist_len : self.hist_len+self.feed_len]

        # ─── 2) variable selection ────────────────────────────────────
        xh = torch.cat([x_hist, emb_hist], dim=-1)
        xf = torch.cat([x_fut,  emb_fut ], dim=-1)
        h_sel = self.var_select_past(xh)
        f_sel = self.var_select_future(xf)

        # ─── 3) encode with BiLSTM ────────────────────────────────────
        H_hist, (h_n, c_n) = self.enc_hist(h_sel)
        H_fut,  _          = self.enc_fut(f_sel)

        # ─── 4) init decoder states from encoder final states ────────
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        c_fwd, c_bwd = c_n[-2], c_n[-1]
        h0 = self.init_h(torch.cat([h_fwd, h_bwd], dim=-1))
        c0 = self.init_c(torch.cat([c_fwd, c_bwd], dim=-1))

        # ─── 5) decode with dual attention each step ─────────────────
        s_h, s_c = h0, c0
        s_f, s_cf = h0.clone(), c0.clone()
        outputs = []
        for _ in range(self.fut_len):
            ctx_h, _ = self.attn_hist(H_hist, s_h, mask=mask_hist)
            ctx_f, _ = self.attn_fut( H_fut,  s_f, mask=mask_fut)
            s_h, s_c = self.decoder.lstm_h(ctx_h, (s_h, s_c))
            s_f, s_cf = self.decoder.lstm_f(ctx_f, (s_f, s_cf))
            comb     = torch.cat([s_h, s_f], dim=-1)
            y_t      = self.decoder.ffn(comb).squeeze(-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, fut_len)

if __name__ == "__main__":
    print("This file only defines model classes. Import it instead.")