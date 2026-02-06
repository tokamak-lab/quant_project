import warnings

import torch
import torch.nn as nn


class LitPatcherConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int = 128,
        pw: int = 4,
        ph: int = 20,
        conv_channels: int = 32,
        kernel_size: tuple[int, int] = (2, 3),
        padding: tuple[int, int] = (0, 1),
        use_internal_pos: bool = True,
        use_abs_pos: bool = True,
        max_blocks: int = 256,
    ):
        super().__init__()
        self.pw = pw
        self.ph = ph
        self.use_internal_pos = use_internal_pos
        self.use_abs_pos = use_abs_pos

        kt, kh = kernel_size
        pt, ph_pad = padding
        time_out = pw + 2 * pt - kt + 1
        depth_out = ph + 2 * ph_pad - kh + 1

        self.conv = nn.Conv2d(c_in, conv_channels, kernel_size=kernel_size, padding=padding)
        self.proj = nn.Linear(conv_channels * time_out * depth_out, d_model)

        self.time_emb = nn.Embedding(max_blocks, d_model)
        self.side_emb = nn.Embedding(2, d_model)

        if use_internal_pos:
            self.level_emb = nn.Embedding(ph, c_in)
            self.tick_emb = nn.Embedding(pw, c_in)
        else:
            self.level_emb = None
            self.tick_emb = None

        if use_abs_pos:
            self.abs_pos = nn.Embedding(2 * max_blocks, d_model)
        else:
            self.abs_pos = None

    def _process_side(self, side_blocks: torch.Tensor, side_id: int) -> torch.Tensor:
        b, blk, pw, ph, c = side_blocks.shape
        if self.use_internal_pos:
            lvl_idx = torch.arange(ph, device=side_blocks.device)
            tick_idx = torch.arange(pw, device=side_blocks.device)
            lvl_pos = self.level_emb(lvl_idx).view(1, 1, 1, ph, c)
            tick_pos = self.tick_emb(tick_idx).view(1, 1, pw, 1, c)
            side_blocks = side_blocks + lvl_pos + tick_pos

        side_blocks = side_blocks.reshape(b * blk, pw, ph, c).permute(0, 3, 1, 2)
        conv_out = self.conv(side_blocks)
        flat = conv_out.flatten(1)
        proj = self.proj(flat).view(b, blk, -1)

        time_pos = self.time_emb(torch.arange(blk, device=side_blocks.device)).unsqueeze(0)
        proj = proj + time_pos
        proj = proj + self.side_emb(torch.tensor([side_id], device=side_blocks.device))
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, c = x.shape
        if h != 2 * self.ph:
            raise ValueError("H must be 2*ph (ask + bid)")
        if t % self.pw != 0:
            raise ValueError("T must be divisible by pw")

        blocks_per_seq = t // self.pw
        blocks = x.view(b, blocks_per_seq, self.pw, h, c)
        ask_blocks = blocks[:, :, :, : self.ph, :]
        bid_blocks = blocks[:, :, :, self.ph :, :]

        ask_tokens = self._process_side(ask_blocks, side_id=0)
        bid_tokens = self._process_side(bid_blocks, side_id=1)
        tokens = torch.cat([ask_tokens, bid_tokens], dim=1)

        if self.abs_pos is not None:
            if tokens.size(1) > self.abs_pos.num_embeddings:
                raise ValueError("Number of tokens exceeds abs_pos size")
            pos_idx = torch.arange(tokens.size(1), device=tokens.device)
            tokens = tokens + self.abs_pos(pos_idx)

        return tokens


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.query(tokens).mean(dim=-1), dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bs,bsd->bd", attn, tokens)


class MLPHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OvRHeadSeparate(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [head(x) for head in self.heads]
        return torch.cat(logits, dim=1)


class LitTransModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int = 128,
        pw: int = 4,
        ph: int = 20,
        conv_channels: int = 32,
        pooling: str = "attn",
        num_classes: int = 3,
        n_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_ovr: bool = False,
    ):
        super().__init__()
        warnings.filterwarnings(
            "ignore",
            message=(
                "enable_nested_tensor is True, but self.use_nested_tensor is False"
            ),
            category=UserWarning,
            module="torch.nn.modules.transformer",
        )
        self.use_ovr = use_ovr
        self.patcher = LitPatcherConv(
            c_in=c_in,
            d_model=d_model,
            pw=pw,
            ph=ph,
            conv_channels=conv_channels,
            kernel_size=(2, 3),
            padding=(0, 1),
            use_internal_pos=True,
            use_abs_pos=True,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pooling = pooling
        self.attn_pool = AttentionPool(d_model, dropout=dropout) if pooling == "attn" else None
        self.head = (
            OvRHeadSeparate(d_model, hidden=d_model, num_classes=num_classes, dropout=dropout)
            if use_ovr
            else MLPHead(d_model, hidden=d_model, num_classes=num_classes, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patcher(x)
        tokens = self.encoder(tokens)
        pooled = self.attn_pool(tokens) if self.pooling == "attn" else tokens.mean(dim=1)
        return self.head(pooled)


class LitLSTMModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int = 128,
        pw: int = 4,
        ph: int = 20,
        conv_channels: int = 32,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_ovr: bool = False,
    ):
        super().__init__()
        self.use_ovr = use_ovr
        self.patcher = LitPatcherConv(
            c_in=c_in,
            d_model=d_model,
            pw=pw,
            ph=ph,
            conv_channels=conv_channels,
            kernel_size=(2, 3),
            padding=(0, 1),
            use_internal_pos=True,
            use_abs_pos=True,
        )
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.pool = AttentionPool(d_model, dropout=dropout)
        self.head = (
            OvRHeadSeparate(d_model, hidden=d_model, num_classes=num_classes, dropout=dropout)
            if use_ovr
            else MLPHead(d_model, hidden=d_model, num_classes=num_classes, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patcher(x)
        out, _ = self.lstm(tokens)
        pooled = self.pool(out)
        return self.head(pooled)
