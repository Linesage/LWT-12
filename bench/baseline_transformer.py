"""Плотный float-трансформер той же формы — база для сравнения стоимости.

Сознательно минимальный: один слой self-attention + FFN, тот же `d_model`,
словарь и длина. Цель — не побить его по качеству, а измерить стоимость шага и
памяти на сопоставимой форме.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    """Стандартный causal-трансформер: обучаемые эмбеддинги, MHA, FFN, голова."""

    def __init__(self, vocab: int, d_model: int, n_layers: int = 1,
                 n_heads: int = 8, d_ff_mult: int = 4, seq_len: int = 16,
                 dtype=torch.float32):
        super().__init__()
        self.d_model, self.seq_len = d_model, seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff_mult * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        nn.init.normal_(self.pos, std=0.02)
        self.to(dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        x = self.embed(tokens) * math.sqrt(self.d_model) + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, device=tokens.device, dtype=torch.bool),
                          diagonal=1)
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x)).reshape(-1, self.head.out_features)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_step(model, opt, tokens, targets) -> float:
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(tokens).float(), targets.reshape(-1),
                           ignore_index=-100)
    loss.backward()
    opt.step()
    return loss.detach().item()
