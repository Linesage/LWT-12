"""Игрушечные задачи. В targets -100 = «здесь ничего не предсказываем»."""

from __future__ import annotations

import functools

import torch

from .config import DEV

IGNORE = -100


@functools.cache
def markov_table(vocab: int, n_succ: int = 3, seed: int = 7) -> torch.Tensor:
    """Фиксированная разреженная цепь Маркова: у токена ровно `n_succ` следующих.

    Энтропия предсказания ограничена `log(n_succ)`, поэтому у модели есть
    измеримый потолок, а не просто шум.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    succ = torch.randint(2, vocab, (vocab, n_succ), device=DEV, generator=g)
    p = torch.zeros(vocab, vocab, device=DEV)
    p.scatter_(1, succ, 1.0 / n_succ)
    return p


def task_batch(kind: str, vocab: int, seq_len: int, batch: int = 8, gen=None):
    """Возвращает `(tokens, targets)`."""
    if kind == "recall":
        n = (seq_len - 1) // 2
        keys = torch.stack([
            torch.randperm(vocab - 2, device=DEV, generator=gen)[:n] + 2
            for _ in range(batch)])
        vals = torch.randint(2, vocab, (batch, n), device=DEV, generator=gen)
        pairs = torch.stack([keys, vals], dim=-1).reshape(batch, -1)
        qi = torch.randint(0, n, (batch,), device=DEV, generator=gen)
        ar = torch.arange(batch, device=DEV)
        tok = torch.cat([pairs, keys[ar, qi][:, None]], dim=1)
        tgt = torch.full_like(tok, IGNORE)
        tgt[:, -1] = vals[ar, qi]
        return tok, tgt
    if kind == "copy":
        h = seq_len // 2
        a = torch.randint(2, vocab, (batch, h), device=DEV, generator=gen)
        tok = torch.cat([a, a], dim=1)
        tgt = torch.full_like(tok, IGNORE)
        tgt[:, h - 1:-1] = a
        return tok, tgt
    if kind == "induction":
        tok = torch.randint(2, vocab, (batch, seq_len), device=DEV, generator=gen)
        pos = torch.randint(0, max(1, seq_len // 2 - 2), (batch,), device=DEV,
                            generator=gen)
        ar = torch.arange(batch, device=DEV)
        a, b = tok[ar, pos], tok[ar, pos + 1]
        tok[:, -1] = a
        tgt = torch.full_like(tok, IGNORE)
        tgt[:, -1] = b
        return tok, tgt
    if kind == "induction_unique":
        # `a` встречается РОВНО один раз до запроса: ответ определён однозначно.
        # В обычном `induction` при повторах `a` есть несколько валидных `b`,
        # поэтому потолок точности там ниже 1.0 по построению задачи.
        tok = torch.stack([
            torch.randperm(vocab - 2, device=DEV, generator=gen)[:seq_len] + 2
            for _ in range(batch)])
        pos = torch.randint(0, max(1, seq_len - 3), (batch,), device=DEV,
                            generator=gen)
        ar = torch.arange(batch, device=DEV)
        a, b = tok[ar, pos], tok[ar, pos + 1]
        tok[:, -1] = a
        tgt = torch.full_like(tok, IGNORE)
        tgt[:, -1] = b
        return tok, tgt
    if kind == "markov":
        p = markov_table(vocab)
        tok = torch.zeros(batch, seq_len, device=DEV, dtype=torch.long)
        tok[:, 0] = torch.randint(2, vocab, (batch,), device=DEV, generator=gen)
        for t in range(1, seq_len):
            tok[:, t] = torch.multinomial(p[tok[:, t - 1]], 1,
                                          generator=gen).squeeze(1)
        tgt = torch.roll(tok, -1, dims=1)
        tgt[:, -1] = IGNORE
        return tok, tgt
    raise ValueError(kind)


def task_accuracy(model, tokens, targets) -> float:
    """Точность на позициях, где target != -100."""
    pred = model.forward(tokens).logits.argmax(-1)
    t = targets.reshape(-1)
    m = t >= 0
    if int(m.sum()) == 0:
        return float("nan")
    return (pred[m] == t[m]).float().mean().item()
