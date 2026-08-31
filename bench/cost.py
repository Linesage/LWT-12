"""Замер стоимости шага обучения: бинарная модель против float-трансформера.

Сравниваются сопоставимые формы (одинаковые `d_model`, словарь, длина, батч).
Измеряется время шага, пиковая память и объём весов + состояния оптимизатора.
"""

from __future__ import annotations

import time

import torch

from bench.baseline_transformer import TinyTransformer, train_step
from hdc.config import DEV, HDCConfig
from hdc.model import HDCBitTransformer
from hdc.tasks import task_batch
from hdc.train import TrainConfig
from hdc.train import train_step as bit_train_step


def timed(fn, iters: int = 20, warmup: int = 3) -> float:
    """Среднее время вызова в миллисекундах."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def measure(vocab=32, d_model=1024, seq_len=16, batch=32, n_layers=1,
            iters=20, dtype=torch.float32) -> dict:
    """Один замер на заданной форме. Возвращает словарь метрик."""
    torch.manual_seed(0)
    gen = torch.Generator(device=DEV).manual_seed(1)
    tok, tgt = task_batch("induction_unique", vocab, seq_len, batch, gen)

    # --- float-трансформер ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ft = TinyTransformer(vocab, d_model, n_layers=n_layers, seq_len=seq_len,
                         dtype=dtype).to(DEV)
    opt = torch.optim.AdamW(ft.parameters(), lr=3e-4)
    train_step(ft, opt, tok, tgt)          # инициализация состояния Adam
    ms_float = timed(lambda: train_step(ft, opt, tok, tgt), iters)
    peak_float = torch.cuda.max_memory_allocated() / 1024 ** 2
    n_float = ft.n_params
    del ft, opt

    # --- бинарная модель ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = HDCConfig(d_hidden=d_model, n_layers=n_layers, n_slots=256,
                    topk_slots=8, topk_attn=1, vocab=vocab, seq_len=seq_len,
                    res_ratio_mem=0.75)
    bt = HDCBitTransformer(cfg, seed=0)
    bit_train_step(bt, tok, tgt)
    ms_bit = timed(lambda: bit_train_step(bt, tok, tgt), iters)
    peak_bit = torch.cuda.max_memory_allocated() / 1024 ** 2
    n_bit = bt.n_bits
    del bt
    torch.cuda.empty_cache()

    return dict(
        d_model=d_model, seq_len=seq_len, batch=batch, n_layers=n_layers,
        float_params=n_float, bit_params=n_bit,
        float_weight_mib=n_float * dtype.itemsize / 1024 ** 2,
        float_optim_mib=n_float * 8 / 1024 ** 2,      # 2 момента fp32
        bit_weight_mib=n_bit / 8 / 1024 ** 2,
        bit_state_mib=n_bit * 5 / 1024 ** 2,          # momentum/flip_rate/age
        ms_float=ms_float, ms_bit=ms_bit,
        peak_float_mib=peak_float, peak_bit_mib=peak_bit,
    )
