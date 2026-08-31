"""Бюджет памяти для больших конфигураций.

Считает, что помещается в VRAM при разных схемах хранения. Числа проверяются
фактическим выделением тензоров, а не только арифметикой.
"""

from __future__ import annotations

import torch

from hdc.config import HDCConfig
from hdc.model import BitParam

GIB = 1024 ** 3


def vram_gib() -> float:
    return torch.cuda.get_device_properties(0).total_memory / GIB


def memory_budget(cfg: HDCConfig, track_history: bool = False,
                  layerwise_votes: bool = False) -> dict:
    """Оценка памяти обучения без запуска модели.

    * `sign` — мастер-копия весов в int8 (1 байт на бит). Упаковка в uint8 дала
      бы 1 бит, но тогда каждое обращение требует распаковки.
    * `signal`/`margin_sum` — аккумуляторы шага, int16 каждый.
    * `momentum`/`flip_rate`/`age` — только для мета-политики.
    * `layerwise_votes` — считать голоса по одному слою за раз: аккумуляторы
      нужны лишь для того слоя, который сейчас обрабатывается.
    """
    bits = cfg.bit_params
    weights = bits / GIB                       # int8
    packed = bits / 8 / GIB
    votes_full = bits * BitParam.STEP_BYTES / GIB
    votes = votes_full / max(cfg.n_layers, 1) if layerwise_votes else votes_full
    history = bits * BitParam.PERSIST_BYTES / GIB if track_history else 0.0
    total = weights + votes + history
    return dict(bits=bits, bits_g=bits / 1e9, weights_gib=weights,
                packed_gib=packed, votes_gib=votes, history_gib=history,
                total_gib=total, fits=total < 0.85 * vram_gib())


def config_for_bits(target_bits: float, d_hidden: int, n_layers: int,
                    vocab: int, seq_len: int = 2048) -> HDCConfig:
    """Подбирает `n_slots` так, чтобы получить заданное число бит-параметров.

    `bit_params = L*(2*R*H + 3H) + V*H`, отсюда `R` выражается напрямую.
    """
    slots = int((((target_bits - vocab * d_hidden) / n_layers)
                 - 3 * d_hidden) / (2 * d_hidden))
    if slots < 1:
        raise ValueError("целевое число бит слишком мало для такой формы")
    return HDCConfig(d_hidden=d_hidden, n_layers=n_layers, n_slots=slots,
                     vocab=vocab, seq_len=seq_len)
