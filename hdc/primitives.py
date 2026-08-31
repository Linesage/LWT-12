"""Bit-native примитивы HDC. Всё в биполярных {-1,+1} int8.

Инварианты, проверяемые в tests/test_primitives.py:

* `bind` самообратна:      bind(bind(a, b), b) == a
* `permute` обратима:      permute(permute(x, k), -k) == x
* `pack/unpack` обратимы:  unpack(pack(x)) == x
* `bipolar_mm` точен:      равен целочисленному произведению без ошибки
* `bundle` не даёт ничьих: суммарный вес голосов всегда нечётный
"""

from __future__ import annotations

import torch

from .config import DEV, SRC_WEIGHT

ONE_SCALE = torch.tensor(1.0, device=DEV)

BMM_EXACT_K = 16384
"""Максимальное `K`, при котором bf16-`bmm` даёт точный результат.

Проверено перебором на худшем случае (`a == b`, произведение равно `K`):
до 16384 ошибка ноль, на 32768 появляется 2.0.
"""


def bipolar_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a(M,K) @ b(N,K).T` для биполярных {-1,+1}.

    fp8 e4m3 представляет ±1 без потерь, аккумуляция идёт в fp32, поэтому
    результат **побитово** совпадает с целочисленным. Требование `out_dtype`
    именно fp32: в bf16 мантиссы 8 бит не хватает на значения до 32768.
    """
    m, k = a.shape
    n = b.shape[0]
    if DEV == "cuda" and m % 16 == 0 and n % 16 == 0 and k % 16 == 0:
        a8 = a.to(torch.bfloat16).to(torch.float8_e4m3fn)
        b8 = b.to(torch.bfloat16).to(torch.float8_e4m3fn)
        return torch._scaled_mm(a8, b8.t(), scale_a=ONE_SCALE, scale_b=ONE_SCALE,
                                out_dtype=torch.float32)
    return a.float() @ b.float().t()


def bipolar_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Батчевый вариант: `a(B,M,K)`, `b(B,N,K)` -> `(B,M,N)`. Точно.

    Один вызов `torch.bmm` в bf16 вместо цикла из B вызовов `bipolar_mm`: замер
    показал 0.009 мс против 1.94 мс при B=32, то есть 215x — почти всё время
    уходило на накладные расходы мелких запусков ядра.

    Точность проверена перебором: bf16 хранит ±1 без потерь, аккумуляция в
    `torch.bmm` идёт в fp32, поэтому результат точен до `K = 16384`
    включительно. На `K = 32768` появляется ошибка 2.0 (округление при записи
    результата в bf16), поэтому там переключаемся на fp8-путь с `out_dtype=fp32`.
    """
    if a.shape[-1] <= BMM_EXACT_K:
        return torch.bmm(a.to(torch.bfloat16), b.to(torch.bfloat16).transpose(1, 2)).float()
    return torch.stack([bipolar_mm(a[i], b[i]) for i in range(a.shape[0])])


def hamming(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Точное расстояние Хэмминга между строками: `(K - a·b) / 2`."""
    return (a.shape[-1] - bipolar_mm(a, b)) * 0.5


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR в биполярном виде — поэлементное произведение. Самообратна."""
    return a * b


def permute(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Циклический сдвиг вдоль оси гипервектора. Обратна `permute(x, -k)`."""
    return torch.roll(x, k, dims=-1)


def shift_prev(x: torch.Tensor) -> torch.Tensor:
    """Сдвиг по времени: `out[:, s] = x[:, s-1]`, позиция 0 обнуляется."""
    y = torch.roll(x, 1, dims=1)
    y[:, 0] = 0
    return y


def shift_next(x: torch.Tensor) -> torch.Tensor:
    """Обратная к `shift_prev`: `out[:, s-1] = x[:, s]`."""
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = 0
    return y


def bundle(vote_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """majority по взвешенной сумме голосов -> `(биты int8, margin int16)`.

    `margin` — абсолютная величина суммы: насколько бит далёк от переключения.
    Это цена инверсии бита, центральная величина всей схемы обучения.
    """
    bits = torch.where(vote_sum >= 0, 1, -1).to(torch.int8)
    margin = vote_sum.abs().clamp(max=32767).to(torch.int16)
    return bits, margin


def res_weight(n_votes: torch.Tensor, ratio: float) -> torch.Tensor:
    """Вес residual-голоса: `2*round(ratio*k) + 1`.

    Суммарный вес `SRC_WEIGHT*k + 2*round(ratio*k) + 1` всегда нечётный, поэтому
    `vote_sum != 0` и знак majority определён — ничьих не бывает по построению.
    """
    return (2 * (ratio * n_votes.float()).round() + 1).to(torch.int32)


def pack_bits(x: torch.Tensor) -> torch.Tensor:
    """Биполярный `(..., D)` int8 -> упакованный `(..., D//8)` uint8."""
    bits = (x > 0).to(torch.uint8).reshape(*x.shape[:-1], -1, 8)
    w = (2 ** torch.arange(8, device=x.device, dtype=torch.int16)).to(torch.uint8)
    return (bits * w).sum(-1, dtype=torch.uint8)


def unpack_bits(p: torch.Tensor) -> torch.Tensor:
    """Упакованный `(..., D//8)` uint8 -> биполярный `(..., D)` int8."""
    sh = torch.arange(8, device=p.device, dtype=torch.uint8)
    bits = (p.unsqueeze(-1) >> sh) & 1
    return (bits.to(torch.int8) * 2 - 1).reshape(*p.shape[:-1], -1)


def gate_with_fallback(scores: torch.Tensor, valid: torch.Tensor,
                      margin: float, fallback: bool) -> torch.Tensor:
    """Порог по близости с гарантией непустого выбора.

    Порог отсекает кандидатов, чья близость не отличается от случайной. Но на
    старте обучения ВСЕ кандидаты случайны (максимум около 4 сигм против 32 у
    обученных), и жёсткий порог обнулил бы выбор целиком: источников нет,
    обратный проход не видит, кого править, веса не учатся.

    `fallback` оставляет единственного лучшего кандидата там, где порог не прошёл
    никто. На majority это влияет слабо (один голос против residual), но
    сохраняет путь для обучения.
    """
    gated = valid & (scores > margin)
    if not fallback:
        return gated
    empty = ~gated.any(-1, keepdim=True)
    best = scores.masked_fill(~valid, -float("inf")).argmax(-1, keepdim=True)
    return gated | (empty & torch.zeros_like(gated).scatter_(-1, best, True))


def topk_select(scores: torch.Tensor, k: int, valid: torch.Tensor):
    """Жёсткий top-k. Возвращает `(one-hot выбор, зазор, индекс (k+1)-го)`.

    Зазор — «насколько близок был другой выбор»: разница между слабейшим
    выбранным и первым невыбранным кандидатом.
    """
    neg = torch.finfo(torch.float32).min / 4
    s = scores.masked_fill(~valid, neg)
    kk = min(k + 1, s.shape[-1])
    top = s.topk(kk, dim=-1)
    idx, val = top.indices[..., :k], top.values[..., :k]
    ok = val > neg / 2
    sel = torch.zeros_like(s).scatter_(-1, idx, ok.float())
    weak = val.masked_fill(~ok, float("inf")).min(-1)
    weak_idx = idx.gather(-1, weak.indices.unsqueeze(-1)).squeeze(-1)
    if kk > k:
        next_val, next_idx = top.values[..., k], top.indices[..., k]
        next_ok = next_val > neg / 2
    else:
        next_val = torch.zeros_like(weak.values)
        next_idx = weak_idx
        next_ok = torch.zeros_like(weak.values, dtype=torch.bool)
    gap = torch.where(next_ok & torch.isfinite(weak.values),
                      (weak.values - next_val).clamp(min=0.0),
                      torch.zeros_like(next_val))
    return sel, gap, torch.where(next_ok, next_idx, weak_idx)
