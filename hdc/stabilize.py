"""Стабилизация GF(2)-динамики: temporal consensus, trust region, чекпоинты.

Диагноз проблемы: апелляция гарантирует невозрастание loss **на текущем батче**,
но это шумный локальный критерий. Разные батчи дают разные направления, и модель
последовательно улучшает каждый батч, разрушая обобщённое решение. Замер: пик
точности 0.859, финал 0.36.

Три независимых механизма против этого:

1. `TemporalConsensus` — аналог momentum для GF(2). Бит переворачивается не
   потому, что один батч потребовал, а если требование подтверждено несколькими
   батчами подряд. Голоса накапливаются, majority берётся по времени.
2. `TrustRegion` — ограничение `d_H(W_t+1, W_t) <= K` с адаптацией `K` по
   валидационному EMA: улучшается — расширяем шаг, деградирует — сжимаем.
3. `Checkpoint` — хранение лучшего состояния по валидации. Отделяет вопрос
   «модель не может найти хорошую область» от «модель не может в ней удержаться».
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import DEV
from .primitives import pack_bits, unpack_bits


class TemporalConsensus:
    """Аналог momentum для GF(2): majority vote по требованиям последних шагов.

    Первая версия фильтровала уже отобранные по бюджету маски и требовала
    `min_votes` подтверждений. Замер показал, почему это не работает: маски
    пересчитываются каждый шаг, один и тот же бит редко попадает в top-k
    несколько раз подряд, и число флипов падало с 4398 до 357 — обучение
    останавливалось.

    Правильная схема: копить **сырые счётчики голосов** по окну, а бюджет
    применять к накопленной сумме. Тогда количество флипов не меняется, а
    меняется их отбор: проходят биты, чьё требование устойчиво во времени.

        score[j] = Σ_{t-window+1..t} votes_t[j]

    Бит с одним сильным требованием проигрывает биту, который требуют стабильно.
    """

    def __init__(self, window: int = 4, min_votes: int = 2):
        if window < 1:
            raise ValueError(f"window={window} должен быть >= 1")
        self.window, self.min_votes = window, min_votes
        self._hist: dict[str, list[torch.Tensor]] = {}

    def accumulate(self, model) -> dict[str, torch.Tensor]:
        """Добавляет текущие голоса в окно и возвращает накопленные суммы."""
        totals = {}
        for name, param in model.params.items():
            hist = self._hist.setdefault(name, [])
            hist.append(param.signal.int())
            if len(hist) > self.window:
                hist.pop(0)
            totals[name] = torch.stack(hist).sum(0)
        return totals

    def select(self, model, budget: float) -> tuple[dict, float]:
        """Отбор `budget` доли бит по накопленным голосам.

        Дополнительно требуется `min_votes` голосов суммарно: бит, замеченный
        один раз за всё окно, скорее шум, чем направление.
        """
        totals = self.accumulate(model)
        masks, n = {}, 0
        for name, param in model.params.items():
            score = totals[name].reshape(-1)
            k = max(1, int(round(budget * param.numel)))
            top = score.topk(k)
            mask = torch.zeros(param.numel, device=DEV, dtype=torch.bool)
            mask[top.indices[top.values >= self.min_votes]] = True
            masks[name] = mask.reshape(param.shape)
            n += int(mask.sum())
        return masks, n / model.n_bits

    def forget(self, masks: dict[str, torch.Tensor]) -> None:
        """Обнуляет историю по применённым битам, чтобы не флипать их снова."""
        for name, mask in masks.items():
            for hist in self._hist.get(name, []):
                hist.masked_fill_(mask, 0.0)


@dataclass
class TrustRegion:
    """Адаптивный радиус шага в Хэмминге: `d_H(W_t+1, W_t) <= K`.

    Радиус растёт при устойчивом улучшении валидации и сжимается при
    деградации — как в классическом trust region, только расстояние измеряется в
    битах.

    Важная деталь, найденная замером: сравнивать надо с **лучшим достигнутым**
    значением, а не с EMA. При сравнении с EMA после застоя `val_loss`
    перестаёт меняться, `improved` остаётся True (равенство не хуже), радиус
    растёт до максимума и шаг превращается в случайный скачок: замер показал
    рост бюджета 0.008 -> 0.05 при accept-rate, упавшем до 0.22.
    """

    budget: float = 8e-3
    min_budget: float = 5e-4
    max_budget: float = 2e-2
    grow: float = 1.2
    shrink: float = 0.6
    patience: int = 2
    _best: float | None = field(default=None, repr=False)
    _good: int = field(default=0, repr=False)

    def update(self, val_loss: float) -> dict:
        """Принимает валидационный loss, возвращает диагностику решения."""
        improved = self._best is None or val_loss < self._best - 1e-6
        if improved:
            self._best = val_loss
            self._good += 1
            if self._good >= self.patience:
                self.budget = min(self.max_budget, self.budget * self.grow)
                self._good = 0
        else:
            self._good = 0
            self.budget = max(self.min_budget, self.budget * self.shrink)
        return dict(budget=self.budget, val_best=self._best,
                    improved=float(improved))


class Checkpoint:
    """Лучшее состояние по валидации: упакованные биты + метрика.

    Хранение в упакованном виде — 1 бит на параметр, поэтому чекпоинт стоит
    столько же, сколько сама модель (для целевой конфигурации 144 MiB).
    """

    def __init__(self, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(mode)
        self.mode = mode
        self.best: float | None = None
        self.step: int = -1
        self._state: dict[str, torch.Tensor] = {}

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def maybe_save(self, model, value: float, step: int) -> bool:
        if not self._better(value):
            return False
        self.best, self.step = value, step
        self._state = {n: pack_bits(p.sign) for n, p in model.params.items()}
        return True

    def restore(self, model) -> None:
        if not self._state:
            raise RuntimeError("чекпоинт пуст")
        for name, packed in self._state.items():
            model.params[name].sign.copy_(unpack_bits(packed))

    @property
    def saved(self) -> bool:
        return bool(self._state)


@torch.no_grad()
def hamming_distance_to(model, packed_state: dict[str, torch.Tensor]) -> int:
    """Расстояние Хэмминга от текущих весов до сохранённого состояния."""
    total = 0
    for name, packed in packed_state.items():
        total += int((model.params[name].sign != unpack_bits(packed)).sum())
    return total
