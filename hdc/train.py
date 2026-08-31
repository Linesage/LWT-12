"""Обучающий цикл: точный GF(2)-backward + стабилизация динамики.

Ни autograd, ни мета-оптимизатора. Апелляция по батчу — необходимое, но
недостаточное условие: она гарантирует невозрастание loss на текущем батче, а
разные батчи дают разные направления, и модель успешно улучшает каждый батч,
разрушая обобщённое решение. Поэтому решение о флипе принимается по трём
критериям, а не одному:

* согласованность во времени (`TemporalConsensus`) — бит меняется, только если
  несколько батчей подряд требуют одного и того же;
* адаптивный радиус (`TrustRegion`) — шаг в Хэмминге сжимается при деградации
  валидации и расширяется при устойчивом улучшении;
* чекпоинт по валидации (`Checkpoint`) — итоговая модель берётся из лучшей
  точки, а не из последнего шага.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import DEV, HDCConfig
from .errors import apply_flips, consensus_flip_masks, exact_backward
from .model import HDCBitTransformer
from .stabilize import Checkpoint, TemporalConsensus, TrustRegion
from .tasks import task_accuracy, task_batch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.float(), targets.reshape(-1),
                                 ignore_index=-100))


@dataclass
class TrainConfig:
    """Гиперпараметры обучения. Все три механизма стабилизации отключаемы."""

    steps: int = 300
    batch: int = 32
    budget: float = 8e-3
    iters: int = 8
    gates: bool = True
    eval_every: int = 10
    eval_batch: int = 128
    # стабилизация
    temporal_window: int = 0      # 0 == выключено
    temporal_votes: int = 3
    trust_region: bool = False
    checkpoint: bool = False
    restore_best: bool = False


@torch.no_grad()
def train_step(model, tokens, targets, budget: float = 8e-3, iters: int = 8,
               gates: bool = True) -> dict:
    """Один шаг обучения без стабилизации — минимальная единица для замеров."""
    loss, stat = collect_votes(model, tokens, targets, iters, gates)
    masks, frac = consensus_flip_masks(model, budget)
    apply_flips(model, masks)
    after = cross_entropy(model.forward(tokens).logits, targets)
    accepted = after < loss
    if not accepted:
        apply_flips(model, masks)
    return dict(loss=loss, loss_after=min(loss, after), delta=loss - after,
                accepted=float(accepted), flip_frac=frac, **stat)


@torch.no_grad()
def collect_votes(model, tokens, targets, iters: int = 8,
                  gates: bool = True) -> tuple[float, dict]:
    """forward + точный обратный проход. Возвращает `(loss, статистика)`."""
    trace = model.forward(tokens)
    loss = cross_entropy(trace.logits, targets)
    return loss, exact_backward(model, trace, targets, iters, gates)


@torch.no_grad()
def train(kind: str, cfg: HDCConfig | None = None, tcfg: TrainConfig | None = None,
          seed: int = 0, progress=None):
    """Полный прогон. Возвращает `(лог шагов, модель, чекпоинт)`."""
    tcfg = tcfg or TrainConfig()
    cfg = cfg or HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=8,
                           topk_attn=1, vocab=32, seq_len=16, res_ratio_mem=0.75)
    torch.manual_seed(seed)
    model = HDCBitTransformer(cfg, seed=seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)

    val_tok, val_tgt = task_batch(kind, cfg.vocab, cfg.seq_len, tcfg.eval_batch,
                                  torch.Generator(device=DEV).manual_seed(999))
    consensus = (TemporalConsensus(tcfg.temporal_window, tcfg.temporal_votes)
                 if tcfg.temporal_window > 0 else None)
    region = TrustRegion(budget=tcfg.budget) if tcfg.trust_region else None
    ckpt = Checkpoint(mode="max") if tcfg.checkpoint else None

    rows, acc, val_loss = [], float("nan"), float("nan")
    for step in range(tcfg.steps):
        tok, tgt = task_batch(kind, cfg.vocab, cfg.seq_len, tcfg.batch, gen)
        l0, stat = collect_votes(model, tok, tgt, tcfg.iters, tcfg.gates)

        budget = region.budget if region is not None else tcfg.budget
        if consensus is not None:
            masks, frac = consensus.select(model, budget)
        else:
            masks, frac = consensus_flip_masks(model, budget)

        apply_flips(model, masks)
        l1 = cross_entropy(model.forward(tok).logits, tgt)
        accepted = l1 < l0
        if accepted:
            if consensus is not None:
                consensus.forget(masks)
        else:
            apply_flips(model, masks)

        n_flip = sum(int(v.sum()) for v in masks.values())
        rec = dict(step=step, loss=l0, loss_after=min(l0, l1), delta=l0 - l1,
                   accepted=float(accepted), flip_frac=frac,
                   n_flip=n_flip, budget=budget, task=kind, **stat)

        if step % tcfg.eval_every == 0 or step == tcfg.steps - 1:
            acc = task_accuracy(model, val_tok, val_tgt)
            val_loss = cross_entropy(model.forward(val_tok).logits, val_tgt)
            if region is not None:
                rec.update(region.update(val_loss))
            if ckpt is not None:
                rec["saved"] = float(ckpt.maybe_save(model, acc, step))
        rec.update(acc_eval=acc, val_loss=val_loss)
        rows.append(rec)
        if progress is not None:
            progress(rec)

    if tcfg.restore_best and ckpt is not None and ckpt.saved:
        ckpt.restore(model)
    return rows, model, ckpt
