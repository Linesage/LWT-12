"""Регрессионные тесты обучения: ловят возврат найденных багов."""

import pandas as pd
import pytest
import torch

from hdc.config import HDCConfig
from hdc.train import TrainConfig, train


@pytest.fixture(scope="module")
def solved():
    """Рабочая конфигурация: k_attn=1 (без разбавления), консенсус слотов памяти."""
    return HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=8,
                     topk_attn=1, vocab=32, seq_len=16, res_ratio_mem=0.75)


def test_learns_induction(solved):
    """Обучение обязано уйти далеко от случайной угадайки (1/32 = 0.031).

    Порог 0.4 выбран с запасом: замеры дают 0.64-0.88 на трёх сидах. Падение
    ниже означает возврат одного из исправленных багов (знак голосов роли,
    разбавление attention, затирание состояния памятью).
    """
    rows, _model, _ck = train("induction_unique", solved,
                              TrainConfig(steps=300, budget=8e-3), seed=0)
    log = pd.DataFrame(rows)
    assert log.acc_eval.max() > 0.5, f"обучение сломано: {log.acc_eval.max():.3f}"
    assert log.loss.tail(20).mean() < log.loss.iloc[0], "loss не убывает"


@pytest.mark.parametrize("kind,seq_len,floor", [
    ("induction_unique", 16, 0.5),
    ("recall", 17, 0.3),
    ("copy", 16, 0.25),
])
def test_learns_all_retrieval_tasks(kind, seq_len, floor):
    """Все задачи на извлечение обучаются заметно выше случайной угадайки (1/32).

    Пороги взяты с запасом от замеров: induction 0.86, recall 0.66, copy 0.46
    в среднем по двум сидам (потолки архитектуры 1.00 / 0.84 / 0.77).
    """
    cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=8,
                    topk_attn=1, vocab=32, seq_len=seq_len, res_ratio_mem=0.75)
    rows, _model, _ck = train(kind, cfg, TrainConfig(steps=300, budget=8e-3), seed=0)
    log = pd.DataFrame(rows)
    assert log.acc_eval.max() > floor, f"{kind}: {log.acc_eval.max():.3f} < {floor}"


def test_accuracy_is_unstable_without_trust_region():
    """Без trust region точность сильно проседает от пика к финалу.

    Причина: апелляция гарантирует невозрастание loss на ТЕКУЩЕМ батче, но это
    шумный локальный критерий — разные батчи задают разные направления. Замер по
    трём сидам: пик 0.828, финал 0.333. С trust region просадка падает до 0.153.
    """
    cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=8,
                    topk_attn=1, vocab=32, seq_len=16, res_ratio_mem=0.75)
    rows, _model, _ck = train("induction_unique", cfg,
                              TrainConfig(steps=300, budget=8e-3), seed=0)
    log = pd.DataFrame(rows)
    drop = log.acc_eval.max() - log.acc_eval.iloc[-1]
    assert drop > 0.1, "нестабильность исправлена — обновить тест и train()"


def test_loss_never_increases_after_appeal(solved):
    """Апелляция гарантирует монотонность: принятый шаг не может ухудшить loss."""
    rows, _model, _ck = train("induction_unique", solved,
                              TrainConfig(steps=60), seed=0)
    for rec in rows:
        if rec["accepted"] > 0:
            assert rec["loss_after"] <= rec["loss"] + 1e-9


def test_diluted_attention_fails(solved):
    """Контрольный тест: с k_attn=4 та же схема НЕ учится.

    Держит в тестах знание о том, что топ-k в attention разбавляет сигнал —
    иначе этот параметр легко «оптимизировать» обратно.
    """
    diluted = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=1,
                        topk_attn=4, vocab=32, seq_len=16)
    rows, _model, _ck = train("induction_unique", diluted,
                              TrainConfig(steps=200, budget=8e-3), seed=0)
    log = pd.DataFrame(rows)
    assert log.acc_eval.max() < 0.4, "тест устарел: разбавление больше не мешает"


def test_no_float_weights_after_training(solved):
    rows, model, _ck = train("induction_unique", solved,
                             TrainConfig(steps=50), seed=0)
    for name, p in model.params.items():
        assert p.sign.dtype == torch.int8, name
        assert set(p.sign.unique().tolist()) <= {-1, 1}, name
