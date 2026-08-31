"""Тесты механизмов стабилизации GF(2)-динамики."""

import pandas as pd
import pytest
import torch

from hdc.config import DEV, HDCConfig
from hdc.model import HDCBitTransformer
from hdc.stabilize import (Checkpoint, TemporalConsensus, TrustRegion,
                           hamming_distance_to)
from hdc.train import TrainConfig, train


@pytest.fixture
def cfg():
    return HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=8,
                     topk_attn=1, vocab=32, seq_len=16, res_ratio_mem=0.75)


class TestTemporalConsensus:
    """Аналог momentum для GF(2): накопление голосов по окну шагов."""

    def test_accumulates_over_window(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        tc = TemporalConsensus(window=3, min_votes=2)
        p = m.params["rv0"]
        for _ in range(3):
            p.signal.zero_()
            p.signal[0, 721] = 1.0
            tc.accumulate(m)
        totals = tc.accumulate(m)
        assert float(totals["rv0"][0, 721]) == 3.0, "окно не сохраняет историю"

    def test_window_forgets_old(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        tc = TemporalConsensus(window=2, min_votes=1)
        p = m.params["rv0"]
        p.signal.zero_()
        p.signal[0, 5] = 1.0
        tc.accumulate(m)
        p.signal.zero_()
        for _ in range(2):
            tc.accumulate(m)
        totals = tc.accumulate(m)
        assert float(totals["rv0"][0, 5]) == 0.0, "старые голоса не забываются"

    def test_min_votes_filters_singletons(self, cfg):
        """Бит с одним голосом за всё окно не проходит порог."""
        m = HDCBitTransformer(cfg, seed=0)
        tc = TemporalConsensus(window=4, min_votes=2)
        for p in m.params.values():
            p.signal.zero_()
        m.params["rv0"].signal[0, 10] = 1.0
        masks, _frac = tc.select(m, budget=0.5)
        assert not bool(masks["rv0"][0, 10]), "одиночный голос прошёл фильтр"

    def test_forget_clears_applied(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        tc = TemporalConsensus(window=4, min_votes=1)
        p = m.params["rv0"]
        p.signal.zero_()
        p.signal[0, 3] = 5.0
        masks, _frac = tc.select(m, budget=0.01)
        assert bool(masks["rv0"][0, 3])
        tc.forget(masks)
        p.signal.zero_()
        totals = tc.accumulate(m)
        assert float(totals["rv0"][0, 3]) == 0.0, "применённый бит остался в истории"


class TestTrustRegion:
    """Радиус шага должен реагировать на валидацию, а не на loss батча."""

    def test_grows_on_improvement(self):
        tr = TrustRegion(budget=1e-2, patience=2)
        start = tr.budget
        for loss in (3.0, 2.9, 2.8, 2.7):
            tr.update(loss)
        assert tr.budget > start, "радиус не растёт при устойчивом улучшении"

    def test_shrinks_on_degradation(self):
        tr = TrustRegion(budget=1e-2)
        tr.update(2.0)
        before = tr.budget
        tr.update(3.0)
        assert tr.budget < before, "радиус не сжимается при деградации"

    def test_plateau_shrinks_not_grows(self):
        """Ключевой случай: на плато радиус должен СЖИМАТЬСЯ.

        Первая версия сравнивала с EMA, и при застое `improved` оставался True —
        радиус разгонялся до максимума, а accept-rate падал до 0.22.
        """
        tr = TrustRegion(budget=1e-2)
        tr.update(2.5)
        for _ in range(6):
            tr.update(2.5)
        assert tr.budget < 1e-2, "плато не сжимает радиус"

    def test_respects_bounds(self):
        tr = TrustRegion(budget=1e-2, min_budget=1e-3, max_budget=2e-2)
        for _ in range(50):
            tr.update(9.0)
        assert tr.budget >= tr.min_budget
        for i in range(50):
            tr.update(1.0 - i * 0.01)
        assert tr.budget <= tr.max_budget


class TestCheckpoint:
    def test_saves_and_restores_exactly(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        ck = Checkpoint(mode="max")
        assert ck.maybe_save(m, 0.5, step=0)
        before = {n: p.sign.clone() for n, p in m.params.items()}
        for p in m.params.values():
            p.sign.mul_(-1)
        ck.restore(m)
        for n, p in m.params.items():
            assert torch.equal(p.sign, before[n]), f"{n}: восстановление неточно"

    def test_keeps_only_best(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        ck = Checkpoint(mode="max")
        assert ck.maybe_save(m, 0.5, 0)
        assert not ck.maybe_save(m, 0.4, 1), "сохранил результат хуже лучшего"
        assert ck.maybe_save(m, 0.7, 2)
        assert ck.best == 0.7 and ck.step == 2

    def test_packed_storage_is_one_bit_per_param(self, cfg):
        m = HDCBitTransformer(cfg, seed=0)
        ck = Checkpoint()
        ck.maybe_save(m, 1.0, 0)
        stored = sum(t.numel() for t in ck._state.values())
        assert stored == m.n_bits // 8, "чекпоинт хранится не упакованным"

    def test_hamming_distance_measures_drift(self, cfg):
        from hdc.primitives import pack_bits
        m = HDCBitTransformer(cfg, seed=0)
        state = {n: pack_bits(p.sign) for n, p in m.params.items()}
        assert hamming_distance_to(m, state) == 0
        m.params["rv0"].sign[0, :7] *= -1
        assert hamming_distance_to(m, state) == 7


class TestStabilizationEffect:
    """Замеренный эффект механизмов. Три сида, induction_unique, 200 шагов.

    | ветка      | пик   | финал | просадка |
    |------------|-------|-------|----------|
    | baseline   | 0.828 | 0.333 | 0.495    |
    | temporal   | 0.724 | 0.286 | 0.438    |
    | trust      | 0.781 | 0.628 | **0.153**|
    | both       | 0.570 | 0.328 | 0.243    |

    Вывод: подтверждается гипотеза, что модель НАХОДИТ хорошую область, но не
    удерживается в ней. Trust region сокращает просадку в 3.2 раза почти без
    потери пика. Temporal consensus в текущем виде только замедляет обучение.
    """

    def test_trust_region_reduces_drop(self, cfg):
        base, _m, _ck = train("induction_unique", cfg,
                              TrainConfig(steps=200, checkpoint=True), seed=0)
        trust, _m2, _ck2 = train("induction_unique", cfg,
                                 TrainConfig(steps=200, checkpoint=True,
                                             trust_region=True), seed=0)
        b, t = pd.DataFrame(base), pd.DataFrame(trust)
        drop_b = b.acc_eval.max() - b.acc_eval.iloc[-1]
        drop_t = t.acc_eval.max() - t.acc_eval.iloc[-1]
        assert drop_t < drop_b, f"trust region не помог: {drop_t:.3f} vs {drop_b:.3f}"

    def test_checkpoint_recovers_best(self, cfg):
        """Чекпоинт обязан вернуть лучшую точку, а не последнюю."""
        rows, model, ck = train("induction_unique", cfg,
                                TrainConfig(steps=120, checkpoint=True,
                                            restore_best=True), seed=0)
        log = pd.DataFrame(rows)
        from hdc.tasks import task_accuracy, task_batch
        val = task_batch("induction_unique", cfg.vocab, cfg.seq_len, 128,
                         torch.Generator(device=DEV).manual_seed(999))
        assert task_accuracy(model, *val) == pytest.approx(ck.best, abs=1e-6)
        assert ck.best >= log.acc_eval.iloc[-1]
