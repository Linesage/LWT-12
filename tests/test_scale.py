"""Бюджет памяти для больших конфигураций и фактическая проверка выделения."""

import gc

import pytest
import torch

from bench.scale import config_for_bits, memory_budget, vram_gib
from hdc.model import BitParam, HDCBitTransformer


class TestMemoryAccounting:
    """Арифметика бюджета должна совпадать с фактическим выделением."""

    def test_config_hits_target_bits(self):
        cfg = config_for_bits(3e9, 32768, 32, 32768)
        assert cfg.bit_params == pytest.approx(3e9, rel=1e-3)

    def test_budget_matches_allocation(self):
        """Предсказанный объём совпадает с фактическим в пределах 5%."""
        gc.collect()
        torch.cuda.empty_cache()
        cfg = config_for_bits(1e9, 8192, 8, 8192)
        predicted = memory_budget(cfg)
        before = torch.cuda.memory_allocated()
        model = HDCBitTransformer(cfg, seed=0)
        actual = (torch.cuda.memory_allocated() - before) / 1024 ** 3
        del model
        gc.collect()
        torch.cuda.empty_cache()
        assert actual == pytest.approx(predicted["total_gib"], rel=0.05)

    def test_history_off_by_default(self):
        """momentum/flip_rate/age не выделяются без мета-политики.

        Они нужны только как её признаки и стоят `PERSIST_BYTES` = 5 байт на бит:
        на модели 30 Гбит это 140 GiB, то есть больше, чем вся VRAM.
        """
        cfg = config_for_bits(1e8, 4096, 4, 4096)
        model = HDCBitTransformer(cfg, seed=0)
        param = next(iter(model.params.values()))
        assert param.momentum is None and param.flip_rate is None
        assert memory_budget(cfg)["history_gib"] == 0.0

    def test_votes_are_int16(self):
        """Счётчики голосов целые: fp32 давал бы вдвое больший расход."""
        cfg = config_for_bits(1e8, 4096, 4, 4096)
        param = next(iter(HDCBitTransformer(cfg, seed=0).params.values()))
        assert param.signal.dtype == torch.int16
        assert param.margin_sum.dtype == torch.int16
        assert BitParam.STEP_BYTES == 4


class TestScalingLimits:
    """Что реально помещается в 95 GiB.

    | модель  | веса int8 | голоса (все слои) | голоса послойно | итого |
    |---------|-----------|-------------------|-----------------|-------|
    | 3 Гбит  | 2.8 GiB   | 11.2 GiB          | 0.3 GiB         | 14.0  |
    | 10 Гбит | 9.3 GiB   | 37.2 GiB          | 1.2 GiB         | 46.6  |
    | 30 Гбит | 27.9 GiB  | 111.8 GiB         | 3.5 GiB         | 139.7 |

    Проверено фактическим выделением: 3 Гбит занимает 14.0 GiB, 8 Гбит — 37.3
    GiB, 30 Гбит даёт OOM. Целевые 30 Гбит требуют либо послойного подсчёта
    голосов (не реализовано), либо упакованных весов с распаковкой на лету, либо
    нескольких GPU.
    """

    @pytest.mark.parametrize("bits_g,fits", [(3, True), (8, True), (30, False)])
    def test_predicted_fit(self, bits_g, fits):
        cfg = config_for_bits(bits_g * 1e9, 32768, 32, 32768)
        assert memory_budget(cfg)["fits"] == fits

    def test_layerwise_would_fit_30g(self):
        """Послойный подсчёт голосов снял бы ограничение: 139.7 -> 31.4 GiB."""
        cfg = config_for_bits(30e9, 32768, 32, 32768)
        assert not memory_budget(cfg, layerwise_votes=False)["fits"]
        assert memory_budget(cfg, layerwise_votes=True)["fits"]

    @pytest.mark.slow
    def test_actually_allocates_8g(self):
        gc.collect()
        torch.cuda.empty_cache()
        cfg = config_for_bits(8e9, 32768, 32, 32768)
        model = HDCBitTransformer(cfg, seed=0)
        assert model.n_bits == pytest.approx(8e9, rel=1e-3)
        del model
        gc.collect()
        torch.cuda.empty_cache()
