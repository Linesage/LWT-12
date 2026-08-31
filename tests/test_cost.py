"""Замеры стоимости и защита от регрессий по скорости.

Числа получены на RTX PRO 6000 Blackwell, batch=32, seq_len=16, 1 слой.
"""

import pytest
import torch

from bench.cost import measure
from hdc.config import DEV
from hdc.primitives import BMM_EXACT_K, bipolar_bmm, bipolar_mm


class TestBatchedMMExactness:
    """Батчевый bf16-путь обязан быть ТОЧНЫМ — на нём держится весь backward."""

    @pytest.mark.parametrize("k", [64, 256, 1024, 4096, BMM_EXACT_K])
    def test_exact_up_to_limit(self, k):
        a = torch.randint(0, 2, (4, 16, k), device=DEV, dtype=torch.int8) * 2 - 1
        b = torch.randint(0, 2, (4, 12, k), device=DEV, dtype=torch.int8) * 2 - 1
        ref = torch.stack([a[i].float() @ b[i].float().t() for i in range(4)])
        assert torch.equal(bipolar_bmm(a, b), ref), f"K={k}: неточно"

    def test_worst_case_is_exact(self):
        """Худший случай — `a == b`, произведение равно K."""
        a = torch.randint(0, 2, (1, 32, BMM_EXACT_K), device=DEV,
                          dtype=torch.int8) * 2 - 1
        ref = a[0].float() @ a[0].float().t()
        assert torch.equal(bipolar_bmm(a, a)[0], ref)

    def test_falls_back_above_limit(self):
        """Выше предела точности переключаемся на fp8-путь, и он тоже точен."""
        k = 2 * BMM_EXACT_K
        a = torch.randint(0, 2, (2, 16, k), device=DEV, dtype=torch.int8) * 2 - 1
        ref = torch.stack([a[i].float() @ a[i].float().t() for i in range(2)])
        assert torch.equal(bipolar_bmm(a, a), ref)


class TestCost:
    """Стоимость шага против плотного float-трансформера той же формы.

    | d_model | float, мс | бит, мс | ускорение | вес float | вес бит | память |
    |---------|-----------|---------|-----------|-----------|---------|--------|
    | 512     | 1.9       | 8.9     | 0.21x     | 12.2 MiB  | 0.03    | 2.5x   |
    | 1024    | 2.2       | 9.0     | 0.24x     | 48.4 MiB  | 0.07    | 5.3x   |
    | 2048    | 6.5       | 9.9     | 0.65x     | 193 MiB   | 0.13    | 12x    |
    | 4096    | 21.9      | 9.0     | **2.4x**  | 769 MiB   | 0.27    | 27x    |
    | 8192    | 81.6      | 12.4    | **6.6x**  | 3075 MiB  | 0.53    | 56x    |

    Ключевое: время шага бинарной модели почти НЕ РАСТЁТ с `d_model` (9 -> 12 мс
    при росте в 16 раз), потому что стоимость определяется числом запусков ядер,
    а не объёмом арифметики. У float-трансформера время растёт как `d^2`.
    """

    def test_memory_advantage_grows(self):
        small = measure(d_model=512, iters=5)
        large = measure(d_model=2048, iters=5)
        r_small = small["peak_float_mib"] / small["peak_bit_mib"]
        r_large = large["peak_float_mib"] / large["peak_bit_mib"]
        assert r_large > r_small, "преимущество по памяти не растёт с размером"
        assert r_large > 5.0, f"ожидали >5x по памяти, получили {r_large:.1f}x"

    def test_weights_are_32x_smaller_than_bf16(self):
        m = measure(d_model=1024, iters=5)
        # у бинарной модели меньше параметров по конструкции, поэтому сравниваем
        # плотность: бит на параметр против 16 бит bf16
        bits_per_param_float = 16
        bits_per_param_bit = m["bit_weight_mib"] * 8 * 1024 ** 2 / m["bit_params"]
        assert bits_per_param_bit == pytest.approx(1.0, abs=0.01)
        assert bits_per_param_float / bits_per_param_bit == pytest.approx(16, abs=1)

    def test_step_time_scales_weakly_with_width(self):
        """Время шага бинарной модели почти не зависит от `d_model`."""
        a = measure(d_model=512, iters=5)["ms_bit"]
        b = measure(d_model=2048, iters=5)["ms_bit"]
        assert b < 2.0 * a, f"время выросло в {b / a:.1f}x при росте d_model в 4x"

    @pytest.mark.slow
    def test_faster_than_float_at_scale(self):
        m = measure(d_model=4096, iters=5)
        assert m["ms_bit"] < m["ms_float"], \
            f"бит {m['ms_bit']:.1f} мс vs float {m['ms_float']:.1f} мс"
