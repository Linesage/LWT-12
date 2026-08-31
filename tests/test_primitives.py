"""Инварианты примитивов. Каждый — точное тождество, не приближение."""

import math

import pytest
import torch

from hdc import (DEV, bind, bipolar_mm, bundle, hamming, pack_bits, permute,
                 res_weight, shift_next, shift_prev, topk_select, unpack_bits)
from hdc.primitives import SRC_WEIGHT


def bipolar(*shape, gen=None):
    return torch.randint(0, 2, shape, device=DEV, dtype=torch.int8,
                         generator=gen) * 2 - 1


class TestInvertibility:
    """bind/permute/pack обратимы над GF(2) — основа точного backward."""

    def test_bind_self_inverse(self):
        x, r = bipolar(64, 1024), bipolar(1, 1024)
        assert torch.equal(bind(bind(x, r), r), x)

    def test_bind_commutes(self):
        a, b = bipolar(8, 256), bipolar(8, 256)
        assert torch.equal(bind(a, b), bind(b, a))

    @pytest.mark.parametrize("k", [1, 3, -2, 511])
    def test_permute_inverse(self, k):
        x = bipolar(16, 512)
        assert torch.equal(permute(permute(x, k), -k), x)

    def test_permute_composition(self):
        """permute(permute(x,a),b) == permute(x,a+b) — сдвиги складываются."""
        x = bipolar(4, 256)
        assert torch.equal(permute(permute(x, 3), 5), permute(x, 8))

    def test_pack_unpack(self):
        x = bipolar(32, 2048)
        assert torch.equal(unpack_bits(pack_bits(x)), x)

    def test_pack_size(self):
        x = bipolar(8, 1024)
        assert pack_bits(x).shape == (8, 128)
        assert pack_bits(x).dtype == torch.uint8

    def test_shift_prev_next_inverse(self):
        """shift_next обратна shift_prev на всех позициях, кроме краёв."""
        x = bipolar(4, 8, 64).reshape(4, 8, 64)
        y = shift_next(shift_prev(x))
        assert torch.equal(y[:, :-1], x[:, :-1])


class TestChainCollapse:
    """Цепочка bind+permute любой глубины СВОРАЧИВАЕТСЯ в один слой.

    Это прямое следствие линейности над GF(2). Практический вывод: глубина без
    bundling между слоями не добавляет выразительности.
    """

    def test_depth_two(self):
        x, w1, w2 = bipolar(4, 512), bipolar(1, 512), bipolar(1, 512)
        s1, s2 = 3, 7
        chain = permute(bind(permute(bind(x, w1), s1), w2), s2)
        w_eff = bind(permute(w1, s1 + s2), permute(w2, s2))
        assert torch.equal(chain, bind(permute(x, s1 + s2), w_eff))

    def test_depth_six(self):
        x = bipolar(4, 512)
        ws = [bipolar(1, 512) for _ in range(6)]
        shifts = [1, 3, 7, 2, 5, 11]
        y = x
        for w, s in zip(ws, shifts):
            y = permute(bind(y, w), s)
        acc = torch.ones(1, 512, device=DEV, dtype=torch.int8)
        total = 0
        for w, s in zip(ws, shifts):
            acc = bind(permute(acc, s), permute(w, s))
            total += s
        assert torch.equal(y, bind(permute(x, total), acc))


class TestExactCorrection:
    """a_new = a XOR error — решение уравнения, а не оценка направления."""

    def test_single_layer(self):
        a, b = bipolar(8, 1024), bipolar(1, 1024)
        target = bipolar(8, 1024)
        error = bind(a, b) != target
        a_new = torch.where(error, -a, a)
        assert torch.equal(bind(a_new, b), target)

    def test_through_chain(self):
        """Через цепочку глубины 6 коррекция всё ещё ПОБИТОВО точна."""
        x = bipolar(8, 1024)
        ws = [bipolar(1, 1024) for _ in range(6)]
        shifts = [1, 3, 7, 2, 5, 11]

        def fwd(v):
            for w, s in zip(ws, shifts):
                v = permute(bind(v, w), s)
            return v

        target = bipolar(8, 1024)
        err = fwd(x) != target
        for s in reversed(shifts):
            err = torch.roll(err, -s, dims=-1)
        assert torch.equal(fwd(torch.where(err, -x, x)), target)


class TestBipolarMM:
    """fp8-тензорные ядра дают ТОЧНЫЙ XNOR-popcount."""

    @pytest.mark.parametrize("shape", [(512, 512, 1024), (256, 128, 2048)])
    def test_exact_vs_int(self, shape):
        m, n, k = shape
        a, b = bipolar(m, k), bipolar(n, k)
        assert torch.equal(bipolar_mm(a, b), (a.float() @ b.float().t()))

    def test_hamming_matches_cdist(self):
        a, b = bipolar(64, 1024).float(), bipolar(32, 1024).float()
        assert torch.equal(hamming(a, b), torch.cdist(a, b, p=0))

    def test_hamming_self_zero(self):
        a = bipolar(16, 512)
        assert torch.equal(hamming(a, a).diag(), torch.zeros(16, device=DEV))

    def test_non_multiple_of_16_falls_back(self):
        """Формы не кратные 16 идут по fp32-пути и тоже точны."""
        a, b = bipolar(17, 2048), bipolar(33, 2048)
        assert torch.equal(bipolar_mm(a, b), a.float() @ b.float().t())


class TestBundle:
    """majority никогда не даёт ничью — суммарный вес голосов нечётный."""

    @pytest.mark.parametrize("k", [1, 2, 7, 16, 32])
    @pytest.mark.parametrize("ratio", [0.25, 0.5, 2.0])
    def test_total_weight_odd(self, k, ratio):
        n = torch.full((8,), k, device=DEV, dtype=torch.int32)
        w = res_weight(n, ratio)
        total = SRC_WEIGHT * k + w.float()
        assert torch.all(total % 2 == 1), f"чётная сумма голосов: {total}"

    def test_no_ties(self):
        votes = torch.randint(0, 2, (7, 64, 512), device=DEV,
                              dtype=torch.int8) * 2 - 1
        vote_sum = SRC_WEIGHT * votes.sum(0).float() + 1.0
        _bits, margin = bundle(vote_sum)
        assert int((margin == 0).sum()) == 0

    def test_margin_is_abs_sum(self):
        vote_sum = torch.tensor([[3.0, -5.0, 1.0]], device=DEV)
        bits, margin = bundle(vote_sum)
        assert bits.tolist() == [[1, -1, 1]]
        assert margin.tolist() == [[3, 5, 1]]

    def test_margin_scale_sqrt(self):
        """Для случайных голосов margin ~ sqrt(суммарного веса)."""
        n = 9
        votes = torch.randint(0, 2, (n, 256, 4096), device=DEV,
                              dtype=torch.int8) * 2 - 1
        _bits, margin = bundle(SRC_WEIGHT * votes.sum(0).float())
        expected = SRC_WEIGHT * math.sqrt(n * 2 / math.pi)
        assert 0.6 * expected < margin.float().mean() < 1.6 * expected


class TestTopK:
    def test_selects_k(self):
        scores = torch.randn(4, 32, device=DEV)
        valid = torch.ones_like(scores, dtype=torch.bool)
        sel, _gap, _nxt = topk_select(scores, 5, valid)
        assert torch.all(sel.sum(-1) == 5)

    def test_respects_mask(self):
        scores = torch.randn(1, 8, device=DEV)
        valid = torch.zeros_like(scores, dtype=torch.bool)
        valid[0, :3] = True
        sel, _gap, _nxt = topk_select(scores, 5, valid)
        assert int(sel.sum()) == 3, "выбрано больше, чем разрешено маской"
        assert int(sel[0, 3:].sum()) == 0

    def test_picks_largest(self):
        scores = torch.tensor([[1.0, 9.0, 3.0, 7.0]], device=DEV)
        valid = torch.ones_like(scores, dtype=torch.bool)
        sel, gap, _nxt = topk_select(scores, 2, valid)
        assert sel[0].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert gap.item() == pytest.approx(4.0)   # слабейший 7 против третьего 3
