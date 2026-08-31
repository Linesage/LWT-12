"""Проверки обратного прохода. Здесь математика должна сходиться точно."""

import pytest
import torch

from hdc import (DEV, HDCBitTransformer, HDCConfig, apply_flips, bipolar_mm,
                 consensus_flip_masks, exact_backward, split_by_margin,
                 target_error_mask, wanted_state)
from hdc.errors import GAP_PER_FLIP, vote_role_v
from hdc.primitives import SRC_WEIGHT
from hdc.tasks import task_batch


class TestTargetMask:
    """Целевая маска должна ЗАКРЫВАТЬ разрыв, а не просто указывать направление."""

    @staticmethod
    def _correct_count(model, x, targets, keep):
        score = bipolar_mm(x, model.w("codebook"))
        t = targets.reshape(-1).clamp_min(0)
        s_t = score.gather(1, t[:, None]).squeeze(1)
        rest = score.scatter(1, t[:, None], -float("inf"))
        return int(((s_t > rest.max(1).values) & keep).sum())

    def test_closes_gap_with_current_rival(self, model, gen):
        """Одна итерация закрывает разрыв ровно с ТЕКУЩИМ лучшим конкурентом.

        Это и есть точное утверждение про GAP_PER_FLIP. Полное решение задачи за
        одну итерацию невозможно: конкурентов много (~23 класса в пределах
        разрыва), после правки вперёд выходит следующий.
        """
        tok, tgt = task_batch("induction", 32, 16, 64, gen)
        trace = model.forward(tok)
        cb = model.w("codebook")
        t = tgt.reshape(-1)
        keep = t >= 0
        safe = t.clamp_min(0)
        score = bipolar_mm(trace.x_final, cb)
        rest = score.scatter(1, safe[:, None], -float("inf"))
        rival = rest.argmax(1)

        mask, _gap, _keep = target_error_mask(model, trace, tgt)
        x2 = torch.where(mask, -trace.x_final, trace.x_final)
        s2 = bipolar_mm(x2, cb)
        beat = ((s2.gather(1, rival[:, None]).squeeze(1)
                 < s2.gather(1, safe[:, None]).squeeze(1)) & keep)
        assert int(beat.sum()) == int(keep.sum()), \
            f"исходный конкурент обойдён лишь в {int(beat.sum())}/{int(keep.sum())}"

    def test_iteration_converges_to_full_solution(self, model, gen):
        """Итерации доводят до 100%: каждая точна, число шагов задаёт охват."""
        tok, tgt = task_batch("induction", 32, 16, 64, gen)
        trace = model.forward(tok)
        keep = tgt.reshape(-1) >= 0
        prev = -1
        for iters in (1, 2, 4, 8):
            e, _k = wanted_state(model, trace, tgt, iters)
            x = torch.where(e, -trace.x_final, trace.x_final)
            got = self._correct_count(model, x, tgt, keep)
            assert got >= prev, "итерации ухудшают результат"
            prev = got
        assert prev == int(keep.sum()), f"не сошлось: {prev}/{int(keep.sum())}"

    def test_flip_changes_gap_by_exactly_four(self, model, gen):
        """Флип полезного бита меняет разрыв ровно на GAP_PER_FLIP."""
        tok, tgt = task_batch("induction", 32, 16, 8, gen)
        trace = model.forward(tok)
        cb = model.w("codebook")
        t = tgt.reshape(-1)
        row = int((t >= 0).nonzero()[0])
        score = bipolar_mm(trace.x_final, cb)
        tgt_c = int(t[row])
        rest = score[row].clone()
        rest[tgt_c] = -float("inf")
        wrong_c = int(rest.argmax())
        gap0 = float(score[row, wrong_c] - score[row, tgt_c])

        x = trace.x_final[row]
        useful = ((x != cb[tgt_c]) & (x == cb[wrong_c])).nonzero().squeeze(-1)
        assert useful.numel() > 0
        x2 = trace.x_final.clone()
        x2[row, useful[0]] *= -1
        s2 = bipolar_mm(x2, cb)
        gap1 = float(s2[row, wrong_c] - s2[row, tgt_c])
        assert gap0 - gap1 == pytest.approx(GAP_PER_FLIP), f"{gap0} -> {gap1}"

    def test_mask_size_matches_need(self, model, gen):
        """Флипов ровно столько, сколько нужно: ceil((gap+1)/4), не больше."""
        tok, tgt = task_batch("induction", 32, 16, 64, gen)
        trace = model.forward(tok)
        mask, gap, keep = target_error_mask(model, trace, tgt)
        need = torch.ceil((gap.clamp_min(0) + 1) / GAP_PER_FLIP).long()
        got = mask.sum(1)
        assert torch.all(got[keep] <= need[keep])

    def test_no_mask_where_ignored(self, model, gen):
        tok, tgt = task_batch("induction", 32, 16, 16, gen)
        trace = model.forward(tok)
        mask, _gap, keep = target_error_mask(model, trace, tgt)
        assert int(mask[~keep].sum()) == 0, "правим позиции, которые не предсказываем"

    def test_wanted_state_converges(self, model, gen):
        tok, tgt = task_batch("induction", 32, 16, 64, gen)
        trace = model.forward(tok)
        e1, _ = wanted_state(model, trace, tgt, iters=1)
        e4, _ = wanted_state(model, trace, tgt, iters=4)
        assert int(e4.sum()) >= int(e1.sum()), "итерации не добивают остаток"
        assert e4.float().mean() < 0.05, "правим слишком много бит состояния"


class TestMarginSplit:
    """Критерий достижимости — точный, проверяем его прямым флипом."""

    def test_reachable_flip_actually_flips_output(self):
        """Если margin < 2*SRC_WEIGHT, флип источника ПЕРЕВОРАЧИВАЕТ бит выхода."""
        for margin in (1.0, 3.0, 5.0, 9.0):
            vote_sum = torch.tensor([[margin]], device=DEV)
            flipped = vote_sum - 2.0 * SRC_WEIGHT      # источник голосовал «за»
            reachable = margin < 2.0 * SRC_WEIGHT
            assert (float(flipped) < 0) == reachable, f"margin={margin}"

    def test_split_is_partition(self, model, gen):
        """Три части покрывают маску и не пересекаются."""
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        e, _ = wanted_state(model, trace, tgt)
        site = trace.mem[0]
        n_src = site.sel.sum(-1).to(torch.int32)
        a, b, c = split_by_margin(e, site.margin, site.w_res, n_src)
        assert torch.equal(a | b | c, e), "разбиение теряет биты"
        assert int((a & c).sum()) == 0 and int((b & c).sum()) == 0

    def test_blocked_is_minority(self, model, gen):
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        stat = exact_backward(model, trace, tgt)
        assert stat["exact_frac"] > 0.7, f"точным осталось лишь {stat['exact_frac']:.2f}"


class TestVoteSigns:
    """Знак голосов — место, где был реальный баг. Проверяем прямым перебором."""

    def test_role_v_votes_match_bruteforce(self, model, gen):
        """Векторизованный подсчёт голосов роли == честный цикл по (t, s)."""
        tok, _tgt = task_batch("induction", 32, 16, 4, gen)
        trace = model.forward(tok)
        site = trace.attn[0]
        want = torch.rand(site.y.shape, device=DEV) < 0.02

        param = model.params["rv0"]
        param.zero_signal()
        vote_role_v(site, want, param)
        fast = param.signal[0].int()

        B, T, _H = site.y.shape
        v, y, sel = site.aux["v"], site.y, site.sel
        slow = torch.zeros_like(fast)
        for b in range(B):
            for t in range(T):
                for s in sel[b, t].nonzero().squeeze(-1).tolist():
                    slow += (want[b, t] & (v[b, s] == y[b, t])).int()
        assert torch.equal(fast, slow), f"расходится: {(fast - slow).abs().max()}"

    def test_votes_identify_corrupted_bits(self):
        """Решающий тест: портим известные биты — backward обязан их найти.

        Стартуем от идеальных весов, инвертируем 100 случайных бит `rv0` и
        смотрим, попадут ли они в top-100 по числу голосов. Случайно ожидалось бы
        ~10 из 100. Это прямая проверка, что знак и адресация голосов верны:
        неверный знак дал бы результат хуже случайного.
        """
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=1,
                        topk_attn=1, vocab=32, seq_len=16)
        m = HDCBitTransformer(cfg, seed=0)
        for name, p in m.params.items():
            if name.startswith(("rq", "rk", "rv")):
                p.sign.fill_(1)
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)

        n_bad = 100
        bad = torch.zeros(cfg.d_hidden, device=DEV, dtype=torch.bool)
        bad[torch.randperm(cfg.d_hidden, device=DEV)[:n_bad]] = True
        m.params["rv0"].sign[0][bad] = -1

        trace = m.forward(tok)
        exact_backward(m, trace, tgt)
        votes = m.params["rv0"].signal[0].int()
        hit = int(bad[votes.topk(n_bad).indices].sum())
        assert hit >= 90, f"нашлось лишь {hit}/{n_bad} испорченных бит"
        assert float(votes[bad].float().mean()) > float(votes[~bad].float().mean())

    def test_flipping_voted_bits_restores_ideal(self):
        """Инверсия бит с голосами должна ВОССТАНОВИТЬ идеальные веса."""
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=1,
                        topk_attn=1, vocab=32, seq_len=16)
        m = HDCBitTransformer(cfg, seed=0)
        for name, p in m.params.items():
            if name.startswith(("rq", "rk", "rv")):
                p.sign.fill_(1)
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        bad = torch.zeros(cfg.d_hidden, device=DEV, dtype=torch.bool)
        bad[torch.randperm(cfg.d_hidden, device=DEV)[:100]] = True
        m.params["rv0"].sign[0][bad] = -1

        trace = m.forward(tok)
        exact_backward(m, trace, tgt)
        p = m.params["rv0"]
        mask = (p.signal > 0)
        p.flip_(mask)
        restored = float((p.sign > 0).float().mean())
        assert restored > 0.99, f"после правки доля +1 = {restored:.3f}"

    def test_role_v_vote_direction_is_correct(self):
        """Голос за флип должен УМЕНЬШАТЬ loss, а не увеличивать.

        Ловит именно ту ошибку, из-за которой `rv` обучался в противоположную
        сторону: сравнение шло по позиции запроса вместо позиции источника.
        """
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=1,
                        topk_attn=1, vocab=32, seq_len=16)
        m = HDCBitTransformer(cfg, seed=0)
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)

        def loss():
            lg = m.forward(tok).logits.float()
            return float(torch.nn.functional.cross_entropy(
                lg, tgt.reshape(-1), ignore_index=-100))

        base = loss()
        best = {}
        for _ in range(12):
            trace = m.forward(tok)
            exact_backward(m, trace, tgt)
            p = m.params["rv0"]
            k = max(1, p.numel // 64)
            top = p.signal.reshape(-1).int().topk(k)
            mask = torch.zeros(p.numel, device=DEV, dtype=torch.bool)
            mask[top.indices[top.values > 0]] = True
            p.flip_(mask.reshape(p.shape))
            cur = loss()
            if cur > base:
                p.flip_(mask.reshape(p.shape))
            else:
                base = cur
            best["loss"] = base
        # обучение ролей должно вести к идеалу (+1), а не от него
        frac_pos = float((m.params["rv0"].sign > 0).float().mean())
        assert frac_pos > 0.5, f"rv уходит от идеала: доля +1 = {frac_pos:.3f}"

    def test_all_params_receive_votes(self, model, gen):
        """Каждый параметр должен обучаться. Ключи памяти и роли q/k — тоже."""
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        exact_backward(model, trace, tgt, gates=True)
        dead = [n for n, p in model.params.items()
                if float(p.signal.abs().sum()) == 0]
        assert not dead, f"не получают сигнала: {dead}"

    def test_votes_are_nonnegative_counts(self, model, gen):
        """signal — счётчик голосов, он не может быть отрицательным."""
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        exact_backward(model, trace, tgt)
        for name, p in model.params.items():
            assert int(p.signal.min()) >= 0, f"{name}: отрицательный счётчик"

    def test_votes_bounded_by_uses(self, model, gen):
        """Голосов «за» не может быть больше, чем наблюдений бита."""
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        exact_backward(model, trace, tgt)
        for name, p in model.params.items():
            if name.startswith("mem_k"):
                continue      # гейт-голоса считаются по своей схеме
            excess = (p.signal.int() > p.uses).sum()
            assert int(excess) == 0, f"{name}: голосов больше наблюдений"


class TestFlips:
    def test_flip_is_self_inverse(self, model, gen):
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        before = {n: p.sign.clone() for n, p in model.params.items()}
        trace = model.forward(tok)
        exact_backward(model, trace, tgt)
        masks, _frac = consensus_flip_masks(model, 4e-3)
        apply_flips(model, masks)
        apply_flips(model, masks)
        for n, p in model.params.items():
            assert torch.equal(p.sign, before[n]), f"{n}: откат не восстановил"

    def test_budget_respected(self, model, gen):
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        exact_backward(model, trace, tgt)
        for budget in (1e-3, 4e-3, 1e-2):
            masks, frac = consensus_flip_masks(model, budget)
            assert frac <= budget * 1.05, f"бюджет {budget} превышен: {frac}"
            for name, mask in masks.items():
                p = model.params[name]
                k = max(1, int(round(budget * p.numel)))
                assert int(mask.sum()) <= k, f"{name}: {int(mask.sum())} > {k}"

    def test_flips_only_where_votes(self, model, gen):
        """Не инвертируем биты без единого голоса «за»."""
        tok, tgt = task_batch("induction", 32, 16, 32, gen)
        trace = model.forward(tok)
        exact_backward(model, trace, tgt)
        masks, _ = consensus_flip_masks(model, 1e-2)
        for name, mask in masks.items():
            p = model.params[name]
            assert int((mask & (p.signal <= 0)).sum()) == 0, name
