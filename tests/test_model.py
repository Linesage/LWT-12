"""Проверки самой модели: бинарность, причинность, механизм induction."""

import pytest
import torch

from hdc import DEV, HDCBitTransformer, HDCConfig, bipolar_mm
from hdc.tasks import task_accuracy, task_batch


class TestBinary:
    """Модель обязана быть бинарной насквозь — иначе весь смысл теряется."""

    def test_weights_are_int8_bipolar(self, model):
        for name, p in model.params.items():
            assert p.sign.dtype == torch.int8, name
            assert set(p.sign.unique().tolist()) <= {-1, 1}, name

    def test_state_is_bipolar(self, model, gen):
        tok, _ = task_batch("induction", 32, 16, 8, gen)
        trace = model.forward(tok)
        assert trace.x_final.dtype == torch.int8
        assert set(trace.x_final.unique().tolist()) <= {-1, 1}
        for site in trace.attn + trace.mem:
            assert set(site.y.unique().tolist()) <= {-1, 1}

    def test_no_autograd_in_forward(self, model, gen):
        tok, _ = task_batch("induction", 32, 16, 8, gen)
        trace = model.forward(tok)
        assert not trace.logits.requires_grad
        assert trace.logits.grad_fn is None

    def test_packed_size(self, model):
        """Упакованный вес — ровно 1 бит на параметр."""
        assert model.n_bits == sum(p.numel for p in model.params.values())


class TestCausality:
    def test_attention_is_causal(self, model, gen):
        tok, _ = task_batch("induction", 32, 16, 4, gen)
        trace = model.forward(tok)
        sel = trace.attn[0].sel
        t_idx = torch.arange(sel.shape[1], device=DEV)
        future = sel * (t_idx[None, None, :] > t_idx[None, :, None]).float()
        assert float(future.sum()) == 0.0, "attention смотрит в будущее"

    def test_prefix_invariance(self, model):
        """Изменение будущих токенов не меняет предсказание в позиции t."""
        tok = torch.randint(2, 32, (1, 16), device=DEV)
        base = model.forward(tok).logits.reshape(1, 16, -1)
        tok2 = tok.clone()
        tok2[0, 10:] = torch.randint(2, 32, (6,), device=DEV)
        alt = model.forward(tok2).logits.reshape(1, 16, -1)
        assert torch.equal(base[0, :10], alt[0, :10])


class TestInductionMechanism:
    """Архитектурный потолок: что модель может при ИДЕАЛЬНЫХ весах.

    Если с идеальными ролями точность равна случайной угадайке — задача не
    решается в принципе, и никакой оптимизатор не поможет. Этот тест ловит
    именно такие ошибки конфигурации.
    """

    @staticmethod
    def ideal(H=4096, **kw):
        kw.setdefault("topk_slots", 8)
        kw.setdefault("res_ratio_mem", 0.75)
        cfg = HDCConfig(d_hidden=H, n_layers=1, n_slots=64,
                        topk_attn=1, vocab=32, seq_len=16, **kw)
        m = HDCBitTransformer(cfg, seed=0)
        for name, p in m.params.items():
            if name.startswith(("rq", "rk", "rv")):
                p.sign.fill_(1)      # q=x_t, k=x_{s-1}, v=x_s — чистая induction
        return m

    def test_attention_finds_previous_occurrence(self):
        m = self.ideal()
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, _tgt = task_batch("induction", 32, 16, 64, gen)
        sel = m.forward(tok).attn[0].sel[:, -1].argmax(-1)
        a = tok[:, -1]
        hits = 0
        for i in range(tok.shape[0]):
            want = (tok[i, :-1] == a[i]).nonzero().squeeze(-1) + 1
            hits += int(sel[i]) in want.tolist()
        assert hits == tok.shape[0], f"attention нашёл позицию лишь в {hits}/64"

    def test_ideal_weights_solve_unique_task(self):
        """На однозначной задаче идеальные веса дают ПОЧТИ 1.0."""
        m = self.ideal()
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        acc = task_accuracy(m, tok, tgt)
        assert acc > 0.9, f"архитектура не решает задачу даже идеально: {acc:.3f}"

    @pytest.mark.parametrize("k_attn", [1, 2, 4])
    def test_topk_attn_dilutes_signal(self, k_attn):
        """Лишние источники в attention РАЗБАВЛЯЮТ сигнал до неузнаваемости.

        При `topk_attn=1` идеальные веса дают 1.0, при 2 — уже 0.28, при 4 — 0.12.
        Причина: нужен ровно один источник (позиция после предыдущего вхождения),
        остальные `k-1` подмешивают случайные гипервекторы с тем же весом 2, и
        majority теряет целевой вектор. Значит `topk_attn` — не «сколько внимания»,
        а «сколько шума добавить»: для точного извлечения он должен быть 1.
        """
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=8,
                        res_ratio_mem=0.75, topk_attn=k_attn, vocab=32, seq_len=16)
        m = HDCBitTransformer(cfg, seed=0)
        for name, p in m.params.items():
            if name.startswith(("rq", "rk", "rv")):
                p.sign.fill_(1)
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        acc = task_accuracy(m, tok, tgt)
        if k_attn == 1:
            assert acc > 0.95, f"k_attn=1 должен давать точное извлечение: {acc}"
        else:
            assert acc < 0.5, f"тест устарел: k_attn={k_attn} больше не разбавляет"

    @pytest.mark.parametrize("k_slots,ratio_mem,works", [
        (1, 0.5, False),    # один слот весом 4 перебивает residual 3
        (8, 0.75, True),    # нужно 3.2 согласных слота из 8
        (16, 0.9, True),
    ])
    def test_memory_needs_slot_consensus(self, k_slots, ratio_mem, works):
        """Одиночный слот памяти не должен перебивать результат attention.

        При `topk_slots=1, ratio=0.5` вес слота 4 против residual 3 — любой шумный
        слот затирает состояние, потолок 0.000. Решение не в увеличении `ratio`
        (это заморозило бы слой, см. TestResidualBalance), а в требовании
        консенсуса: `topk_slots > 1` при `ratio` близком к границе.
        """
        m = self.ideal(topk_slots=k_slots, res_ratio_mem=ratio_mem)
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        acc = task_accuracy(m, tok, tgt)
        assert (acc > 0.9) == works, f"k={k_slots} ratio={ratio_mem}: acc={acc:.3f}"

    def test_attention_output_matches_target_exactly(self):
        """После attention состояние ПОБИТОВО равно кодбуку целевого токена."""
        m = self.ideal()
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, tgt = task_batch("induction_unique", 32, 16, 32, gen)
        trace = m.forward(tok)
        H = m.H
        y = trace.attn[0].y[:, -1]
        d = (H - bipolar_mm(y, m.w("codebook"))) / 2
        exact = int((d.gather(1, tgt[:, -1:]) == 0).sum())
        assert exact >= 30, f"побитовое совпадение лишь в {exact}/32"


class TestRoleSemantics:
    """Что РЕАЛЬНО определяет работу attention — проверено перебором.

    Результат неочевидный: абсолютные значения `rq`/`rk` не важны вовсе, важно
    только их СОГЛАСОВАНИЕ. Оценка `q_t · k_s = (x_t·rq)·(x_{s-1}·rk)` зависит
    от `rq XOR rk`, а не от каждого по отдельности. Это значит, что обучать
    `rq` и `rk` независимо бессмысленно: у задачи есть ровно одна степень
    свободы вместо двух, и 1024 бита `rq` — избыточная параметризация.
    """

    @staticmethod
    def _model(**kw):
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=64, topk_slots=1,
                        topk_attn=1, vocab=32, seq_len=16, **kw)
        m = HDCBitTransformer(cfg, seed=0)
        m.params["rv0"].sign.fill_(1)
        return m, cfg

    def test_only_agreement_matters(self):
        m, cfg = self._model()
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        r = torch.randint(0, 2, (1, cfg.d_hidden), device=DEV,
                          dtype=torch.int8) * 2 - 1
        m.params["rq0"].sign.copy_(r)
        m.params["rk0"].sign.copy_(r)
        assert task_accuracy(m, tok, tgt) > 0.95, "согласованные роли должны работать"

    def test_anti_agreement_breaks(self):
        m, cfg = self._model()
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        r = torch.randint(0, 2, (1, cfg.d_hidden), device=DEV,
                          dtype=torch.int8) * 2 - 1
        m.params["rq0"].sign.copy_(r)
        m.params["rk0"].sign.copy_(-r)
        assert task_accuracy(m, tok, tgt) < 0.1, "противоположные роли должны ломать"

    @pytest.mark.parametrize("n_bad,works", [(300, True), (450, True), (512, False)])
    def test_tolerance_to_role_corruption(self, n_bad, works):
        """До половины бит роли можно испортить без потери точности.

        Порог ровно на H/2: там `rq XOR rk` становится равновероятным, и
        согласование теряется. Отсюда следует, что бит роли несёт около одного
        бита информации на всю роль, а не H бит.
        """
        m, cfg = self._model()
        gen = torch.Generator(device=DEV).manual_seed(3)
        tok, tgt = task_batch("induction_unique", 32, 16, 64, gen)
        for name in ("rq0", "rk0"):
            m.params[name].sign.fill_(1)
        bad = torch.zeros(cfg.d_hidden, device=DEV, dtype=torch.bool)
        bad[torch.randperm(cfg.d_hidden, device=DEV, generator=gen)[:n_bad]] = True
        m.params["rq0"].sign[0][bad] = -1
        acc = task_accuracy(m, tok, tgt)
        assert (acc > 0.9) == works, f"n_bad={n_bad}: acc={acc:.3f}"


class TestResidualBalance:
    """Веса residual определяют, КТО побеждает в majority. Обе крайности ломают.

    Ключевое соотношение: сила источников `2*SRC_WEIGHT*k` против веса residual
    `w_res = 2*round(ratio*k) + 1`.

    * `w_res > 2*SRC_WEIGHT*k` — источники физически не могут перевернуть бит,
      и слой становится тождественным. Проверено: при `k=1, ratio=2.0` residual
      имеет вес 5 против силы источников 4, `mem_v` не получает НИ ОДНОГО голоса.
      Условие `ratio < SRC_WEIGHT/2` (то есть `< 1.0`) — необходимое, потому что
      `w_res ≈ 2*ratio*k`, а сила источников `2*SRC_WEIGHT*k` растёт так же
      линейно по `k`: увеличение `k` НЕ спасает.
    * `w_res` слишком мал — источники затирают состояние (см. предыдущий класс).

    Отсюда правило: `ratio` задаётся вместе с `k`, а не независимо от него.
    """

    @staticmethod
    def source_power(k: int) -> float:
        from hdc.primitives import SRC_WEIGHT
        return 2.0 * SRC_WEIGHT * k

    @pytest.mark.parametrize("k,ratio,can_flip", [
        (1, 2.0, False),    # residual 5 > сила 4 — слой заморожен
        (1, 0.5, True),     # residual 3 < сила 4
        (8, 0.5, True),     # residual 9 < сила 32
        (32, 2.0, False),   # residual 129 > сила 128 — заморожен даже при k=32
        (32, 0.5, True),    # residual 33 < сила 128
    ])
    def test_sources_can_overcome_residual(self, k, ratio, can_flip):
        from hdc.primitives import res_weight
        w = float(res_weight(torch.tensor([k], device=DEV, dtype=torch.int32), ratio))
        assert (self.source_power(k) > w) == can_flip, \
            f"k={k} ratio={ratio}: сила {self.source_power(k)} vs residual {w}"

    def test_config_rejects_frozen_ratio(self):
        """Конфигурация ОБЯЗАНА падать на значениях, замораживающих слой.

        Это единственная защита от того, чтобы кто-то (включая меня) снова
        «оптимизировал» ratio вверх и получил модель, которая молча не учится.
        """
        with pytest.raises(ValueError, match="res_ratio_mem"):
            HDCConfig(res_ratio_mem=1.0)
        with pytest.raises(ValueError, match="res_ratio_attn"):
            HDCConfig(res_ratio_attn=2.0)

    def test_frozen_layer_gives_no_votes(self):
        """Если источники бессильны, обратный проход не может их обучать.

        Проверяем в обход валидатора: важно зафиксировать саму связь
        «замороженный слой -> нет голосов», а не только запрет в конфиге.
        """
        from hdc.errors import exact_backward
        cfg = HDCConfig(d_hidden=1024, n_layers=1, n_slots=256, topk_slots=1,
                        topk_attn=1, vocab=32, seq_len=16, res_ratio_mem=0.9)
        object.__setattr__(cfg, "res_ratio_mem", 2.0)
        m = HDCBitTransformer(cfg, seed=0)
        gen = torch.Generator(device=DEV).manual_seed(1)
        tok, tgt = task_batch("markov", 32, 16, 32, gen)
        exact_backward(m, m.forward(tok), tgt)
        assert float(m.params["mem_v0"].signal.sum()) == 0.0


class TestMemoryPipeline:
    """Память видит НЕ токен, а выход attention — это меняет постановку.

    Задача `markov` требует отображения «текущий токен -> следующий», то есть на
    вход памяти должен приходить именно текущий токен. Но attention стоит ПЕРЕД
    памятью и уже перемешал состояние: замер показывает, что вход памяти
    совпадает с текущим токеном лишь в 35% случаев.

    Если заставить attention пропускать состояние (большой `res_ratio_attn`),
    вход памяти совпадает с токеном в 100%, и вручную заполненная таблица
    переходов даёт 0.364 — выше потолка 1/3, то есть память работает точно.
    """

    @staticmethod
    def _with_table(res_ratio_attn: float):
        from hdc.tasks import markov_table
        V, H = 32, 1024
        cfg = HDCConfig(d_hidden=H, n_layers=1, n_slots=96, topk_slots=1,
                        topk_attn=1, vocab=V, seq_len=16, res_ratio_mem=0.5,
                        res_ratio_attn=0.25)
        object.__setattr__(cfg, "res_ratio_attn", res_ratio_attn)
        m = HDCBitTransformer(cfg, seed=0)
        m.params["rv0"].sign.fill_(1)
        table, cb = markov_table(V), m.w("codebook")
        mk, mv = m.params["mem_k0"], m.params["mem_v0"]
        i = 0
        for tk in range(V):
            for s in table[tk].nonzero().squeeze(-1).tolist():
                mk.sign[i], mv.sign[i] = cb[tk], cb[s]
                i += 1
        return m, cfg

    def test_attention_destroys_memory_input(self):
        from hdc import bipolar_mm
        m, cfg = self._with_table(0.25)
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, _tgt = task_batch("markov", 32, 16, 128, gen)
        trace = m.forward(tok)
        d = (cfg.d_hidden - bipolar_mm(trace.mem[0].x_in, m.w("codebook"))) / 2
        same = float((d.argmin(1) == tok.reshape(-1)).float().mean())
        assert same < 0.6, f"тест устарел: attention больше не мешает ({same:.2f})"

    def test_memory_solves_markov_when_input_preserved(self):
        """С сохранённым входом идеальная таблица достигает потолка задачи."""
        m, _cfg = self._with_table(10.0)
        gen = torch.Generator(device=DEV).manual_seed(7)
        tok, tgt = task_batch("markov", 32, 16, 128, gen)
        acc = task_accuracy(m, tok, tgt)
        assert acc > 0.3, f"память не решает markov даже с готовой таблицей: {acc:.3f}"
