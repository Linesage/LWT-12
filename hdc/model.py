"""Бинарная HDC-модель: все веса — биполярные гипервекторы в int8.

Ни autograd, ни float-весов. Единственная непрерывная величина в forward —
взвешенная сумма голосов перед `bundle`, и она немедленно превращается в биты.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import DEV, HDCConfig
from .primitives import (SRC_WEIGHT, bind, bipolar_bmm, bipolar_mm, bundle,
                         gate_with_fallback, res_weight, shift_prev, topk_select)

BIT_ROLES = ("codebook", "role_q", "role_k", "role_v", "mem_k", "mem_v")


class BitParam:
    """Биполярный {-1,+1} параметр: мастер-копия int8 + состояние обучения.

    `signal` — счётчик голосов «инвертировать этот бит», заполняется обратным
    проходом. `uses` — сколько экземпляров этот бит наблюдали. Их отношение и
    есть консенсус батча; отдельно они не значат почти ничего.
    """

    PERSIST_BYTES = 2 + 2 + 1
    """Байт на бит для состояния между шагами: momentum, flip_rate, age."""

    STEP_BYTES = 2 + 2
    """Байт на бит для аккумуляторов шага: signal и margin_sum, оба int16.

    `signal` — счётчик голосов, его максимум равен числу наблюдений бита за шаг;
    `margin_sum` — сумма margin по этим наблюдениям. Обе величины целые и
    ограничены размером батча, поэтому fp32 был расточительством: на модели
    30 Гбит это разница между 223 и 112 GiB.
    """

    def __init__(self, shape, name: str, role: str, gen=None,
                 track_history: bool = False):
        """`track_history=False` не выделяет momentum/flip_rate/age.

        Эти три тензора нужны исключительно мета-политике как признаки. Без неё
        они стоят 5 байт на бит и ничего не дают: на модели 30 Гбит это 140 GiB
        впустую. По умолчанию выключены.
        """
        self.name, self.role, self.shape = name, role, tuple(shape)
        self.role_id = BIT_ROLES.index(role)
        self.track_history = track_history
        self.sign = (torch.randint(0, 2, shape, device=DEV, dtype=torch.int8,
                                   generator=gen) * 2 - 1)
        if track_history:
            self.momentum = torch.zeros(shape, device=DEV, dtype=torch.bfloat16)
            self.flip_rate = torch.zeros(shape, device=DEV, dtype=torch.bfloat16)
            self.age = torch.zeros(shape, device=DEV, dtype=torch.uint8)
        else:
            self.momentum = self.flip_rate = self.age = None
        self.signal = torch.zeros(shape, device=DEV, dtype=torch.int16)
        self.margin_sum = torch.zeros(shape, device=DEV, dtype=torch.int16)
        self.uses = torch.zeros((shape[0], 1), device=DEV, dtype=torch.int32)

    def zero_signal(self) -> None:
        self.signal.zero_()
        self.margin_sum.zero_()
        self.uses.zero_()

    def flip_(self, mask: torch.Tensor) -> None:
        """Инверсия по маске. Самообратна: повторный вызов возвращает как было."""
        self.sign.mul_(torch.where(mask, -1, 1).to(torch.int8))

    @property
    def numel(self) -> int:
        return self.sign.numel()


@dataclass
class BundleSite:
    """Сайт majority-голосования — всё, что нужно обратному проходу."""

    x_in: torch.Tensor        # состояние-источник residual-голоса, int8
    y: torch.Tensor           # результат голосования, int8
    margin: torch.Tensor      # запас голосования, int16
    sel: torch.Tensor         # one-hot выбранных источников
    gap: torch.Tensor         # зазор до первого невыбранного
    next_idx: torch.Tensor    # индекс первого невыбранного
    w_res: torch.Tensor       # вес residual-голоса
    aux: dict = field(default_factory=dict)


@dataclass
class HammingTrace:
    """Трасса прямого прохода: по два сайта bundling на слой."""

    tokens: torch.Tensor
    attn: list
    mem: list
    x_final: torch.Tensor
    logits: torch.Tensor


class HDCBitTransformer:
    """Трансформер на биполярных гипервекторах.

    Слой:
        q_t = x_t     XOR r_q          # запрос по содержимому
        k_s = x_{s-1} XOR r_k          # ключ = ПРЕДЫДУЩИЙ токен -> induction head
        v_s = x_s     XOR r_v
        x = majority(2*top-k(v) + w_res*x)
        slots = top-k ключей памяти по хэммингу
        x = majority(2*value_slots + w_res*x)
    """

    def __init__(self, cfg: HDCConfig, seed: int = 0,
                 track_history: bool = False):
        self.cfg = cfg
        self.H, self.R, self.V = cfg.d_hidden, cfg.n_slots, cfg.vocab
        self.L, self.T = cfg.n_layers, cfg.seq_len
        gen = torch.Generator(device=DEV).manual_seed(seed)
        self.params: dict[str, BitParam] = {}

        def mk(shape, name, role):
            self.params[name] = BitParam(shape, name, role, gen=gen,
                                         track_history=track_history)

        mk((self.V, self.H), "codebook", "codebook")
        for l in range(self.L):
            mk((1, self.H), f"rq{l}", "role_q")
            mk((1, self.H), f"rk{l}", "role_k")
            mk((1, self.H), f"rv{l}", "role_v")
            mk((self.R, self.H), f"mem_k{l}", "mem_k")
            mk((self.R, self.H), f"mem_v{l}", "mem_v")

    def w(self, name: str) -> torch.Tensor:
        return self.params[name].sign

    @property
    def n_bits(self) -> int:
        return sum(p.numel for p in self.params.values())

    def zero_signal(self) -> None:
        for p in self.params.values():
            p.zero_signal()

    @torch.no_grad()
    def attention_branch(self, x: torch.Tensor, layer: int, causal: torch.Tensor):
        """Ветка attention: индукционная голова. Возвращает `(голоса, сайт-данные)`.

        `k_s = x_{s-1}` — ключ строится из ПРЕДЫДУЩЕГО токена, поэтому запрос по
        содержимому находит позицию после прошлого вхождения, а её значение и
        есть ответ. Это и делает механизм индукционной головой.
        """
        B, T = x.shape[0], x.shape[1]
        q = bind(x, self.w(f"rq{layer}"))
        k = bind(shift_prev(x), self.w(f"rk{layer}"))
        v = bind(x, self.w(f"rv{layer}"))
        scores = bipolar_bmm(q, k) / self.H ** 0.5
        valid = causal.expand(B, T, T)
        if self.cfg.attn_gate_margin > 0:
            # attention молчит там, где нет достаточно похожей позиции: оценка
            # нормирована на sqrt(H), поэтому порог задаётся в сигмах шума
            valid = gate_with_fallback(scores, valid, self.cfg.attn_gate_margin,
                                       self.cfg.gate_fallback)
        sel, gap, nxt = topk_select(scores, min(self.cfg.topk_attn, T), valid)
        votes = torch.bmm(sel, v.float())
        return votes, sel, gap, nxt, {"q": q, "k": k, "v": v}

    @torch.no_grad()
    def memory_branch(self, xf: torch.Tensor, layer: int):
        """Ветка ассоциативной памяти: top-k слотов по хэммингу к состоянию.

        `mem_gate_margin > 0` отсекает слоты, чья близость не отличается от
        случайной. Оценка нормирована на `sqrt(H)`, поэтому для случайных
        гипервекторов она распределена как `N(0, 1)`, и порог задаётся в сигмах.
        Без этого необученная память голосует шумом наравне с attention.
        """
        mk_, mv = self.w(f"mem_k{layer}"), self.w(f"mem_v{layer}")
        scores = bipolar_mm(xf, mk_) / self.H ** 0.5
        valid = torch.ones_like(scores, dtype=torch.bool)
        if self.cfg.mem_gate_margin > 0:
            valid = gate_with_fallback(scores, valid, self.cfg.mem_gate_margin,
                                       self.cfg.gate_fallback)
        sel, gap, nxt = topk_select(scores, min(self.cfg.topk_slots, self.R), valid)
        return sel @ mv.float(), sel, gap, nxt

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> HammingTrace:
        B, T = tokens.shape
        H, N = self.H, B * T
        cb = self.w("codebook")
        x = cb[tokens]
        causal = torch.ones(T, T, device=DEV, dtype=torch.bool).tril()
        attn_sites, mem_sites = [], []

        for l in range(self.L):
            a_votes, a_sel, a_gap, a_nxt, a_aux = self.attention_branch(x, l, causal)

            if self.cfg.parallel_branches:
                # Обе ветки читают ОДИН вход и голосуют в одном majority. Память
                # видит исходное состояние, а не результат attention.
                m_votes, m_sel, m_gap, m_nxt = self.memory_branch(x.reshape(N, H), l)
                wa, wm = self.cfg.branch_weight_attn, self.cfg.branch_weight_mem
                n_votes = (wa * a_sel.sum(-1).reshape(N)
                           + wm * m_sel.sum(-1)).to(torch.int32)
                w_res = res_weight(n_votes, self.cfg.res_ratio_attn)
                vote = (SRC_WEIGHT * (wa * a_votes.reshape(N, H) + wm * m_votes)
                        + w_res.unsqueeze(-1).float() * x.reshape(N, H).float())
                y, margin = bundle(vote)
                attn_sites.append(BundleSite(
                    x_in=x, y=y.reshape(B, T, H), margin=margin.reshape(B, T, H),
                    sel=a_sel, gap=a_gap, next_idx=a_nxt,
                    w_res=w_res.reshape(B, T), aux=a_aux))
                mem_sites.append(BundleSite(
                    x_in=x.reshape(N, H), y=y, margin=margin, sel=m_sel,
                    gap=m_gap, next_idx=m_nxt, w_res=w_res))
                x = y.reshape(B, T, H)
                continue

            # Последовательная схема: память читает выход attention.
            w_res = res_weight(a_sel.sum(-1).to(torch.int32), self.cfg.res_ratio_attn)
            vote = (SRC_WEIGHT * a_votes
                    + w_res.unsqueeze(-1).float() * x.float())
            y, margin = bundle(vote)
            attn_sites.append(BundleSite(x_in=x, y=y, margin=margin, sel=a_sel,
                                         gap=a_gap, next_idx=a_nxt, w_res=w_res,
                                         aux=a_aux))
            xf = y.reshape(N, H)
            m_votes, m_sel, m_gap, m_nxt = self.memory_branch(xf, l)
            sw = res_weight(m_sel.sum(-1).to(torch.int32), self.cfg.res_ratio_mem)
            svote = (SRC_WEIGHT * m_votes + sw.unsqueeze(-1).float() * xf.float())
            sy, smargin = bundle(svote)
            mem_sites.append(BundleSite(x_in=xf, y=sy, margin=smargin, sel=m_sel,
                                        gap=m_gap, next_idx=m_nxt, w_res=sw))
            x = sy.reshape(B, T, H)

        xfin = x.reshape(N, H)
        return HammingTrace(tokens=tokens, attn=attn_sites, mem=mem_sites,
                            x_final=xfin,
                            logits=bipolar_mm(xfin, cb) / self.cfg.tau)
