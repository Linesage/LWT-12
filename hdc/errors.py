"""Точный backward над GF(2): маска битовых ошибок вместо градиента.

`bind` и `permute` линейны и обратимы над GF(2), поэтому маска ошибки проходит
через них БЕЗ приближения: если `y = a XOR b` и известен `target`, то
`a_new = a XOR (y XOR target)` — решение уравнения, не оценка направления.

Необратим только `bundle` (majority). Но и там критерий точен: флип одного
источника меняет сумму голосов ровно на `2*SRC_WEIGHT`, значит бит выхода
перевернётся тогда и только тогда, когда `margin < 2*SRC_WEIGHT`.
"""

from __future__ import annotations

import dataclasses

import torch

from .config import DEV
from .model import BitParam, BundleSite, HammingTrace
from .primitives import SRC_WEIGHT, bipolar_bmm, bipolar_mm

# Флип бита состояния меняет `x·cb_c` на `-2*x_j*cb_c[j]`, поэтому разрыв
# `score_wrong - score_target` меняется ровно на 4 (по 2 с каждой стороны).
GAP_PER_FLIP = 4.0


@torch.no_grad()
def target_error_mask(model, trace: HammingTrace, targets: torch.Tensor):
    """Точная маска бит состояния, которые надо инвертировать.

    Требовать `x == cb[target]` нельзя: система переопределена, одно состояние
    не совпадёт с кодбуком по всем позициям. Настоящая цель — **порядок**:
    `hamming(x, cb[tgt])` меньше, чем у любого конкурента.

    Полезен флип бита `j`, если `cb_tgt[j] != x[j]` и `cb_wrong[j] == x[j]`:
    тогда разрыв уменьшается ровно на `GAP_PER_FLIP`. Нужное число флипов —
    `ceil((gap + 1) / GAP_PER_FLIP)`. Всё точно, комбинаторно.
    """
    cb = model.w("codebook")
    t = targets.reshape(-1)
    keep = t >= 0
    safe = t.clamp_min(0)
    score = bipolar_mm(trace.x_final, cb)
    s_tgt = score.gather(1, safe[:, None]).squeeze(1)
    rest = score.scatter(1, safe[:, None], -float("inf"))
    s_wrong, wrong = rest.max(1)
    gap = s_wrong - s_tgt

    x = trace.x_final
    useful = (x != cb[safe]) & (x == cb[wrong])
    need = torch.ceil((gap.clamp_min(0) + 1) / GAP_PER_FLIP).to(torch.int64)
    need = torch.where(keep, need, torch.zeros_like(need))

    # Все полезные биты дают одинаковые -GAP_PER_FLIP, выбор среди них не
    # информативен; случайная выборка избавляет от перекоса по позициям.
    rank = torch.where(useful, torch.rand(x.shape, device=DEV),
                       torch.full(x.shape, -1.0, device=DEV))
    order = rank.argsort(dim=1, descending=True)
    pos = torch.empty_like(order)
    pos.scatter_(1, order, torch.arange(x.shape[1], device=DEV).expand_as(order))
    return useful & (pos < need[:, None]), gap, keep


@torch.no_grad()
def wanted_state(model, trace: HammingTrace, targets: torch.Tensor,
                 iters: int = 8):
    """Точное целевое состояние: какие биты `x_final` надо перевернуть.

    Одна итерация закрывает разрыв ровно с ТЕКУЩИМ лучшим конкурентом — это
    проверяемо точно (см. test_closes_gap_with_current_rival). Но конкурентов
    много: замер показывает ~23 класса в пределах разрыва, и после правки
    вперёд выходит следующий. Поэтому итерируем: каждый шаг точен, а число
    шагов определяется тем, сколько классов надо обойти.

    Сходимость проверяется по остатку: цикл прерывается, когда правок больше не
    требуется.
    """
    x = trace.x_final.clone()
    keep = targets.reshape(-1) >= 0
    for _ in range(iters):
        probe = dataclasses.replace(
            trace, x_final=x,
            logits=bipolar_mm(x, model.w("codebook")) / model.cfg.tau)
        mask, _gap, keep = target_error_mask(model, probe, targets)
        if int(mask.sum()) == 0:
            break
        x = torch.where(mask, -x, x)
    return (x != trace.x_final) & keep[:, None], keep


@torch.no_grad()
def split_by_margin(error: torch.Tensor, margin: torch.Tensor,
                    w_res: torch.Tensor, n_src: torch.Tensor):
    """Делит маску ошибки по тому, СКОЛЬКО согласованных флипов нужно.

    Критерий точен: флип одного источника меняет сумму голосов на
    `2*SRC_WEIGHT`, флип бита состояния — на `2*w_res`. Значит:

    * `reach_res`  — хватит флипа бита состояния (`margin < 2*w_res`);
    * `reach_src`  — хватит флипов источников: нужно `ceil(margin/(2*SRC_WEIGHT))`
      согласованных флипов, и столько источников в наличии;
    * `blocked`    — не хватает даже всех источников вместе. Здесь и только
      здесь точность заканчивается.

    Прежняя версия требовала, чтобы хватило **одного** источника, и при
    `res_ratio_mem=2.0` (вес residual 29 против margin ~27) это давало ноль
    голосов слотам памяти: половина модели не обучалась. Правильный порог —
    суммарная сила доступных источников.
    """
    m = margin.to(torch.float32)
    reach_res = error & (m < 2.0 * w_res.float().unsqueeze(-1))
    src_power = 2.0 * SRC_WEIGHT * n_src.float().unsqueeze(-1)
    reach_src = error & (m < src_power)
    blocked = error & ~(reach_src | reach_res)
    return reach_src, reach_res, blocked


@torch.no_grad()
def vote_bundle_sources(sel: torch.Tensor, want: torch.Tensor, y: torch.Tensor,
                        src: torch.Tensor, param: BitParam) -> None:
    """Голоса источникам bundling за инверсию их бит.

    Флип бита источника помогает только если источник голосовал ЗА текущее
    (ошибочное) значение выхода: `src[r, j] == y[q, j]`. Флип «против» лишь
    увеличит margin и уведёт бит дальше от переключения.

    `sel` — (Q, R) one-hot, `want` — (Q, H) маска ошибки, `y` — (Q, H) выход,
    `src` — (R, H) значения источников. Суммирование строго по парам (q, r),
    где `sel[q, r] = 1`: голос получает источник `r`, а не запрос `q`.
    """
    want_f = want.to(torch.float32)
    pos = sel.t() @ (want_f * (y > 0).to(torch.float32))    # (R,H) хотим -> y=+1
    neg = sel.t() @ (want_f * (y < 0).to(torch.float32))    # (R,H) хотим -> y=-1
    # источник голосовал за +1 там, где src=+1: его флип уберёт голос за +1
    param.signal.add_(torch.where(src > 0, pos, neg).to(torch.int16))
    param.uses.add_(sel.sum(0, keepdim=True).t().to(torch.int32))


@torch.no_grad()
def vote_role_v(site: BundleSite, want: torch.Tensor, param: BitParam) -> None:
    """Голоса ролевому гипервектору `rv`.

    Здесь был баг с неверным знаком: код сравнивал `v[b, t]` с `y[b, t]`, то есть
    значение в позиции **запроса**, тогда как голосует значение в позиции
    **источника** `s`, выбранного attention. Правильный подсчёт — сумма по парам
    `(t, s)` с `sel[t, s] = 1` от индикатора `v[b, s, j] == y[b, t, j]`.

    В матричной форме: `sel^T · want` даёт вклад в источники, но индикатор
    зависит и от `t`, и от `s`, поэтому раскладываем по знаку `y`:

        votes[j] = Σ_b Σ_s ( Σ_t sel[b,t,s]·want[b,t,j]·[y[b,t,j]=+1] )·[v[b,s,j]=+1]
                 + Σ_b Σ_s ( ... [y=-1] )·[v[b,s,j]=-1]

    `v = bind(x, rv)`, поэтому голос за флип `rv[j]` — это голос за флип бита `j`
    у всех источников одновременно.
    """
    want_f = want.to(torch.float32)
    y_pos = (site.y > 0).to(torch.float32)
    # (B,S,H): сколько запросов с ошибочным битом j выбрали источник s
    to_src_pos = torch.bmm(site.sel.transpose(1, 2), want_f * y_pos)
    to_src_neg = torch.bmm(site.sel.transpose(1, 2), want_f * (1.0 - y_pos))
    v_pos = (site.aux["v"] > 0).to(torch.float32)
    votes = (to_src_pos * v_pos + to_src_neg * (1.0 - v_pos)).sum((0, 1))
    param.signal.add_(votes.to(torch.int16).unsqueeze(0))
    param.uses.add_(int(site.y.shape[0] * site.y.shape[1]))


@torch.no_grad()
def vote_gate_memory(model, site: BundleSite, want: torch.Tensor,
                     layer: int) -> None:
    """Голоса ключам памяти: поднять слоты, чьи значения ведут к нужному состоянию.

    Ключи влияют только на ВЫБОР слотов, поэтому в поток ошибки значений они не
    попадают — без этой функции `mem_k` не обучается вообще (48% модели).

    Правило точное: `score_r = x · k_r`, флип бита `j` ключа меняет её на
    `-2*x_j*k_r[j]`, то есть флип там, где `k_r[j] != x[j]`, поднимает оценку на
    +2. Цель — слот, чьё значение ближе к желаемому состоянию `x* = x XOR want`.
    """
    mk = model.params[f"mem_k{layer}"]
    mv = model.params[f"mem_v{layer}"]
    x = site.x_in
    x_want = torch.where(want, -x, x)
    fit = bipolar_mm(x_want, mv.sign)            # (N,R): больше = value полезнее
    best = fit.argmax(1)                          # (N,)
    already = site.sel.gather(1, best[:, None]).squeeze(1) > 0
    act = ~already
    if not bool(act.any()):
        return
    rows = best[act]
    disagree = (mk.sign[rows] != x[act]).to(torch.float32)
    mk.signal.index_add_(0, rows, disagree.to(torch.int16))
    mk.uses.index_add_(0, rows,
                       torch.ones(rows.shape[0], 1, device=DEV, dtype=torch.int32))


@torch.no_grad()
def vote_gate_attention(model, site: BundleSite, want: torch.Tensor,
                        layer: int) -> None:
    """Голоса ролям `rq`/`rk`: выбирать полезную позицию вместо текущей.

    Оценка позиции `s` равна `q_t · k_s`, где `q_t = x_t XOR rq`,
    `k_s = x_{s-1} XOR rk`. Флип бита `j` роли `rq` меняет `q_t[j]`, а значит и
    оценку — но **у всех** позиций сразу, на `-2*q_t[j]*k_s[j]`. Поэтому нужен
    не «где q расходится с k», а **разность** вкладов: бит полезен, если он
    поднимает нужную позицию сильнее, чем текущую выбранную.

    Разность вкладов на бит `j`:
        d[j] = q_t[j]*k_sel[j] - q_t[j]*k_best[j]
    Флип полезен там, где `d[j] > 0` (сейчас бит помогает НЕ той позиции).
    Прежняя версия сравнивала q с k напрямую и давала голоса всем битам
    поголовно (1024 из 1024), то есть чистый шум.
    """
    rq, rk = model.params[f"rq{layer}"], model.params[f"rk{layer}"]
    B, T, H = want.shape
    x = site.x_in
    x_want = torch.where(want, -x, x)
    fit = bipolar_bmm(x_want, site.aux["v"])
    causal = torch.ones(T, T, device=DEV, dtype=torch.bool).tril()
    best = fit.masked_fill(~causal, -float("inf")).argmax(-1)          # (B,T)

    sel_idx = site.sel.argmax(-1)                                     # (B,T)
    act = (best != sel_idx) & want.any(-1)                            # (B,T)
    if not bool(act.any()):
        return

    k = site.aux["k"]
    idx_best = best.unsqueeze(-1).expand(-1, -1, H)
    idx_sel = sel_idx.unsqueeze(-1).expand(-1, -1, H)
    k_best = torch.gather(k, 1, idx_best).float()
    k_sel = torch.gather(k, 1, idx_sel).float()
    q = site.aux["q"].float()

    # вклад бита в разрыв «нужная минус текущая»; флип помогает там, где он < 0
    delta_q = q * (k_best - k_sel)
    votes_q = ((delta_q < 0) & act.unsqueeze(-1)).sum((0, 1)).to(torch.int16)
    rq.signal.add_(votes_q.unsqueeze(0))
    rq.uses.add_(int(act.sum()))

    # для `rk` вклад несимметричен: флип бита меняет ТОЛЬКО k_s, а какая позиция
    # затронута — зависит от s. Поднять `best` можно флипом бита, где
    # `q_t[j] != k_best[j]`; понизить `sel` — где `q_t[j] == k_sel[j]`.
    votes_k = ((q != k_best) & act.unsqueeze(-1)).sum((0, 1)).to(torch.int16)
    rk.signal.add_(votes_k.unsqueeze(0))
    rk.uses.add_(int(act.sum()))


@torch.no_grad()
def exact_backward(model, trace: HammingTrace, targets: torch.Tensor,
                   iters: int = 8, gates: bool = True) -> dict:
    """Полный проход маски ошибки от выхода к весам. Ни одного градиента.

    В параллельной схеме на слой приходится ОДИН сайт majority: обе ветки
    голосуют вместе, поэтому маска ошибки делится между ними, а не проходит
    последовательно. В последовательной схеме сайтов два, и маска идёт через них
    по очереди.
    """
    model.zero_signal()
    E, _keep = wanted_state(model, trace, targets, iters)
    B, T = trace.tokens.shape
    H = model.H
    stat = dict(bits_wanted=int(E.sum()), reach=0, blocked=0)
    parallel = model.cfg.parallel_branches

    for l in reversed(range(model.L)):
        mem_site, attn_site = trace.mem[l], trace.attn[l]

        if parallel:
            # один сайт, два набора источников: суммарная сила определяет,
            # достижим ли бит вообще
            n_src = (mem_site.sel.sum(-1)
                     + attn_site.sel.sum(-1).reshape(-1)).to(torch.int32)
            reach_src, reach_res, blocked = split_by_margin(
                E, mem_site.margin, mem_site.w_res, n_src)
            stat["reach"] += int((reach_src | reach_res).sum())
            stat["blocked"] += int(blocked.sum())
            want = reach_src | blocked

            mv = model.params[f"mem_v{l}"]
            vote_bundle_sources(mem_site.sel, want, mem_site.y, mv.sign, mv)
            vote_role_v(attn_site, want.reshape(B, T, H),
                        model.params[f"rv{l}"])
            if gates:
                vote_gate_memory(model, mem_site, E, l)
                vote_gate_attention(model, attn_site, E.reshape(B, T, H), l)

            to_src = torch.bmm(attn_site.sel.transpose(1, 2),
                               want.reshape(B, T, H).to(torch.float32)) > 0
            E = ((reach_res | blocked).reshape(B, T, H) | to_src).reshape(B * T, H)
            continue

        n_src = mem_site.sel.sum(-1).to(torch.int32)
        reach_src, reach_res, blocked = split_by_margin(
            E, mem_site.margin, mem_site.w_res, n_src)
        stat["reach"] += int((reach_src | reach_res).sum())
        stat["blocked"] += int(blocked.sum())
        mv = model.params[f"mem_v{l}"]
        vote_bundle_sources(mem_site.sel, reach_src | blocked, mem_site.y,
                            mv.sign, mv)
        if gates:
            vote_gate_memory(model, mem_site, E, l)
        E = reach_res | blocked

        E3 = E.reshape(B, T, H)
        n_src = attn_site.sel.sum(-1).to(torch.int32)
        reach_src, reach_res, blocked = split_by_margin(
            E3, attn_site.margin, attn_site.w_res, n_src)
        stat["reach"] += int((reach_src | reach_res).sum())
        stat["blocked"] += int(blocked.sum())
        want3 = reach_src | blocked
        vote_role_v(attn_site, want3, model.params[f"rv{l}"])
        if gates:
            vote_gate_attention(model, attn_site, E3, l)
        to_src = torch.bmm(attn_site.sel.transpose(1, 2),
                           want3.to(torch.float32)) > 0
        E = ((reach_res | blocked) | to_src).reshape(B * T, H)

    cb = model.params["codebook"]
    rows = trace.tokens.reshape(-1)
    cb.signal.index_add_(0, rows, E.to(torch.int16))
    cb.uses.index_add_(0, rows,
                       torch.ones(E.shape[0], 1, device=DEV, dtype=torch.int32))

    n = max(stat["reach"] + stat["blocked"], 1)
    stat["exact_frac"] = stat["reach"] / n
    stat["trained"] = sum(1 for p in model.params.values()
                          if int(p.signal.abs().sum()) > 0)
    return stat


@torch.no_grad()
def consensus_flip_masks(model, budget: float) -> tuple[dict, float]:
    """Маски флипов по консенсусу батча: верхние `budget` бит по числу голосов."""
    masks, n = {}, 0
    for name, p in model.params.items():
        k = max(1, int(round(budget * p.numel)))
        top = p.signal.reshape(-1).int().topk(k)
        mask = torch.zeros(p.numel, device=DEV, dtype=torch.bool)
        mask[top.indices[top.values > 0]] = True
        masks[name] = mask.reshape(p.shape)
        n += int(mask.sum())
    return masks, n / model.n_bits


def apply_flips(model, masks: dict) -> None:
    """Применение/откат маски: инверсия самообратна, откат бесплатен."""
    for name, mask in masks.items():
        model.params[name].flip_(mask)
