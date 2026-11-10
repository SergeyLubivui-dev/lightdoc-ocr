"""Привязка значений полей к подписям по ПОСЛЕДОВАТЕЛЬНОСТИ, а не по жёстким зонам шаблона.

1. locate_labels: у типа документа задан упорядоченный список подписей; нечётко ищем их в строках OCR и выбираем набор,
   сохраняющий порядок (динамическое программирование) — пропущенная/искажённая подпись остальным не мешает.
2. assign_values: значения (строки OCR) выравниваются по подписям монотонным выравниванием (как редакционное расстояние):
   подпись может остаться без значения, лишняя строка (шум) — без подписи, подпись может взять 2 соседние строки.
   Систематический сдвиг печати (значения ниже/правее подписей) оценивается и учитывается — зона «плывёт» вместе с печатью.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import numpy as np
from .ocr import Line
from .textutils import _L2C

_EXTRA = {"U": "И", "N": "П", "R": "Я", "G": "Г", "D": "Д", "L": "Л", "Z": "З", "W": "Ш", "F": "Ф", "Y": "У", "J": "Й", "I": "И", "Q": "О", "S": "С", "0": "О"}


def fold2(s: str) -> str:
    """Свёртка для сравнения подписей: регистр, латинские «двойники» -> кириллица, знаки/пробелы убраны."""
    s = s.upper()
    s = "".join(_L2C.get(ch, _EXTRA.get(ch, ch)) for ch in s)
    s = s.replace("Ё", "Е").replace("Й", "И")
    return re.sub(r"[^0-9А-Я]", "", s)


@dataclass
class LabelSpec:
    key: str
    patterns: list[str]
    multi: int = 1            # сколько строк значения может занимать поле
    optional: bool = True


def match_score(spec: LabelSpec, text: str) -> float:
    f = fold2(text)
    if not f: return 0.0
    best = 0.0
    for p in spec.patterns:
        pf = fold2(p)
        if len(pf) <= 7:
            if len(f) > len(pf) + 6: continue
            r = SequenceMatcher(None, pf, f).ratio()
        else:
            if len(f) <= len(pf):
                r = SequenceMatcher(None, pf, f).ratio()
            else:
                r = 0.0
                for n in (len(pf) - 2, len(pf), len(pf) + 2):
                    for i in range(0, max(1, len(f) - n + 1)):
                        r = max(r, SequenceMatcher(None, pf, f[i:i + n]).ratio())
                        if r > 0.97: break
        best = max(best, r)
    return best


def locate_labels(lines: list[Line], specs: list[LabelSpec], thr: float = 0.62) -> dict[str, tuple[Line, float]]:
    """Монотонное (по y) сопоставление подписей и строк: максимум суммарного сходства."""
    ls = sorted(lines, key=lambda l: l.cy)
    L, N = len(specs), len(ls)
    S = np.zeros((L, N))
    for i, sp in enumerate(specs):
        for j, l in enumerate(ls): S[i, j] = match_score(sp, l.text)
    f = np.zeros((L + 1, N + 1)); back = {}
    for i in range(1, L + 1):
        for j in range(1, N + 1):
            best, bk = f[i, j - 1], "skip_line"
            if f[i - 1, j] > best: best, bk = f[i - 1, j], "skip_label"
            if S[i - 1, j - 1] >= thr and f[i - 1, j - 1] + S[i - 1, j - 1] > best:
                best, bk = f[i - 1, j - 1] + S[i - 1, j - 1], "match"
            f[i, j] = best; back[(i, j)] = bk
    out = {}; i, j = L, N
    while i > 0 and j > 0:
        bk = back[(i, j)]
        if bk == "match": out[specs[i - 1].key] = (ls[j - 1], float(S[i - 1, j - 1])); i -= 1; j -= 1
        elif bk == "skip_line": j -= 1
        else: i -= 1
    return out


def label_positions(specs: list[LabelSpec], found: dict[str, tuple[Line, float]]) -> list[tuple[float, bool]]:
    """cy каждой подписи; для не найденных — интерполяция/экстраполяция по соседям (индекс -> y)."""
    idx = [i for i, s in enumerate(specs) if s.key in found]
    ys = [found[specs[i].key][0].cy for i in idx]
    if len(idx) == 0: return [(0.0, False)] * len(specs)
    if len(idx) == 1: sp = 0.04
    else: sp = float(np.median(np.diff(ys) / np.diff(idx)))
    out = []
    for i in range(len(specs)):
        if i in idx: out.append((ys[idx.index(i)], True)); continue
        lo = max([k for k in idx if k < i], default=None); hi = min([k for k in idx if k > i], default=None)
        if lo is not None and hi is not None:
            y = found[specs[lo].key][0].cy + (found[specs[hi].key][0].cy - found[specs[lo].key][0].cy) * (i - lo) / (hi - lo)
        elif lo is not None: y = found[specs[lo].key][0].cy + sp * (i - lo)
        else: y = found[specs[hi].key][0].cy - sp * (hi - i)
        out.append((y, False))
    return out


def assign_values(specs: list[LabelSpec], pos: list[tuple[float, bool]], values: list[Line],
                  win=(-0.3, 1.15), skip_label=0.9, skip_value=0.55, multi_gap=0.03) -> tuple[dict[str, list[Line]], dict]:
    """Монотонное выравнивание подписей и строк значений. Возвращает {key: [строки значения]} и диагностику (сдвиг)."""
    vals = sorted(values, key=lambda v: (v.cy, v.x0))
    L, N = len(specs), len(vals)
    ys = [p[0] for p in pos]
    spac = [(ys[i + 1] - ys[i]) if i + 1 < L else (ys[i] - ys[i - 1] if i else 0.04) for i in range(L)]
    spac = [max(0.02, s) for s in spac]

    def run(shift):
        def cost(i, j):
            t = (vals[j].cy - ys[i]) / spac[i] - shift
            lo, hi = win
            if t < lo: return (lo - t) ** 2 * 2.0
            if t > hi: return (t - hi) ** 2 * 2.0
            return 0.05 * abs(t)
        INF = 1e9
        f = np.full((L + 1, N + 1), INF); f[0, 0] = 0; bk = {}
        for i in range(L + 1):
            for j in range(N + 1):
                if f[i, j] >= INF: continue
                c = f[i, j]
                if i < L and c + skip_label < f[i + 1, j]: f[i + 1, j] = c + skip_label; bk[(i + 1, j)] = (i, j, 0)
                if j < N and c + skip_value < f[i, j + 1]: f[i, j + 1] = c + skip_value; bk[(i, j + 1)] = (i, j, -1)
                if i < L and j < N:
                    for k in range(1, specs[i].multi + 1):
                        if j + k > N: break
                        if k > 1 and abs(vals[j + k - 1].cy - vals[j].cy) > multi_gap: break
                        cc = c + cost(i, j) + 0.12 * (k - 1)
                        if cc < f[i + 1, j + k]: f[i + 1, j + k] = cc; bk[(i + 1, j + k)] = (i, j, k)
        i, j = L, N; assign = {}
        while (i, j) != (0, 0):
            pi, pj, k = bk[(i, j)]
            if k > 0: assign[specs[pi].key] = vals[pj:pj + k]
            i, j = pi, pj
        return assign, float(f[L, N])

    assign, c0 = run(0.0)
    ts = [(v[0].cy - ys[specs.index(next(s for s in specs if s.key == k))]) / spac[specs.index(next(s for s in specs if s.key == k))] for k, v in assign.items()]
    shift = float(np.median(ts)) if ts else 0.0
    shift = float(np.clip(shift, -0.3, 0.6))
    assign, c1 = run(shift)
    return assign, {"shift_rel": round(shift, 3), "cost": round(c1, 3), "labels_with_value": len(assign)}
