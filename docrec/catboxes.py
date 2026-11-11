"""Поле 9 водительского удостоверения (лицевая сторона): цепочка рамок с буквами категорий.

Алгоритм (без OCR как основы):
 1. по морфологии (вертикальные отрезки рамок длиной ≈ высоте рамки; буквенные штрихи короче) находим левые/правые стороны рамок;
 2. сторонки группируем в строки (по y), в строке — в рамки (внутренняя ширина большая, зазор между соседними рамками малый);
    верх/низ рамки проверяем по горизонтальной «чернильной» линии (хотя бы один из двух);
 3. содержимое каждой рамки сопоставляется с шаблонами 16 категорий (шаблоны отрисованы шрифтами из системы, NCC + штраф за пропорции);
 4. рамки по порядку (строка за строкой) выравниваются по ФИКСИРОВАННОМУ порядку категорий на карточке (монотонное DP — пропуски разрешены,
    повторы и перестановки запрещены), OCR/шаблон — только оценка соответствия;
 5. открыта/закрыта: плотность чернил внутри рамки относительно соседей (Otsu по плотностям, если разброс большой; иначе все найденные рамки открыты).
Возвращает список рамок: category, open, confidence, box, row + общую уверенность.
"""
from __future__ import annotations
import os
import numpy as np
import cv2

CARD_ORDER = ["A", "A1", "B", "B1", "C", "C1", "D", "D1", "BE", "CE", "C1E", "DE", "D1E", "M", "Tm", "Tb"]
_TH, _TW = 24, 48
_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
]
_TPL = None


def _norm_mask(m: np.ndarray) -> tuple[np.ndarray, float]:
    """бинарная маска (0/255) -> обрезка по чернилам -> размер 24x48, лёгкое размытие; + пропорция ширина/высота"""
    ys, xs = np.nonzero(m > 0)
    if len(xs) < 6: return np.zeros((_TH, _TW), np.float32), 0.0
    c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = c.shape[1] / max(1, c.shape[0])
    r = cv2.resize(c.astype(np.float32) / 255.0, (_TW, _TH), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(r, (3, 3), 0.8), float(a)


def _templates():
    global _TPL
    if _TPL is not None: return _TPL
    from PIL import Image, ImageDraw, ImageFont
    out = {c: [] for c in CARD_ORDER}
    for fp in _FONTS:
        if not os.path.exists(fp): continue
        try: ft = ImageFont.truetype(fp, 64)
        except Exception: continue
        for c in CARD_ORDER:
            im = Image.new("L", (400, 120), 0); ImageDraw.Draw(im).text((20, 10), c, fill=255, font=ft)
            m = (np.array(im) > 128).astype(np.uint8) * 255
            out[c].append(_norm_mask(m))
    _TPL = out
    return out


def classify_glyph(mask: np.ndarray) -> dict[str, float]:
    """оценка 0..1 для каждой категории по маске чернил внутри рамки"""
    f, a = _norm_mask(mask)
    if a == 0.0: return {c: 0.0 for c in CARD_ORDER}
    fz = f - f.mean(); nf = np.sqrt((fz ** 2).sum()) + 1e-6
    res = {}
    for c, lst in _templates().items():
        best = -1.0
        for t, ta in lst:
            tz = t - t.mean(); ncc = float((fz * tz).sum() / (nf * (np.sqrt((tz ** 2).sum()) + 1e-6)))
            ncc -= 0.6 * abs(np.log(max(a, 1e-3) / max(ta, 1e-3)))     # пропорции: «C1E» ≠ «C»
            best = max(best, ncc)
        res[c] = best
    return res


# ----------------------------------------------------------------------------------- поиск рамок
def _ink(gray: np.ndarray) -> np.ndarray:
    bg = cv2.medianBlur(gray, 31) if min(gray.shape) > 31 else gray
    d = bg.astype(np.int16) - gray.astype(np.int16)
    thr = max(14, int(np.percentile(d, 93) * 0.55))
    return (d > thr).astype(np.uint8) * 255


def find_boxes(card: np.ndarray, region=(0.36, 0.80, 0.995, 1.0), h_rel=0.060):
    """-> список рамок (x0,y0,x1,y1) в координатах карточки + отладочная информация"""
    H, W = card.shape[:2]
    rx0, ry0, rx1, ry1 = int(region[0] * W), int(region[1] * H), int(region[2] * W), min(H, int(region[3] * H))
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
    g = g[ry0:ry1, rx0:rx1]
    ink = _ink(g)
    hb = h_rel * H
    ink_c = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    kh = max(9, int(round(0.70 * hb)))
    vert = cv2.morphologyEx(ink_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(vert, connectivity=8)
    bars = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > 0.22 * hb or h < 0.6 * hb: continue
        if h > 1.5 * hb:       # левые/правые стороны двух строк слиплись (зазор между строками 2-5 px + размытие): режем по минимуму чернил
            if h > 3.4 * hb: continue
            k = int(round(h / (1.07 * hb))); prof = ink[y:y + h, x:x + w].mean(axis=1).astype(np.float32); cuts = [0]
            for j in range(1, k):
                c0 = int(j * h / k); lo, hi = max(cuts[-1] + int(0.5 * hb), c0 - int(0.25 * hb)), min(h - 1, c0 + int(0.25 * hb))
                if hi > lo: cuts.append(lo + int(np.argmin(prof[lo:hi + 1])))
            cuts.append(h)
            for a_, b_ in zip(cuts[:-1], cuts[1:]):
                if b_ - a_ >= 0.6 * hb: bars.append((x + w / 2.0, y + a_, y + b_))
            continue
        bars.append((x + w / 2.0, y, y + h))
    if len(bars) < 2: return [], {"bars": len(bars)}
    bars.sort(key=lambda b: (b[1] + b[2]) / 2)
    rows = []                                  # группировка в строки по середине по y
    for b in bars:
        cy = (b[1] + b[2]) / 2
        if rows and abs(cy - np.mean([(r[1] + r[2]) / 2 for r in rows[-1]])) < 0.5 * hb: rows[-1].append(b)
        else: rows.append([b])
    boxes = []
    hl = cv2.morphologyEx(ink_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(0.30 * hb)), 1)))
    for ri, r in enumerate(rows):
        r = sorted(r, key=lambda b: b[0]); y0 = float(np.median([b[1] for b in r])); y1 = float(np.median([b[2] for b in r]))
        i = 0
        while i < len(r) - 1:
            L, R = r[i], r[i + 1]; d = R[0] - L[0]
            if d < 0.30 * hb:        # слишком близко — это сторона соседней рамки (R_i→L_{i+1}) или двойная линия: сдвигаемся
                i += 1; continue
            if d > 3.2 * hb: i += 1; continue
            xa, xb = int(L[0]), int(R[0]); ya, yb = int(y0), int(min(y1, g.shape[0] - 1))
            top = float((hl[max(0, ya - 2):ya + 3, xa:xb] > 0).any(axis=0).mean()) if xb > xa else 0.0
            bot = float((hl[max(0, yb - 2):yb + 3, xa:xb] > 0).any(axis=0).mean()) if xb > xa else 0.0
            if max(top, bot) >= 0.45:
                boxes.append({"row": ri, "box": (xa + rx0, ya + ry0, xb + rx0, yb + ry0), "frame": round(max(top, bot), 2)})
                i += 2
            else:
                i += 1
    # нумерация строк заново (после отбрасывания пустых)
    rr = sorted({b["row"] for b in boxes}); idx = {r: k for k, r in enumerate(rr)}
    for b in boxes: b["row"] = idx[b["row"]]
    boxes.sort(key=lambda b: (b["row"], b["box"][0]))
    return boxes, {"bars": len(bars), "rows": len(rr)}


def _align(scores: list[dict[str, float]]):
    """монотонное выравнивание рамок по фиксированному порядку CARD_ORDER: каждая рамка получает категорию строго правее предыдущей"""
    n, m = len(scores), len(CARD_ORDER)
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG); dp[0, :] = 0.0       # dp[i][j]: первые i рамок, последняя занимает позицию < j
    bk = np.zeros((n + 1, m + 1), np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            keep = dp[i, j - 1]                              # позиция j-1 не занята
            take = dp[i - 1, j - 1] + scores[i - 1][CARD_ORDER[j - 1]]
            if take >= keep: dp[i, j] = take; bk[i, j] = 1
            else: dp[i, j] = keep; bk[i, j] = 0
    out = [None] * n; i, j = n, m
    if dp[n, m] <= NEG / 2: return None
    while i > 0 and j > 0:
        if bk[i, j] == 1: out[i - 1] = CARD_ORDER[j - 1]; i -= 1; j -= 1
        else: j -= 1
    return out


def read_categories(card: np.ndarray, region=(0.36, 0.80, 0.995, 1.0), cell=None, hint=None):
    """-> dict(boxes=[{category, open, confidence, box, row, density}], categories=[...], confidence, note) или None, если цепочки рамок нет"""
    H, W = card.shape[:2]
    boxes, info = find_boxes(card, region)
    if not boxes or len(boxes) > len(CARD_ORDER): return None
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
    scores, dens = [], []
    for b in boxes:
        x0, y0, x1, y1 = b["box"]; ins = max(3, int(0.09 * (y1 - y0)))
        cell = g[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
        if cell.size == 0 or min(cell.shape) < 6: scores.append({c: 0.0 for c in CARD_ORDER}); dens.append(0.0); continue
        t, m = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # убрать остатки рамки по краям ячейки
        m[:2, :] = 0; m[-2:, :] = 0; m[:, :2] = 0; m[:, -2:] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        scores.append(classify_glyph(m))
        # «сила» чернил рамки+буквы: фон минус самые тёмные пиксели ячейки с рамкой (слабая/серая печать = закрытая категория)
        full = g[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2]
        dens.append(float(np.median(full) - np.percentile(full, 3)))
    if hint and len(hint) == len(boxes):      # OCR строки дал столько же токенов, сколько рамок: это подсказка (бонус) классу, а не приговор
        for i, hcat in enumerate(hint):
            if hcat in scores[i]: scores[i][hcat] += 0.3
    al = _align(scores)
    if al is None: return None
    ocr_notes = {}
    if cell is not None:      # OCR ячейки — только арбитр для неуверенных рамок: бонус классу, который она назвала, затем пере-выравнивание по порядку
        from .doctypes.driver_license import _norm_cat
        def _conf(i):
            sc = scores[i]; o = sorted(sc.values(), reverse=True); return sc[al[i]] - (o[1] if len(o) > 1 and o[0] == sc[al[i]] else o[0])
        weak = [i for i in range(len(boxes)) if _conf(i) < 0.18]
        for i in weak:
            x0, y0, x1, y1 = boxes[i]["box"]; ins = max(3, int(0.09 * (y1 - y0))); im = g[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
            if im.size == 0: continue
            votes = {}
            for psm in (13, 7):
                try: t, _c = cell(im, "ABCDEMTb1m", psm)
                except Exception: t = None
                n = _norm_cat(t) if t else None
                if n: votes[n] = votes.get(n, 0) + 1
            for n, v in votes.items():
                if n in scores[i]: scores[i][n] += 0.22 * v; ocr_notes[i] = n
        if ocr_notes:
            al2 = _align(scores)
            if al2 is not None: al = al2
    dens_a = np.array(dens)
    open_flags = [True] * len(boxes)
    if len(boxes) >= 3 and dens_a.max() > 0 and dens_a.min() < 0.6 * dens_a.max():     # заметный разброс силы печати: слабые = «закрытые» (серые буквы)
        srt = np.sort(dens_a); k = int(np.argmax(np.diff(srt)))                       # порог — самый большой разрыв в отсортированных значениях
        if srt[k + 1] > 1.35 * srt[k]:
            thr = 0.5 * (srt[k] + srt[k + 1]); open_flags = [bool(d >= thr) for d in dens_a]
    res = []
    pos = [CARD_ORDER.index(c) for c in al]
    for i, (b, c, sc, d, o) in enumerate(zip(boxes, al, scores, dens, open_flags)):
        s_best = sc[c]
        lo = pos[i - 1] if i > 0 else -1; hi = pos[i + 1] if i + 1 < len(pos) else len(CARD_ORDER)
        alts = [sc[k] for j, k in enumerate(CARD_ORDER) if lo < j < hi and k != c]        # альтернативы, допустимые порядком соседей
        margin = s_best - max(alts) if alts else 0.5
        conf = float(np.clip(0.6 * s_best + 0.7 * margin + 0.25 * b["frame"], 0.0, 1.0))
        res.append({"category": c, "open": o, "confidence": round(conf, 2), "box": [int(v) for v in b["box"]], "row": b["row"], "ink_strength": round(float(d), 1), "margin": round(float(margin), 2), **({"ocr_vote": ocr_notes[i]} if i in ocr_notes else {})})
    cats = [r["category"] for r in res if r["open"]]
    return {"boxes": res, "categories": cats, "confidence": round(float(np.mean([r["confidence"] for r in res])), 2), "info": info}
