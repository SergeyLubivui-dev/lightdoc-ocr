"""Несколько разворотов на одном листе (скан/копия двух разворотов рядом, друг под другом): поиск «колонок/строк» по пустым промежуткам (XY-cut по маске «чернил»).
Один разворот в кадре → возвращается как есть (весь кадр). Работает на листе с однородным светлым фоном; фото на столе/ткани (фон везде тёмный) не делится."""
from __future__ import annotations
import cv2
import numpy as np

MIN_GAP = 0.018      # минимальная ширина промежутка (доля стороны региона)
GAP_DENS = 0.05      # плотность «чернил» внутри промежутка (после отбрасывания краёв скана)
MIN_SIDE = 0.2       # минимальная сторона куска (доля стороны кадра)
MIN_AREA = 0.08      # минимальная площадь куска (доля кадра)


def _ink_mask(img: np.ndarray, side: int = 512):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape; s = side / max(H, W)
    sm = cv2.resize(g, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
    bg = float(np.percentile(sm, 90))
    m = (cv2.GaussianBlur(sm, (0, 0), 1.0) < bg - 30).astype(np.uint8)
    e = max(1, int(0.015 * min(sm.shape)))          # рамка скана по краю кадра — не документ
    m[:e] = 0; m[-e:] = 0; m[:, :e] = 0; m[:, -e:] = 0
    return m, s, bg


def _gaps(dens: np.ndarray, min_len: int):
    out = []; i = 0; n = len(dens)
    while i < n:
        if dens[i] <= GAP_DENS:
            j = i
            while j < n and dens[j] <= GAP_DENS: j += 1
            if j - i >= min_len: out.append((i, j))
            i = j
        else: i += 1
    return out


def _tight(m, x0, y0, x1, y1):
    sub = m[y0:y1, x0:x1]
    if sub.sum() == 0: return None
    cols = np.where(sub.mean(0) > 0.02)[0]; rows = np.where(sub.mean(1) > 0.02)[0]
    if len(cols) == 0 or len(rows) == 0: return None
    return x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1


def _cut(m, box, depth=0):
    x0, y0, x1, y1 = box
    sub = m[y0:y1, x0:x1]; h, w = sub.shape
    if depth >= 3: return [box]
    best = None
    for axis in (1, 0):             # сначала вертикальные промежутки (колонки), затем горизонтальные
        dens = np.convolve(sub.mean(axis=0 if axis == 1 else 1), np.ones(3) / 3, mode="same")
        L = w if axis == 1 else h
        for a, b in _gaps(dens, max(2, int(MIN_GAP * L))):
            if a < 0.12 * L or b > 0.88 * L: continue          # промежуток у самого края региона — это поле, а не разделитель
            sc = b - a
            if best is None or sc > best[0]: best = (sc, axis, a, b)
        if best is not None: break
    if best is None: return [box]
    _, axis, a, b = best
    if axis == 1: parts = [(x0, y0, x0 + a, y1), (x0 + b, y0, x1, y1)]
    else: parts = [(x0, y0, x1, y0 + a), (x0, y0 + b, x1, y1)]
    out = []
    for p in parts:
        t = _tight(m, *p)
        if t is not None: out += _cut(m, t, depth + 1)
    return out


def _plausible(keep) -> bool:
    """Куски должны быть похожи на листы-развороты: пропорции 1.15–2.1 (в любой ориентации), близкие размеры, выровнены по одной оси.
    Страницы с колонками текста, фото на столе и один разворот с «пустым» местом сюда не попадают."""
    areas = []
    for (x0, y0, x1, y1) in keep:
        w, h = x1 - x0, y1 - y0; a = max(w, h) / max(1, min(w, h))
        if not (1.15 <= a <= 2.1): return False
        areas.append(w * h)
    if min(areas) < 0.45 * max(areas): return False
    xs = [(b[0], b[2]) for b in keep]; ys = [(b[1], b[3]) for b in keep]
    ov = lambda iv: (min(i[1] for i in iv) - max(i[0] for i in iv)) / max(1, min(i[1] - i[0] for i in iv))
    return ov(ys) >= 0.7 or ov(xs) >= 0.7


def split_spreads(img: np.ndarray) -> list[dict]:
    """-> [{bbox:(x0,y0,x1,y1) в пикселях исходника, crop}] в порядке слева направо/сверху вниз. Пусто/один элемент — документ один (весь кадр)."""
    H, W = img.shape[:2]
    m, s, bg = _ink_mask(img)
    h, w = m.shape
    if bg < 150: return [dict(bbox=(0, 0, W, H), crop=img)]            # тёмный фон — не лист
    if m.mean() < 0.03: return [dict(bbox=(0, 0, W, H), crop=img)]
    t = _tight(m, 0, 0, w, h)
    if t is None: return [dict(bbox=(0, 0, W, H), crop=img)]
    parts = _cut(m, t)
    keep = []
    for (x0, y0, x1, y1) in parts:
        if (x1 - x0) < MIN_SIDE * w or (y1 - y0) < MIN_SIDE * h or (x1 - x0) * (y1 - y0) < MIN_AREA * w * h: continue
        keep.append((x0, y0, x1, y1))
    if len(keep) < 2 or not _plausible(keep): return [dict(bbox=(0, 0, W, H), crop=img)]
    keep.sort(key=lambda b: (round(b[1] / (0.5 * h)), b[0]))
    out = []
    for (x0, y0, x1, y1) in keep:
        pad = int(0.004 * max(w, h))
        X0, Y0, X1, Y1 = [int(round(v / s)) for v in (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))]
        X1, Y1 = min(W, X1), min(H, Y1)
        out.append(dict(bbox=(X0, Y0, X1, Y1), crop=img[Y0:Y1, X0:X1]))
    return out
