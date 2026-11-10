"""Блики на ламинате: маска пересвеченных областей и пометка полей, значения которых лежат под бликом (partial, без «дорисовки»)."""
from __future__ import annotations
import cv2
import numpy as np


def glare_mask(card: np.ndarray, min_area: float = 0.0015):
    """Пересвеченные (почти белые и малонасыщенные) крупные пятна. -> (mask uint8 0/255, fraction)."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    # у бланка светлая бумага: блик — это значительно ярче и белее фона бумаги
    v = hsv[..., 2].astype(np.float32); s = hsv[..., 1].astype(np.float32)
    paper = float(np.percentile(v, 60))
    m = ((v >= max(246.0, paper + 8)) & (s <= 18)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m); H, W = m.shape
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area * H * W: out[lab == i] = 255
    if out.any(): out = cv2.dilate(out, np.ones((9, 9), np.uint8))
    return out, float((out > 0).mean())


def flag_fields(F, specs, pos, mask, ex, x_range=(0.3, 1.0), band=0.028, thr=0.25):
    """Поле partial, если в полосе его подписи (по y) блик закрывает >= thr области значений. Значения не придумываем."""
    H, W = mask.shape
    frac = float((mask > 0).mean())
    ys, xs = np.where(mask > 0)
    ex.extra["glare"] = {"fraction": round(frac, 4), "bbox_norm": ([round(float(xs.min()) / W, 3), round(float(ys.min()) / H, 3), round(float(xs.max()) / W, 3), round(float(ys.max()) / H, 3)] if len(xs) else None), "flagged": []}
    if not len(xs): return
    for sp, p in zip(specs, pos):
        if sp.key not in F: continue
        y0, y1 = int(max(0, p[0] - band * 0.4) * H), int(min(1, p[0] + band * 1.6) * H)
        zone = mask[y0:y1, int(x_range[0] * W):int(x_range[1] * W)]
        if zone.size and (zone > 0).mean() >= thr:
            f = F[sp.key]; f.partial = True; f.confidence = min(f.confidence, 0.5)
            f.note = ((f.note + "; ") if f.note else "") + "блик на ламинате закрывает часть значения — возможна неполная/неточная печать"
            ex.extra["glare"]["flagged"].append(sp.key)
