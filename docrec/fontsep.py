"""Разделение печати по «шрифту»: подписи бланка (старая гравированная антиква: толстые штрихи, засечки, разная толщина) и значения,
которые впечатаны принтером (гротеск типа Arial: одинаковая толщина штриха, крупнее, часто темнее). Подписи и значения нередко налезают друг на друга
(«класс|НЕ УСТАНОВЛЕНО», «№|TCRUE-RU»), поэтому классифицируем не слова, а связные компоненты, а между соседями в строке сглаживаем (Витерби по строке).

Признаки компоненты: относительная толщина штриха sw/h (по distance transform), высота. Два класса оцениваются по самой картинке (EM, 2 гауссианы),
поэтому разметка не привязана к конкретному бланку; если классы не разделяются (мало данных, один шрифт) — возвращается ok=False, и вызывающий код
остаётся на прежнем разделении по яркости (ink.split_ink).
Выход: два изображения (BGR, белый фон) — «только подписи» и «только значения» в разрешении входа + диагностика."""
from __future__ import annotations
import cv2
import numpy as np


def _components(g, H, W):
    gs = cv2.GaussianBlur(g, (0, 0), 0.8)
    bs = int(max(15, round(41 * H / 1500))) | 1
    th = cv2.adaptiveThreshold(gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 16)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    # слипшиеся по вертикали компоненты (хвост буквы подписи касается цифры значения) режем по самому узкому месту
    hs = st[1:, cv2.CC_STAT_HEIGHT]; hs = hs[(hs > 0.010 * H) & (hs < 0.06 * H)]
    Hm = float(np.median(hs)) if len(hs) else 0.0
    cut = False
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if Hm and h > 1.7 * Hm and h <= 0.12 * H and a > 80:
            prof = (lab[y:y + h, x:x + w] == i).sum(1).astype(float)
            lo, hi = int(0.25 * h), int(0.75 * h)
            if hi - lo < 3: continue
            j = lo + int(np.argmin(prof[lo:hi]))
            if prof[j] < 0.35 * prof.max():
                th[y + j - 1:y + j + 1, x:x + w][lab[y + j - 1:y + j + 1, x:x + w] == i] = 0; cut = True
    if cut:
        n, lab, st, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    dt = cv2.distanceTransform(th, cv2.DIST_L2, 5)
    rows = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 25 or h < 0.010 * H or h > 0.06 * H or w > 0.5 * W or a / float(w * h) > 0.9: continue
        m = lab[y:y + h, x:x + w] == i
        d_ = dt[y:y + h, x:x + w][m]
        sw = 2.0 * float(np.median(d_[d_ >= np.percentile(d_, 60)]))
        rows.append((i, x, y, w, h, a, sw, sw / h))
    return th, lab, np.array(rows, dtype=np.float64)


def _em2(X, thr0, iters=30):
    """2 гауссианы (диагональные); resp = P(класс «подпись» — большая относительная толщина)."""
    resp = (X[:, 0] > thr0).astype(float)
    mu = var = pi = None
    for _ in range(iters):
        mu, var, pi = [], [], []
        for c in (0, 1):
            w = (resp if c == 1 else 1 - resp) + 1e-6
            m = (w[:, None] * X).sum(0) / w.sum(); v = (w[:, None] * (X - m) ** 2).sum(0) / w.sum() + 1e-4
            mu.append(m); var.append(v); pi.append(float(w.mean()))
        ll = np.array([-0.5 * (((X - mu[c]) ** 2) / var[c] + np.log(var[c])).sum(1) + np.log(pi[c]) for c in (0, 1)])
        resp = 1 / (1 + np.exp(np.clip(ll[0] - ll[1], -30, 30)))
    return mu, var, pi, ll


def separate(card_bgr: np.ndarray, work_w: int = 1056, monotone: bool = True, use_h: bool = True, hvar: float = 4.0):
    """-> dict(ok, label_img, value_img, label_mask, value_mask, info) в разрешении входа."""
    H0, W0 = card_bgr.shape[:2]
    S = work_w / float(W0)
    im = cv2.resize(card_bgr, None, fx=S, fy=S, interpolation=cv2.INTER_AREA if S < 1 else cv2.INTER_CUBIC)
    H, W = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th, lab, R = _components(g, H, W)
    info = {"components": int(len(R))}
    if len(R) < 60: return dict(ok=False, info=info)
    X = np.stack([R[:, 7], np.log(R[:, 4])] if use_h else [R[:, 7]], 1)
    q = np.clip((X[:, 0] - 0.05) / 0.30 * 255, 0, 255).astype(np.uint8)
    t0, _ = cv2.threshold(q.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); t0 = t0 / 255 * 0.30 + 0.05
    mu, var, pi, ll = _em2(X, t0)
    # какой кластер — «подписи»: толще штрих относительно высоты (засечки, контраст штрихов)
    if mu[1][0] < mu[0][0]: mu, var, pi, ll = mu[::-1], var[::-1], pi[::-1], ll[::-1]
    dprime = float(abs(mu[1][0] - mu[0][0]) / np.sqrt(0.5 * (var[0][0] + var[1][0])))
    info.update(sw_h_values=round(float(mu[0][0]), 3), sw_h_labels=round(float(mu[1][0]), 3), dprime=round(dprime, 2), share_labels=round(pi[1], 2))
    if dprime < 1.2 or min(pi) < 0.12: return dict(ok=False, info=info)
    if use_h:    # высота — слабый довод (строчные буквы значений ниже заглавных подписей): ослабляем её вклад, главный — относительная толщина штриха
        var = [np.array([v[0], v[1] * hvar]) for v in var]
        ll = np.array([-0.5 * (((X - mu[c]) ** 2) / var[c] + np.log(var[c])).sum(1) + np.log(pi[c]) for c in (0, 1)])
    emis = np.clip(ll[1] - ll[0], -6, 6)     # > 0 — подпись
    cy = R[:, 2] + R[:, 4] / 2; Hm = float(np.median(R[:, 4]))
    idx = np.argsort(cy); cur = [idx[0]]; rowsets = []
    for k in idx[1:]:
        if cy[k] - np.mean(cy[cur]) < 0.45 * Hm: cur.append(k)
        else: rowsets.append(cur); cur = [k]
    rowsets.append(cur)
    final = np.zeros(len(R), bool)
    for rs in rowsets:
        rs = sorted(rs, key=lambda k: R[k, 1]); n_ = len(rs)
        cost = np.zeros((n_, 2)); bp = np.zeros((n_, 2), int)
        for t, k in enumerate(rs):
            e = np.array([emis[k], -emis[k]]) * min(1.0, R[k, 5] / 150.0)
            if t == 0: cost[t] = e; continue
            gap = R[k, 1] - (R[rs[t - 1], 1] + R[rs[t - 1], 3])
            sw_ = 1.5 + (2.5 if gap < 0.15 * Hm else 0.0) - (1.0 if gap > 0.6 * Hm else 0.0)
            for c in (0, 1):
                stay = cost[t - 1][c]; alt = cost[t - 1][1 - c] + sw_
                if monotone and c == 1 and gap <= 2.5 * Hm: alt = 1e9      # в строке бланка порядок «подпись, затем значение»: назад (значение -> подпись) только через широкий просвет (вторая колонка)
                if stay <= alt: cost[t][c] = stay + e[c]; bp[t][c] = c
                else: cost[t][c] = alt + e[c]; bp[t][c] = 1 - c
        c = int(np.argmin(cost[-1]))
        for t in range(n_ - 1, -1, -1):
            final[rs[t]] = (c == 1); c = bp[t][c]
    # короткие обрывки (≤2 компоненты) между соседями другого класса — шум классификации: переворачиваем
    for rs in rowsets:
        rs = sorted(rs, key=lambda k: R[k, 1]); changed = True
        while changed:
            changed = False; runs = []
            for k in rs:
                if runs and final[runs[-1][-1]] == final[k]: runs[-1].append(k)
                else: runs.append([k])
            for ri, r in enumerate(runs):
                if len(r) <= 2 and len(runs) > 1:
                    gaps = []
                    if ri > 0: gaps.append(R[r[0], 1] - (R[runs[ri - 1][-1], 1] + R[runs[ri - 1][-1], 3]))
                    if ri < len(runs) - 1: gaps.append(R[runs[ri + 1][0], 1] - (R[r[-1], 1] + R[r[-1], 3]))
                    if min(gaps) < 0.5 * Hm:
                        for k in r: final[k] = not final[k]
                        changed = True; break
    # все чернильные пиксели (в т.ч. мелкие осколки) -> класс ближайшей классифицированной компоненты
    seed = np.zeros((H, W), np.uint8)     # 1 value, 2 label
    ids = R[:, 0].astype(np.int32)
    lut = np.zeros(lab.max() + 1, np.uint8); lut[ids[~final]] = 1; lut[ids[final]] = 2
    seed = lut[lab]
    src = (seed == 0).astype(np.uint8)    # нули — «источники» для distanceTransform
    _, near = cv2.distanceTransformWithLabels((seed == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # near: метка ближайшего «нулевого» пикселя; построим таблицу класса по этим меткам
    cls_of = np.zeros(near.max() + 1, np.uint8)
    zy, zx = np.where(seed > 0)
    # метки нулевых пикселей: у каждого нулевого пикселя метка = его номер; берём класс по его координатам
    zeros_y, zeros_x = np.where(seed > 0)
    # distanceTransformWithLabels присваивает меткам нулевых пикселей; класс ищем через повторный запрос по самим пикселям
    cls_pix = seed[zeros_y, zeros_x]; lbl_pix = near[zeros_y, zeros_x]
    cls_of[lbl_pix] = cls_pix
    ink_cls = cls_of[near]
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    ink = th > 0
    near_ok = dist <= max(4.0, 0.006 * H)
    val_mask = ink & (ink_cls == 1) & near_ok | (seed == 1)
    lab_mask = ink & (ink_cls == 2) & near_ok | (seed == 2)
    k = max(3, int(round(0.004 * H)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    vm = cv2.dilate(val_mask.astype(np.uint8), ker) > 0
    lm = cv2.dilate(lab_mask.astype(np.uint8), ker) > 0
    # слои — из ОРИГИНАЛЬНЫХ яркостей (сглаженные штрихи для OCR), фон белый
    gl = g.astype(np.float32)
    bgv = cv2.medianBlur(g, 61).astype(np.float32)
    norm = np.clip(gl / np.maximum(bgv, 1) * 255.0, 0, 255)        # выравнивание фона/блика
    def layer(m, other):
        # чужие пиксели внутри «маски» (пересечение расширений) не берём
        mm = m & ~(cv2.dilate(other.astype(np.uint8), np.ones((1, 1), np.uint8)) > 0) | (m & ~other)
        out = np.where(m, norm, 255).astype(np.uint8)
        lo = np.percentile(out[m], 2) if m.any() else 0
        out = np.clip((out.astype(np.float32) - lo) * (255.0 / max(1.0, 215 - lo)), 0, 255).astype(np.uint8)
        out[~m] = 255
        return out
    vl = layer(vm & ~lab_mask, lab_mask); ll_ = layer(lm & ~val_mask, val_mask)
    up = lambda a: cv2.resize(a, (W0, H0), interpolation=cv2.INTER_CUBIC if S < 1 else cv2.INTER_AREA) if (W0, H0) != (W, H) else a
    vi = cv2.cvtColor(up(vl), cv2.COLOR_GRAY2BGR); li = cv2.cvtColor(up(ll_), cv2.COLOR_GRAY2BGR)
    info.update(value_comps=int((~final).sum()), label_comps=int(final.sum()))
    return dict(ok=True, label_img=li, value_img=vi, info=info,
                value_mask=cv2.resize(vm.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0,
                label_mask=cv2.resize(lm.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0)
