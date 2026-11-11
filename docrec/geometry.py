"""Поиск границ карточки (4 угла), перспективная коррекция."""
from __future__ import annotations
import os
import cv2
import numpy as np

ID1_ASPECT = 85.6 / 54.0  # 1.585


def order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _line_intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1], np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1], np.float32)


def _refine_quad(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Уточняем углы: по точкам контура подгоняем 4 прямые (устойчиво к скруглённым углам)."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    diag = float(np.linalg.norm(quad[0] - quad[2]))
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a; L = np.linalg.norm(ab)
        if L < 1:
            return quad
        u = ab / L; n = np.array([-u[1], u[0]])
        rel = pts - a
        t = rel @ u; dist = np.abs(rel @ n)
        sel = pts[(dist < 0.03 * diag) & (t > 0.12 * L) & (t < 0.88 * L)]
        if len(sel) < 8:
            return quad
        lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = []
    for i in range(4):
        p = _line_intersect(lines[i - 1], lines[i])
        if p is None:
            return quad
        out.append(p)
    out = np.float32(out)
    # sanity: не уехали далеко от исходного
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.08 * diag:
        return quad
    return out


def snap_edges(gray: np.ndarray, quad: np.ndarray, iters: int = 2) -> np.ndarray:
    """Подтягиваем стороны четырёхугольника к реальным краям по градиенту яркости (точность до пикселя)."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h, w = g.shape
    diag = float(np.linalg.norm(quad[0] - quad[2]))
    q = quad.copy()
    for it in range(iters):
        r = max(4, int(diag * (0.035 if it == 0 else 0.012)))
        offs = np.arange(-r, r + 1)
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            ab = b - a; L = np.linalg.norm(ab)
            if L < 10: return quad
            u = ab / L; n = np.array([-u[1], u[0]], np.float32)
            ts = np.linspace(0.12, 0.88, 36)
            pts = []; peaks = []
            for t in ts:
                p0 = a + ab * t
                xs = np.clip((p0[0] + offs * n[0]).round().astype(int), 0, w - 1)
                ys = np.clip((p0[1] + offs * n[1]).round().astype(int), 0, h - 1)
                grad = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1]) * (1 - 0.35 * np.abs(offs) / r)
                k = int(np.argmax(grad))
                pts.append(p0 + n * offs[k]); peaks.append(grad[k])
            pts = np.array(pts, np.float32); peaks = np.array(peaks)
            keep = peaks > max(8.0, 0.4 * np.median(peaks))
            if keep.sum() < 10: return quad
            lines.append(cv2.fitLine(pts[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        new = []
        for i in range(4):
            p_ = _line_intersect(lines[i - 1], lines[i])
            if p_ is None: return quad
            new.append(p_)
        new = np.float32(new)
        if np.max(np.linalg.norm(new - q, axis=1)) > 0.07 * diag: return quad
        q = new
    return q


def _quad_aspect(q: np.ndarray) -> float:
    w = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    h = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    return max(w, h) / max(1e-6, min(w, h))


def _candidates_from_mask(mask, area_img):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.04 * area_img:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(np.float32); break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        quad = order_corners(quad)
        quad = order_corners(_refine_quad(c, quad))
        qa = cv2.contourArea(quad)
        if qa < 1:
            continue
        fill = min(area, qa) / max(area, qa)
        out.append((quad, area / area_img, fill))
    return out


def find_card_quad(img: np.ndarray, target_aspects=(ID1_ASPECT,), max_side=800):
    """Возвращает (quad[4,2] в координатах исходного изображения, score 0..1, метод)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    blur = cv2.GaussianBlur(small, (0, 0), 2.0)
    masks = {}
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks["otsu_light"] = m
    masks["otsu_dark"] = 255 - m
    # расстояние от медианного цвета рамки кадра (фон)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(3, int(0.03 * min(sh, sw)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist8 = np.clip(dist * 255 / max(1.0, dist.max()), 0, 255).astype(np.uint8)
    _, m2 = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks["bgdist"] = m2
    # Canny как запасной вариант
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m3 = np.zeros_like(gray)
    cv2.drawContours(m3, cnts, -1, 255, -1)
    masks["canny"] = m3

    best = (None, 0.0, None)
    area_img = float(sh * sw)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    for name, mk in masks.items():
        for quad, area_ratio, fill in _candidates_from_mask(mk, area_img):
            asp = _quad_aspect(quad)
            asp_err = min(abs(asp - a) / a for a in target_aspects)
            if asp_err > 0.22:
                continue
            # не касается ли границ кадра (признак того, что это фон, а не карта)
            touch = sum([(quad[:, 0] < 2).any(), (quad[:, 0] > sw - 3).any(), (quad[:, 1] < 2).any(), (quad[:, 1] > sh - 3).any()])
            score = (0.45 * min(1.0, area_ratio / 0.25) + 0.35 * fill + 0.20 * (1 - asp_err / 0.22)) - 0.25 * touch
            if score > best[1]:
                best = (quad, score, name)
    if best[0] is not None:
        snapped = order_corners(snap_edges(gray_s, best[0]))
        if cv2.contourArea(snapped) > 0.7 * cv2.contourArea(best[0]):
            best = (snapped, best[1], best[2] + "+snap")
    if best[0] is None:
        # fallback: весь кадр
        q = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
        return q / s, 0.0, "fallback_full_frame"
    return best[0] / s, float(min(1.0, best[1])), best[2]


def warp_card(img: np.ndarray, quad: np.ndarray, size=(1026, 648)) -> np.ndarray:
    """Перспективное выпрямление. quad: TL,TR,BR,BL. Длинная сторона квада = длинная сторона `size`."""
    q = order_corners(quad)
    w = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    h = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    W, H = size
    # длинная сторона карточки совпадает с длинной стороной целевого размера (ID-1 — горизонталь, СТС — вертикаль)
    if (h > w) == (W >= H):
        q = np.float32([q[3], q[0], q[1], q[2]])
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    up = max(W, H) / max(w, h, 1.0)
    if up > 2.2:     # мелкая карточка в кадре: сильное увеличение — Lanczos + мягкая резкость (иначе цифры расплываются)
        out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        bl = cv2.GaussianBlur(out, (0, 0), 1.6)
        return cv2.addWeighted(out, 1.6, bl, -0.6, 0)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def refine_quad_edges(img: np.ndarray, quad: np.ndarray, win_rel=0.045, segs=14):
    """Подтягивает каждую сторону четырёхугольника к настоящему краю карточки: вдоль нормали ищем самый резкий перепад (градиент Lab) —
    край карточки резче, чем тень/ореол на листе. Точки по сегментам стороны -> Huber-прямая -> пересечения. Возвращает (quad, ok)."""
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.0)
    grad = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3); grad += gx * gx + gy * gy
    grad = np.sqrt(grad)
    short = float(min(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[1] - quad[2])))
    cur = np.float32(quad); area0 = abs(cv2.contourArea(cur))
    for it, win in enumerate((win_rel, win_rel * 0.5)):
        W = max(4, int(win * short)); lines = []
        for i in range(4):
            a, b = cur[i], cur[(i + 1) % 4]; ab = b - a; L = float(np.linalg.norm(ab)); u = ab / L; nrm = np.array([-u[1], u[0]], np.float32)
            ctr = cur.mean(axis=0)
            if float((ctr - a) @ nrm) < 0: nrm = -nrm           # нормаль внутрь
            pts = []; g_ref = []; g_org = []
            for k in range(segs):
                t0 = 0.16 + 0.68 * k / segs; t1 = 0.16 + 0.68 * (k + 1) / segs
                prof = np.zeros(2 * W + 1, np.float32)
                for t in np.linspace(t0, t1, 7):
                    base = a + u * (t * L)
                    xs = base[0] + nrm[0] * np.arange(-W, W + 1); ys = base[1] + nrm[1] * np.arange(-W, W + 1)
                    prof += cv2.remap(grad, xs.reshape(1, -1).astype(np.float32), ys.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
                prof /= 7
                if prof.max() < 6.0: continue
                pm_ = np.convolve(prof, np.ones(3) / 3, mode="same"); thr_ = 0.5 * pm_.max()       # самый ВНЕШНИЙ заметный пик (нормаль внутрь): детали у края (таблица, цифры) не перетягивают сторону
                j = next((q for q in range(1, len(pm_) - 1) if pm_[q] >= thr_ and pm_[q] >= pm_[q - 1] and pm_[q] >= pm_[q + 1]), int(np.argmax(prof)))
                pts.append(a + u * (0.5 * (t0 + t1) * L) + nrm * (j - W)); g_ref.append(float(pm_[j])); g_org.append(float(pm_[max(0, W - 1):W + 2].max()))
            if len(pts) < 6: return quad, False
            # переносим сторону, только если новый край заметно резче текущего (тень/ореол даёт мягкий перепад; настоящий край карточки — резкий)
            if np.mean(g_ref) >= 1.4 * np.mean(g_org): lines.append(cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
            else: lines.append(np.float32([u[0], u[1], a[0], a[1]]))
        new = []
        for i in range(4):
            p_ = _line_intersect(lines[i - 1], lines[i])
            if p_ is None: return quad, False
            new.append(p_)
        new = order_corners(np.float32(new)); a1 = abs(cv2.contourArea(new))
        if not (0.55 * area0 <= a1 <= 1.08 * area0) or np.max(np.linalg.norm(new - cur, axis=1)) > 0.2 * short: return quad, False
        cur = new
    return cur.astype(np.float32), True


def paper_card_quad(img: np.ndarray, bbox=None, min_ratio=0.003, max_ratio=0.35):
    """Бледная карточка на белом листе (скан А4): маска «не бумага» (расстояние цвета от цвета бумаги, оцененного по кольцу вокруг карточки),
    затем 4 стороны по контуру (Huber-прямые по прямым участкам, скруглённые углы отбрасываются) -> перспективный четырёхугольник TL,TR,BR,BL.
    -> (quad[4,2] в координатах img, info) | None."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bbox is None:
        bg = float(np.percentile(g, 92)); m = (g.astype(np.float32) < bg - 28).astype(np.uint8)
        k = max(5, int(0.02 * min(h, w))); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2: return None
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); bbox = tuple(int(st[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    x, y, bw, bh = bbox
    pad = int(0.15 * max(bw, bh)) + 8
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    ring = np.ones(roi.shape[:2], np.uint8); ring[max(0, y - y0 - 6):y - y0 + bh + 6, max(0, x - x0 - 6):x - x0 + bw + 6] = 0
    if ring.sum() < 200: return None
    paper = np.median(roi[ring > 0], axis=0)
    d = cv2.GaussianBlur(np.abs(roi - paper).max(axis=2), (0, 0), 1.2)
    ringd = d[ring > 0]; thr = max(9.0, float(np.percentile(ringd, 99)) + 4.0)
    m = (d > thr).astype(np.uint8)
    k = max(3, int(0.012 * max(bw, bh))); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    cx_, cy_ = x - x0 + bw // 2, y - y0 + bh // 2
    j = int(lab[cy_, cx_]) or (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    comp = (lab == j).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea); area = cv2.contourArea(c)
    ratio = area / float(w * h)
    if not (min_ratio <= ratio <= max_ratio): return None
    hull = cv2.convexHull(c)
    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32); quad = order_corners(box)
    pts = c.reshape(-1, 2).astype(np.float32)
    lines = []; diag = float(np.linalg.norm(quad[0] - quad[2]))
    for it in range(3):          # итеративно: полосы вдоль сторон -> Huber-прямые -> новые углы
        lines = []
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]; ab = b - a; L = float(np.linalg.norm(ab))
            u = ab / L; nrm = np.array([-u[1], u[0]], np.float32); rel = pts - a; t = rel @ u; dist = np.abs(rel @ nrm)
            sel = pts[(dist < (0.05 - 0.012 * it) * diag) & (t > 0.18 * L) & (t < 0.82 * L)]      # середина стороны: углы скруглены
            if len(sel) < 12: lines = None; break
            lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        if lines is None: break
        new = []
        for i in range(4):
            p_ = _line_intersect(lines[i - 1], lines[i])
            if p_ is None: new = None; break
            new.append(p_)
        if new is None: break
        new = order_corners(np.float32(new))
        if np.max(np.linalg.norm(new - quad, axis=1)) > 0.12 * diag: break
        quad = new
    refined = False
    try: quad, refined = (refine_quad_edges(roi.astype(np.uint8), quad) if os.environ.get("DOCREC_QUAD_EDGE_REFINE", "0") == "1" else (quad, False))      # по умолчанию выкл.: на теневых сторонах не помогает, а сдвиг ~1° ломает таблицу оборота ВУ
    except Exception: refined = False
    quad = quad + np.float32([x0, y0])
    qa = float(cv2.contourArea(quad)); fill = min(area, qa) / max(area, qa, 1.0)
    return quad.astype(np.float32), {"ratio": round(ratio, 4), "fill": round(fill, 3), "aspect": round(_quad_aspect(quad), 3), "thr": round(thr, 1), "edge_refined": bool(refined), "bbox": [int(x), int(y), int(bw), int(bh)]}


def small_card_crop(img: np.ndarray, max_ratio: float = 0.2, long_side: int = 1100):
    """Маленькая карточка посреди большого светлого листа (скан А4): находим единственный тёмный блок карточной формы, вырезаем с полями и увеличиваем
    (Lanczos + резкость) до long_side по длинной стороне. -> (crop, meta) | None."""
    h, w = img.shape[:2]
    if max(h, w) < 900: return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); bg = float(np.percentile(g, 92))
    if bg < 170: return None                  # не светлый лист
    m = (g.astype(np.float32) < bg - 28).astype(np.uint8)
    k = max(5, int(0.02 * min(h, w))); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, bw, bh = (int(st[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    ratio = bw * bh / float(w * h); asp = max(bw, bh) / max(1, min(bw, bh))
    if ratio > max_ratio or ratio < 0.003 or not (1.2 <= asp <= 2.3): return None
    # точная рамка карточки: светлый корпус карточки чуть темнее листа — порог мягче, берём компонент, пересекающийся с найденным блоком
    pq = None
    try: pq = paper_card_quad(img, (x, y, bw, bh))
    except Exception: pq = None
    soft = (g.astype(np.float32) < bg - 7).astype(np.uint8)
    soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); soft = cv2.morphologyEx(soft, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    nn, lab2, st2, _ = cv2.connectedComponentsWithStats(soft)
    cx_, cy_ = x + bw // 2, y + bh // 2; j = int(lab2[cy_, cx_]) if lab2[cy_, cx_] > 0 else 0
    quad = None
    if j > 0 and st2[j, cv2.CC_STAT_AREA] < 3.0 * bw * bh:
        ys_, xs_ = np.where(lab2 == j); pts = np.stack([xs_, ys_], 1).astype(np.float32)
        hull = cv2.convexHull(pts); ap = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        quad = ap[:, 0, :] if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(hull))
    if quad is None: quad = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
    quad_method = "blob"
    if pq is not None and pq[1]["fill"] >= 0.9 and 1.3 <= pq[1]["aspect"] <= 1.9: quad, quad_method = pq[0], "paper_lines"      # 4 прямые по контуру карточки (перспектива/поворот)
    quad = np.float32(quad)
    pw, ph = int(0.08 * bw), int(0.08 * bh)
    x0, y0, x1, y1 = max(0, x - pw), max(0, y - ph), min(w, x + bw + pw), min(h, y + bh + ph)
    crop = img[y0:y1, x0:x1]; s = min(4.0, long_side / max(crop.shape[:2]))
    if quad_method == "paper_lines":      # карточка в итоге должна быть ~1000 px по длинной стороне (ниже порога 1170 px, после которого тип СТС/паспорт переключается на «hi»-размер; выше OCR мелкого шрифта хуже)
        Lq = float(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[0] - quad[3])))
        s = min(4.0, max(s, float(os.environ.get("DOCREC_CARD_LONG", "1000")) / max(Lq, 1.0)))
    quad = (quad - np.float32([x0, y0])) * s
    if s > 1.05:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
        crop = cv2.addWeighted(crop, 1.5, cv2.GaussianBlur(crop, (0, 0), 1.8), -0.5, 0)
    else: s = 1.0
    return crop, {"quad": quad.tolist(), "bbox_src": [x0, y0, x1, y1], "scale": round(float(s), 2), "area_ratio": round(ratio, 4), "quad_method": quad_method, "note": "маленькая карточка на большом листе: вырезана и увеличена (Lanczos + резкость) перед OCR"}
