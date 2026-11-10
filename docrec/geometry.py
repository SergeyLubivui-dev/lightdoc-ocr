"""Поиск границ карточки (4 угла), перспективная коррекция."""
from __future__ import annotations
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
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
