"""Уточнение окна фотографии по рамке (краям), а не по фиксированному шаблону.

Идея: фото — прямоугольник с мягкими углами; его внешнюю границу видно как резкий перепад цвета (Lab) между фоном карточки и снимком.
Для каждой из 4 сторон: профиль среднего перепада поперёк стороны (по средним 60% длины, без скруглённых углов), берём ВНЕШНИЙ заметный пик
в окне вокруг ожидаемого положения. Лицо (YuNet, если найдено) центрует/проверяет окно: оно обязано целиком содержать лицо и быть не меньше его.
Результат принимаем, только если размеры в разумных пределах относительно ожидания; иначе возвращаем исходную зону.
"""
from __future__ import annotations
import numpy as np
import cv2


def _profile(g: np.ndarray, axis: int, lo: int, hi: int, a: int, b: int, sign: int) -> np.ndarray:
    """g: карта (H,W) градиента по x (axis=1) или y (axis=0), усредняем по полосе [lo:hi] в поперечном направлении"""
    band = g[lo:hi, a:b] if axis == 1 else g[a:b, lo:hi]
    prof = band.mean(axis=0 if axis == 1 else 1)
    return sign * prof


def _outer_peak(p: np.ndarray, from_start: bool, frac=0.45):
    """первый (со стороны границы области) локальный максимум, не ниже frac от наибольшего"""
    if len(p) < 5 or p.max() <= 0: return None
    pm = np.convolve(p, np.ones(3) / 3, mode="same"); thr = frac * pm.max()
    idx = range(1, len(pm) - 1) if from_start else range(len(pm) - 2, 0, -1)
    for i in idx:
        if pm[i] >= thr and pm[i] >= pm[i - 1] and pm[i] >= pm[i + 1]: return i
    return None


def refine_photo_box(card: np.ndarray, zone_px, face=None, margin=0.14, min_frac=0.7, max_frac=1.35):
    """zone_px = (x0,y0,x1,y1) пикселях; face = (fx,fy,fw,fh) в тех же координатах или None. -> (box, info)"""
    H, W = card.shape[:2]
    x0, y0, x1, y1 = zone_px; zw, zh = x1 - x0, y1 - y0
    mx, my = int(margin * zw), int(margin * zh)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    # «тёплость» Lab (b*+0.5a*): снимок (бежевый/телесный) против бледно-синего/сиреневого фона карточки; яркость для краёв хуже (волосы, ореол)
    lab = cv2.cvtColor(card[Y0:Y1, X0:X1], cv2.COLOR_BGR2LAB).astype(np.float32)
    warm = cv2.GaussianBlur(lab[..., 2] + 0.5 * lab[..., 1], (0, 0), 2.2)
    gxs = cv2.Sobel(warm, cv2.CV_32F, 1, 0, ksize=3); gys = cv2.Sobel(warm, cv2.CV_32F, 0, 1, ksize=3)
    cy0, cy1 = int((y0 - Y0) + 0.20 * zh), int((y0 - Y0) + 0.80 * zh)
    cx0, cx1 = int((x0 - X0) + 0.20 * zw), int((x0 - X0) + 0.80 * zw)
    info = {"method": "edges"}
    pk = {}
    # окна поиска вокруг ожидаемых сторон (±12 % размера): берём самый сильный перепад (внешний ореол/виньетка слабее границы снимка)
    wx, wy = int(0.12 * zw), int(0.12 * zh)
    spec = (("l", gxs, 1, cy0, cy1, x0 - X0 - wx, x0 - X0 + wx, 1), ("r", gxs, 1, cy0, cy1, x1 - X0 - wx, x1 - X0 + wx, -1),
            ("t", gys, 0, cx0, cx1, y0 - Y0 - wy, y0 - Y0 + wy, 1), ("b", gys, 0, cx0, cx1, y1 - Y0 - wy, y1 - Y0 + wy, -1))
    for name, g, axis, lo, hi, a, b, sg in spec:
        lim = g.shape[1] if axis == 1 else g.shape[0]
        a = int(np.clip(a, 0, lim - 8)); b = int(np.clip(b, a + 6, lim))
        p = _profile(g, axis, lo, hi, a, b, sg)
        pm = np.convolve(p, np.ones(3) / 3, mode="same")
        pk[name] = None if pm.max() <= 0 else a + int(np.argmax(pm))
    if any(v is None for v in pk.values()):
        info.update({"method": "zone", "reason": "edge not found"}); return (x0, y0, x1, y1), info
    bx0, bx1, by0, by1 = X0 + pk["l"], X0 + pk["r"], Y0 + pk["t"], Y0 + pk["b"]
    bw, bh = bx1 - bx0, by1 - by0
    if not (min_frac * zw <= bw <= max_frac * zw and min_frac * zh <= bh <= max_frac * zh):
        info.update({"method": "zone", "reason": f"edges {bw}x{bh} vs expected {zw}x{zh}"}); return (x0, y0, x1, y1), info
    if face is not None:
        fx, fy, fw, fh = face
        if not (bx0 <= fx and fx + fw <= bx1 and by0 <= fy and fy + fh <= by1):
            info.update({"method": "zone", "reason": "face not inside edges"}); return (x0, y0, x1, y1), info
        info["face_center_offset_px"] = [round((fx + fw / 2) - (bx0 + bx1) / 2, 1), round((fy + fh / 2) - (by0 + by1) / 2, 1)]
    info["edges_px"] = [int(bx0), int(by0), int(bx1), int(by1)]
    return (int(bx0), int(by0), int(bx1), int(by1)), info
