"""Разделение «двух красок» на бланках: тёмная типографская печать (подписи полей) и бледная машинная печать значений.
Нужно для документов, где значения напечатаны блёклым шрифтом поверх/между подписями (СТС и т.п.)."""
from __future__ import annotations
import cv2
import numpy as np


def split_ink(card_bgr: np.ndarray, label_th: float = 90.0, dilate: int = 5, gain: float = 2.0):
    """-> (label_img, value_img) — оба BGR. Подписи = пиксели заметно темнее фона; значения = остальное, с вырезанными подписями."""
    g = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    k = max(15, (int(max(g.shape) * 0.04) | 1))
    bg = cv2.medianBlur(g, min(k, 61) | 1)
    d = cv2.subtract(bg, g).astype(np.float32)
    dark = (d > label_th).astype(np.uint8) * 255
    dark_d = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    vv = d.copy(); vv[dark_d > 0] = 0
    value = np.clip(255 - vv * gain, 0, 255).astype(np.uint8)
    ll = np.clip(255 - np.where(d > label_th * 0.6, d, 0) * 1.2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ll, cv2.COLOR_GRAY2BGR), cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
