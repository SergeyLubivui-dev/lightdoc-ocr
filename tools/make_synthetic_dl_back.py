"""Синтетическая ОБОРОТНАЯ сторона российского водительского удостоверения (вымышленные данные): штрихкод слева, таблица категорий (колонки 9–12:
категория, пиктограмма, дата выдачи, дата окончания, особые отметки), поле 14, красный номер. Карточка — маленькая (≈300–600 px) в центре белого листа А4.
Запуск: python tools/make_synthetic_dl_back.py -> samples/synth_dlback{1,2,3}.jpg + .gt.json"""
import sys, os, json
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CATS = ["A", "A1", "B", "B1", "C", "C1", "D", "D1", "BE", "CE", "C1E", "DE", "D1E", "M", "Tm", "Tb"]
EXPIRY = "06.05.2036"
ROWS = {"B": ("06.07.1999", EXPIRY), "B1": ("07.05.2016", EXPIRY), "C": ("06.07.1999", EXPIRY), "C1": ("07.05.2016", EXPIRY), "BE": ("06.05.2026", EXPIRY),
        "CE": ("28.11.2006", EXPIRY), "C1E": ("07.05.2016", EXPIRY), "M": ("07.05.2016", EXPIRY)}     # вымышленные даты
NUMBER = "77 00 123456"; YEAR14 = "1999"
CW, CH = 1026, 648


def card(seed=0, rows=ROWS):
    rng = np.random.default_rng(seed)
    bg = np.zeros((CH, CW, 3), np.float32) + (232, 236, 240)
    bg += cv2.resize(rng.normal(0, 4, (CH // 30, CW // 30, 3)).astype(np.float32), (CW, CH), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    ink = (60, 55, 60); grid = (120, 118, 125)
    x0, x1, y0 = int(0.40 * CW), int(0.97 * CW), int(0.10 * CH); hdr = 34; pitch = (int(0.93 * CH) - y0 - hdr) / 16
    xs = [x0, x0 + int(0.11 * CW), x0 + int(0.17 * CW), x0 + int(0.32 * CW), x0 + int(0.47 * CW), x1]
    ys = [y0, y0 + hdr] + [y0 + hdr + pitch * (i + 1) for i in range(16)]
    for y in ys: d.line((x0, y, x1, y), fill=grid, width=2)
    for x in xs: d.line((x, y0, x, ys[-1]), fill=grid, width=2)
    f = ImageFont.truetype(FB, 17)
    for x, t in zip(xs[:-1], ["9.", "", "10.", "11.", "12."]): d.text((x + 10, y0 + 6), t, font=f, fill=ink)
    for i, c in enumerate(CATS):
        yc = ys[i + 1] + pitch / 2
        d.text((xs[0] + 10, yc - 11), c, font=ImageFont.truetype(FB, 22), fill=ink)
        d.rectangle((xs[1] + 14, yc - 8, xs[1] + 40, yc + 8), outline=grid, width=2)     # пиктограмма (ненадёжный признак)
        if c in rows:
            for x, t in ((xs[2], rows[c][0]), (xs[3], rows[c][1])): d.text((x + 10, yc - 11), t, font=ImageFont.truetype(FM, 21), fill=ink)
    for k in range(70):                                                                  # «штрихкод»: PDF417-подобные столбцы
        x = int(0.04 * CW) + k * 5; h = rng.integers(0, 2, 20)
        for j, v in enumerate(h):
            if v: d.rectangle((x, int(0.12 * CH) + j * 17, x + 3, int(0.12 * CH) + j * 17 + 14), fill=(30, 30, 30))
    d.text((int(0.04 * CW), int(0.72 * CH)), "14.", font=ImageFont.truetype(FB, 20), fill=ink); d.text((int(0.10 * CW), int(0.72 * CH)), YEAR14, font=ImageFont.truetype(FM, 28), fill=ink)
    d.text((int(0.63 * CW), int(0.945 * CH)), NUMBER, font=ImageFont.truetype(FB, 30), fill=(40, 40, 200))
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (CH, CW, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def sheet(c, name, card_w=300, angle=2.0, tilt=0.03, SW=1240, SH=1754, q=80, seed=0):
    rng = np.random.default_rng(seed)
    out = np.full((SH, SW, 3), 250, np.float32) + rng.normal(0, 2, (SH, SW, 3))
    cw = card_w; c = cv2.resize(c, (int(cw), int(round(cw * c.shape[0] / c.shape[1]))), interpolation=cv2.INTER_AREA)    # предфильтр до целевого размера (без алиасинга)
    h, w = c.shape[:2]; ch = cw * h / w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32([[0, 0], [cw, 0], [cw * (1 - tilt / 2), ch], [cw * tilt / 2, ch]])
    ang = np.deg2rad(angle); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = ((dst - dst.mean(0)) @ R.T + np.array([SW * .5, SH * .48])).astype(np.float32)
    Mx = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(c, Mx, (SW, SH), flags=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(cv2.warpPerspective(np.full((h, w), 255, np.uint8), Mx, (SW, SH)), (0, 0), 0.7)[..., None] / 255.0
    out = np.clip(out * (1 - m) + warped * m, 0, 255)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    cv2.imwrite(name, out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return dst


def gt():
    cats = [{"category": c, "open": c in ROWS, "date_from": ROWS[c][0] if c in ROWS else None, "date_to": ROWS[c][1] if c in ROWS else None} for c in CATS]
    return {"categories": cats, "number": NUMBER, "field_14": YEAR14, "expiry": EXPIRY}


if __name__ == "__main__":
    for i, (w, a, tl, q) in enumerate([(300, 2.0, 0.03, 85), (450, -3.0, 0.05, 80), (600, 1.0, 0.02, 80)], 1):
        c = card(i); dst = sheet(c, f"samples/synth_dlback{i}.jpg", w, a, tl, q=q, seed=i)
        g = gt(); g["corners_TL_TR_BR_BL"] = dst.tolist(); json.dump(g, open(f"samples/synth_dlback{i}.gt.json", "w"), ensure_ascii=False, indent=1)
