"""Синтетический разворот страниц 4–5 паспорта (вымышленные данные) на листе А4: слева пустая страница с бледным номером «4» в круге, справа — страница регистрации
(«МЕСТО ЖИТЕЛЬСТВА», номер «5», два штампа: ЗАРЕГИСТРИРОВАН и СНЯТ С РЕГИСТРАЦИОННОГО УЧЕТА), внизу обеих страниц — перфорация серии/номера (точечные цифры).
Запуск: python tools/make_synthetic_passport_pages.py -> samples/synth_pass_pages{1,2}.jpg + .gt.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.make_synthetic_passport import FM, FS
W, H = 900, 1300
SERIES, NUMBER = "45 07", "654321"     # вымышленные


def dots(img, d, text, x, y, size=44, step=6, r=2, fill=(110, 105, 118), vertical=False):
    tw = Image.new("L", (int(size * 0.75 * len(text)) + 20, size + 12), 0); ImageDraw.Draw(tw).text((6, 2), text, font=ImageFont.truetype(FM, size), fill=255)
    a = np.array(tw)
    if vertical: a = np.rot90(a, 1)
    ys, xs = np.mgrid[0:a.shape[0]:step, 0:a.shape[1]:step]
    for yy, xx in zip(ys.ravel(), xs.ravel()):
        if a[yy, xx] > 128: d.ellipse((x + xx - r, y + yy - r, x + xx + r, y + yy + r), fill=fill)


def page(num, kind, seed=0):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (238, 232, 242)
    bg += cv2.resize(rng.normal(0, 4, (H // 50, W // 50, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    pale = (225, 218, 230); blue = (150, 60, 40); lab = (95, 90, 105)
    d.ellipse((W // 2 - 110, 70, W // 2 + 110, 290), outline=pale, width=8)                 # бледный круг
    f = ImageFont.truetype(FM, 170); tw = d.textlength(str(num), font=f)
    d.text((W // 2 - tw / 2, 100), str(num), font=f, fill=(226, 219, 231))                   # бледная цифра (контраст ~12 уровней)
    if kind == "registration":
        d.text((W // 2 - 150, 320), "МЕСТО ЖИТЕЛЬСТВА", font=ImageFont.truetype(FS, 26), fill=lab)
        d.rectangle((90, 400, 810, 640), outline=blue, width=3)
        for k, t in enumerate(["Отдел УФМС России по Республике", "Крым в Сакском районе"]): d.text((110, 415 + 36 * k), t, font=ImageFont.truetype(FM, 24), fill=blue)
        d.text((110, 500), "ЗАРЕГИСТРИРОВАН", font=ImageFont.truetype(FM, 30), fill=blue)
        d.text((110, 550), "14.03.2019", font=ImageFont.truetype(FM, 26), fill=(40, 30, 30))
        d.rectangle((90, 680, 810, 920), outline=blue, width=3)
        for k, t in enumerate(["Отдел УФМС России по Республике", "Крым в Сакском районе"]): d.text((110, 695 + 36 * k), t, font=ImageFont.truetype(FM, 24), fill=blue)
        d.text((110, 780), "СНЯТ С РЕГИСТРАЦИОННОГО УЧЕТА", font=ImageFont.truetype(FM, 26), fill=blue)
        d.text((110, 830), "02.09.2022", font=ImageFont.truetype(FM, 26), fill=(40, 30, 30))
    dots(img, d, None if False else f"{SERIES} {NUMBER}", 250, H - 120)
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def sheet(name, seed=0, SW=2100, SH=2970, rot=0.0):
    rng = np.random.default_rng(seed)
    out = np.full((SH, SW, 3), 248, np.float32)
    L, R = page(4, "empty", seed), page(5, "registration", seed + 1)
    spread = np.hstack([L, np.full((H, 16, 3), 70, np.uint8), R]).astype(np.float32) * 0.9
    sc = (SW - 160) / spread.shape[1]; sp = cv2.resize(spread, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = sp.shape[:2]; cx, cy = SW // 2, SH // 3
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0); M[:, 2] += (cx - w / 2, cy - h / 2)
    warped = cv2.warpAffine(sp, M, (SW, SH), flags=cv2.INTER_CUBIC, borderValue=(-1, -1, -1))
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (SW, SH), flags=cv2.INTER_NEAREST) > 0.5
    out[m] = warped[m]
    out = cv2.GaussianBlur(out, (0, 0), 1.0) + rng.normal(0, 4, out.shape)
    cv2.imwrite(name, np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    for i, rot in enumerate([0.5, -0.8], 1):
        sheet(f"samples/synth_pass_pages{i}.jpg", seed=i, rot=rot)
        json.dump({"pages": [{"page_number": 4, "kind": "empty", "stamps": 0}, {"page_number": 5, "kind": "registration", "stamps": 2, "stamp_types": ["ЗАРЕГИСТРИРОВАН", "СНЯТ С РЕГИСТРАЦИОННОГО УЧЕТА"]}],
                   "perforated": f"{SERIES} {NUMBER}"}, open(f"samples/synth_pass_pages{i}.gt.json", "w"), ensure_ascii=False, indent=1)
