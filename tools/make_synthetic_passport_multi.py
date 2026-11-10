"""Синтетический лист с ДВУМЯ разворотами паспорта рядом (вымышленные данные): слева — страницы регистрации (штамп с повёрнутым «ЗАРЕГИСТРИРОВАН», вертикальная серия),
справа — основной разворот (без МРЗ). Серый «ксерокс» на белом листе. Запуск: python tools/make_synthetic_passport_multi.py -> samples/synth_pass_multi{1,2}.jpg + .gt.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.make_synthetic_passport import spread, P1, P2, FM, FS
W, H = 900, 1300


def registration(p, seed=0):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (238, 232, 242)
    bg += cv2.resize(rng.normal(0, 5, (H // 50, W // 50, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    ink = (35, 30, 45); lab = (95, 90, 105); blue = (150, 60, 40)
    d.text((300, 60), "МЕСТО ЖИТЕЛЬСТВА", font=ImageFont.truetype(FS, 24), fill=lab)
    d.rectangle((420, 130, 840, 520), outline=blue, width=4)                      # рамка штампа
    for k, t in enumerate(["Отдел УФМС России по", "Тверской области в", "гор. Твери"]): d.text((440, 150 + 40 * k), t, font=ImageFont.truetype(FM, 26), fill=blue)
    tw = Image.new("RGB", (330, 50), (238, 232, 242)); ImageDraw.Draw(tw).text((0, 6), "ЗАРЕГИСТРИРОВАН", font=ImageFont.truetype(FM, 28), fill=blue)
    img.paste(tw.rotate(-90, expand=True), (760, 270))
    d.text((440, 470), "20.05.2018", font=ImageFont.truetype(FM, 26), fill=ink)
    red = (40, 40, 200)
    for y0 in (700,):
        s = p["series"] + " " + p["number"]
        tw = Image.new("RGB", (400, 40), (238, 232, 242)); ImageDraw.Draw(tw).text((0, 4), s, font=ImageFont.truetype(FM, 30), fill=red)
        img.paste(tw.rotate(90, expand=True), (30, y0))
    d.ellipse((330, 800, 520, 990), outline=(120, 40, 140), width=4)
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def sheet(left, right, name, seed=0, SW=2048, SH=1489, angle_l=0.6, angle_r=-0.4):
    rng = np.random.default_rng(seed)
    out = np.full((SH, SW), 246, np.float32)
    def put(card, x0, x1, ang):
        g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY if card.shape[2] == 3 else cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = g.shape; sc = min((SH - 90) / h, (SW // 2 - 150) / w); g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) * 0.82   # серый «ксерокс»
        g = np.pad(g, 22, constant_values=40)                                       # тёмная рамка обложки
        h, w = g.shape; cx, cy = (x0 + x1) // 2, SH // 2
        Mx = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0); Mx[:, 2] += (cx - w / 2, cy - h / 2)
        warped = cv2.warpAffine(g, Mx, (SW, SH), flags=cv2.INTER_CUBIC, borderValue=-1)
        m = cv2.warpAffine(np.ones_like(g), Mx, (SW, SH), flags=cv2.INTER_NEAREST) > 0.5
        out[m] = warped[m]
    put(left, 40, SW // 2 - 24, angle_l); put(right, SW // 2 + 24, SW - 40, angle_r)
    out = cv2.GaussianBlur(out, (0, 0), 1.3) + rng.normal(0, 6, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(name, cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])


if __name__ == "__main__":
    for i, (p, ang) in enumerate([(P2, (0.6, -0.4)), (P1, (-0.5, 0.5))], 1):
        sheet(registration(p, i), cv2.cvtColor(spread(p, False, i), cv2.COLOR_RGB2BGR), f"samples/synth_pass_multi{i}.jpg", seed=i, angle_l=ang[0], angle_r=ang[1])
        gt = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in p.items()}
        json.dump({"fields": gt, "pages": ["registration", "main_spread"], "summary": {"has_main_spread": True, "has_registration": True}}, open(f"samples/synth_pass_multi{i}.gt.json", "w"), ensure_ascii=False, indent=1)
