"""Синтетический СТС (лицевая сторона) с ВЫМЫШЛЕННЫМИ данными.
Имитирует особенности реального бланка: тёмные печатные подписи + бледные «напечатанные» значения,
которые смещены относительно подписей (между строками), красный номер внизу, значения у правого края обрезаны.
Запуск: python tools/make_synthetic_sts.py -> samples/synth_sts*.jpg + .gt.json"""
import cv2, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
FB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FM = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
W, H = 880, 1250
GT_OVERRIDE = None
GT = {"plate": "А123ВС77", "vin": "XTA219090E1234567", "make_model": "ЛАДА 2190 / LADA 2190", "make_model_ru": "ЛАДА 2190", "make_model_lat": "LADA 2190", "vehicle_type": "Легковой седан", "category": "B", "year": "2014",
      "chassis": "ОТСУТСТВУЕТ", "body": "XTA219090E1234567", "color": "Серебристый", "power": {"kw": 64.0, "hp": 87.0}, "eco_class": "четвёртый",
      "max_mass_kg": 1560, "curb_mass_kg": 1060, "pts": "77УТ123456", "sts_number": "77 12 123456"}
# (ключ, подпись, y подписи, значение)
ROWS = [("plate", "Регистрационный знак", .215, "А123ВС77"), ("vin", "Идентификационный номер (VIN)", .237, "XTA219090E1234567"),
        ("make_model", "Марка, модель", .323, "ЛАДА 2190\nLADA 2190"), ("vehicle_type", "Тип ТС", .358, "Легковой седан"),
        ("category", "Категория ТС (ABCD, прицеп)", .385, "B"), ("year", "Год выпуска ТС", .4405, "2014"),
        ("chassis", "Шасси (рама) №", .4785, "ОТСУТСТВУЕТ"), ("body", "Кузов (кабина, прицеп) №", .524, "XTA219090E1234567"),
        ("color", "Цвет", .5727, "Серебристый"), ("power", "Мощность двигателя, кВт/л. с.", .6215, "64/87"),
        ("eco_class", "Экологический класс", .6635, "четвертый"), ("max_mass_kg", "Технически допустимая max масса, кг", .6965, "1560"),
        ("curb_mass_kg", "Масса в снаряженном состоянии, кг", .7445, "1060"), ("temp_reg", "Срок временной регистрации", .784, "")]


def card_image(seed=0, value_shift=0.014, model=None, pitch=22):
    """model=(ru, lat): две строки марки/модели; pitch — шаг строк в пикселях (17 = строки касаются друг друга, как на реальном бланке)."""
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (236, 228, 238)       # светлая розово-сиреневая «гознаковая» бумага (BGR)
    bg += cv2.resize(rng.normal(0, 4, (H // 40, W // 40, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    def txt(x, y, t, size, font, fill, wrap=None):
        f = ImageFont.truetype(font, size)
        if wrap:
            words, line, ln = t.split(), "", []
            for w in words:
                if f.getlength((line + " " + w).strip()) > wrap: ln.append(line); line = w
                else: line = (line + " " + w).strip()
            ln.append(line); t = "\n".join(ln)
        d.multiline_text((x, y), t, font=f, fill=fill, spacing=1)
    txt(W * .12, H * .06, "СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ ТС", 27, FB, (40, 40, 50))
    txt(W * .12, H * .095, "CERTIFICAT D'IMMATRICULATION", 15, FR, (70, 70, 80))
    for key, lab, y, val in ROWS:
        if key == "make_model" and model: val = model[0] + "\n" + model[1]
        txt(W * .05, H * y - 14, lab, 17, FB, (45, 40, 50), wrap=W * .28)                  # печатные подписи: тёмные, мелкие
        if val:
            col = (172, 158, 148)                                                         # значения: бледные
            for k, line in enumerate(val.split("\n")):
                t = line
                if key in ("max_mass_kg", "curb_mass_kg"): t = line + " .. "
                txt(W * .34, H * (y + value_shift) - 14 + (pitch if key == "make_model" else 22) * k, t, 22, FM, col)
    txt(W * .05, H * .80, "ПТС: 77УТ123456", 22, FM, (172, 158, 148))
    txt(W * .30, H * .915, "77 12 123456", 38, FB, (40, 40, 205))                         # красный номер
    a = np.array(img).astype(np.float32); a += rng.normal(0, 2, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def scene(card, name, W_=1200, H_=1700, angle=6, tilt=0.05, scale=0.62, seed=0, bgc=(40, 44, 48), gt=None):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H_, W_, 3), np.float32) + bgc
    bg += cv2.resize(rng.normal(0, 1, (H_ // 8, W_ // 8, 3)).astype(np.float32), (W_, H_), interpolation=cv2.INTER_CUBIC) * 7 + rng.normal(0, 3, (H_, W_, 3))
    h, w = card.shape[:2]; cw = W_ * scale; ch = cw * h / w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [cw, 0], [cw * (1 - tilt / 2), ch], [cw * tilt / 2, ch]])
    ang = np.deg2rad(angle); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (dst - dst.mean(0)) @ R.T + np.array([W_ * .5, H_ * .5])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(card, M, (W_, H_), flags=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W_, H_)), (0, 0), 0.8)[..., None] / 255.0
    out = cv2.GaussianBlur(np.clip(bg * (1 - m) + warped * m, 0, 255), (0, 0), 0.9) + rng.normal(0, 2.5, (H_, W_, 3))
    cv2.imwrite(name, np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump({"corners_TL_TR_BR_BL": dst.tolist(), "fields": gt or GT_OVERRIDE or GT}, open(name.replace(".jpg", ".gt.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    scene(card_image(0, 0.014), "samples/synth_sts1.jpg", angle=6, seed=1)
    scene(card_image(1, 0.020), "samples/synth_sts2.jpg", angle=-9, tilt=0.09, scale=0.58, seed=2)   # другой сдвиг значений
    # двуязычная марка/модель, строки вплотную (как на реальном бланке); суффикс S5 (буква+цифра) и 55 (две цифры — ложного «S» быть не должно)
    for n, (ru, la, suf, a, tl) in enumerate([("КАМАЗ", "KAMAZ", "S5", 4, 0.06), ("КАМАЗ", "KAMAZ", "55", -5, 0.08)], 3):
        g = dict(GT); g.update(make_model=f"{ru} 5490-{suf} / {la} 5490-{suf}", make_model_ru=f"{ru} 5490-{suf}", make_model_lat=f"{la} 5490-{suf}")
        scene(card_image(n, 0.017, model=(f"{ru} 5490-{suf}", f"{la} 5490-{suf}"), pitch=17), f"samples/synth_sts{n}.jpg", angle=a, tilt=tl, scale=0.6, seed=n, gt=g)
