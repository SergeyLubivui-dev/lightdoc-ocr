"""Синтетический СТС, ОБОРОТ НОВОГО ОБРАЗЦА (вымышленные данные): подписи «Субъект Российской Федерации / Населенный пункт / Корпус (строение) / Квартира /
Код подразделения ГИБДД / Дата выдачи», значения чёрной матричной печатью на отдельных строках, ФИО физлица в пяти строках (RU/LAT), красный номер сверху +
вертикальный чёрный справа, «Л.С.» переносится на следующую строку. Запуск: python tools/make_synthetic_sts_back_b.py -> samples/synth_sts_backb*.jpg + .gt.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import make_synthetic_sts as MS
from make_synthetic_sts import scene, FB, FM
W, H = 880, 1250
SER = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
GT = {"owner_type": "individual", "owner_surname": "ПЕТРОВ", "owner_name": "ИГОРЬ", "owner_patronymic": "ОЛЕГОВИЧ", "owner_surname_lat": "PETROV", "owner_name_lat": "IGOR'",
      "region": "Тверская область", "locality": "г Кимры", "street": "ул Садовая", "house": "12", "building": "", "apartment": "7",
      "special_marks": "МОЩНОСТЬ 250.0 КВТ/339.9 Л.С.", "power_kw": 250.0, "power_hp": 339.9, "gibdd_code": "1123456", "issue_date": "15.05.2020", "sts_number": "77 30 123456"}


def card_image(seed=0, pale=False):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (236, 226, 238)
    bg += cv2.resize(rng.normal(0, 4, (H // 40, W // 40, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    def txt(x, y, t, size, font, fill): d.text((x, y), t, font=ImageFont.truetype(font, size), fill=fill)
    lab = (95, 80, 95); val = (60, 55, 60) if pale else (20, 20, 22)
    txt(W * .08, H * .14, "77 30 123456", 40, FB, (60, 40, 170))
    txt(W * .09, H * .185, "СОБСТВЕННИК (владелец)", 24, SER, lab)
    for k, (t, x) in enumerate([("ПЕТРОВ", .16), ("PETROV", .19), ("ИГОРЬ", .16), ("IGOR'", .19), ("ОЛЕГОВИЧ", .16)]): txt(W * x, H * (.235 + .03 * k), t, 24, FM, val)
    rows = [("Субъект Российской Федерации", .405, "Тверская область"), ("Населенный пункт", .485, "г Кимры"), ("Улица", .525, "ул Садовая")]
    for lb, y, v in rows:
        txt(W * .09, H * y, lb, 22, SER, lab)
        if lb.startswith("Субъект"): txt(W * .16, H * (y + .035), v, 24, FM, val)
        elif lb.startswith("Нас"): txt(W * .45, H * y, v, 24, FM, val)
        else: txt(W * .26, H * (y - .006), v, 24, FM, val)
    txt(W * .09, H * .562, "Дом", 22, SER, lab); txt(W * .22, H * .56, "12", 24, FM, val)
    txt(W * .09, H * .598, "Корпус (строение)", 22, SER, lab)
    txt(W * .09, H * .634, "Квартира", 22, SER, lab); txt(W * .30, H * .632, "7", 24, FM, val)
    txt(W * .09, H * .672, "Особые отметки", 22, SER, lab)
    txt(W * .16, H * .70, "МОЩНОСТЬ 250.0 КВТ/339.9 Л.", 24, FM, val); txt(W * .16, H * .73, "С.", 24, FM, val)
    txt(W * .09, H * .826, "Код подразделения ГИБДД", 22, SER, lab); txt(W * .70, H * .826, "1123456", 24, FM, (110, 100, 105))
    d.line((W * .70, H * .84, W * .90, H * .84), fill=(90, 90, 100), width=2)
    txt(W * .44, H * .89, "15.05.2020", 24, FM, val); txt(W * .09, H * .903, "Дата выдачи", 22, SER, lab)
    txt(W * .45, H * .93, "ООО «ЗНАК», Москва, 2023", 14, SER, (150, 140, 150))
    # вертикальный чёрный номер справа (читается снизу вверх)
    tmp = Image.new("RGB", (330, 60), (236, 226, 238)); ImageDraw.Draw(tmp).text((6, 4), "77 30 123456", font=ImageFont.truetype(FB, 40), fill=(15, 15, 15))
    tmp = tmp.rotate(90, expand=True); img.paste(tmp, (int(W * .86), int(H * .32)))
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    MS.GT_OVERRIDE = GT
    scene(card_image(0), "samples/synth_sts_backb1.jpg", angle=4, seed=11)
    scene(card_image(1, pale=True), "samples/synth_sts_backb2.jpg", angle=-6, tilt=0.06, scale=0.6, seed=12)
