"""Синтетический СТС, ОБОРОТ (вымышленные данные): тёмные печатные подписи, бледные матричные значения со смещением, красный номер сверху,
чёрный снизу, зачёркнутый код ГИБДД, дата. Запуск: python tools/make_synthetic_sts_back.py -> samples/synth_sts_back*.jpg + .gt.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import make_synthetic_sts as MS
from make_synthetic_sts import scene, FB, FR, FM
W, H = 880, 1250
GT = {"owner": 'ООО"Ромашка"', "owner_lat": 'OOO"ROMASHKA"', "region": "Тверская область", "district": "р-н Кимрский", "locality": "г Кимры", "street": "ул Садовая",
      "house": "12", "building": "", "apartment": "", "special_marks": "", "gibdd_code": "1123456", "issue_date": "2020-05-15", "sts_number": "77 30 123456"}


def card_image(seed=0, shift=0.012):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (236, 228, 238)
    bg += cv2.resize(rng.normal(0, 4, (H // 40, W // 40, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    SER = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    def txt(x, y, t, size, font, fill): d.text((x, y), t, font=ImageFont.truetype(font, size), fill=fill)
    pale = (172, 150, 140)
    txt(W * .08, H * .105, "77 30 123456", 36, FB, (60, 40, 170))
    txt(W * .08, H * .19, "СОБСТВЕННИК (владелец)", 24, SER, (45, 40, 50))
    txt(W * .20, H * (.245 + shift), 'ООО"Ромашка"', 22, FM, pale); txt(W * .20, H * (.285 + shift), 'OOO"ROMASHKA"', 22, FM, pale)
    rows = [("Республика, край, область", .405, "Тверская область", .22), ("Район", .485, "р-н Кимрский", .36), ("Нас. пункт", .526, "г Кимры", .42), ("Улица", .566, "ул Садовая", .30)]
    for lab, y, val, x in rows:
        txt(W * .07, H * y - 14, lab, 22, SER, (45, 40, 50)); txt(W * x, H * (y + shift + (0.035 if lab.startswith("Респ") else 0.0)) - 14, val, 22, FM, pale)
    txt(W * .07, H * .609 - 14, "Дом", 22, SER, (45, 40, 50)); txt(W * .27, H * .609 - 8, "12", 22, FM, pale)
    txt(W * .34, H * .609 - 14, "корп.", 22, SER, (45, 40, 50)); txt(W * .58, H * .609 - 14, "кв.", 22, SER, (45, 40, 50))
    txt(W * .07, H * .645 - 14, "Особые отметки", 22, SER, (45, 40, 50))
    txt(W * .76, H * .808, "1123456", 24, FM, (130, 115, 110))
    d.line((W * .76, H * .823, W * .95, H * .823), fill=(90, 90, 100), width=2)
    txt(W * .07, H * .855 - 14, "Код подразделения ГИБДД", 22, SER, (45, 40, 50))
    txt(W * .24, H * .935 - 14, "15", 24, FM, pale); txt(W * .5, H * .94 - 14, "мая", 24, FM, pale); txt(W * .76, H * .935 - 14, "20", 26, FB, (45, 40, 50)); txt(W * .84, H * .94 - 14, "20", 24, FM, pale); txt(W * .93, H * .935 - 14, "г.", 26, FB, (45, 40, 50))
    txt(W * .3, H * .965 - 10, "77 30 123456", 38, FB, (20, 20, 20))
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    MS.GT_OVERRIDE = GT
    scene(card_image(0, 0.012), "samples/synth_sts_back1.jpg", angle=5, seed=3)
    scene(card_image(1, 0.018), "samples/synth_sts_back2.jpg", angle=-8, tilt=0.08, scale=0.58, seed=4)
