"""Генератор синтетической страницы договора с ВЫМЫШЛЕННЫМИ реквизитами (с корректными контрольными суммами).
python tools/make_synthetic_page.py -> samples/synth_page.jpg (+ samples/synth_page_rot90.jpg для проверки автоповорота)"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont

FONTS = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
BOLD = ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]


def font(paths, size):
    for p in paths:
        if os.path.exists(p): return ImageFont.truetype(p, size)
    raise SystemExit("нужен TTF-шрифт с кириллицей (Arial / DejaVu Sans)")


def inn10(base9):
    d = [int(c) for c in base9]; c = sum(a * b for a, b in zip(d, (2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
    return base9 + str(c)


def account(prefix, bik, body):  # 20 цифр, ключ по БИК
    w = (7, 1, 3) * 8; head = prefix
    s = bik[-3:] + head + "0" + body
    key = (sum(int(ch) * w[i] % 10 for i, ch in enumerate(s)) % 10 * 3) % 10
    return head + str(key) + body


def build(path, rot=0):
    random.seed(7)
    bik = "044525225"
    inn1, inn2 = inn10("770812345"), inn10("781055501")
    rs1 = account("40702810", bik, "00001234567"[:11].rjust(11, "0"))
    # к/с: ключ по «0» + 5-6 цифры БИК
    w = (7, 1, 3) * 8; s = "0" + bik[4:6] + "30101810" + "0" + "40000000225"
    ks = "30101810" + str((sum(int(ch) * w[i] % 10 for i, ch in enumerate(s)) % 10 * 3) % 10) + "40000000225"
    im = Image.new("RGB", (1600, 1200), (250, 250, 247)); d = ImageDraw.Draw(im)
    f, fb = font(FONTS, 24), font(BOLD, 28)
    d.text((300, 60), "7. Юридические адреса, реквизиты и подписи сторон", font=fb, fill=(20, 20, 20))
    cols = [(120, "ЗАКАЗЧИК", ["ООО «Ромашка»", "Юридический адрес: 101000, г. Москва,", "ул. Примерная, 1", f"ИНН {inn1}, КПП 770801001", f"Расчетный счет: {rs1}", f"Корр. счет {ks}", f"БИК {bik}", "ПАО Сбербанк г. Москва", "Тел. +7 (495) 000-00-01", "", "Директор ____________ А.В. Иванов"]),
            (860, "ИСПОЛНИТЕЛЬ", ["Петров Пётр Петрович", "Адрес: 190000, г. Санкт-Петербург,", "ул. Образцовая, 2", f"ИНН {inn2}", f"Паспорт: 40 12 № 123456, выдан 01.02.2015", f"Р/с {account('40817810', bik, '00007654321'.rjust(11, '0'))}", f"БИК {bik}, к/с {ks}", "e-mail: petrov@example.com", "", "____________ П.П. Петров"])]
    for x, title, lines in cols:
        d.text((x, 160), title, font=fb, fill=(0, 0, 0))
        for i, t in enumerate(lines): d.text((x, 220 + 46 * i), t, font=f, fill=(25, 25, 25))
    im = im.rotate(rot, expand=True) if rot else im
    im.save(path, quality=92)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(here, "samples", "synth_page.jpg")); build(os.path.join(here, "samples", "synth_page_rot90.jpg"), rot=90)
    print("OK")
