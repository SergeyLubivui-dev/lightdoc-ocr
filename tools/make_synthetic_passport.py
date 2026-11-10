"""Синтетический разворот паспорта РФ с ВЫМЫШЛЕННЫМИ данными (две страницы друг над другом, красный вертикальный номер, МРЗ с верными контрольными цифрами).
Запуск: python tools/make_synthetic_passport.py -> samples/synth_pass{1,2}.jpg + .gt.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from docrec import mrz as M
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
W, H = 900, 1300
P1 = dict(issued_by=["ОТДЕЛОМ УФМС РОССИИ ПО ТВЕРСКОЙ", "ОБЛАСТИ В ГОР. ТВЕРИ"], issue_date="12.03.2015", dept_code="690-001", surname="ТЕСТОВ", name="ИВАН", patronymic="ПЕТРОВИЧ",
          sex="МУЖ.", birth_date="15.06.1985", birth_place=["ГОР. ТВЕРЬ"], series="28 15", number="123456")
P2 = dict(issued_by=["ГУ МВД РОССИИ ПО Г.МОСКВЕ"], issue_date="04.09.2019", dept_code="770-099", surname="ОБРАЗЦОВА", name="МАРИЯ", patronymic="ИГОРЕВНА",
          sex="ЖЕН.", birth_date="02.11.1992", birth_place=["ГОР. КАЗАНЬ", "РЕСП. ТАТАРСТАН"], series="45 19", number="654321")


def mrz_lines(p):
    n = lambda s: M.translit_ru(s)
    l1 = ("PNRUS" + n(p["surname"]) + "<<" + n(p["name"]) + "<" + n(p["patronymic"]) + "<" * 44)[:44]
    d, b = p["series"].replace(" ", "")[:3] + p["number"], p["birth_date"]
    bd = b[8:10] + b[3:5] + b[0:2]; idt = p["issue_date"][8:10] + p["issue_date"][3:5] + p["issue_date"][0:2]
    dept = p["dept_code"].replace("-", "")
    s4 = p["series"].replace(" ", "")[3]
    l2 = d + str(M.check_digit(d)) + "RUS" + bd + str(M.check_digit(bd)) + ("M" if p["sex"] == "МУЖ." else "F") + "<" * 7 + s4 + idt + dept + "<"
    opt = l2[28:42]; l2 += str(M.check_digit(opt))
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    l2 += str(M.check_digit(comp))
    return l1, l2


def spread(p, with_mrz=True, seed=0):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.float32) + (238, 232, 242)
    bg += cv2.resize(rng.normal(0, 5, (H // 50, W // 50, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    lab = (95, 90, 105); ink = (35, 30, 45); red = (40, 40, 200)
    def T(x, y, t, size, font, fill):
        d.text((x, y), t, font=ImageFont.truetype(font, size), fill=fill)
    T(190, 40, "Р О С С И Й С К А Я   Ф Е Д Е Р А Ц И Я", 22, FS, (110, 90, 120))
    T(60, 100, "Паспорт выдан", 14, FS, lab)
    for k, t in enumerate(p["issued_by"]): T(250, 90 + 42 * k, t, 27, FM, ink)
    T(60, 270, "Дата выдачи", 14, FS, lab); T(180, 255, p["issue_date"], 28, FM, ink)
    T(390, 270, "Код подразделения", 14, FS, lab); T(600, 255, p["dept_code"], 28, FM, ink)
    T(560, 440, "Личный код", 14, FS, lab)
    d.ellipse((330, 330, 520, 520), outline=(120, 40, 140), width=4); T(375, 405, "М.П.", 18, FS, (120, 40, 140))   # «печать»
    d.rectangle((20, 560, W - 20, 585), fill=(215, 170, 185))                                           # красная полоса между страницами
    T(300, 625, "Фамилия", 14, FS, lab); T(500, 610, p["surname"], 28, FM, ink)
    T(300, 700, "Имя", 14, FS, lab); T(500, 685, p["name"], 28, FM, ink)
    T(300, 750, "Отчество", 14, FS, lab); T(500, 735, p["patronymic"], 28, FM, ink)
    T(300, 805, "Пол", 14, FS, lab); T(340, 790, p["sex"], 28, FM, ink)
    T(470, 805, "Дата рождения", 14, FS, lab); T(620, 790, p["birth_date"], 28, FM, ink)
    T(300, 870, "Место рождения", 14, FS, lab)
    for k, t in enumerate(p["birth_place"]): T(500, 855 + 42 * k, t, 28, FM, ink)
    # красный вертикальный номер (читается сверху вниз: поворот текста на -90°)
    for y0, y1 in ((60, 520), (620, 1010)):
        s = p["series"] + " " + p["number"]
        tw = Image.new("RGB", (400, 40), (238, 232, 242)); ImageDraw.Draw(tw).text((0, 4), s, font=ImageFont.truetype(FM, 30), fill=red)
        tw = tw.rotate(-90, expand=True); img.paste(tw, (W - 70, y0 + 40))
    face = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "face.png"))
    if face is not None:
        face = cv2.resize(face, (230, 300)); a = np.array(img); a[630:930, 40:270] = cv2.cvtColor(face, cv2.COLOR_BGR2RGB); img = Image.fromarray(a); d = ImageDraw.Draw(img)
    if with_mrz:
        l1, l2 = mrz_lines(p)
        T(40, 1130, l1, 29, FR, ink); T(40, 1180, l2, 29, FR, ink)
    a = np.array(img).astype(np.float32) + rng.normal(0, 2, (H, W, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def scene(card, name, angle, rot180=False, W_=1400, H_=1900, seed=0, tilt=0.03):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H_, W_, 3), np.float32) + (70, 85, 95)
    bg += cv2.resize(rng.normal(0, 1, (H_ // 8, W_ // 8, 3)).astype(np.float32), (W_, H_), interpolation=cv2.INTER_CUBIC) * 6 + rng.normal(0, 3, (H_, W_, 3))
    if rot180: card = cv2.rotate(card, cv2.ROTATE_180)
    h, w = card.shape[:2]; cw = W_ * 0.8; ch = cw * h / w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32([[0, 0], [cw, 0], [cw * (1 - tilt / 2), ch], [cw * tilt / 2, ch]])
    ang = np.deg2rad(angle); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (dst - dst.mean(0)) @ R.T + np.array([W_ * .5, H_ * .5])
    Mx = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(card, Mx, (W_, H_), flags=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(cv2.warpPerspective(np.full((h, w), 255, np.uint8), Mx, (W_, H_)), (0, 0), 0.8)[..., None] / 255.0
    out = cv2.GaussianBlur(np.clip(bg * (1 - m) + warped * m, 0, 255), (0, 0), 0.9) + rng.normal(0, 2.5, (H_, W_, 3))
    cv2.imwrite(name, np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    for i, (p, mz, ang, r180) in enumerate([(P1, True, -4.0, False), (P2, False, 3.0, True)], 1):
        scene(spread(p, mz, i), f"samples/synth_pass{i}.jpg", ang, r180, seed=i)
        gt = {k: (" ".join(v) if isinstance(v, list) else v) for k, v in p.items()}
        json.dump(gt, open(f"samples/synth_pass{i}.gt.json", "w"), ensure_ascii=False, indent=1)
