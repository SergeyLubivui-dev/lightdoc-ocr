"""Синтетическая оценка цепочки рамок категорий (поле 9, лицевая сторона ВУ). Вымышленные данные, образцы на диск не пишутся.
Варианты: случайное подмножество открытых категорий (1..16), перенос на 2-ю строку, размытие, уменьшение масштаба (маленькая карточка),
небольшой поворот/шум, два стиля: (A) печатаются только открытые, (B) печатаются все 16, закрытые — бледные.
Оценивается итоговый список открытых категорий (точное совпадение) и доля рамок с верной категорией."""
import sys, os, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from docrec import catboxes as CB

FONTS = [p for p in (
    "/usr/share/fonts/truetype/sand-box/google/Ubuntu/Ubuntu-Bold.ttf", "/usr/share/fonts/truetype/sand-box/custom/Source Sans Pro/SourceSansPro-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf") if os.path.exists(p)] or ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def render(open_set, style, font, per_line=10, seed=0, blur=0.0, scale=1.0, angle=0.0, noise=3.0):
    rng = np.random.default_rng(seed)
    W, H = 1026, 648
    bg = np.full((H, W, 3), (222, 214, 226), np.float32) + rng.normal(0, 2, (H, W, 1))
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    ft = ImageFont.truetype(font, 24); lab = ImageFont.truetype(font, 22)
    d.text((372, 528), "9.", font=lab, fill=(150, 70, 140))
    cats = CB.CARD_ORDER if style == "B" else [c for c in CB.CARD_ORDER if c in open_set]
    x, y, n, bh = 430, 557, 0, 45
    for c in cats:
        tw = d.textlength(c, font=ft); bw = int(tw + 17)
        if n == per_line: x, y, n = 430, y + bh + 5, 0
        is_open = c in open_set
        col = (45, 45, 60) if is_open else (176, 170, 182)
        d.rectangle([x, y, x + bw, y + bh], outline=col, width=2)
        d.text((x + 8, y + 9), c, font=ft, fill=col)
        x += bw + 6; n += 1
    a = np.array(img).astype(np.float32)
    if abs(angle) > 0: a = cv2.warpAffine(a, cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0), (W, H), borderMode=cv2.BORDER_REPLICATE)
    if scale < 1.0:      # «маленькая карточка»: уменьшили → увеличили обратно (потеря резкости как после апскейла)
        a = cv2.resize(cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_CUBIC)
    if blur > 0: a = cv2.GaussianBlur(a, (0, 0), blur)
    a += rng.normal(0, noise, a.shape)
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    try:
        from docrec import DocumentPipeline
        cell = DocumentPipeline(engine='tess').tess.read_cell
    except Exception:
        cell = None
    random.seed(7)
    cases = []
    for i in range(40):
        k = random.choice([1, 2, 3, 5, 8, 10, 12, 12, 14, 16]); sub = set(random.sample(CB.CARD_ORDER, k))
        if i % 10 == 0: sub = set(CB.CARD_ORDER)
        style = "A" if i < 28 else "B"
        cases.append(dict(open_set=sub, style=style, font=random.choice(FONTS), per_line=random.choice([8, 10, 10, 11]), seed=i,
                          blur=random.choice([0, 0, 0.7, 1.2, 1.6]), scale=random.choice([1.0, 1.0, 0.7, 0.55]), angle=random.uniform(-1.2, 1.2), noise=random.choice([2, 4, 6])))
    ok = 0; box_ok = box_tot = 0
    for i, c in enumerate(cases):
        img = render(**c)
        r = CB.read_categories(img, region=(0.36, 0.74, 0.995, 1.0), cell=cell)
        want = [x for x in CB.CARD_ORDER if x in c["open_set"]]
        got = r["categories"] if r else []
        good = got == want; ok += good
        if r:
            want_all = CB.CARD_ORDER if c["style"] == "B" else want
            for b, w in zip(r["boxes"], want_all): box_tot += 1; box_ok += b["category"] == w and b["open"] == (w in c["open_set"])
            box_tot += max(0, len(want_all) - len(r["boxes"]))
        else: box_tot += len(CB.CARD_ORDER if c["style"] == "B" else want)
        if not good: print(f"  case {i} style={c['style']} blur={c['blur']} scale={c['scale']}: want {want} got {got}")
    print(f"CATEGORY BOX CHAINS: exact list {ok}/{len(cases)}; boxes {box_ok}/{box_tot}")
    return ok, len(cases)


if __name__ == "__main__":
    main()
