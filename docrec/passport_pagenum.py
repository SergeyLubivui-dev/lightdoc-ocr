"""Нумерация страниц паспорта РФ на развороте/листе: бледная крупная цифра в круге вверху страницы (контраст-усиление + только цифры), заголовок страницы
(МЕСТО ЖИТЕЛЬСТВА 5–12, ВОИНСКАЯ ОБЯЗАННОСТЬ 13, СЕМЕЙНОЕ ПОЛОЖЕНИЕ 14, ДЕТИ 16–17), чётность (левые страницы чётные, правые нечётные),
вид страницы (empty / registration / main / other), штампы регистрации с типом и органом, перфорированная серия/номер (точечные цифры) внизу."""
from __future__ import annotations
import re
from collections import Counter
import cv2
import numpy as np
from .textutils import partial_ratio

HEADERS = [
    (r"МЕСТО\s*ЖИТЕЛЬСТВА", "registration", tuple(range(5, 13)), "МЕСТО ЖИТЕЛЬСТВА"),
    (r"ВОИНСКАЯ\s*ОБЯЗАННОСТЬ", "military", (13,), "ВОИНСКАЯ ОБЯЗАННОСТЬ"),
    (r"СЕМЕЙНОЕ\s*ПОЛОЖЕНИЕ", "family", (14,), "СЕМЕЙНОЕ ПОЛОЖЕНИЕ"),
    (r"(?:^|\s)ДЕТИ(?:\s|$)", "children", (16, 17), "ДЕТИ"),
]
SN_RE = re.compile(r"(?<!\d)(\d{2})\s*(\d{2})\s*(\d{6})(?!\d)")
MAX_PAGE = 20


def _up(t: str) -> str:
    return t.upper().replace("Ё", "Е")


def header_of(lines, y_max: float = 0.35):
    """Заголовок страницы по строкам верхней трети (нечёткое сравнение). -> (kind, range, text, score) | None"""
    best = None
    for l in lines:
        if l.cy > y_max: continue
        t = _up(l.text)
        for pat, kind, rng, word in HEADERS:
            ok = bool(re.search(pat, t)) or (len(word) >= 6 and partial_ratio(word, t) >= 0.84 and len(t) <= len(word) + 6)
            if ok and (best is None or l.score > best[3]): best = (kind, rng, word, float(l.score))
    return best


def split_pages(img: np.ndarray):
    """Разворот (альбомный) -> [(x0, x1)] левой и правой страниц по минимуму плотности чернил у сгиба; книжный кадр — одна страница."""
    H, W = img.shape[:2]
    if W < 1.05 * H: return [(0, W)]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); bg = cv2.GaussianBlur(g, (0, 0), max(H, W) / 25)
    ink = (g.astype(np.float32) < 0.8 * bg).astype(np.float32)
    col = cv2.GaussianBlur(ink.mean(0).reshape(1, -1), (0, 0), W / 80).ravel()
    a, b = int(0.42 * W), int(0.58 * W); x = a + int(np.argmin(col[a:b]))
    return [(0, x), (x, W)]


def pale_digit_crops(page: np.ndarray):
    """Кандидаты кропов с бледной цифрой номера: внутри найденного круга и верхняя центральная полоса; контраст выровнен, цифра тёмная на белом."""
    H, W = page.shape[:2]
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    top = g[: int(0.42 * H)]
    bg = cv2.GaussianBlur(top, (0, 0), max(top.shape) / 12)
    flat = np.clip(128 + (top.astype(np.float32) - bg.astype(np.float32)) * 255 / np.maximum(bg.astype(np.float32), 30) * 1.0, 0, 255).astype(np.uint8)
    flat = cv2.GaussianBlur(flat, (0, 0), 1.2)
    boxes = []
    circ = cv2.HoughCircles(cv2.createCLAHE(3.0, (8, 8)).apply(flat), cv2.HOUGH_GRADIENT, dp=1.2, minDist=W * 0.2, param1=80, param2=28, minRadius=int(0.05 * W), maxRadius=int(0.16 * W))
    if circ is not None:
        for cx, cy, r in circ[0][:3]:
            if 0.2 * W < cx < 0.8 * W and cy < 0.4 * H:
                s = int(0.62 * r); boxes.append((int(cx - s), int(cy - s), int(cx + s), int(cy + s), "circle"))
    boxes.append((int(0.30 * W), 0, int(0.70 * W), int(0.30 * H), "band"))
    out = []
    for x0, y0, x1, y1, kind in boxes:
        x0, y0 = max(0, x0), max(0, y0); c = flat[y0:min(y1, flat.shape[0]), x0:x1]
        if c.size == 0: continue
        c = cv2.createCLAHE(4.0, (4, 4)).apply(c)
        c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
        sc = 160 / max(c.shape[0], 1) if c.shape[0] < 160 else 1.0
        c = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (bw == 0).mean() > 0.5: bw = 255 - bw   # цифра должна быть малой тёмной долей
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        out.append((kind, cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255), cv2.copyMakeBorder(c, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)))
    return out


def read_page_number(pipe, page: np.ndarray):
    """-> (number|None, confidence, note). Голосование Tesseract(только цифры) и Paddle по нескольким кропам; допустимо 1..20."""
    votes = Counter(); detail = []
    cell = pipe._cell_fn()
    for kind, bw, gray in pale_digit_crops(page):
        for name, im in (("bw", bw), ("gray", gray)):
            for psm in (8, 7):
                if cell is None: break
                try:
                    t, sc = cell(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), "0123456789", psm, 1)
                except Exception: continue
                d = re.sub(r"\D", "", t)
                if d and 1 <= int(d) <= MAX_PAGE: votes[int(d)] += (1.0 + (0.5 if kind == "circle" else 0)) * max(0.2, sc / 100 if sc > 1 else sc); detail.append((kind, name, "tess", d))
        try:
            t, sc = pipe.paddle.recognize_crop(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
            d = re.sub(r"\D", "", t.replace("O", "0").replace("О", "0"))
            if d and 1 <= int(d) <= MAX_PAGE: votes[int(d)] += (0.8 + (0.4 if kind == "circle" else 0)) * sc; detail.append((kind, "bw", "paddle", d))
        except Exception: pass
    if not votes: return None, 0.0, "цифра в круге не прочитана"
    (n, v), = votes.most_common(1); tot = sum(votes.values())
    conf = min(0.95, 0.25 + 0.7 * v / max(tot, 1e-6)) * min(1.0, 0.5 + 0.25 * v)
    return n, round(float(conf), 3), "бледная цифра в круге, усиление контраста, голосов за: " + str(len([d for d in detail if int(d[3]) == n])) + " из " + str(len(detail))


def resolve_numbers(items):
    """items: [{'number','conf','header_range','side'('left'|'right'|None)}] слева направо -> заполняет number/source по заголовку и чётности."""
    for it in items:
        it["source"] = "ocr_pale_digit" if it["number"] else None
    for i, it in enumerate(items):
        if it["number"]:
            # согласуем с чётностью стороны
            if it.get("side") == "left" and it["number"] % 2: it["conf"] *= 0.6; it["parity_mismatch"] = True
            if it.get("side") == "right" and it["number"] % 2 == 0: it["conf"] *= 0.6; it["parity_mismatch"] = True
    for i, it in enumerate(items):
        if it["number"] is None:
            nb = [(j, o) for j, o in enumerate(items) if o["number"] and j != i and len(items) == 2]
            if nb:
                j, o = nb[0]; n = o["number"] + (1 if j < i else -1)
                if 1 <= n <= MAX_PAGE and (n % 2 == 0) == (it.get("side") == "left" or it.get("side") is None and n % 2 == 0):
                    it["number"], it["conf"], it["source"] = n, 0.5, "neighbor_parity"; continue
            r = it.get("header_range")
            if r:
                cand = [n for n in r if it.get("side") is None or (it["side"] == "left") == (n % 2 == 0)]
                if len(cand) == 1: it["number"], it["conf"], it["source"] = cand[0], 0.5, "header_single"
                elif cand: it["number_range"] = [min(cand), max(cand)]; it["source"] = "header_range+parity"; it["conf"] = 0.4
    return items


def _sn_dots(pipe, page: np.ndarray):
    """Перфорация (точечные цифры) серии/номера внизу страницы: бинаризация + морфологическое замыкание склеивает точки в штрихи, затем OCR цифр
    (Paddle + Tesseract только цифры) по разным ядрам замыкания и 4 ориентациям; итог — позиционное голосование по 10 цифрам. -> dict(series, number, score)|None"""
    H, W = page.shape[:2]; band = page[int(0.82 * H):]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY); bg = cv2.GaussianBlur(g, (0, 0), 25)
    d = np.clip(128 + (g.astype(np.float32) - bg) * 3, 0, 255).astype(np.uint8)
    cands = []; cell = pipe._cell_fn()
    for k in (3, 5, 7):
        bw = (d < 105).astype(np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        im = 255 - bw
        for rot in (None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            r = im if rot is None else cv2.rotate(im, rot)
            if r.shape[0] > 3 * r.shape[1]: continue
            r = cv2.copyMakeBorder(cv2.resize(r, None, fx=2, fy=2, interpolation=cv2.INTER_AREA), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            rb = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
            try: ls = pipe.paddle(rb, merge=False)
            except Exception: ls = []
            for l in ls:
                m = SN_RE.search(re.sub(r"[^\d ]", "", l.text)) or SN_RE.search(re.sub(r"\D", "", l.text))
                if m: cands.append(("".join(m.groups()), float(l.score)))
            if cell is not None and rot is None:
                try:
                    t, sc = cell(rb, "0123456789 ", 7, 1)
                    m = SN_RE.search(re.sub(r"[^\d ]", "", t))
                    if m: cands.append(("".join(m.groups()), float(sc / 100 if sc > 1 else sc)))
                except Exception: pass
    if not cands: return None
    digs = []; share = []
    for i in range(10):
        v = Counter()
        for c, sc in cands: v[c[i]] += sc
        ch, w = v.most_common(1)[0]; digs.append(ch); share.append(w / sum(v.values()))
    val = "".join(digs)
    return dict(series=f"{val[:2]} {val[2:4]}", number=val[4:], score=float(max(sc for _, sc in cands) * float(np.mean(share))))


def perforated_sn(pipe, page):
    try: return _sn_dots(pipe, page)
    except Exception: return None


def _ocr_page(pipe, page: np.ndarray, mx: int = 1500):
    H0, W0 = page.shape[:2]; s = mx / max(H0, W0) if max(H0, W0) > mx * 1.2 else 1.0
    im = cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else page
    return [l for l in pipe.paddle(im, merge=False) if l.score >= 0.4]


def _orient_score(lines) -> float:
    sc = 0.0
    for l in lines:
        t = _up(l.text)
        for pat, _k, _r, word in HEADERS:
            if re.search(pat, t) or (len(word) >= 6 and partial_ratio(word, t) >= 0.84): sc += 2.0
        if re.search(r"УФМС|ЗАРЕГИСТР|ОТДЕЛ|РЕГИСТРАЦИОН|ПАСПОРТ|ФАМИЛИЯ|ВЫДАН", t): sc += 1.0
        sc += 0.02 * len(re.findall(r"[А-ЯЁ]{3,}", t))
    return sc


def content_bbox(img: np.ndarray, pad: float = 0.01):
    """Рамка содержимого на светлом листе: тёмные/цветные пиксели относительно фона листа, закрытие, крайние ряды/столбцы с заметной долей."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = g.shape
    bg = float(np.percentile(g, 97)); m = (g < bg - 18).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    rows, cols = m.mean(1), m.mean(0)
    ys, xs = np.where(rows > 0.03)[0], np.where(cols > 0.03)[0]
    if len(ys) < 10 or len(xs) < 10: return 0, 0, W, H
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    return max(0, int(x0 - pad * W)), max(0, int(y0 - pad * H)), min(W, int(x1 + pad * W)), min(H, int(y1 + pad * H))


def find_orientation(pipe, img: np.ndarray):
    """-> (rotated image, degrees, lines). Перебор 0/180 (альбомный) или 90/270 (книжный) кадра, ранний выход при найденном заголовке."""
    H, W = img.shape[:2]
    order = (0, 180, 90, 270) if W >= H else (90, 270, 0, 180)
    rots = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    best = None
    for deg in order:
        im = img if rots[deg] is None else cv2.rotate(img, rots[deg])
        ls = _ocr_page(pipe, im, 1100); sc = _orient_score(ls)
        if best is None or sc > best[0]: best = (sc, deg, im, ls)
        if sc >= 3.0: break
    return best[2], best[1], best[3]


def analyze_pages(pipe, img: np.ndarray, parts=None):
    """Лист/разворот без основного разворота паспорта -> dict(pages=[...], summary={...}) либо None, если признаков страниц паспорта нет.
    parts — куски от split_spreads (если лист разрезан по сгибу на две страницы)."""
    from .passport_pages import read_registration
    H0, W0 = img.shape[:2]; pages_img = []
    if parts and len(parts) >= 2:
        for p in sorted(parts, key=lambda p: p["bbox"][0]): pages_img.append((p["crop"], p["bbox"]))
        rot = 0
    else:
        crop, bbox = img, [0, 0, W0, H0]
        if parts and len(parts) == 1: crop, bbox = parts[0]["crop"], list(parts[0]["bbox"])
        cx0, cy0, cx1, cy1 = content_bbox(crop); crop = crop[cy0:cy1, cx0:cx1]; bbox = [bbox[0] + cx0, bbox[1] + cy0, bbox[0] + cx1, bbox[1] + cy1]
        crop, rot, _ = find_orientation(pipe, crop)
        spans = split_pages(crop)
        ox = bbox[0]
        for (a, b) in spans: pages_img.append((crop[:, a:b], [ox + a, bbox[1], ox + b, bbox[3]]))
    items = []; raw = []
    for k, (pg, bbox) in enumerate(pages_img):
        side = None if len(pages_img) == 1 else ("left" if k == 0 else "right")
        ls = _ocr_page(pipe, pg); hdr = header_of(ls)
        num, conf, note = read_page_number(pipe, pg)
        items.append(dict(number=num, conf=conf, header_range=hdr[1] if hdr else None, side=side))
        raw.append((pg, bbox, ls, hdr, note))
    resolve_numbers(items)
    pages = []; sns = []
    for k, ((pg, bbox, ls, hdr, note), it) in enumerate(zip(raw, items), 1):
        d = dict(index=k, bbox=[int(v) for v in bbox], side=it.get("side"), page_number=it["number"], page_number_source=it["source"],
                 page_number_confidence=round(float(it["conf"]), 3) if it["number"] else 0.0)
        if it["source"] == "ocr_pale_digit": d["page_number_note"] = note
        if it.get("number_range"): d["page_number_range"] = it["number_range"]
        if it.get("parity_mismatch"): d["page_number_note"] = (d.get("page_number_note", "") + "; номер не совпал с чётностью стороны (слева чётные, справа нечётные)").strip("; ")
        reg = None
        if hdr or len(ls) > 3: reg = read_registration(pipe, pg)
        stamps = (reg or {}).get("stamp_list", [])
        if hdr and hdr[0] == "registration" or stamps or (reg and reg["kind"] == "registration"): kind = "registration"
        elif hdr: kind = "other"; d["section"] = hdr[0]
        elif len(ls) <= 3: kind = "empty"
        else: kind = "other"
        d.update(kind=kind, found=True, header=hdr[2] if hdr else None, stamps=stamps, stamps_n=len(stamps))
        if reg and reg.get("fields", {}).get("registration_authority") and not stamps: d["registration_authority"] = reg["fields"]["registration_authority"]
        sn = perforated_sn(pipe, pg)
        if sn: d["perforated_series_number"] = dict(value=f"{sn['series']} {sn['number']}", confidence=round(min(0.7, sn["score"] * 0.8), 3), note="точечная перфорация внизу страницы (замыкание + OCR цифр), best-effort"); sns.append(d["perforated_series_number"]["value"])
        pages.append(d)
    # одной «цифры» мало (в таблицах/штрихкодах цифр много): нужен заголовок страницы, штамп или уверенное чтение номера
    if not any(p["header"] or p["stamps_n"] or (p["page_number"] and p["page_number_confidence"] >= 0.7 and p["page_number_source"] == "ocr_pale_digit") for p in pages): return None
    summ = dict(pages_n=len(pages), page_numbers=[p["page_number"] for p in pages], has_registration=any(p["kind"] == "registration" for p in pages),
                has_empty=any(p["kind"] == "empty" for p in pages), registration_stamps=sum(p["stamps_n"] for p in pages), rotation_applied=rot)
    if sns:
        pf = [(p["perforated_series_number"]["confidence"], p["perforated_series_number"]["value"]) for p in pages if p.get("perforated_series_number")]
        summ["perforated_series_number"] = max(pf)[1]; summ["perforated_consistent"] = len(set(v for _, v in pf)) == 1
    return dict(pages=pages, summary=summ)
