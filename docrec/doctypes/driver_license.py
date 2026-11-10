"""Водительское удостоверение РФ, пластиковая карточка (образцы 2011+), лицевая сторона.

Шаблон: зоны полей заданы в ОТНОСИТЕЛЬНЫХ координатах нормализованной карточки 1026x648 (x0, y0, x1, y1).
Значения поля = строки OCR, центр которых попал в зону; дальше разбор регулярками/валидаторами.
Нумерация полей на карточке: 1 фамилия, 2 имя-отчество, 3 дата и место рождения, 4a/4b даты выдачи/окончания,
4c орган выдачи (ГИБДД xxxx), 5 серия и номер, 8 регион проживания, 9 категории.
"""
from __future__ import annotations
import re
from datetime import date
import numpy as np
import cv2
from ..ocr import Line
from ..textutils import (repair_ru_from_lat, snap_lat_to_ru, split_given_names, translit, script_of, to_cyr, to_lat, digits_fix, partial_ratio, translit_match)
from .base import DocType, Extraction, Field, register, in_zone, mean_conf

# --- ШАБЛОН (правится под реальные фото; все числа — доли ширины/высоты карточки) ---
ZONES = {
    "f1_surname":   (0.37, 0.212, 0.98, 0.295),
    "f2_name":      (0.37, 0.295, 0.98, 0.405),
    "f3_birth":     (0.37, 0.405, 0.98, 0.540),
    "f4ab_dates":   (0.37, 0.540, 0.98, 0.600),
    "f4c_issuer":   (0.37, 0.600, 0.98, 0.668),
    "f5_number":    (0.37, 0.668, 0.98, 0.725),
    "f8_region":    (0.37, 0.725, 0.98, 0.820),
    "f9_categories": (0.37, 0.820, 0.98, 0.960),
}
PHOTO_ZONE = (0.068, 0.230, 0.312, 0.790)
CATEGORIES = ["A", "A1", "B", "B1", "BE", "C", "C1", "CE", "C1E", "D", "D1", "DE", "D1E", "M", "Tm", "Tb"]
_CAT_RE = re.compile(r"C1E|D1E|C1|D1|CE|DE|BE|B1|A1|Tm|Tb|[ABCDM]")
_LABEL_RE = re.compile(r"^\s*\d{1,2}\s*[abcаbс]?\s*[.)]?\s*$", re.I)
_DATE_RE = re.compile(r"(\d{2})\s*[.,:;]?\s*(\d{2})\s*[.,:;]?\s*(\d{4})")


def parse_date(s: str):
    s2 = digits_fix(s)
    m = re.search(r"(?<!\d)(\d{2})[.,:;\s]{0,2}(\d{2})[.,:;\s]{0,2}(\d{4})(?!\d)", s2)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        dt = date(y, mo, d)
    except ValueError:
        return None
    if not (1900 <= y <= 2100):
        return None
    return f"{d:02d}.{mo:02d}.{y}", dt, m.span()


_PREFIX_RE = re.compile(r"^\s*(?:\d{1,2}\s*[abcабс]?\s*[.)]|4\s*[abcабс](?=\s)|[iIl1ТtT]\s*[abcабс]?\s*\)|[A-Za-z]{1,2}\)|\d{1,2}\s*[.,)]?\s*[|\]\[—–-]+)\s*", re.I)


def _strip_label(l: Line) -> Line:
    """Если детектор приклеил номер поля к значению («4a) 20.10.2011») — отрезаем его."""
    m = _PREFIX_RE.match(l.text)
    if m and m.end() < len(l.text) and not re.match(r"^\d{2}\.\d{2}\.\d{4}", l.text):
        import copy
        n = copy.copy(l); n.text = l.text[m.end():]
        return n
    return l


def _gray2x(img, scale=2, clahe=True):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if clahe:
        g = cv2.createCLAHE(2.0, (4, 4)).apply(g)
    g = cv2.copyMakeBorder(g, 8, 8, 24, 24, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _clean(s: str) -> str:
    s = re.sub(r"^[^\wА-Яа-яЁё]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;_-|")
    return s



def _reread(card, rect, shift, cell, whitelist, psm=7):
    """Перечитать прямоугольник (в координатах шаблона) одной строкой с белым списком символов."""
    ca, cb, cdx = shift; H, W = card.shape[:2]
    x0, y0, x1, y1 = rect
    X0, X1 = int(max(0, (x0 + cdx) * W)), int(min(W, (x1 + cdx) * W)); Y0, Y1 = int(max(0, (ca * y0 + cb) * H)), int(min(H, (ca * y1 + cb) * H))
    if X1 - X0 < 10 or Y1 - Y0 < 8: return "", 0.0
    return cell(card[Y0:Y1, X0:X1], whitelist, psm)


_CATS_SET = set(CATEGORIES)
_CYR2CAT = {**dict(zip('ВСЕМАТДвсемат', 'BCEMATDBCEMAT')), 'д': 'D', 'В': 'B'}
_DELIMS = re.compile(r"[|\[\]()/,;:\\]+")


def _norm_cat(tok: str):
    """Токен OCR -> каноническая категория или None (двойники кириллица->латиница, 8->B, l/I/| ->1)."""
    t = "".join(_CYR2CAT.get(ch, ch) for ch in to_lat(tok.strip()))
    t = re.sub(r"[^A-Za-z0-9]", "", t)
    if not t: return None
    t = t.replace("l", "1").replace("I", "1") if len(t) > 1 else t
    if t == "8": t = "B"
    tl = t.lower()
    if tl in ("tm", "tb"): return "T" + tl[1]
    tu = t.upper()
    tu = tu.replace("O", "0") if "0" in tu else tu
    return tu if tu in _CATS_SET else None


def parse_categories_text(text: str) -> list[str]:
    text = re.sub(r"^\W*\S{0,2}[.)]\s+", "", text.strip())   # метка поля «9.» (может быть прочитана как «э.»)
    toks = [x for x in re.split(r"\s+", _DELIMS.sub(" ", text)) if x]
    out = []
    for tk in toks:
        c = _norm_cat(tk)
        if c is None and len(tk) > 2:  # слипшиеся «BB1C» — жадный разбор
            comp = re.sub(r"[^A-Za-z0-9]", "", to_lat(tk).replace("В", "B"))
            parts = _CAT_RE.findall(comp)
            if "".join(parts) == comp: out += parts; continue
        if c: out.append(c)
    return list(dict.fromkeys(out))


def _cell_images(card, line, shift):
    """Нарезка ячеек строки категорий по вертикальным границам рамок."""
    ca, cb, cdx = shift
    H, W = card.shape[:2]
    cy_ = ca * (line.y0 + line.y1) / 2 + cb
    y0 = int(max(0, (cy_ - 0.045) * H)); y1 = int(min(H, (cy_ + 0.045) * H))
    x0 = int(max(0, (line.x0 + cdx + 0.03) * W)); x1 = int(min(W, (line.x1 + cdx + 0.03) * W))
    if y1 - y0 < 8 or x1 - x0 < 20: return []
    row = card[y0:y1, x0:x1]
    g = cv2.GaussianBlur(cv2.cvtColor(row, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    k = max(8, int(26 * W / 1026))                      # рамка ячейки ~32 px при ширине карточки 1026; штрихи букв короче
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    rows = np.where((v > 0).sum(1) >= 3)[0]
    if len(rows) < k // 2: return []
    ry0, ry1 = int(rows.min()), int(rows.max()) + 1       # вертикальная протяжённость рамок
    row = row[ry0:ry1]; v = v[ry0:ry1]
    cs = (v > 0).mean(0) > 0.6
    bands, i = [], 0
    while i < len(cs):
        if cs[i]:
            j = i
            while j < len(cs) and cs[j]: j += 1
            bands.append((i, j)); i = j
        else: i += 1
    cells = []
    for (a0, a1), (b0, b1) in zip(bands, bands[1:]):
        if b0 - a1 >= 8:
            c = row[:, a1 + 1:b0 - 1]
            cells.append(c)
    return cells


def parse_categories_zone(card, zone_lines, shift, cell, rec):
    best, conf = [], 0.0
    line_tokens = []
    for l in zone_lines:
        c = parse_categories_text(l.text)
        if len(c) > len(best): best, conf, line = c, l.score, l
        if len(c) >= len(line_tokens): line_tokens = c
    note = None
    if cell is not None and zone_lines:
        line = max(zone_lines, key=lambda l: len(parse_categories_text(l.text)) * 10 + l.score)
        imgs = _cell_images(card, line, shift)
        if 1 <= len(imgs) <= 16:
            raw_tokens = [x for x in re.split(r"\s+", _DELIMS.sub(" ", re.sub(r"^\W*\S{0,2}[.)]\s+", "", line.text))) if x]
            aligned = len(raw_tokens) == len(imgs)
            res = []
            for k, im in enumerate(imgs):
                votes = {}
                for ins in (2, 3):
                    inner = im[ins:-ins, ins:-ins] if im.shape[0] > 2 * ins + 6 and im.shape[1] > 2 * ins + 6 else im
                    for psm in (13, 7):
                        t, c = cell(inner, "ABCDEMTb1m", psm)
                        n = _norm_cat(t) if t else None
                        if n: votes[n] = votes.get(n, 0) + 1
                if aligned:
                    n = _norm_cat(raw_tokens[k])
                    if n: votes[n] = votes.get(n, 0) + 1.5
                if votes: res.append(max(votes, key=votes.get))
            if len(res) >= len(best):
                if res != best: note = f"cell-voting ({len(imgs)} cells); line parse={best}"
                best = list(dict.fromkeys(res)); conf = max(conf, 0.85 if len(res) == len(imgs) else 0.7)
    if not best and rec is not None:
        pass
    return best, conf, note


# Ожидаемые положения подписей-номеров полей слева (центры по Y, левый край X) на эталонной карточке.
LABEL_CY = {"1": 0.225, "2": 0.324, "3": 0.416, "4a": 0.553, "4c": 0.604, "5": 0.697, "8": 0.745, "6": 0.814, "7": 0.866}
LABEL_X0 = 0.352
_LABEL_KEY = re.compile(r"^\s*(\d)\s*([abcабс]?)\s*[.)]?\s*$", re.I)


def calibrate(lines):
    """Оцениваем сдвиг/масштаб по Y и сдвиг по X по найденным подписям полей (1. 2. 3. 4a) 4c) 5. 8.).
    Нужно, когда рамка карточки найдена с погрешностью в 1–3 %. Возвращает (a, b, dx): y_real = a*y_tpl + b."""
    ys, xs = [], []
    for l in lines:
        if l.x0 > 0.45: continue
        m = _LABEL_KEY.match(l.text)
        if not m: continue
        key = m.group(1) + m.group(2).lower().replace("а", "a").replace("б", "b").replace("с", "c")
        if key in LABEL_CY and 0.30 < l.x0 < 0.42:
            ys.append((LABEL_CY[key], l.cy)); xs.append(l.x0 - LABEL_X0)
    if len(ys) < 3:
        return 1.0, 0.0, 0.0, len(ys)
    t, r = np.array(ys).T
    a, b = np.polyfit(t, r, 1)
    res = np.abs(r - (a * t + b))
    keep = res < max(0.01, 2.5 * np.median(res))
    if keep.sum() >= 3 and not keep.all():
        a, b = np.polyfit(t[keep], r[keep], 1)
    if not (0.92 <= a <= 1.08) or abs(b) > 0.08:
        return 1.0, 0.0, 0.0, len(ys)
    return float(a), float(b), float(np.median(xs)), len(ys)


def to_template(lines, a, b, dx):
    import copy
    out = []
    for l in lines:
        n = copy.copy(l)
        n.y0, n.y1, n.cy = (l.y0 - b) / a, (l.y1 - b) / a, (l.cy - b) / a
        n.x0, n.x1, n.cx = l.x0 - dx, l.x1 - dx, l.cx - dx
        out.append(n)
    return out


@register
class DriverLicenseRF(DocType):
    name = "driver_license_rf"
    title_ru = "Водительское удостоверение РФ (карточка 2011+, лицевая сторона)"
    keywords = [("ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ", 4), ("ГИБДД", 1.2), ("GIBDD", 1.2), ("RUS", 0.6)]
    implemented = True
    expected_fields = ["surname", "surname_lat", "given_names", "given_names_lat", "birth_date", "birth_place", "birth_place_lat", "issue_date", "expiry_date", "issuer", "issuer_code", "number", "region", "region_lat", "categories"]

    # ---- классификация: ключевые слова + структурные признаки
    def classify(self, lines):
        base = super().classify(lines)
        txt = " ".join(l.text for l in lines)
        struct = 0.0
        n_dates = len(_DATE_RE.findall(digits_fix(txt)))
        struct += 0.35 * min(1.0, n_dates / 3)
        labels = sum(bool(re.match(r"^\s*(4\s*[abcаbс]|[1-9])\s*[.)]?\s*$", l.text.strip(), re.I)) for l in lines)
        struct += 0.25 * min(1.0, labels / 5)
        struct += 0.40 * (1.0 if re.search(r"(?<!\d)\d{2}\s?\d{2}\s\d{6}(?!\d)|(?<!\d)\d{2}\s\d{2}\s?\d{6}(?!\d)", digits_fix(txt)) else 0.0)
        # заголовок решает; структура страхует, когда заголовок прочитан плохо
        return float(min(1.0, 0.75 * base + 0.5 * struct))

    # ---- извлечение
    def extract(self, card: np.ndarray, lines: list[Line], rec=None, cell=None) -> Extraction:
        ex = Extraction(photo_zone=PHOTO_ZONE)
        F = ex.fields
        ca, cb, cdx, n_lab = calibrate(lines)
        ex.extra["template_calibration"] = {"labels_used": n_lab, "scale_y": round(ca, 4), "shift_y": round(cb, 4), "shift_x": round(cdx, 4)}
        real_lines = lines
        lines = to_template(lines, ca, cb, cdx)
        pz = PHOTO_ZONE
        ex.photo_zone = (pz[0] + cdx, ca * pz[1] + cb, pz[2] + cdx, ca * pz[3] + cb)
        # лишние «метки» (1. 2. 3 4a) и всё слева от значений убираем
        lines = [_strip_label(l) for l in lines]
        vals = [l for l in lines if l.cx >= 0.40 and l.x1 > 0.42 and not _LABEL_RE.match(l.text)]
        zone = {k: sorted([l for l in vals if in_zone(l, z)], key=lambda l: l.cy) for k, z in ZONES.items()}

        def by_script(ls):
            ru = [l for l in ls if script_of(l.text) in ("ru", "mixed")]
            la = [l for l in ls if script_of(l.text) == "lat"]
            # «none» (только цифры/знаки) в обе корзины не кладём
            return ru, la

        def txt(ls, kind):
            t = " ".join(l.text for l in ls)
            return _clean(to_cyr(t).upper() if kind == "ru" else to_lat(t).upper())

        # 1. фамилия
        H_, W_ = card.shape[:2]

        def reread_latin(ru_line, la_lines):
            """Мелкая латинская строка под кириллической часто не детектируется — читаем её по шаблону."""
            if rec is None or ru_line is None:
                return la_lines
            if la_lines and translit_match(ru_line.text, " ".join(l.text for l in la_lines)) >= 0.7:
                return la_lines
            x0, x1 = max(0, ru_line.x0 - 0.006 + cdx), min(1.0, max(ru_line.x1, ru_line.x0 + 0.1) + 0.03 + cdx)
            y0, y1 = ca * (ru_line.y1 - 0.004) + cb, ca * (ru_line.y1 + 0.050) + cb
            crop = card[int(y0 * H_):int(y1 * H_), int(x0 * W_):int(x1 * W_)]
            if crop.size == 0:
                return la_lines
            t, sc = rec(_gray2x(crop, 3, False))
            if t and script_of(t) != "ru" and translit_match(ru_line.text, t) >= 0.5 and (not la_lines or translit_match(ru_line.text, t) > translit_match(ru_line.text, " ".join(l.text for l in la_lines))):
                n = Line(text=t, score=sc, box=np.zeros((4, 2), np.float32)); n.x0, n.x1, n.y0, n.y1 = x0, x1, y0, y1; n.cx, n.cy = (x0 + x1) / 2, (y0 + y1) / 2
                return [n]
            return la_lines

        ru, la = by_script(zone["f1_surname"]); la = reread_latin(ru[0] if ru else None, la)
        if ru: F["surname"] = Field(txt(ru[:1], "ru"), mean_conf(ru[:1]))
        if la: F["surname_lat"] = Field(txt(la, "lat"), mean_conf(la))
        # 2. имя + отчество
        ru, la = by_script(zone["f2_name"]); la = reread_latin(ru[0] if ru else None, la)
        if ru:
            full = txt(ru[:1], "ru")
            full2 = split_given_names(full, txt(la, "lat") if la else None)
            F["given_names"] = Field(full2, mean_conf(ru[:1]), "space restored via ICAO/suffix" if full2 != full else None); full = full2
            parts = full.split()
            if parts: F["name"] = Field(parts[0], mean_conf(ru[:1]))
            if len(parts) > 1: F["patronymic"] = Field(" ".join(parts[1:]), mean_conf(ru[:1]))
        if la: F["given_names_lat"] = Field(txt(la, "lat"), mean_conf(la))
        # транслитерационная сверка (диагностика качества OCR)
        if "surname" in F and "surname_lat" in F:
            m = translit_match(F["surname"].value, F["surname_lat"].value)
            F["surname"].confidence *= (0.6 + 0.4 * m); F["surname_lat"].confidence *= (0.6 + 0.4 * m)
            if m < 0.99: F["surname_lat"].note = f"translit(ru)={translit(F['surname'].value)} sim={m:.2f}"
        if "given_names" in F and "given_names_lat" in F:
            m = translit_match(F["given_names"].value, F["given_names_lat"].value)
            F["given_names"].confidence *= (0.6 + 0.4 * m); F["given_names_lat"].confidence *= (0.6 + 0.4 * m)
            if m < 0.99: F["given_names_lat"].note = f"translit(ru)={translit(F['given_names'].value)} sim={m:.2f}"

        # 3. дата и место рождения
        birth_lines = zone["f3_birth"]
        bdate = None; rest = []
        for l in birth_lines:
            pd = parse_date(l.text)
            if pd and bdate is None:
                bdate = (pd, l)
                tail = (l.text[:0] + digits_fix(l.text)[pd[2][1]:]).strip()
                if len(re.sub(r"[\W\d_]", "", l.text.replace(l.text[pd[2][0]:pd[2][1]], ""))) >= 3:
                    rest.append(l)
            else:
                rest.append(l)
        if bdate:
            F["birth_date"] = Field(bdate[0][0], bdate[1].score, None)
        ru, la = by_script([l for l in rest if not parse_date(l.text)])
        if ru and not la: la = reread_latin(ru[-1], la)
        if ru: F["birth_place"] = Field(txt(ru, "ru"), mean_conf(ru))
        if la: F["birth_place_lat"] = Field(txt(la, "lat"), mean_conf(la))

        # 4a / 4b: даты выдачи и окончания
        dates = []
        for l in zone["f4ab_dates"] + [l for l in zone["f3_birth"] if l.cx > 0.6]:
            pd = parse_date(l.text)
            if pd: dates.append((l.cx, pd, l))
        dates.sort(key=lambda t: t[0])
        if dates:
            F["issue_date"] = Field(dates[0][1][0], dates[0][2].score)
        if len(dates) > 1:
            F["expiry_date"] = Field(dates[-1][1][0], dates[-1][2].score)
        # fallback: ищем все даты в правой части и расставляем по возрастанию
        if "issue_date" not in F or "expiry_date" not in F or "birth_date" not in F:
            alld = []
            for l in vals:
                pd = parse_date(l.text)
                if pd: alld.append((pd[1], pd[0], l))
            alld = sorted({a[1]: a for a in alld}.values(), key=lambda a: a[0])
            if len(alld) >= 3:
                for k, a in zip(("birth_date", "issue_date", "expiry_date"), alld[:3] if "birth_date" not in F else alld[-2:] if len(alld) > 2 else alld):
                    F.setdefault(k, Field(a[1], a[2].score * 0.8, "fallback by date order"))
        # --- запасное перечитывание: поле не найдено (Tesseract иногда теряет строку) -> одна строка по зоне, белый список символов
        if cell is not None:
            for key, rect in (("birth_date", (0.39, 0.386, 0.62, 0.442)), ("issue_date", (0.39, 0.534, 0.585, 0.588)), ("expiry_date", (0.65, 0.534, 0.86, 0.588))):
                if key not in F:
                    t_, c_ = _reread(card, rect, (ca, cb, cdx), cell, "0123456789.")
                    pd = parse_date(t_)
                    if pd: F[key] = Field(pd[0], max(0.55, c_ * 0.9), "re-read by zone")
            if "number" not in F:
                t_, c_ = _reread(card, (0.39, 0.685, 0.64, 0.742), (ca, cb, cdx), cell, "0123456789 ")
                dg = re.sub(r"\D", "", t_)
                if len(dg) == 10: F["number"] = Field(f"{dg[:2]} {dg[2:4]} {dg[4:]}", max(0.55, c_ * 0.9), "re-read by zone")
        # логические проверки
        def dt(k): 
            p = parse_date(F[k].value) if k in F else None
            return p[1] if p else None
        b, i, e = dt("birth_date"), dt("issue_date"), dt("expiry_date")
        if i and e and e <= i: F["expiry_date"].confidence *= 0.4; F["issue_date"].confidence *= 0.4
        if b and i and b >= i: F["birth_date"].confidence *= 0.4

        # 4c. орган выдачи
        ru, la = by_script(zone["f4c_issuer"])
        if ru:
            t = txt(ru[:1], "ru")
            # «ГНБДД», «ГИБДО» и т.п. → ГИБДД (нечёткая привязка слова-аббревиатуры)
            t = " ".join("ГИБДД" if partial_ratio("ГИБДД", w) >= 0.72 and not re.search(r"\d", w) else w for w in t.split())
            F["issuer"] = Field(t, mean_conf(ru[:1]))
            m = re.search(r"(\d[\d\sO]{3,5})\s*$", digits_fix(t))
            code = re.sub(r"\D", "", m.group(1)) if m else ""
            if len(code) >= 4:
                code = code[-4:]
                F["issuer_code"] = Field(code, mean_conf(ru[:1]))
                if la:
                    lc = re.sub(r"\D", "", digits_fix(la[0].text))[-4:]
                    if lc == code: F["issuer_code"].confidence = min(1.0, F["issuer_code"].confidence * 0.5 + 0.5 * max(la[0].score, F["issuer_code"].confidence) + 0.1)
                    else: F["issuer_code"].confidence *= 0.6; F["issuer_code"].note = f"lat line gives '{lc}'"
        elif la:
            F["issuer"] = Field(txt(la[:1], "lat"), mean_conf(la[:1]))
        if la: F["issuer_lat"] = Field(txt(la[:1], "lat"), mean_conf(la[:1]))

        # 5. серия и номер
        cand = zone["f5_number"] + [l for l in vals if l not in zone["f5_number"]]
        for l in cand:
            digs = re.sub(r"\D", "", digits_fix(l.text)) if len(re.sub(r"\s", "", l.text)) <= 14 else ""
            if len(digs) == 10 and not parse_date(l.text):
                F["number"] = Field(f"{digs[:2]} {digs[2:4]} {digs[4:]}", l.score * (1.0 if l in zone["f5_number"] else 0.7))
                break
        # 8. регион проживания
        ru, la = by_script(zone["f8_region"])
        if ru and not la: la = reread_latin(ru[-1], la)
        if ru:
            t = txt(ru, "ru"); t = re.sub(r"\.(?=\S)", ". ", t); t = re.sub(r"\s+\(", " (", t)
            F["region"] = Field(t, mean_conf(ru))
        if la:
            t = re.sub(r"\.(?=\S)", ". ", txt(la, "lat"))
            F["region_lat"] = Field(t, mean_conf(la))
        # 9. категории: разбор строки + пословное (по ячейкам с рамками) перечитывание и голосование
        zl = zone["f9_categories"]
        if not zl and cell is not None:
            zl = [Line(text="", score=0.0, box=np.zeros((4, 2), np.float32), cx=0.6, cy=0.865, x0=0.35, x1=0.82, y0=0.853, y1=0.877)]
        cats, cats_conf, cats_note = parse_categories_zone(card, zl, (ca, cb, cdx), cell, rec)
        if cats:
            F["categories"] = Field(cats, cats_conf, cats_note)
        # сверка латиницы с кириллицей по ICAO-транслитерации (исправляет «KAZANSKIL» → «KAZANSKII»)
        for ru_k, la_k in (("surname", "surname_lat"), ("given_names", "given_names_lat"), ("birth_place", "birth_place_lat"),
                           ("region", "region_lat"), ("issuer", "issuer_lat")):
            if ru_k in F and la_k in F and isinstance(F[ru_k].value, str) and isinstance(F[la_k].value, str):
                if F[ru_k].confidence + 0.02 < F[la_k].confidence:   # кириллица менее уверенна -> чиним её по латинице
                    new_ru, chr_ = repair_ru_from_lat(F[ru_k].value, F[la_k].value)
                    if chr_:
                        F[ru_k].note = f"было '{F[ru_k].value}', исправлено по латинице «{F[la_k].value}»"; F[ru_k].value = new_ru
                        F[ru_k].confidence = max(F[ru_k].confidence, 0.8 * F[la_k].confidence); F[la_k].note = None
                        if ru_k == "given_names":
                            parts = new_ru.split(); F["name"].value = parts[0]
                            if len(parts) > 1: F["patronymic"].value = " ".join(parts[1:])
                        continue
                new, ch = snap_lat_to_ru(F[la_k].value, F[ru_k].value)
                if ch:
                    F[la_k].note = f"было '{F[la_k].value}', исправлено по транслитерации «{F[ru_k].value}»"
                    F[la_k].value = new
        for k in ("birth_date", "issue_date", "expiry_date"):
            if k in F: F[k].valid = parse_date(F[k].value) is not None
        if all(k in F for k in ("birth_date", "issue_date", "expiry_date")):
            b_, i_, e_ = (parse_date(F[k].value) for k in ("birth_date", "issue_date", "expiry_date"))
            if b_ and i_ and e_:
                ok = b_[1] < i_[1] < e_[1]
                for k in ("birth_date", "issue_date", "expiry_date"): F[k].valid = F[k].valid and ok
        if "number" in F: F["number"].valid = bool(re.fullmatch(r"\d{2} \d{2} \d{6}", F["number"].value))
        if "issuer_code" in F: F["issuer_code"].valid = bool(re.fullmatch(r"\d{4}", str(F["issuer_code"].value)))
        if "categories" in F: F["categories"].valid = all(c in CATEGORIES for c in F["categories"].value)
        ex.extra["checks"] = {"dates_order_ok": bool(b and i and e and b < i < e)}
        return ex
