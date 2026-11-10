"""СТС (свидетельство о регистрации ТС), лицевая сторона. Поля привязываются к подписям по ПОСЛЕДОВАТЕЛЬНОСТИ (anchors.py);
значения печатаются бледным шрифтом поверх/между подписями и обычно смещены относительно них — поэтому слои печати разделяются (ink.py)."""
from __future__ import annotations
import re
import numpy as np
from ..ocr import Line
from ..textutils import fold
from .. import anchors, vehicle as V
from ..ink import split_ink
from .. import makemodel as MM
from .base import DocType, Extraction, Field, register

SPECS = [
    anchors.LabelSpec("plate", ["Регистрационный знак"]),
    anchors.LabelSpec("vin", ["Идентификационный номер (VIN)"]),
    anchors.LabelSpec("make_model", ["Марка, модель"], multi=2),
    anchors.LabelSpec("vehicle_type", ["Тип ТС"]),
    anchors.LabelSpec("category", ["Категория ТС (ABCD, прицеп)"]),
    anchors.LabelSpec("year", ["Год выпуска ТС"]),
    anchors.LabelSpec("chassis", ["Шасси (рама) №"]),
    anchors.LabelSpec("body", ["Кузов (кабина, прицеп) №"]),
    anchors.LabelSpec("color", ["Цвет"]),
    anchors.LabelSpec("power", ["Мощность двигателя, кВт/л. с."]),
    anchors.LabelSpec("eco_class", ["Экологический класс"]),
    anchors.LabelSpec("max_mass_kg", ["Технически допустимая max масса, кг"]),
    anchors.LabelSpec("curb_mass_kg", ["Масса в снаряженном состоянии, кг"]),
    anchors.LabelSpec("temp_reg", ["Срок временной регистрации"]),
]
MAKES = ["КАМАЗ", "ГАЗ", "ЗИЛ", "УАЗ", "ВАЗ", "ЛАДА", "МАЗ", "УРАЛ", "ПАЗ", "ЛИАЗ", "НЕФАЗ", "ГАЗЕЛЬ", "KAMAZ", "MAN", "SCANIA", "VOLVO", "DAF", "IVECO", "MERCEDES-BENZ", "RENAULT", "FORD",
         "TOYOTA", "HYUNDAI", "KIA", "NISSAN", "MAZDA", "HONDA", "VOLKSWAGEN", "SKODA", "CHEVROLET", "BMW", "AUDI", "LADA", "GAZ", "UAZ", "VAZ", "MAZ", "URAL", "SITRAK", "SHACMAN", "HOWO", "FAW", "JAC"]


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[|_*~`^<>]+", " ", t)).strip(" .,:;-")


def _norm_ident(raw: str):
    s = re.sub(r"[^0-9A-Za-zА-Яа-я]", "", raw).upper()
    s = "".join(V._VIN_FIX.get(c, c) if c in "ХСТАВЕКМНРУ" else c for c in s)
    if not s: return None, False
    if 8 <= len(s) <= 13 and _lev2(anchors.fold2(s), "ОТСУТСТВУЕТ") <= 3: return "ОТСУТСТВУЕТ", True
    return s, 6 <= len(s) <= 17


def _norm_make(raw: str):
    t = _clean_text(raw)
    m = re.search(r"([A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9\-]{1,})\s+(\d{3,5}(?:\s?[-–—]\s?[A-Za-z0-9]{1,3})?)", t)
    if not m: m = re.search(r"([A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{1,})", t)
    if not m: return None, False
    w = m.group(1).upper(); mod = re.sub(r"\s", "", re.sub(r"[–—]", "-", (m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else ""))).upper()
    w2 = re.sub(r"[0-9]", lambda c: {"7": "Z", "3": "З", "0": "О"}.get(c.group(0), c.group(0)), w)
    best = min(MAKES, key=lambda k: _lev2(fold(k), fold(w2)))
    if _lev2(fold(best), fold(w2)) <= 1: w = best
    return (w + (" " + mod if mod else "")).strip(), bool(mod) and w in MAKES


def _lev2(a, b):
    from ..textutils import _lev
    return _lev(a, b)


def _norm_temp(raw: str):
    m = re.search(r"\d{2}\.\d{2}\.\d{4}", raw)
    return (m.group(0), True) if m else (None, False)


NORMALIZERS = {
    "plate": lambda t: (lambda r: (r[0], r[1], r[2]))(V.norm_plate(t)),
    "vin": V.norm_vin,
    "vehicle_type": lambda t: (*V.norm_vehicle_type(t), None),
    "category": lambda t: (lambda r: ((f"{r[0]['category'] or ''}/{r[0]['class'] or ''}".strip("/") if r[0] else None), r[1], None))(V.norm_category(t)),
    "year": lambda t: (*V.norm_year(t), None),
    "chassis": lambda t: (*_norm_ident(t), None),
    "body": lambda t: (*_norm_ident(t), None),
    "color": lambda t: (*V.norm_color(t), None),
    "power": V.norm_power,
    "eco_class": lambda t: (*V.norm_eco(t), None),
    "max_mass_kg": lambda t: (*V.norm_mass(t), None),
    "curb_mass_kg": lambda t: (*V.norm_mass(t), None),
    "make_model": lambda t: (lambda r: (*r, "суффикс модели после дефиса мог быть прочитан неточно (S↔5)" if r[0] and re.search(r"-\d{2}$", r[0]) else None))(_norm_make(t)),
    "temp_reg": lambda t: (*_norm_temp(t), None),
}
# какие поля важны для решения «слой значений прочитан достаточно хорошо»
CORE = ("plate", "vin", "year", "chassis", "body", "power", "eco_class", "max_mass_kg")


def _tuple3(r):
    return (r + (None,))[:3] if len(r) < 3 else r


def _usable_lines(ls, W_edge=0.995):
    out = []
    for l in ls:
        al = re.sub(r"[\W_]", "", l.text)
        if not al or (l.x1 - l.x0) < 0.012 and len(al) < 2: continue
        out.append(l)
    return out


@register
class STS(DocType):
    name = "sts"; title_ru = "Свидетельство о регистрации ТС (СТС), лицевая сторона"
    card_size = (704, 1000); aspect = 1000 / 704
    keywords = [("СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ ТС", 3), ("CERTIFICAT D'IMMATRICULATION", 1), ("РЕГИСТРАЦИОННЫЙ ЗНАК", 1.5), ("ИДЕНТИФИКАЦИОННЫЙ НОМЕР VIN", 1.5),
                ("МАРКА МОДЕЛЬ", 1), ("ЭКОЛОГИЧЕСКИЙ КЛАСС", 1), ("СРОК ВРЕМЕННОЙ РЕГИСТРАЦИИ", 1)]
    implemented = True
    hybrid_internal = True
    expected_fields = ["plate", "vin", "make_model", "make_model_ru", "make_model_lat", "vehicle_type", "category", "year", "chassis", "body", "color", "power", "eco_class", "max_mass_kg", "curb_mass_kg", "pts", "sts_number"]

    # ---- одна «попытка»: линии значений от одного движка -> поля-кандидаты
    def _fields_from(self, specs, pos, vlines, W):
        assign, diag = anchors.assign_values(specs, pos, vlines)
        cand = {}
        for key, ls in assign.items():
            fn = NORMALIZERS.get(key)
            if fn is None: continue
            txt = " ".join(l.text for l in ls)
            if key == "make_model":   # две строки: кириллица и латиница дублируют друг друга — берём первую, где есть марка и модель
                tl = [l.text for l in ls]
                txt = next((t for t in tl if _norm_make(t)[1]), None) or max(tl, key=lambda t: len(re.sub(r"\W", "", t)))
            val, ok, note = _tuple3(fn(txt))
            if val is None: continue
            conf = float(np.mean([l.score for l in ls]))
            partial = bool(max(l.x1 for l in ls) >= 0.985 or txt.rstrip().endswith("."))
            cand[key] = dict(value=val, valid=bool(ok), conf=conf, note=note, partial=partial, raw=txt, lines=ls)
        return cand, diag

    def extract(self, card, lines, rec=None, cell=None, ocr=None, paddle=None) -> Extraction:
        ex = Extraction(photo_zone=None); F = ex.fields; ex.extra["side"] = "front"
        H, W = card.shape[:2]
        Limg, Vimg = split_ink(card)
        # 1) подписи: слой подписей, монотонное сопоставление
        llines = ocr(Limg) if ocr is not None else lines
        found = anchors.locate_labels(llines, SPECS)
        if len(found) < 4:                       # слой подписей не помог — пробуем исходные строки
            found2 = anchors.locate_labels(lines, SPECS)
            if len(found2) > len(found): found = found2
        pos = anchors.label_positions(SPECS, found)
        ex.extra["labels"] = {"found": [k for k in (s.key for s in SPECS) if k in found], "missing": [s.key for s in SPECS if s.key not in found]}
        # 2) значения: Tesseract по слою значений (дёшево) -> при слабом результате Paddle (гибрид)
        cands_all = []; engines = []; used_paddle = False
        if paddle is not None:
            pv = _usable_lines(paddle(Vimg))
            if pv:
                c, d = self._fields_from(SPECS, pos, pv, W); cands_all.append(("paddle", c)); engines.append(("paddle", d)); used_paddle = True
        good = sum(1 for k in CORE if k in (cands_all[0][1] if cands_all else {}) and cands_all[0][1][k]["valid"] and cands_all[0][1][k]["conf"] >= 0.7)
        tv = []
        if ocr is not None and good < len(CORE) - 1:
            tv = _usable_lines(ocr(Vimg))
            if tv:
                c, d = self._fields_from(SPECS, pos, tv, W); cands_all.append(("tesseract", c)); engines.append(("tesseract", d))
        ex.extra["value_engines"] = [{"engine": n, **d} for n, d in engines]
        # 3) выбор лучшего кандидата по каждому полю
        best = {}
        for src, cand in cands_all:
            for k, c in cand.items():
                sc = (c["valid"], c["conf"] if c["valid"] else c["conf"] * 0.5)
                if k not in best or sc > best[k][0]: best[k] = (sc, src, c)
        # 4) повторное чтение «потерянных» значений кропом по расчётной полосе (подписи с известным y)
        reread = getattr(paddle, "recognize_crop", None) or rec or (cell and (lambda im: cell(im, None, 7, 2)))
        if reread is not None:
            shift = (engines[-1][1].get("shift_rel", 0.4) if engines else 0.4)
            for i, sp in enumerate(SPECS):
                if (sp.key in best and best[sp.key][2]["valid"]) or sp.key not in ("make_model", "color", "vehicle_type", "category", "plate", "year"): continue
                ycen = pos[i][0] + shift * 0.04
                wins = ((-0.012, 0.024), (-0.02, 0.012), (0.0, 0.03), (0.002, 0.036), (0.008, 0.032))
                if sp.key == "make_model":   # две плотные строки (кириллица+латиница): окна на нижней/средней части, подпись выше значения
                    ycen = pos[i][0]; wins = ((0.007, 0.033), (0.011, 0.033), (0.015, 0.041), (0.011, 0.037), (0.007, 0.029), (0.015, 0.033))
                tries = []
                for dy0, dy1 in wins:
                    y0, y1 = int(max(0, (ycen + dy0) * H)), int(min(H, (ycen + dy1) * H))
                    x0 = int(0.28 * W) if sp.key not in ("make_model",) else int(0.45 * W)
                    crop = Vimg[y0:y1, x0:int(0.99 * W)]
                    if crop.size == 0: continue
                    r = reread(crop); t, sc = (r if isinstance(r, tuple) else (r, 0.5))
                    fn = NORMALIZERS[sp.key]; val, ok, note = _tuple3(fn(t)) if t else (None, False, None)
                    if val is not None: tries.append((ok, sc, val, note, t))
                if tries:
                    from collections import Counter
                    cnt = Counter(x[2] for x in tries if x[1] >= 0.5)
                    ok, sc, val, note, t = max(tries, key=lambda x: (x[0], cnt.get(x[2], 0), x[1]))
                    if sp.key in best and not ok: continue
                    best[sp.key] = ((ok, sc * 0.9), "region-reread", dict(value=val, valid=bool(ok), conf=sc * 0.9, note=((note + "; ") if note else "") + "перечитано кропом", partial=False, raw=t, lines=[]))
        for k, (_, src, c) in best.items():
            val = c["value"]
            conf = c["conf"] * (1.0 if c["valid"] else 0.6) * (0.85 if c["partial"] else 1.0)
            note = c["note"]
            if src != "tesseract": note = ((note + "; ") if note else "") + f"engine={src}"
            if c["partial"]: note = ((note + "; ") if note else "") + "значение у края карточки — возможно обрезано"
            F[k] = Field(val, conf, note, valid=c["valid"], partial=True if c["partial"] else None)
        # 4a) цвет: если подпись не найдена или уверенность низкая — несколько окон/слоёв + словарь
        if paddle is not None and ("color" not in F or F["color"].confidence < 0.8):
            try:
                ic = [s_.key for s_ in SPECS].index("color"); sh = (engines[-1][1].get("shift_rel", 0.4) if engines else 0.4)
                cr = MM.read_color(card, Vimg, pos[ic][0] + sh * 0.04, paddle, V.norm_color)
            except Exception as e:
                cr = None; ex.extra["color_error"] = repr(e)[:120]
            if cr and ("color" not in F or cr["conf"] > F["color"].confidence):
                F["color"] = Field(cr["value"], cr["conf"], f"словарь цветов; {cr['n']} совпад. чтений в разных окнах/слоях (доля {cr['share']:.0%})", valid=True)
        # 4b) марка/модель: две плотные строки (кириллица + латиница) — отдельный разрез, перечитывание и сверка строк (makemodel.py)
        if paddle is not None:
            try:
                mm = MM.read_make_model(card, Vimg, pos[[s_.key for s_ in SPECS].index("make_model")][0], paddle, cell)
            except Exception as e:   # не должно ронять весь СТС
                mm = None; ex.extra["make_model_error"] = repr(e)[:120]
            if mm and mm["combined"] and (mm["valid"] or "make_model" not in F or mm["conf"] > F["make_model"].confidence):
                note = "; ".join(mm["notes"]) or None
                if mm["agree"] is False: note = (note + "; " if note else "") + "модель в строках ru/lat прочитана по-разному"
                for key, val in (("make_model_ru", mm["ru"]), ("make_model_lat", mm["lat"])):
                    if val: F[key] = Field(val, mm["conf"], note if key == "make_model_ru" or not mm["ru"] else None, valid=mm["valid"])
                F["make_model"] = Field(mm["combined"], mm["conf"], ("две строки бланка (кириллица / латиница), сверены по транслитерации" if mm["lines"] == 2 else "одна строка бланка") + ("; " + note if note else ""), valid=mm["valid"])
                ex.extra["make_model_check"] = {k: mm[k] for k in ("lines", "agree", "y_lab", "block", "cuts", "candidates")}
        if "power" in F and isinstance(F["power"].value, dict) and "hp" not in F["power"].value: F["power"].partial = True
        # 5) ПТС и номер СТС — по шаблону строки, независимо от подписей
        pool = list(lines) + llines + (tv or [])
        self._pts_and_number(F, pool, llines, cands_all, paddle, Vimg, H)
        ex.extra["checks"] = {"used_paddle": used_paddle, "sequence": engines[-1][1] if engines else None}
        return ex

    def _pts_and_number(self, F, pool, llines, cands_all, paddle, Vimg, H):
        # ПТС: строка внизу, «ПТС: 77АА123456»
        cands = []
        for l in pool:
            if l.cy < 0.74: continue
            t = l.text
            m = re.search(r"[:\-–]\s*([0-9A-Za-zА-Яа-я\s]{9,14})$", t) or re.search(r"([0-9]{2}\s?[0-9A-Za-zА-Яа-я]{2}\s?[0-9]{6})", t)
            if m:
                v, ok = V.norm_pts(m.group(1))
                if v: cands.append((ok, l.score, v))
        if not any(c[0] for c in cands) and paddle is not None:
            band = Vimg[int(0.80 * H):int(0.90 * H)]
            pad = max(0, int(0.5 * band.shape[1]) - band.shape[0])   # тонкие полосы раздувают RAM детектора (апскейл) — дополняем белым
            if pad: band = np.vstack([band, np.full((pad, band.shape[1], 3), 255, band.dtype)])
            for l in _usable_lines(paddle(band)):
                m = re.search(r"([0-9]{2}\s?[0-9A-Za-zА-Яа-я]{2}\s?[0-9]{6})", l.text)
                if m:
                    v, ok = V.norm_pts(m.group(1))
                    if v: cands.append((ok, l.score, v))
        if cands:
            ok, sc, v = max(cands); F["pts"] = Field(v, sc * (1.0 if ok else 0.6), None, valid=ok)
        # номер СТС (красный, внизу): «99 00 123456» — склеиваем токены одной строки y>0.85
        for src in (llines, pool):
            row = sorted([l for l in src if l.cy > 0.86], key=lambda l: (round(l.cy, 2), l.x0))
            txt = " ".join(l.text for l in row)
            m = re.search(r"(\d{2})\s*(\d{2})\s*(\d{6})", txt.replace("O", "0").replace("О", "0"))
            if m:
                sc = float(np.mean([l.score for l in row]))
                F["sts_number"] = Field(f"{m.group(1)} {m.group(2)} {m.group(3)}", sc, None, valid=True)
                break
