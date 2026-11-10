"""СТС, оборотная сторона: собственник, адрес (республика/край/область, район, нас. пункт, улица, дом/корп./кв.), особые отметки,
код подразделения ГИБДД, дата выдачи, номер (сверху красный и снизу чёрный — должны совпасть).
Значения — бледная матричная печать, смещённая относительно подписей: привязка по последовательности (anchors) + слой значений (ink)."""
from __future__ import annotations
import re
from datetime import date
import cv2
import numpy as np
from ..ocr import Line
from ..textutils import fold, partial_ratio, _lev, to_cyr
from .. import anchors
from ..ink import split_ink
from .. import lexicon as LX
from ..glare import glare_mask, flag_fields
from .base import DocType, Extraction, Field, register

SPECS = [anchors.LabelSpec("owner", ["Собственник (владелец)"], multi=2), anchors.LabelSpec("region", ["Республика, край, область"]),
         anchors.LabelSpec("district", ["Район"]), anchors.LabelSpec("locality", ["Нас. пункт"]), anchors.LabelSpec("street", ["Улица"]),
         anchors.LabelSpec("house", ["Дом корп. кв."]), anchors.LabelSpec("marks", ["Особые отметки"])]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
NUM_RE = re.compile(r"(?<!\d)(\d{2})\s*(\d{2})\s*(\d{6})(?!\d)")


def _dedupe(primary: list[Line], extra: list[Line]) -> list[Line]:
    out = list(primary)
    for e in extra:
        if not any(min(e.x1, p.x1) - max(e.x0, p.x0) > 0.4 * (e.x1 - e.x0) and min(e.y1, p.y1) - max(e.y0, p.y0) > 0.5 * (e.y1 - e.y0) for p in primary): out.append(e)
    return out


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[|_*~`^<>]+", " ", t)).strip(" .,:;-")


def _ru(t: str) -> str:
    t = _clean(t)
    t = re.sub(r'^[0OОоo]{3}(?=\s*")', "ООО", t) if re.search(r"[А-Яа-я]{3}", t) else re.sub(r'^[0О]{3}(?=\s*")', "OOO", t)
    t = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', t)
    t = re.sub(r'^(ООО|ОАО|ЗАО|ПАО|АО|OOO|OAO|ZAO|PAO)\s+"', r'\1"', t)
    return to_cyr(t) if re.search(r"[А-Яа-я]{3}", t) else t


def _num(t: str):
    m = NUM_RE.search(t.replace("O", "0").replace("О", "0"))
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else None


def _month(tokens: list[str]):
    best = (0.0, None)
    for t in tokens:
        f = fold(t)
        if len(f) < 3: continue
        for i, mname in enumerate(MONTHS):
            r = 1 - _lev(f, fold(mname)) / max(len(f), len(mname))
            if r > best[0]: best = (r, i + 1)
    return best[1] if best[0] >= 0.66 else None, best[0]


_LEGAL = re.compile(r"^(ООО|ОАО|ЗАО|ПАО|АО|НАО|АНО|НКО|ГУП|МУП|ФГУП|МБУ|ГБУ|ФГБУ|ГКУ|МКУ|ТСЖ|СНТ|ПК|КФХ|ОГУП|ГУ|МУ|OOO|OAO|ZAO|PAO|LLC|JSC)\b", re.I)


def owner_type(text: str):
    """-> ('legal_entity'|'sole_proprietor'|'individual'|'unknown', note). Юрлицо: ООО/АО/… (в т.ч. «ООО"ИМЯ"» слитно); ИП; иначе ФИО."""
    t = re.sub(r"\s+", " ", (text or "").strip().replace("0", "О") if re.match(r"^[0О]{3}", text or "") else (text or "").strip())
    if re.match(r"^(ИП|IP)\b", t, re.I) or re.match(r"^ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ", t, re.I): return "sole_proprietor", "ИП"
    if _LEGAL.match(t) or re.search(r'["«»“”]', t) and re.search(r"[А-ЯA-Z]{2,}", t) and len(t.split()) <= 4: return "legal_entity", "организационно-правовая форма/кавычки в наименовании"
    if len(t.split()) >= 2: return "individual", "ФИО"
    return "unknown", None


def join_breaks(text: str):
    """Склейка переноса слова без дефиса между строками/токенами: «СОБСТВЕН НИК» -> «СОБСТВЕННИК», только если склейка есть в словаре отметок,
    а ни одна из частей сама словом не является. -> (text, [склейки])."""
    lex = LX.get("sts_back/marks_words")
    toks = text.split(); out = []; joins = []; i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            a, b = toks[i], toks[i + 1]; ab = (a + b).upper().strip(".,")
            if ab in lex and a.strip(".,") not in lex and b.strip(".,") not in lex and len(a) >= 3 and len(b) >= 2:
                out.append(a + b); joins.append(f"{a} {b} → {a + b}"); i += 2; continue
        out.append(toks[i]); i += 1
    return " ".join(out), joins


@register
class STSBack(DocType):
    name = "sts_back"; title_ru = "Свидетельство о регистрации ТС (СТС), оборотная сторона"
    card_size = (704, 1000); aspect = 1000 / 704
    keywords = [("СОБСТВЕННИК (ВЛАДЕЛЕЦ)", 3), ("РЕСПУБЛИКА, КРАЙ, ОБЛАСТЬ", 2), ("НАС. ПУНКТ", 1), ("ОСОБЫЕ ОТМЕТКИ", 2), ("КОД ПОДРАЗДЕЛЕНИЯ ГИБДД", 2), ("УЛИЦА", 0.5)]
    implemented = True; hybrid_internal = True
    expected_fields = ["sts_number", "owner", "region", "district", "locality", "street", "house", "gibdd_code", "issue_date"]

    def extract(self, card, lines, rec=None, cell=None, ocr=None, paddle=None) -> Extraction:
        ex = Extraction(); F = ex.fields; ex.extra["side"] = "back"
        H, W = card.shape[:2]
        L, V = split_ink(card)
        read = (lambda im: paddle(im, merge=False)) if paddle is not None else (lambda im: ocr(im) if ocr else lines)
        card_lines = read(card)
        v_lines = read(V)
        found = anchors.locate_labels([l for l in card_lines if l.x0 < 0.4], SPECS, thr=0.6)
        pos = anchors.label_positions(SPECS, found)
        ex.extra["labels"] = {"found": [s.key for s in SPECS if s.key in found], "missing": [s.key for s in SPECS if s.key not in found]}
        # --- значения: слой значений + недостающее из полной карточки (без подписей)
        lab_words = ["Собственник", "владелец", "Республика", "край", "область", "Район", "Нас", "пункт", "Улица", "Дом", "корп", "кв", "Особые", "отметки", "Код", "подразделения", "ГИБДД"]
        def is_label(l):
            if l.x0 < 0.12 and l.cy > 0.18: return True
            t = fold(l.text)
            return any(partial_ratio(w, l.text) > 0.85 and len(re.sub(r"\W", "", l.text)) <= len(w) + 3 for w in lab_words) and l.score > 0.4 and l.x0 < 0.3
        extra = [l for l in card_lines if not is_label(l) and l.cy > 0.18]
        pool = _dedupe([l for l in v_lines if l.cy > 0.18], extra)
        ys = [p[0] for p in pos]
        y_house = ys[SPECS.index(next(s for s in SPECS if s.key == "house"))]
        y_marks = ys[SPECS.index(next(s for s in SPECS if s.key == "marks"))]
        vals = [l for l in pool if l.x0 > 0.12 and l.cy < y_marks - 0.012 and abs(l.cy - y_house) > 0.015 and re.search(r"[A-Za-zА-Яа-я]{2}", l.text)]
        assign, diag = anchors.assign_values(SPECS[:5], pos[:5], vals, win=(-0.9, 1.6), multi_gap=0.06, skip_value=1.5)
        ex.extra["sequence"] = diag
        def put(key, ls, joiner=" "):
            if not ls: return None
            txt = _ru(joiner.join(l.text for l in sorted(ls, key=lambda l: (l.cy, l.x0))))
            if key in ("locality", "street"): txt = re.sub(r"^(г|с|д|п|х|ст|ул|пер|пр|ш|пос)(?=[А-ЯЁ][а-яё])", r"\1 ", txt)
            if txt: F[key] = Field(txt, float(np.mean([l.score for l in ls])), None, valid=None)
            return txt
        own = assign.get("owner", [])
        if own:
            ru = [l for l in own if re.search(r"[А-Яа-я]{3}", l.text)]; la = [l for l in own if l not in ru]
            if ru: put("owner", ru[:1])
            if la: put("owner_lat", la[:1])
            if not ru and la: put("owner", la[:1])
        if "owner" in F or "owner_lat" in F:
            ot, why = owner_type(F["owner"].value if "owner" in F else F["owner_lat"].value)
            F["owner_type"] = Field(ot, min([F[k].confidence for k in ("owner", "owner_lat") if k in F]), why, valid=ot != "unknown")
        for k in ("region", "district", "locality", "street"): put(k, assign.get(k, []))
        for k, voc in (("street", ["ул", "пр-кт", "пер", "б-р", "ш", "наб", "пл", "проезд", "тер", "мкр", "аллея", "линия", "пр"]), ("locality", ["г", "п", "пос", "с", "д", "х", "ст", "аул", "пгт", "рп"]), ("district", ["р-н", "район"])):
            if k in F:
                parts = F[k].value.split(" ", 1)
                if parts[0].lower() not in voc and len(parts[0]) <= 5:
                    best = min(voc, key=lambda v: _lev(parts[0].lower(), v))
                    if _lev(parts[0].lower(), best) <= 1 and len(best) >= 2: F[k].value = best + (" " + parts[1] if len(parts) > 1 else ""); F[k].note = f"префикс исправлен ('{parts[0]}' -> '{best}')"
        # валидаторы форматов адреса
        PRE = {"region": r"(республика|край|область|округ|г\.? ?(москва|санкт|севастополь)|москва|санкт|севастополь)", "district": r"^(р-н|район|г\b|г\.|городской)", "locality": r"^(г|п|пос|с|д|х|ст|ст-ца|аул|дер|пгт|рп|кп|снт)\b", "street": r"^(ул|пр-кт|пр|пер|б-р|ш|наб|пл|проезд|тер|мкр|аллея|линия)\b"}
        for k in ("region", "district", "locality", "street"):
            if k in F:
                v = F[k].value
                if k == "region": v = v[:1].upper() + v[1:]
                else:
                    parts = v.split(" ", 1)
                    if len(parts) == 2: v = parts[0] + " " + parts[1][:1].upper() + parts[1][1:]
                F[k].value = v
        if "region" in F and isinstance(F["region"].value, str) and len(F["region"].value) >= 5:   # субъект РФ — по лексикону (фраза целиком)
            from .. import lexicon as LX
            sn = LX.snap(F["region"].value, "sts_back/regions", thr=0.2)
            if sn["status"] == "snapped":
                F["region"].note = ((F["region"].note + "; ") if F["region"].note else "") + LX.note_for(sn, "регион"); F["region"].value = sn["value"]
            elif sn["status"] == "ambiguous":
                F["region"].note = ((F["region"].note + "; ") if F["region"].note else "") + LX.note_for(sn, "регион"); F["region"].confidence *= 0.9
        for k, pat in PRE.items():
            if k in F:
                F[k].valid = bool(re.search(pat, F[k].value, re.I))
                if k == "district": F[k].value = re.sub(r"^р\s*-?\s*н\b", "р-н", F[k].value, flags=re.I)
        # --- дом / корп. / кв. : по x-позициям подписей в строке «Дом»
        row = [l for l in card_lines if abs(l.cy - y_house) < 0.025]
        korp_l = next((l for l in row if partial_ratio("корп.", l.text) > 0.8 and l.x0 > 0.25), None)
        kv_l = next((l for l in row if partial_ratio("кв.", l.text) > 0.8 and l.x0 > 0.5), None)
        x_korp0, x_korp1 = (korp_l.x0, korp_l.x1) if korp_l else (0.34, 0.47)
        x_kv0, x_kv1 = (kv_l.x0, kv_l.x1) if kv_l else (0.59, 0.67)
        y0, y1 = int(max(0, y_house - 0.028) * H), int(min(1, y_house + 0.022) * H)
        dom_l = next((l for l in row if partial_ratio("Дом", l.text) > 0.8 and l.x0 < 0.2 and len(l.text) <= 6), None)
        x_h0 = max(0.17, dom_l.x1 + 0.015) if dom_l else 0.2
        def read_zone(xa, xb, min_conf):
            """Нормализованный кроп (карточка и слой значений), 4x, чтение Paddle по строке; пустое поле -> ''."""
            cands = []
            for l in card_lines + v_lines:
                if abs(l.cy - y_house) < 0.02 and l.x0 >= xa - 0.015 and l.x1 <= xb + 0.04:
                    t = re.sub(r"[^0-9A-Za-zА-Яа-я/\-]", "", l.text)
                    if t and re.search(r"\d", t) and len(t) <= 6 and l.score >= min_conf: cands.append((t, l.score))
            if cands: return max(cands, key=lambda c: c[1])
            for im in (card, V):
                crop = im[y0:y1, int(xa * W):int(xb * W)]
                if crop.size == 0 or paddle is None or not hasattr(paddle, "recognize_crop"): continue
                g = cv2.normalize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX)
                big = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                t, sc = paddle.recognize_crop(big)
                t = re.sub(r"[^0-9A-Za-zА-Яа-я/\-]", "", t).replace("g", "9").replace("q", "9")
                if t and re.search(r"\d", t) and len(t) <= 6 and sc >= min_conf: cands.append((t, sc))
            return max(cands, key=lambda c: c[1]) if cands else ("", 0.8)
        hv, hs = read_zone(x_h0, x_korp0 - 0.015, 0.3)
        F["house"] = Field(hv, hs, "пусто" if not hv else "из слоя значений (бледная печать)", valid=(bool(re.fullmatch(r"\d{1,4}[А-Яа-яA-Za-z/\-\d]{0,3}", hv)) if hv else None))
        kv_, ks = read_zone(x_korp1 + 0.015, x_kv0 - 0.015, 0.5)
        F["building"] = Field(kv_, ks, "пусто" if not kv_ else None, valid=True if not kv_ else None)
        ap, aps = read_zone(x_kv1 + 0.01, 0.97, 0.5)
        F["apartment"] = Field(ap, aps, "пусто" if not ap else None, valid=True if not ap else None)
        # --- особые отметки: ниже подписи до полосы с рисунком
        mk = [l for l in pool if y_marks - 0.01 < l.cy < y_marks + 0.075 and l.x0 > 0.3 and re.search(r"[А-Яа-яA-Za-z]{3}", l.text)]
        mk.sort(key=lambda l: (round(l.cy, 2), l.x0))
        mk_txt, joins = join_breaks(_ru(" ".join(l.text for l in mk))) if mk else ("", [])
        F["special_marks"] = Field(mk_txt, float(np.mean([l.score for l in mk])) if mk else 0.8, ("пусто" if not mk else (("склеены переносы без дефиса: " + ", ".join(joins)) if joins else None)), valid=True)
        # --- код подразделения ГИБДД (7 цифр, зачёркнут линией) и подпись ниже
        gl = [l for l in card_lines if re.search(r"ГИБДД|ГИБД", l.text.upper().replace("T", "Т")) and l.cy > 0.7]
        y_g = gl[0].cy if gl else 0.855
        cands = []
        for l in card_lines + v_lines:
            if y_g - 0.075 < l.cy < y_g - 0.005 and l.x0 > 0.45:
                d = re.sub(r"\D", "", l.text)
                if len(d) == 7: cands.append((d, l.score))
        if not cands:   # перечитать полосу
            y0g, y1g = int((y_g - 0.06) * H), int((y_g - 0.01) * H)
            crop = card[y0g:y1g, int(0.55 * W):int(0.98 * W)]
            if crop.size and paddle is not None and hasattr(paddle, "recognize_crop"):
                g = cv2.normalize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX)
                for src in (g, V[y0g:y1g, int(0.55 * W):int(0.98 * W)][:, :, 1]):
                    big = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    t, s_ = paddle.recognize_crop(big)
                    if len(re.sub(r"\D", "", t)) == 7: cands.append((re.sub(r"\D", "", t), s_ * 0.8))
        if cands:
            d, s = max(cands, key=lambda c: c[1]); F["gibdd_code"] = Field(d, s, "зачёркнут линией на бланке", valid=True)
        # --- дата: «30» сентября 20 21 г.
        band = [l for l in card_lines + v_lines if y_g + 0.04 < l.cy < 0.965 and l.x0 > 0.05]
        toks = [(t, l) for l in band for t in re.split(r"[\s\"“”«»_]+", l.text) if t]
        day = next(((t, l) for t, l in sorted(toks, key=lambda x: x[1].x0) if re.fullmatch(r"\d{1,2}", re.sub(r"\D", "", t)) and 1 <= int(re.sub(r"\D", "", t) or 0) <= 31 and l.x0 < 0.55 and len(re.sub(r"\D", "", t)) <= 2 and re.sub(r"\D", "", t) == t.strip("_ ")), None)
        if day is None:
            day = next(((t, l) for t, l in sorted(toks, key=lambda x: x[1].x0) if re.search(r"\d{2}", t) and l.x0 < 0.55 and 1 <= int(re.sub(r"\D", "", t)[:2] or 0) <= 31), None)
        mo, mr = _month([t for t, _ in toks])
        yl = sorted([l for l in band if l.x1 > 0.72 and l.x0 > 0.65], key=lambda l: l.x0)
        yd = "".join(re.sub(r"\D", "", l.text) for l in yl)
        yr = int(yd[:4]) if len(yd) >= 4 and yd[:2] == "20" else (2000 + int(yd[:2]) if len(yd) == 2 else None)
        if day and mo and yr:
            dd = int(re.sub(r"\D", "", day[0])[:2])
            try:
                dt = date(yr, mo, dd); ok = 2005 <= yr <= date.today().year
                F["issue_date"] = Field(dt.isoformat(), float(np.mean([day[1].score, 0.5 + 0.5 * mr, 0.8])), f"{dd} {MONTHS[mo-1]} {yr}", valid=ok)
            except ValueError: pass
        elif day and mo:   # год обрезан краем снимка — отдаём часть и помечаем partial
            dd = int(re.sub(r"\D", "", day[0])[:2])
            F["issue_date"] = Field(f"{dd:02d}.{mo:02d}.????", 0.4, f"{dd} {MONTHS[mo-1]}, год не прочитан (возможно, обрезан краем кадра)", valid=False, partial=True)
        if "issue_date" in F and band and max(l.y1 for l in band) > 0.985: F["issue_date"].partial = True; F["issue_date"].note = ((F["issue_date"].note or "") + "; строка даты у самого края кадра — возможна обрезка").strip("; ")
        # --- номер: красный сверху и чёрный снизу
        top = bot = None
        for l in card_lines + v_lines:
            n = _num(l.text)
            if n and l.cy < 0.3 and (top is None or l.score > top[1]): top = (n, l.score)
            if n and l.cy > 0.9 and (bot is None or l.score > bot[1]): bot = (n, l.score)
        if top: F["sts_number_top"] = Field(top[0], top[1], "красный", valid=True)
        if bot: F["sts_number_bottom"] = Field(bot[0], bot[1], "чёрный, внизу", valid=True)
        if top or bot:
            if top and bot:
                ag = top[0] == bot[0]
                F["sts_number"] = Field(top[0] if ag else (top[0] if top[1] >= bot[1] else bot[0]), (min(1.0, (top[1] + bot[1]) / 2 + 0.03) if ag else 0.5 * max(top[1], bot[1])), "верх = низ" if ag else f"РАСХОЖДЕНИЕ: верх {top[0]}, низ {bot[0]}", valid=ag)
            else:
                one = top or bot; F["sts_number"] = Field(one[0], one[1] * 0.8, "найден только один номер", valid=True)
        try:
            gm, gf = glare_mask(card)
            if gf > 0: flag_fields(F, SPECS, pos, gm, ex)
        except Exception as e:
            ex.extra["glare_error"] = repr(e)[:100]
        return ex
