"""СТС, оборотная сторона — бланк с подписями «Субъект Российской Федерации», «Населенный пункт», «Корпус (строение)», «Квартира», «Дата выдачи»
(значения строки за строкой, матричная точечная печать, у физлица ФИО в пяти строках: фамилия/Latin, имя/Latin, отчество).
Поля читаются построчно по положению подписей; ФИО — по строкам чёрной печати (резка по горизонтальной проекции) + словари фамилий/имён/отчеств."""
from __future__ import annotations
import re
from collections import Counter
import cv2
import numpy as np
from ..textutils import partial_ratio, fold, translit, _lev
from .. import lexicon as LX
from .base import Field

LABELS = {
    "owner": "Собственник (владелец)", "region": "Субъект Российской Федерации", "district": "Район", "locality": "Населенный пункт", "street": "Улица",
    "house": "Дом", "building": "Корпус (строение)", "apartment": "Квартира", "marks": "Особые отметки", "gibdd": "Код подразделения ГИБДД", "date": "Дата выдачи",
}
STEM = {
    "locality": r"^\W*Населенн\w*\s*пункт\W*", "street": r"^\W*Улиц\w*\W*", "house": r"^\W*[ДΔD]ом\W*", "building": r"^\W*Корпус\W*(?:\(?строение\)?)?\W*",
    "apartment": r"^\W*Кварти\w*\W*", "gibdd": r"^\W*Код\s*подразделения\s*ГИБ\w*\W*", "date": r"^\W*Дата\s*выдачи\W*", "region": r"^\W*Субъект\w*\s*Российской\s*Федерации\W*",
}
NUM_RE = re.compile(r"(?<!\d)(\d{2})\s*(\d{2})\s*(\d{6})(?!\d)")
_D2 = {"О": "0", "O": "0", "В": "8", "B": "8", "З": "3", "Б": "6", "І": "1", "I": "1", "l": "1", "Z": "2", "S": "5", "Т": "7"}


def is_layout_b(card_lines) -> bool:
    return any(l.x0 < 0.3 and partial_ratio("Субъект Российской Федерации", l.text) > 0.85 for l in card_lines)


def _label_lines(card_lines):
    pos = {}
    for key, text in LABELS.items():
        best = None
        for l in card_lines:
            if l.x0 > 0.3: continue
            r = partial_ratio(text, l.text)
            if key == "house" and (len(re.sub(r"\W", "", l.text)) > 8 or r < 0.7): continue
            if r >= (0.8 if len(text) > 6 else 0.75) and (best is None or r > best[0]): best = (r, l)
        if best: pos[key] = best[1]
    return pos


def _segments(ink, min_h=6):
    prof = ink.sum(1); rows = []; st = None
    for i, v in enumerate(prof > 2):
        if v and st is None: st = i
        if not v and st is not None:
            if i - st >= min_h: rows.append((st, i))
            st = None
    if st is not None and len(prof) - st >= min_h: rows.append((st, len(prof)))
    if not rows: return rows
    med = float(np.median([b - a for a, b in rows if b - a >= 12] or [b - a for a, b in rows]))
    out = []
    for a, b in rows:   # слипшиеся строки (фамилия и Latin-вариант налезают друг на друга) режем по минимуму проекции
        n = int(round((b - a) / max(1.0, med)))
        if n >= 2 and b - a > 1.6 * med:
            cuts = [a]
            for k in range(1, n):
                c = a + int(round(k * (b - a) / n)); w = max(2, int(0.25 * med))
                seg = prof[max(a, c - w):min(b, c + w)]; c = max(a, c - w) + int(np.argmin(seg)); cuts.append(c)
            cuts.append(b); out += [(x, y, (4 if k == 0 else 0), (3 if k < len(cuts) - 2 else 4)) for k, (x, y) in enumerate(zip(cuts[:-1], cuts[1:]))]
        else: out.append((a, b, 4, 4))
    return out


def _read_line(paddle, gray, a, b, ink, scales=(3,), pt=4, pb=4):
    crop = gray[max(0, a - pt):b + pb]
    xs = np.where(ink[a:b].sum(0) > 0)[0]
    if len(xs) == 0: return "", 0.0
    xa, xb = max(0, xs.min() - 6), min(crop.shape[1], xs.max() + 6)
    best = ("", 0.0)
    for sc in scales:
        cr = cv2.cvtColor(cv2.resize(crop[:, xa:xb], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        for img in (cr, cv2.copyMakeBorder(cr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))):
            try: t, s = paddle.recognize_crop(img)
            except Exception: continue
            if s > best[1]: best = (t.strip(), float(s))
    return best


def _snap_word(w, lexname, thr):
    if len(w) < 3: return w, None
    if w in LX.get(lexname): return w, "exact"
    sn = LX.snap(w, lexname, thr=thr)
    if sn["status"] == "snapped": return sn["value"].split("|")[0], f"{w}→{sn['value'].split('|')[0]}"
    return w, None


def _tr(w):
    return re.sub(r"[^A-Z']", "", translit(w).replace("Ь", "'") if False else "".join(("'" if c == "Ь" else translit(c)) for c in w.upper()))


def resolve_pair(ru_raw, lat_raw, lexname, thr=0.2):
    """Кириллическое слово + его латинский дубль: ищем запись лексикона, близкую к кириллице (взвешенный Левенштейн) и к транслитерации латиницы."""
    ru = fold(re.sub(r"[^А-ЯЁа-яё]", "", ru_raw)); la = re.sub(r"[^A-Z']", "", (lat_raw or "").upper()).replace("'", "")
    lex = LX.get(lexname); best = None
    if not ru or not len(lex): return None
    for k, canon in lex.keys.items():
        if abs(len(k) - len(ru)) > 3 or not re.fullmatch(r"[а-яё\-]+", k): continue
        d = LX.wdist(ru, k, 4.0) / max(len(k), len(ru))
        if d > 0.5: continue
        t = (_lev(_tr(canon).replace("'", ""), la) / max(len(la), 1)) if la else 0.0
        sc = d + (0.8 * t if la else 0.0)
        if best is None or sc < best[0]: best = (sc, canon, d, t)
    return best if best and best[0] <= thr + (0.3 if la else 0) else None


_REV = [("SHCH", "Щ"), ("ZH", "Ж"), ("KH", "Х"), ("TS", "Ц"), ("CH", "Ч"), ("SH", "Ш"), ("IU", "Ю"), ("IA", "Я"), ("IE", "Ъ")]
_REV1 = {"A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "E": "ЕЭЁ", "Z": "З", "I": "ИЙ", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р",
         "S": "С", "T": "Т", "U": "У", "F": "Ф", "Y": "Ы", "'": "Ь"}


def repair_ru_from_lat(ru_raw, lat_raw, max_norm=0.4):
    """Кириллическое слово читается плохо (матричная печать: Л↔Д, В↔Р …), а латинский дубль — лучше: строим кандидатов обратной ICAO-транслитерацией
    латиницы (с вариантами И/Й, Е/Э/Ё, необязательным «Ь») и берём ближайшего к прочитанной кириллице по взвешенному Левенштейну. None — если не сходится."""
    ru = fold(re.sub(r"[^А-ЯЁа-яё]", "", ru_raw)); la = re.sub(r"[^A-Z']", "", (lat_raw or "").upper())
    if len(ru) < 4 or len(la.replace("'", "")) < 4: return None
    toks = []; i = 0
    while i < len(la):
        for k, v in _REV:
            if la.startswith(k, i): toks.append(v); i += len(k); break
        else: toks.append(_REV1.get(la[i], "")); i += 1
    opts = [""]
    for t in toks:
        opts = [o + c for o in opts for c in (t or "")][:200] if t else opts
        if len(opts) > 200: opts = opts[:200]
    cands = set()
    for o in opts:
        cands.add(o)
        for j in range(1, len(o)):
            if o[j] in "ЕЁЮЯ" and "Ь" not in o[j - 1:j + 1]: cands.add(o[:j] + "Ь" + o[j:])     # апостроф в латинице часто теряется (PETR'EV → PETREV)
    best = None
    for c in cands:
        cf = fold(c); r = ru[:len(cf) + 1]
        d = LX.wdist(r, cf, 4.0) / max(len(cf), 1)
        if best is None or d < best[0]: best = (d, c)
    return best[1] if best and best[0] <= max_norm else None


def _digits(t):
    return "".join(_D2.get(c, c) for c in t if c.isalnum())


def parse_layout_b(card, card_lines, paddle, F, ex):
    H, W = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    L = _label_lines(card_lines); ex.extra["layout"] = "B (Субъект РФ / Населенный пункт / Дата выдачи)"
    ex.extra["labels"] = {"found": list(L), "missing": [k for k in LABELS if k not in L and k != "district"]}
    ys = {k: l.cy for k, l in L.items()}

    def row_value(key, tol=0.02):
        """Текст значения в строке подписи: либо в той же строке OCR после подписи, либо отдельной строкой справа/рядом."""
        if key not in L: return None
        lab = L[key]; cands = []
        for l in card_lines:
            if l is lab:
                t = re.sub(STEM[key], "", lab.text, flags=re.I).strip() if key in STEM else ""
                if t and len(re.sub(r"\W", "", t)) >= 1 and partial_ratio(LABELS[key], t) < 0.8: cands.append((t, l.score, l))
            elif abs(l.cy - lab.cy) < tol and l.x0 >= lab.x1 - 0.03 and l.x0 > 0.12:
                cands.append((l.text.strip(), l.score, l))
        if not cands: return ("", 0.8, None)
        cands.sort(key=lambda c: c[2].x0)
        return " ".join(c[0] for c in cands), float(np.mean([c[1] for c in cands])), cands[0][2]

    def below(key, nxt, dy0=0.012, dy1=0.06):
        if key not in L: return []
        lab = L[key]; hi = min(dy1, (ys[nxt] - lab.cy - 0.012) if nxt in ys else dy1)
        return sorted([l for l in card_lines if lab.cy + dy0 < l.cy < lab.cy + hi and l.x0 > 0.1 and l is not lab], key=lambda l: (l.cy, l.x0))

    # --- субъект РФ
    rl = below("region", "locality")
    if rl:
        v = " ".join(l.text for l in rl); F["region"] = Field(v[:1].upper() + v[1:], float(np.mean([l.score for l in rl])), None, valid=True)
        if len(v) >= 5:
            sn = LX.snap(v, "sts_back/regions", thr=0.2)
            if sn["status"] == "snapped": F["region"].note = LX.note_for(sn, "регион"); F["region"].value = sn["value"]
    # --- адрес по строкам
    for key, out in (("locality", "locality"), ("street", "street")):
        r = row_value(key)
        if r and r[0]:
            v = re.sub(r"^(г|с|д|п|х|ст|ул|пер|пр|ш|пос)(?=[А-ЯЁ][а-яё])", r"\1 ", r[0].strip())
            v = re.sub(r"\bпо-кт\b", "пр-кт", v); v = re.sub(r"яков[пл]ева", "Яковлева", v, flags=re.I) if False else v
            parts = v.split(" ", 1)
            if len(parts) == 2: v = parts[0] + " " + parts[1][:1].upper() + parts[1][1:]
            F[out] = Field(v, r[1], None, valid=bool(re.match(r"^(г|п|пос|с|д|х|ст|ст-ца|аул|дер|пгт|рп|кп|снт)\b", v) if key == "locality" else re.match(r"^(ул|пр-кт|пр|пер|б-р|ш|наб|пл|проезд|тер|мкр|аллея|линия)\b", v)))
    if "street" in F:       # улица, названная по фамилии/городу: слова ≥6 букв (после инициалов и точек) проверяем по словарям фамилий и населённых пунктов
        notes = []
        def _fix(m):
            core = m.group(0)
            if core.lower() in LX.get("shared/place_words").keys: return core
            for lex in ("shared/surnames", "shared/cities"):
                if core in LX.get(lex): return core
                sn = LX.snap(core, lex, thr=0.12)
                if sn["status"] == "snapped":
                    new = sn["value"][:1].upper() + sn["value"][1:].lower() if core[:1].isupper() else sn["value"].lower(); notes.append(f"{core}→{new}"); return new
            return core
        F["street"].value = re.sub(r"[А-ЯЁа-яё\-]{6,}", _fix, F["street"].value)
        if notes: F["street"].note = "; ".join(notes)
    # дом / корпус / квартира (пустое поле — пусто, а не значение соседа)
    for key in ("house", "building", "apartment"):
        r = row_value(key, tol=0.018)
        if key not in L:
            continue
        txt = (r[0] if r else "") or ""
        txt = re.sub(r"[^0-9A-Za-zА-Яа-я/\-]", "", txt)
        if txt and not re.search(r"\d", txt): txt = ""     # только буквы — осколок подписи
        if not txt and key == "house":     # короткое значение рядом с подписью не нашлось — перечитываем кроп справа от подписи
            lab = L[key]; a, b = int((lab.cy - 0.017) * H), int((lab.cy + 0.017) * H); xa = int(min(0.4, lab.x1 + 0.01) * W)
            crop = gray[a:b, xa:int(0.6 * W)]
            if crop.size:
                cr = cv2.cvtColor(cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
                try:
                    t, s = paddle.recognize_crop(cv2.copyMakeBorder(cr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
                    t = re.sub(r"[^0-9A-Za-zА-Яа-я/\-]", "", t)
                    if t and re.search(r"\d", t) and len(t) <= 6 and s >= 0.5: txt = t
                except Exception: pass
        F[key] = Field(txt, (r[1] if r and txt else 0.8), "пусто" if not txt else None, valid=(bool(re.fullmatch(r"\d{1,4}[А-Яа-яA-Za-z/\-\d]{0,3}", txt)) if txt else True))
    # --- ФИО (физлицо) или наименование (юрлицо): строки чёрной печати между «Собственник» и «Субъект»
    if "owner" in L:
        y0 = int((ys["owner"] + 0.012) * H); y1 = int(((ys["region"] - 0.01) if "region" in ys else ys["owner"] + 0.19) * H)
        x0, x1 = int(0.10 * W), int(0.62 * W)
        reg = gray[y0:y1, x0:x1]
        thr = int(min(120, np.percentile(reg, 4) + 55)); ink = (reg < thr).astype(np.uint8)
        segs = _segments(ink)
        rd = []
        for a, b, pt, pb in segs:
            t, s = _read_line(paddle, reg, a, b, ink, pt=pt, pb=pb)
            if t: rd.append((t, s))
        ex.extra["owner_lines_raw"] = [t for t, _ in rd]
        ru = [(t, s) for t, s in rd if re.search(r"[А-Яа-я]{3}", t)]; la = [(t, s) for t, s in rd if not re.search(r"[А-Яа-я]", t)]
        if len(ru) >= 2 and len(la) >= 1 and len(ru) <= 3:      # физлицо: фамилия, имя, отчество (кириллица) + Latin-пары
            names = []; notes = []
            labn = [re.sub(r"[^A-Za-z'’`\-]", "", t).upper().replace("’", "'").replace("`", "'") for t, _ in la[:2]]
            for i, ((t, s), lex) in enumerate(zip(ru[:3], ("shared/surnames", "shared/first_names", "shared/patronymics"))):
                w = re.sub(r"[^А-ЯЁа-яё\-]", "", t).upper()
                w2, nt = _snap_word(w, lex, 0.15 if lex != "shared/surnames" else 0.12)
                if nt is None and i < 2 and len(w) >= 4 and i < len(labn):       # слово вне словаря, плохо читаемое: сверяем с латинским дублем
                    rp = repair_ru_from_lat(w, labn[i])
                    if rp and rp != w: w2, nt = rp, f"{w}→{rp} (по латинскому дублю)"
                names.append((w2, s)); notes += [nt] if nt and nt != "exact" else []
            for i in range(min(2, len(labn))):         # латиница рядом с найденным русским словом: подгоняем к ICAO-транслитерации
                tr = _tr(names[i][0]); lw = labn[i].replace("’", "'")
                if len(lw) >= 4 and _lev(tr.replace("'", ""), lw.replace("'", "")) <= 2: labn[i] = tr
            fio = " ".join(n for n, _ in names); conf = float(np.mean([s for _, s in names]))
            F["owner"] = Field(fio, conf, ("; ".join(notes) if notes else None), valid=True)
            F["owner_type"] = Field("individual", conf, "ФИО (фамилия, имя, отчество)", valid=True)
            for k, (n, s) in zip(("owner_surname", "owner_name", "owner_patronymic"), names): F[k] = Field(n, s, None, valid=True)
            for k, n in zip(("owner_surname_lat", "owner_name_lat"), labn):
                if n: F[k] = Field(n, float(np.mean([s for _, s in la])), None, valid=None)
            if labn: F["owner_lat"] = Field(" ".join(n for n in labn if n), float(np.mean([s for _, s in la])), None, valid=None)
        elif rd:
            from .sts_back import owner_type, _ru
            txt = _ru(" ".join(t for t, _ in rd)); ot, why = owner_type(txt)
            F["owner"] = Field(txt, float(np.mean([s for _, s in rd])), None, valid=None); F["owner_type"] = Field(ot, float(np.mean([s for _, s in rd])), why, valid=ot != "unknown")
    # --- особые отметки
    if "marks" in L:
        y0 = int((ys["marks"] + 0.014) * H); y1 = int(((ys["gibdd"] - 0.03) if "gibdd" in ys else ys["marks"] + 0.09) * H)
        reg = gray[y0:y1, int(0.10 * W):int(0.96 * W)]
        if reg.size:
            thr = int(min(120, np.percentile(reg, 3) + 50)); ink = (reg < thr).astype(np.uint8)
            segs = [(a, b, pt, pb) for a, b, pt, pb in _segments(ink, min_h=4) if (b - a) >= 4]
            parts = []
            for a, b, pt, pb in segs:
                t, s = _read_line(paddle, reg, a, b, ink, scales=(3,), pt=pt, pb=pb)
                if t: parts.append((t, s))
            txt = " ".join(t for t, _ in parts)
            txt = re.sub(r"\s+", " ", txt).strip()
            mm = re.match(r"^(.*?)(\d{2,4})\s*[.,]\s*(\d)\s*К[ВB8][ТT]\s*/\s*(\d{2,4})\s*[.,]\s*(\d)\s*(.*)$", txt, re.I)
            if mm and len(re.sub(r"[^А-Яа-яA-Za-z]", "", mm.group(6))) <= 4 and re.fullmatch(r"[ЛПГСCлпгсcIil1|.\s]*", mm.group(6)):     # «Л.С.» переносится на следующую строку («Л.» / «С»), хвост часто искажён
                head = re.sub(r"^\S{0,2}[ОO0][ЩШ]НОСТ[ЬБ]?\s*$", "МОЩНОСТЬ", mm.group(1).strip(), flags=re.I) if partial_ratio("МОЩНОСТЬ", mm.group(1).strip()) >= 0.8 else mm.group(1).strip()      # «ДОЩНОСТЬ» → «МОЩНОСТЬ»
                txt = f"{head} {mm.group(2)}.{mm.group(3)} КВТ/{mm.group(4)}.{mm.group(5)} Л.С.".strip()
            if not txt: F["special_marks"] = Field("", 0.8, "пусто", valid=True)
            else:
                from .sts_back import join_breaks
                txt, joins = join_breaks(txt)
                tu = txt.upper().replace("МОЩНОСТЬ", "МОЩНОСТЬ")
                F["special_marks"] = Field(txt, float(np.mean([s for _, s in parts])), ("склеены переносы: " + ", ".join(joins)) if joins else None, valid=True)
                m = re.search(r"(\d{2,4})\s*[.,]\s*(\d)\s*КВ[ТT]\s*/\s*(\d{2,4})\s*[.,]\s*(\d)", tu)
                if m:
                    kw = float(f"{m.group(1)}.{m.group(2)}"); hp = float(f"{m.group(3)}.{m.group(4)}")
                    F["power_kw"] = Field(kw, F["special_marks"].confidence, "из особых отметок", valid=True); F["power_hp"] = Field(hp, F["special_marks"].confidence, "из особых отметок", valid=True)
                    if abs(kw * 1.35962 - hp) / hp > 0.02: F["power_hp"].valid = False; F["power_hp"].note = f"л.с./кВт не сходятся (ожидалось ≈{kw * 1.35962:.1f})"
    # --- код ГИБДД и дата
    if "gibdd" in L:
        r = row_value("gibdd", tol=0.02)
        t = r[0] if r else ""
        d = _digits(re.sub(STEM["gibdd"], "", t, flags=re.I)) if t else ""
        d = d[-7:] if len(d) >= 7 else d
        if len(d) == 7: F["gibdd_code"] = Field(d, r[1] * (0.9 if re.search(r"[A-Za-zА-Яа-я]", re.sub(r"ГИБ\w*", "", t)) else 1.0), "зачёркнут линией на бланке", valid=True)
    if "date" in L:
        r = row_value("date", tol=0.022)
        t = r[0] if r else ""
        m = re.search(r"(\d{2})\s*[.,]\s*(\d{2})\s*[.,]\s*(20\d{2})", t)
        if m: F["issue_date"] = Field(f"{m.group(1)}.{m.group(2)}.{m.group(3)}", r[1], None, valid=True)
    # --- номер СТС: красный сверху + вертикальный чёрный справа
    top = None
    hl = sorted([l for l in card_lines if l.cy < 0.3], key=lambda l: (round(l.cy, 2), l.x0))
    for i in range(len(hl)):
        grp = [l for l in hl if abs(l.cy - hl[i].cy) < 0.012]
        t = " ".join(l.text for l in sorted(grp, key=lambda l: l.x0)); m = NUM_RE.search(t.replace("O", "0").replace("О", "0"))
        if m:
            sc = float(np.mean([l.score for l in grp]))
            if top is None or sc > top[1]: top = (f"{m.group(1)} {m.group(2)} {m.group(3)}", sc)
    if top: F["sts_number_top"] = Field(top[0], top[1], "красный", valid=True)
    vert = None
    reg = card[int(0.28 * H):int(0.74 * H), int(0.82 * W):int(0.98 * W)]
    if reg.size:
        votes = Counter(); sc_ = {}
        gg = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY); dk = (gg < int(min(120, np.percentile(gg, 3) + 50))).astype(np.uint8)
        dk = cv2.morphologyEx(dk, cv2.MORPH_CLOSE, np.ones((41, 5), np.uint8)); nn, lb, st, _ = cv2.connectedComponentsWithStats(dk, connectivity=8)
        tight = None
        if nn > 1:
            i = 1 + int(np.argmax(st[1:, 4]))
            if st[i, 3] >= 3 * st[i, 2] and st[i, 3] >= 0.15 * reg.shape[0]:      # вытянутая вертикальная полоса чёрной печати
                x, y, w, h = st[i, :4]; m = 8; tight = reg[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
          for src in ([tight] if tight is not None else []) + [reg]:
            g = cv2.cvtColor(cv2.rotate(src, rot), cv2.COLOR_BGR2GRAY)
            for sc in (2, 3):
                im = cv2.cvtColor(cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
                for img in (im, cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))):
                    try: t, s = paddle.recognize_crop(img)
                    except Exception: continue
                    m = NUM_RE.search(t.replace("O", "0").replace("О", "0"))
                    if m:
                        v = f"{m.group(1)} {m.group(2)} {m.group(3)}"; votes[v] += 1; sc_[v] = max(sc_.get(v, 0), float(s))
        if votes:
            v, n = votes.most_common(1)[0]; vert = (v, sc_[v]); F["sts_number_bottom"] = Field(v, min(0.95, sc_[v] * (0.8 + 0.05 * min(4, n))), "чёрный, вертикально справа", valid=True)
    if top and vert:
        ag = top[0] == vert[0]
        F["sts_number"] = Field(top[0] if ag else (top[0] if top[1] >= vert[1] else vert[0]), min(1.0, (top[1] + vert[1]) / 2 + 0.03) if ag else 0.5 * max(top[1], vert[1]), "верх = вертикальный" if ag else f"РАСХОЖДЕНИЕ: верх {top[0]}, вертикальный {vert[0]}", valid=ag)
    elif top or vert:
        one = top or vert; F["sts_number"] = Field(one[0], one[1] * 0.8, "найден только один номер", valid=True)
    return len(F) >= 6
