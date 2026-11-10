"""Паспорт гражданина РФ, разворот с фото (две страницы друг над другом) + МРЗ + вертикальный красный номер.
Работает по ВСЕМУ кадру (без поиска рамки): масштаб -> поворот-выравнивание -> PaddleOCR по кадру -> привязка значений по порядку/подписям,
серия и номер — по полосам справа (повёрнутым на 90°), поля сверяются с МРЗ (контрольные цифры), у каждого поля источник и уверенность."""
from __future__ import annotations
import re
from datetime import date
import cv2
import numpy as np
from ..ocr import Line
from ..textutils import fold, partial_ratio, _lev, to_cyr, _L2C
from .. import anchors, mrz as MRZ
from .base import DocType, Extraction, Field, register
from .. import lexicon as _LX

CUES = [r"УФМС", r"\bМВД\b", r"ПАСПОРТ\s*ВЫДАН", r"ДАТА\s*ВЫДАЧИ", r"КОД\s*ПОДРАЗДЕЛЕНИЯ", r"ЛИЧНЫЙ\s*КОД", r"ФАМИЛИЯ", r"ДАТА\s*РОЖДЕНИЯ", r"МЕСТО\s*РОЖДЕНИЯ",
        r"ФЕДЕРАЦИЯ", r"РОССИЙСКАЯ", r"PNRUS", r"\b(?:МУЖ|ЖЕН)\.?\b", r"\b\d{3}\s*[-–—]\s*\d{3}\b", r"<<<"]
SPECS_UPPER = [anchors.LabelSpec("issued_by", ["Паспорт выдан"], multi=3), anchors.LabelSpec("issue_date", ["Дата выдачи"]),
               anchors.LabelSpec("dept_code", ["Код подразделения"]), anchors.LabelSpec("personal_code", ["Личный код"])]
SPECS_LOWER = [anchors.LabelSpec("surname", ["Фамилия"]), anchors.LabelSpec("name", ["Имя"]), anchors.LabelSpec("patronymic", ["Отчество"]),
               anchors.LabelSpec("sex", ["Пол"]), anchors.LabelSpec("birth_date", ["Дата рождения"]), anchors.LabelSpec("birth_place", ["Место рождения"], multi=3)]
LABEL_VOC = ["ПАСПОРТ", "ВЫДАН", "ДАТА", "ВЫДАЧИ", "КОД", "ПОДРАЗДЕЛЕНИЯ", "ЛИЧНЫЙ", "ЛИЧНАЯ", "ПОДПИСЬ", "ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОЛ", "РОЖДЕНИЯ", "МЕСТО", "РОССИЙСКАЯ", "ФЕДЕРАЦИЯ"]
ISSUER_VOC = ["ГУ", "МВД", "РОССИИ", "УФМС", "ОТДЕЛ", "ОТДЕЛОМ", "ОТДЕЛЕНИЕМ", "ВНУТРЕННИХ", "ДЕЛ", "ГОРОДА", "ГОРОДЕ", "ОБЛАСТИ", "РЕСПУБЛИКЕ", "РЕСПУБЛИКИ", "РАЙОНЕ", "РАЙОНА",
              "КРАЮ", "КРАЕ", "ОКРУГА", "ОКРУГЕ", "ОВД", "УВД", "ПО", "ТП", "РОВД", "ГОР", "ГОРОД", "ОБЛ", "АВТОНОМНОМУ", "АВТОНОМНОЙ", "МИГРАЦИОННОЙ", "СЛУЖБЫ", "ФЕДЕРАЛЬНОЙ"]
import os
STRIP_X = 0.78
WORK_SIDE = int(os.environ.get("DOCREC_PASSPORT_SIDE", "1800"))
SEX = {"МУЖ": "МУЖ.", "ЖЕН": "ЖЕН."}


def _cyr_word(w: str) -> str:
    return "".join(_L2C.get(c, c) for c in w.upper())


_EXTRA2 = {"I": "И", "D": "Д", "L": "Л", "Z": "З", "G": "Г", "F": "Ф", "Y": "У", "J": "Й", "N": "И"}


def clean_cyr(t: str, vocab=()) -> str:
    """ВЕРХНИЙ РЕГИСТР, латинские двойники -> кириллица, шум по краям убран, слова словаря чинятся (≤1 правка)."""
    t = re.sub(r"[|_*~`^<>=•©®]+", " ", t)
    words = []
    for w in t.split():
        core = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", w)
        if core and all(c in _L2C or "А" <= c.upper() <= "Я" or c in "Ёё" for c in core):
            w = "".join(_L2C.get(c, c) for c in w.upper())
        elif core and sum("А" <= c.upper() <= "Я" for c in core) >= max(2, len(core) // 2):   # кириллическое слово с «чужими» латинскими буквами
            w = "".join(_L2C.get(c, _EXTRA2.get(c, c)) for c in w.upper())
        else:
            w = w.upper()
        m = re.match(r"^([^А-ЯЁA-Z0-9]*)(.*?)([^А-ЯЁA-Z0-9]*)$", w)
        pre, core2, post = m.groups()
        if vocab and len(core2) >= 2 and core2 not in vocab:
            best = min(vocab, key=lambda v: _lev(core2, v))
            if _lev(core2, best) <= 1 and abs(len(best) - len(core2)) <= 1 and len(core2) >= 3: core2 = best
        words.append(pre + core2 + post)
    out = re.sub(r"\s+", " ", " ".join(words)).strip(" .,;:-_")
    out = re.sub(r"([А-ЯЁ]{3,})(Г\.)(?=[А-ЯЁ])", r"\1 \2", out)
    return out


def upper_frac(t: str) -> float:
    ls = [c for c in t if c.isalpha()]
    return sum(c.isupper() for c in ls) / len(ls) if ls else 0.0


DATE_RE = re.compile(r"(?<!\d)(\d{2})\s*[.,:;\-/*·]\s*(\d{2})\s*[.,:;\-/*·]\s*(\d{4})(?!\d)")
CODE_RE = re.compile(r"(?<!\d)(\d{3})\s*[-–—.]\s*(\d{3})(?!\d)")


# потерянная точка: «1102.1987», «11.021987», «11021987» (в паспорте допустимы только даты, прошедшие проверку календаря)
DATE_LENIENT = [re.compile(r"(?<!\d)(\d{2})(\d{2})\s*[.,:;\-/*·]\s*(\d{4})(?!\d)"), re.compile(r"(?<!\d)(\d{2})\s*[.,:;\-/*·]\s*(\d{2})(\d{4})(?!\d)"),
                re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")]


def find_dates(text: str):
    out = [(f"{a}.{b}.{c}", m.start()) for m in DATE_RE.finditer(text) for a, b, c in [m.groups()]]
    if not out:
        for rx in DATE_LENIENT:
            for m in rx.finditer(text):
                d = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
                if valid_date(d, 1850): out.append((d, m.start()))
            if out: break
    return out


def valid_date(s: str, lo=1890, hi=None):
    try: d, m, y = map(int, s.split("."))
    except Exception: return None
    hi = hi or date.today().year
    try: dt = date(y, m, d)
    except ValueError: return None
    return dt if lo <= y <= hi else None


def deskew_angle(img, rng=10.0, step=0.5) -> float:
    h, w = img.shape[:2]; s = 700 / max(h, w)
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bw = cv2.medianBlur(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 18), 3)
    H, W = bw.shape; best = (0.0, -1.0)
    for a in np.arange(-rng, rng + 1e-6, step):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1), (W, H))
        v = float(np.var(r.sum(1).astype(np.float32)))
        if v > best[1]: best = (float(a), v)
    # уточнение вокруг максимума
    a0 = best[0]
    for a in np.arange(a0 - step, a0 + step + 1e-6, step / 4):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1), (W, H))
        v = float(np.var(r.sum(1).astype(np.float32)))
        if v > best[1]: best = (float(a), v)
    return best[0]


_CUE_PHRASES = ["Паспорт выдан", "Дата выдачи", "Код подразделения", "Личный код", "Фамилия", "Дата рождения", "Место рождения", "Отчество", "Российская Федерация"]


def cue_count(lines) -> int:
    n = 0; txt = " ".join(l.text.upper() for l in lines)
    for c in CUES[:2] + CUES[11:]:
        if re.search(c, txt): n += 1
    for ph in _CUE_PHRASES:
        if any(partial_ratio(ph, l.text) >= 0.78 and len(l.text) <= len(ph) + 14 for l in lines): n += 1
    return n


@register
class PassportRF(DocType):
    name = "passport_rf"; title_ru = "Паспорт гражданина РФ (разворот с фото)"
    card_size = (1500, 1056); aspect = 125 / 88
    keywords = [("ПАСПОРТ ВЫДАН", 2), ("РОССИЙСКАЯ ФЕДЕРАЦИЯ", 1), ("КОД ПОДРАЗДЕЛЕНИЯ", 2), ("ДАТА РОЖДЕНИЯ", 1), ("МЕСТО РОЖДЕНИЯ", 1), ("ФАМИЛИЯ", 0.5), ("ДАТА ВЫДАЧИ", 1), ("PNRUS", 2)]
    implemented = True; hybrid_internal = True
    expected_fields = ["issued_by", "issue_date", "dept_code", "surname", "name", "patronymic", "sex", "birth_date", "birth_place", "series", "number"]

    def extract(self, card, lines, **kw) -> Extraction:   # в карточном конвейере не используется: разворот читается по всему кадру
        return Extraction()


# =====================================================================  чтение разворота
def read_strips(img, paddle) -> dict:
    """Вертикальный красный номер у правого края (на обеих страницах): полоса -> поворот на 90° против часовой -> OCR."""
    H, W = img.shape[:2]; res = {}
    for name, (y0, y1) in {"top": (0.0, 0.5), "bottom": (0.5, 0.95)}.items():
        st = img[int(y0 * H):int(y1 * H), int(STRIP_X * W):]
        if st.size == 0: continue
        g = cv2.normalize(st[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
        rot = cv2.rotate(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = rot.shape[:2]; pad = max(0, int(0.4 * w) - h)   # тонкие полосы раздувают RAM детектора — дополняем белым
        if pad: rot = np.vstack([rot, np.full((pad, w, 3), 255, np.uint8)])
        ls = sorted(paddle(rot, merge=False), key=lambda l: l.x0)
        toks = [(re.sub(r"\D", "", l.text), l.score) for l in ls if re.sub(r"\D", "", l.text)]
        txt = " ".join(t for t, _ in toks)
        m = re.search(r"(?<!\d)(\d{2})\s+(\d{2})\s+(\d{6})(?!\d)", txt) or re.search(r"(?<!\d)(\d{2})(\d{2})(\d{6})(?!\d)", txt.replace(" ", ""))
        if m:
            sc = float(np.mean([s for _, s in toks])) if toks else 0.0
            res[name] = {"series": f"{m.group(1)} {m.group(2)}", "number": m.group(3), "score": sc}
        else:
            nm = [t for t, _ in toks if len(t) == 6]
            if nm: res[name] = {"series": None, "number": nm[0], "score": float(np.mean([s for _, s in toks]))}
    return res


def is_mrz_like(t: str) -> bool:
    """Строка машиночитаемой зоны: «<<», «…RUS930…», «P<RUS…» (в т.ч. с OCR-ошибками; без кириллицы)."""
    u = re.sub(r"\s+", "", t.upper())
    if u.count("<") >= 2: return True
    if re.search(r"[A-Z0-9<]*RUS\d{5,}", u) or re.search(r"\d{8,10}RUS", u): return True
    return bool(re.match(r"^P[<NIM1HV]{0,2}RUS[A-Z0-9<]", u) and len(u) >= 18 and not re.search(r"[А-ЯЁ]", u))


_MRZ_TOKEN = re.compile(r"[A-Z0-9<]{10,}")


def strip_mrz_text(text: str) -> str:
    """Ни одно обычное поле не должно содержать МРЗ: режем значение на первом «МРЗ-подобном» токене (длинная латиница/цифры, «<», «RUS»+цифры)."""
    toks = str(text).split(); out = []
    for t in toks:
        u = t.upper()
        if "<" in u or (len(re.sub(r"\D", "", u)) >= 8) or (_MRZ_TOKEN.fullmatch(re.sub(r"[^A-Z0-9<]", "", u)) and re.search(r"\d", u) and len(u) >= 10) or re.search(r"RUS\d", u):
            break
        out.append(t)
    return " ".join(out)


_LABEL_FRAG = "ДАТАРОЖДЕНИЯМЕСТОПОЛ"


def clean_place(text: str) -> str:
    """Место рождения: убираем обрывки подписей слева («АТА» от «Дата», «ЖДЕНИЯ») и всё, что похоже на МРЗ."""
    from .. import lexicon as LX
    t = strip_mrz_text(text); toks = t.split()
    while toks:
        w = re.sub(r"[^А-ЯЁA-Za-z]", "", toks[0]).upper()
        if w and len(w) <= 6 and w in _LABEL_FRAG and w not in ("Д", "Г", "С", "П") and not (toks[0] in LX.get("shared/place_words")):
            toks.pop(0); continue
        break
    return " ".join(toks)


def _mrz_block(mz, raw_lines):
    """Отдельный JSON-блок МРЗ (не поле): сырые строки, разобранные значения, контрольные цифры, «личные данные» (4-я цифра серии, дата выдачи, код подразделения).
    Используется только для сверки печатных полей."""
    blk = {"found": bool(mz), "used_for": "cross-check only (не распознаваемое поле)", "raw_lines": list(raw_lines or [])}
    if not mz: return blk
    chk = mz.get("checks") or {}
    blk["parsed"] = {k: mz.get(k) for k in ("surname", "name", "patronymic", "sex", "birth_date", "issue_date", "dept_code", "series", "number") if mz.get(k) is not None}
    blk["latin"] = {k: mz.get(k) for k in ("surname_lat", "name_lat", "patronymic_lat") if mz.get(k)}
    blk["personal_data"] = {"series_4th_digit": (mz.get("series") or "").replace(" ", "")[3:4] or None, "issue_date": mz.get("issue_date"), "dept_code": mz.get("dept_code")}
    blk["checks"] = {k: bool(v) for k, v in chk.items()}
    blk["checks_valid"] = bool(chk.get("check_doc") and chk.get("check_birth"))
    blk["line1_found"] = mz.get("line1_found"); blk["line2_found"] = mz.get("line2_found")
    return blk


def mrz_candidates(lines) -> list[str]:
    ml = [l for l in lines if l.cy > 0.55 and (is_mrz_like(l.text) or (l.cy > 0.78 and (l.text.count("<") >= 2 or "RUS" in l.text.upper() or l.text.upper().startswith("PN"))))]
    ml = sorted(ml, key=lambda l: (round(l.cy, 2), l.x0))
    rows, out = [], []
    for l in ml:
        if rows and abs(rows[-1][-1].cy - l.cy) < 0.012: rows[-1].append(l)
        else: rows.append([l])
    return ["".join(x.text.replace(" ", "") for x in sorted(r, key=lambda l: l.x0)) for r in rows]


def sim(a: str, b: str) -> float:
    a, b = fold(a), fold(b)
    if not a or not b: return 0.0
    return 1 - _lev(a, b) / max(len(a), len(b))


def _vote_crop(pipe, crop):
    """Чтение кропа строки в разных масштабах/обработках -> {текст: суммарная уверенность}."""
    from collections import defaultdict
    from ..makemodel import _enhance, _prep
    votes = defaultdict(float)
    if crop.shape[0] < 8 or crop.shape[1] < 8: return votes
    for mode in ("raw", "clahe", "sharp", "gamma"):
        c = _enhance(crop, mode)
        for h_ in (64, 96, 128):
            t, sc, _cf = pipe.paddle.recognize_chars(_prep(c, h_))
            tt = re.sub(r"[^А-ЯЁA-Z\- ]", "", to_cyr(t.upper()) if t else "").strip()
            if len(tt) >= 3 and sc >= 0.35: votes[tt] += sc
    return votes


def _rescue_surname(pipe, m, F_lines):
    """Строку фамилии детектор не нашёл (размытая копия): читаем кроп по расчётной позиции над «Имя»: шаг строк — как между именем и отчеством."""
    H, W = m.shape[:2]
    ln = F_lines["name"]; sp = (F_lines["patronymic"].cy - ln.cy) if "patronymic" in F_lines else 0.03
    sp = min(max(sp, 0.025), 0.045)
    best = None
    for k in (2.0, 1.8, 2.2, 1.6, 1.0):     # между фамилией и именем промежуток больше, чем между именем и отчеством (подписи)
        cy = ln.cy - k * sp
        y0, y1 = int(max(0, (cy - 0.017) * H)), int(min(H, (cy + 0.017) * H))
        x0, x1 = int(max(0, (ln.x0 - 0.03) * W)), int(min(W, (ln.x1 + 0.10) * W))
        votes = _vote_crop(pipe, m[y0:y1, x0:x1])
        for t, w in votes.items():
            if len(re.sub(r"[^А-ЯЁ]", "", t)) >= 4 and not re.search(r"ФАМИЛ|ИМЯ|ОТЧЕСТВ", t) and (best is None or w > best[1]): best = (t, w, votes)
    return best


def _reread_alts(pipe, m, line, text):
    """Варианты чтения слабой строки значения: кроп с разным отступом/масштабом/обработкой -> голосование. -> dict(value, note, factor) или None.
    Значение меняем, только если другой вариант явно лидирует (≥1.5× вес исходного); иначе оставляем и перечисляем альтернативы."""
    from collections import defaultdict
    from ..makemodel import _enhance, _prep
    H, W = m.shape[:2]
    votes = defaultdict(float); base = re.sub(r"[^А-ЯЁA-Z\- ]", "", text.upper()).strip()
    cy = (line.y0 + line.y1) / 2; hh = max(line.y1 - line.y0, 0.012)
    for padk in (0.15, 0.35):
        y0, y1 = int(max(0, (cy - hh / 2 - padk * hh) * H)), int(min(H, (cy + hh / 2 + padk * hh) * H))
        x0, x1 = int(max(0, (line.x0 - 0.02) * W)), int(min(W, (line.x1 + 0.02) * W))
        crop = m[y0:y1, x0:x1]
        if crop.shape[0] < 8 or crop.shape[1] < 8: continue
        for mode in ("raw", "clahe", "sharp", "gamma"):
            c = _enhance(crop, mode)
            for h_ in (64, 96, 128):
                t, sc, _cf = pipe.paddle.recognize_chars(_prep(c, h_))
                tt = re.sub(r"[^А-ЯЁA-Z\- ]", "", to_cyr(t.upper()) if t else "").strip()
                if len(tt) >= 3 and sc >= 0.35: votes[tt] += sc
    if not votes or len(votes) == 1 and base in votes: return None
    votes[base] += line.score
    ranked = sorted(votes.items(), key=lambda x: -x[1]); top, wtop = ranked[0]
    note = "варианты чтения: " + ", ".join(f"«{v}» ({w:.1f})" for v, w in ranked[:4])
    if top != base and wtop >= 1.5 * votes[base]: return dict(value=top, note=note + f"; выбрано «{top}» вместо «{base}»", factor=0.95)
    return dict(value=text, note=note + " (оставлено исходное)" if len(ranked) > 1 else None, factor=0.95) if len(ranked) > 1 else None


def _lexicon_pass(F, checks):
    """Привязка к публичным лексиконам (docrec/lexicons): имена/отчества/места/орган выдачи. Поля, подтверждённые МРЗ, не трогаем.
    Фамилии НЕ подгоняем (редкие/нестандартные фамилии): только повышаем уверенность, если фамилия есть в словаре, иначе — пометка «возможно …»."""
    from .. import lexicon as LX
    def mrz_ok(k): return checks.get(k) in ("agree", "corrected")
    f0 = F.get("patronymic")
    if f0 is not None and f0.value:    # «ВЛАДИМИРОВ ИЧ» -> «ВЛАДИМИРОВИЧ»: хвост отчества отделился пробелом
        ws = str(f0.value).split()
        if len(ws) == 2 and ws[1] in ("ИЧ", "ВИЧ", "ВНА", "НА", "ИЧНА", "ЫЧ", "ЧНА", "ОВНА", "ЕВНА") and len(ws[0]) >= 4:
            f0.note = ((f0.note + "; ") if f0.note else "") + f"склеено «{ws[0]} {ws[1]}» → «{ws[0] + ws[1]}»"; f0.value = ws[0] + ws[1]
    for key, lex in (("name", "shared/first_names"), ("patronymic", "shared/patronymics")):
        f = F.get(key)
        if f is None or not f.value or mrz_ok(key) or (f.note or "").startswith("из МРЗ"): continue
        words = str(f.value).split(); new = []; ch = []
        for w in words:
            parts = w.split("-"); np_ = []
            for part in parts:
                if len(part) < 3 or part in LX.get(lex): np_.append(part); continue
                sn = LX.snap(part, lex, thr=0.1 if lex.endswith("first_names") else 0.12)
                if sn["status"] == "snapped": np_.append(sn["value"].upper()); ch.append((part, sn["value"].upper(), sn["norm"]))
                else: np_.append(part)
            new.append("-".join(np_))
        if ch:
            raw = f.value; f.value = " ".join(new); f.confidence = float(f.confidence) * 0.95
            f.note = ((f.note + "; ") if f.note else "") + "лексикон: " + ", ".join(f"«{a}» → «{b}» (расстояние {n})" for a, b, n in ch) + f"; печать читалась «{raw}»"
    f = F.get("surname")
    if f is not None and f.value and not mrz_ok("surname") and not (f.note or "").startswith("из МРЗ"):
        words = [w for w in re.split(r"[\s-]+", str(f.value)) if len(w) >= 3]
        if words and all(w in LX.get("shared/surnames") for w in words): f.confidence = min(1.0, float(f.confidence) + 0.03)
        elif words:
            alts = []
            for w in words:
                if w in LX.get("shared/surnames"): continue
                sn = LX.snap(w, "shared/surnames", thr=0.12)
                if sn["status"] in ("snapped", "ambiguous") and sn["alts"]: alts.append((w, sn["alts"][0][0], sn["alts"][0][1]))
            if alts:
                f.note = ((f.note + "; ") if f.note else "") + "лексикон: фамилии нет в словаре, возможна путаница букв — " + ", ".join(f"«{a}» ≈ «{b}» ({n})" for a, b, n in alts) + " (не исправлено)"
                f.confidence = float(f.confidence) * 0.95
    f = F.get("issued_by")
    if f is not None and f.value:
        f.value = re.sub(r"\bГОР[-–—]\s*(?=[А-ЯЁ])", "ГОР. ", str(f.value))     # «ГОР- ТВЕРИ» -> «ГОР. ТВЕРИ»
        txt, ch = LX.snap_multi(str(f.value), [("passport/issuer_places", 0.07, 7), ("passport/issuer_words", 0.15, 4)])
        if ch:
            raw = f.value; f.value = txt; f.confidence = float(f.confidence) * 0.97
            f.note = ((f.note + "; ") if f.note else "") + "лексикон: " + ", ".join(f"«{a}» → «{b}» (расстояние {n})" for a, b, n in ch)
    f = F.get("birth_place")
    if f is not None and f.value:
        txt, ch = LX.snap_multi(str(f.value), [("shared/place_words", 0.15, 4), ("shared/months", 0.0, 99)], min_len=4)   # закрытые слова («ОБЛАСТЬ», «ГОР.») — поправляем
        if ch:
            f.value = txt; f.confidence = float(f.confidence) * 0.98
            f.note = ((f.note + "; ") if f.note else "") + "лексикон: " + ", ".join(f"«{a}» → «{b}» (расстояние {n})" for a, b, n in ch)
        # названия мест — открытое множество (села/посёлки в словаре отсутствуют): не подменяем, только подсказываем близкую запись
        hints = []
        for tok in re.split(r"[\s.,]+", str(f.value)):
            if len(tok) < 6 or any(tok in LX.get(n) for n in ("shared/cities", "passport/issuer_places", "shared/place_words", "shared/regions", "shared/ussr_countries")): continue
            sn = LX.snap(tok, "shared/cities", thr=0.12)
            if sn["status"] in ("snapped", "ambiguous") and sn["alts"]: hints.append(f"«{tok}» ≈ «{sn['alts'][0][0]}» ({sn['alts'][0][1]})")
        if hints:
            f.note = ((f.note + "; ") if f.note else "") + "лексикон: названия нет в словаре, возможна путаница букв — " + ", ".join(hints) + " (не исправлено)"; f.confidence = float(f.confidence) * 0.97


def read_passport(pipe, img: np.ndarray) -> dict | None:
    """-> {fields, extra, image(выровненный кадр), photo_box, lines} или None, если это не паспорт."""
    H0, W0 = img.shape[:2]
    s = min(4.0, WORK_SIDE / max(H0, W0)) if max(H0, W0) < WORK_SIDE else (WORK_SIDE / max(H0, W0) if max(H0, W0) > WORK_SIDE * 1.3 else 1.0)
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA) if s != 1 else img
    notes = {"scale": round(s, 2)}
    paddle = pipe.paddle
    def prep(m):
        a = deskew_angle(m)
        if abs(a) >= 0.4:
            Hh, Ww = m.shape[:2]; m = cv2.warpAffine(m, cv2.getRotationMatrix2D((Ww / 2, Hh / 2), a, 1), (Ww, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return m, a
    best = None
    for rot in (0, 180, 90, 270):
        base = {0: im, 180: cv2.rotate(im, cv2.ROTATE_180), 90: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), 270: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)}[rot]
        m, a = prep(base)
        lines = paddle(m, merge=False)
        c = cue_count(lines)
        if best is None or c >= best[0] + 2: best = (c, rot, m, a, lines)
        if best[0] >= 4: break
    cues, rot, m, angle, lines = best
    if cues < 3: return None
    notes.update({"rotation_ccw": rot, "deskew_deg": round(angle, 2), "cues": cues})
    H, W = m.shape[:2]
    F: dict[str, Field] = {}
    # ---------- страницы: делим по подписям (последовательность) — «Фамилия»… ниже красной полосы
    flat = sorted(lines, key=lambda l: (l.cy, l.x0))
    lab_lower = anchors.locate_labels([l for l in flat if l.cy > 0.35 and l.x0 < 0.5], SPECS_LOWER, thr=0.6)
    lab_upper = anchors.locate_labels([l for l in flat if l.cy < 0.45 and l.x0 < 0.5], SPECS_UPPER, thr=0.6)
    split = 0.5
    ys = [v[0].cy for v in lab_lower.values()] if lab_lower else []
    if ys: split = max(0.40, min(0.55, min(ys) - 0.05))
    else:   # подписи нижней страницы не прочитаны (мелкий шрифт/ксерокс): граница — ниже последней отметки верхней страницы (дата, код, «Личный код», «М.П.», подпись)
        ups = [l.cy for l in flat if l.cy < 0.42 and (DATE_RE.search(l.text) or CODE_RE.search(l.text) or re.search(r"ЛИЧН|М\.?\s*П\.?|ПОДПИСЬ|КОД\s*ПОДР", l.text.upper()))]
        if ups: split = max(0.40, min(0.50, max(ups) + 0.07))
    notes["page_split"] = round(float(split), 3)
    mrz_lines = mrz_candidates(lines)
    mrz_y = min([l.cy for l in lines if l.cy > 0.55 and is_mrz_like(l.text)] + [l.cy for l in lines if l.cy > 0.78 and (l.text.count("<") >= 2 or "RUS" in l.text.upper())], default=1.0)
    # ---------- МРЗ
    mz = MRZ.parse(mrz_lines) if mrz_lines else None
    if mz and not (mz.get("line1_found") or mz.get("line2_found")): mz = None
    mz_ok = bool(mz and mz.get("checks") and mz["checks"]["check_doc"] and mz["checks"]["check_birth"])
    # ---------- верхняя страница
    up = [l for l in flat if l.cy < split]
    low = [l for l in flat if split <= l.cy < mrz_y - 0.015]
    dates_up = [(d, l) for l in up for d, _ in find_dates(l.text)]
    issue = None
    for d, l in dates_up:
        if valid_date(d, 1992): issue = (d, l); break
    codes = [(f"{a}-{b}", l) for l in up if l.x0 < 0.95 for a, b in CODE_RE.findall(l.text)]
    def pick(lst): return max(lst, key=lambda t: t[1].score) if lst else None
    cp = pick(codes)
    if cp: F["dept_code"] = Field(cp[0], cp[1].score, "printed", valid=True)
    if issue:
        F["issue_date"] = Field(issue[0], issue[1].score, "printed", valid=valid_date(issue[0], 1992) is not None)
    # выдан: строки между заголовком и строкой даты
    y_lim = (issue[1].cy if issue else (cp[1].cy if cp else 0.22)) - 0.02
    iss = []
    for l in up:
        if l.cy >= y_lim or l.x0 > 0.9: continue
        t = l.text
        if partial_ratio("РОССИЙСКАЯ", t) > 0.8 and len(t) < 16 or partial_ratio("ФЕДЕРАЦИЯ", t) > 0.8 and len(t) < 16 or partial_ratio("Российская Федерация", t) > 0.7 and len(t) < 30: continue
        if partial_ratio("Паспорт выдан", t) > 0.8 and len(t) < 18: continue
        if l.cy < 0.12 and max(sim(t, "ФЕДЕРАЦИЯ"), sim(t, "РОССИЙСКАЯ")) >= 0.7 and not re.search(r"УФМС|МВД|ОТДЕЛ", t.upper()): continue
        if l.cy < 0.12 and sim(t, "РОССИЙСКАЯ ФЕДЕРАЦИЯ") > 0.45 and not re.search(r"УФМС|МВД|ОТДЕЛ", t.upper()): continue
        if upper_frac(t) < 0.55 or not re.search(r"[А-ЯA-Z]{2}", t.upper()): continue
        iss.append(l)
    if iss:
        iss.sort(key=lambda l: l.cy)
        txt = clean_cyr(" ".join(l.text for l in iss), ISSUER_VOC)
        F["issued_by"] = Field(txt, float(np.mean([l.score for l in iss])), "printed")
    # ---------- нижняя страница: значения по порядку
    def is_label(l): return (upper_frac(l.text) < 0.4 and not re.search(r"\d", l.text)) or any(partial_ratio(v, l.text) > 0.85 and len(l.text) < len(v) + 4 and upper_frac(l.text) < 0.6 for v in ("Фамилия", "Имя", "Отчество", "Дата рождения", "Место рождения"))
    vals = []
    for l in low:
        if l.x0 > 0.9 or l.x1 < 0.30: continue
        t = l.text
        if is_label(l) or is_mrz_like(l.text): continue
        tt = re.sub(r"\b(Пол|Дата|рождения|рожд|Место|Фамилия|Имя|Отчество)\b\.?", " ", t, flags=re.I)
        vals.append((l, tt))
    # дата рождения, пол
    bd = None; sexl = None
    for l, tt in vals:
        for d, _ in find_dates(l.text):
            if valid_date(d, 1850): bd = bd or (d, l)
    for l, tt in vals:
        mm = re.search(r"(?<![А-ЯA-Z])(МУЖ|ЖЕН|МУЖ\.|XEH|ЖЕН)", clean_cyr(tt)) or None
        w = clean_cyr(tt).replace(".", "")
        for tok in w.split():
            for k in SEX:
                if (_lev(fold(tok), fold(k)) <= 1 or _LX.wdist(_LX.fold(tok), _LX.fold(k), 1.2) <= 1.05) and len(tok) >= 3 and not (len(tok) > 4):
                    sexl = sexl or (SEX[k], l)
    names, place = [], []
    row_y = bd[1].cy if bd else (sexl[1].cy if sexl else None)
    for l, tt in vals:
        if (bd and l is bd[1]) or (sexl and l is sexl[1]): 
            if l is not (bd[1] if bd else None) or sexl is None or l is not sexl[1]:
                pass
        t = DATE_RE.sub(" ", tt)
        for _rx in DATE_LENIENT:
            t = _rx.sub(lambda m_: " " if valid_date(f"{m_.group(1)}.{m_.group(2)}.{m_.group(3)}", 1850) else m_.group(0), t)
        t = clean_cyr(t)
        w = [x for x in t.split() if not (sexl and len(x) <= 4 and (_lev(fold(x), fold(sexl[0].strip(".")))<=1 or _LX.wdist(_LX.fold(x), _LX.fold(sexl[0].strip(".")), 1.2) <= 1.05))]
        t = " ".join(w)
        if not re.search(r"[А-ЯЁA-Z]{3}", t) or upper_frac(l.text) < 0.55 and not re.search(r"[А-ЯЁ]{3}", t): continue
        if row_y is not None and l.cy < row_y - 0.012: names.append((t, l))
        elif row_y is not None and l.cy > row_y + 0.012: place.append((t, l))
        elif row_y is None: names.append((t, l))
        else:
            # на одной строке с датой/полом: остаток — начало места рождения
            if t.strip(" .") and not re.fullmatch(r"(МУЖ|ЖЕН)\.?", t.strip()): place.append((t, l))
    names = [n for n in names if len(re.sub(r"[^А-ЯЁA-Z]", "", n[0])) >= 3]
    notes["lower_value_lines"] = len(names)
    PAT = re.compile(r"(ВИЧ|ВНА|ЧНА|ИЧ|ИЧНА|ЫЧ|ОГЛЫ|КЫЗЫ)$")
    sl = {}
    if len(names) >= 3:
        # три последних до строки даты (выше могут оказаться водяные знаки)
        sl = dict(zip(("surname", "name", "patronymic"), names[-3:]))
    elif len(names) == 2:
        sl = dict(zip(("name", "patronymic") if PAT.search(names[-1][0].split()[-1]) else ("surname", "name"), names))   # одна из трёх потеряна/размыта — заполним из МРЗ
    elif len(names) == 1:
        sl = {"patronymic" if PAT.search(names[0][0].split()[-1]) else "surname": names[0]}
    if "surname" not in sl and "name" in sl and pipe is not None and not (mz and mz.get("surname")):
        try:
            rs = _rescue_surname(pipe, m, {k: v[1] for k, v in sl.items()})
            if rs:
                top, w, votes = rs; tot = sum(votes.values())
                F["surname"] = Field(top, 0.45, "перечитано кропом по расчётной позиции (строку не нашёл детектор): варианты " + ", ".join(f"«{a}» ({b:.1f})" for a, b in sorted(votes.items(), key=lambda x: -x[1])[:4]))
        except Exception:
            pass
    for k, (t, l) in sl.items():
        F[k] = Field(t, l.score, "printed")
        if l.score < 0.75 and pipe is not None:      # размытая строка: перечитываем в разных масштабах/обработках, смотрим на согласие вариантов
            try:
                alt = _reread_alts(pipe, m, l, t)
                if alt: F[k].note = "printed; " + alt["note"]; F[k].confidence = float(F[k].confidence) * alt["factor"]; F[k].value = alt["value"]
            except Exception:
                pass
    if place:
        place.sort(key=lambda p: p[1].cy)
        txt = clean_place(clean_cyr(" ".join(t for t, _ in place)))
        F["birth_place"] = Field(txt, float(np.mean([l.score for _, l in place])), "printed")
    if bd: F["birth_date"] = Field(bd[0], bd[1].score, "printed", valid=valid_date(bd[0], 1850) is not None)
    if sexl: F["sex"] = Field(sexl[0], sexl[1].score, "printed", valid=True)
    # ---------- серия/номер: полосы справа
    strips = read_strips(m, paddle)
    sn = None
    ex_cmp = {}
    if strips:
        vals_ = list(strips.values())
        if len(vals_) == 2 and vals_[0]["number"] == vals_[1]["number"] and vals_[0]["series"] == vals_[1]["series"]:
            sn = dict(vals_[0]); sn["agree"] = True; sn["pages"] = "обе страницы совпали"
        else:
            sn = max(vals_, key=lambda v: (v["series"] is not None, v["score"])); sn = dict(sn); sn["agree"] = len(vals_) == 1 and None
            sn["pages"] = ("только одна полоса прочитана" if len(vals_) == 1 else "ПОЛОСЫ РАСХОДЯТСЯ: " + " | ".join(f"{k}: {v['series']} {v['number']}" for k, v in strips.items()))
            for k, v in strips.items():
                if mz and mz.get("number") and v["number"] == mz["number"]: pg = sn["pages"]; sn = dict(v); sn["agree"] = None; sn["pages"] = pg + f"; выбрана {k} (совпала с МРЗ)"
    # ---------- сверка с МРЗ
    checks = {}
    if mz:
        mc = mz.get("checks") or {}
        base_conf = 0.9 if mz_ok else 0.55
        def xcheck(key, mval, kind="text"):
            cur = F.get(key)
            if not mval: return
            if cur is None:
                if key in ("surname", "name", "patronymic") and (not mz_ok or not re.fullmatch(r"[А-ЯЁ\- ]{2,}", str(mval))): return   # МРЗ с OCR-ошибками имён в поле не подставляем
                F[key] = Field(mval, base_conf * 0.9, "из МРЗ (в печатной части не прочитано)", valid=None); checks[key] = "mrz_only"; return
            a, b = str(cur.value), str(mval)
            same = (a == b) if kind == "exact" else (sim(a, b) >= 0.75 or (len(b) >= 4 and sim(a[:len(b) + 1], b) >= 0.8))
            if same and kind != "exact" and len(a) == len(b) and a != b and base_conf >= 0.9 and sim(a, b) >= 0.7:
                # те же длины, различия только в «похожих» буквах -> побуквенно берём МРЗ (контрольные суммы МРЗ сошлись)
                conf_pairs = {frozenset(x) for x in ("БВ", "НП", "ИЙ", "ИН", "ШЩ", "ЗЭ", "ЕЁ", "ХЖ", "ОЮ", "СЕ", "ГТ", "ЛД", "ЬЫ", "УЧ", "АЯ", "ЕВ", "РВ", "ИЛ", "ГП")}
                if key in ("name", "patronymic", "surname"):
                    from .. import lexicon as LX
                    lx = {"name": "shared/first_names", "patronymic": "shared/patronymics", "surname": "shared/surnames"}[key]
                    if a in LX.get(lx) and b not in LX.get(lx):    # в МРЗ вариант вне словаря (В/Б, Н/П неоднозначны при транслитерации), печатный — словарный
                        cur.note = f"печать ('{a}') есть в словаре, вариант МРЗ ('{b}') — нет: оставлена печать"; cur.confidence = max(cur.confidence, 0.85); checks[key] = "agree"; return
                diff = [(x, y) for x, y in zip(a, b) if x != y]
                if all(frozenset((x, y)) in conf_pairs for x, y in diff):
                    cur.note = f"исправлено по МРЗ (печать читалась как '{a}')"; cur.value = b; cur.confidence = 0.9; checks[key] = "corrected"; return
            if same:
                if kind == "exact" or sim(a, b) > 0.99: cur.confidence = min(1.0, 1 - (1 - cur.confidence) * 0.3); cur.note = "печать = МРЗ"
                else: cur.confidence = min(1.0, cur.confidence * 0.5 + 0.5 * base_conf + 0.1); cur.note = f"печать≈МРЗ ({mval})"
                checks[key] = "agree"
            else:
                checks[key] = f"MISMATCH print='{a}' mrz='{b}'"
                if kind == "exact" and mz_ok:     # даты/код: МРЗ с проверенными контрольными цифрами надёжнее
                    cur.note = f"исправлено по МРЗ (печать читалась как '{a}')"; cur.value = mval; cur.confidence = base_conf; cur.valid = True
                elif kind == "exact":
                    cur.note = f"РАСХОЖДЕНИЕ с МРЗ ('{mval}'; контрольные цифры МРЗ не сошлись — МРЗ недостоверен)"; cur.confidence *= 0.85
                else:
                    cur.note = f"расхождение с МРЗ (в МРЗ: '{mval}'; МРЗ мог быть прочитан/напечатан неточно)"; cur.confidence *= 0.85
        xcheck("birth_date", mz.get("birth_date"), "exact"); xcheck("issue_date", mz.get("issue_date"), "exact"); xcheck("dept_code", mz.get("dept_code"), "exact")
        xcheck("sex", mz.get("sex"), "exact")
        for k in ("surname", "name", "patronymic"):
            xcheck(k, mz.get(k))
        if "surname" in F and F["surname"].note and "МРЗ" in (F["surname"].note or "") and F["surname"].note.startswith("из МРЗ"): F["surname"].note += "; печатная фамилия не прочитана (размыта?)"
        if mz.get("number") and sn:
            ok = sn["number"] == mz["number"] and (not sn.get("series") or sn["series"][:2] == mz["series3"][:2] and sn["series"].replace(" ", "")[:3] == mz["series3"])
            checks["series_number"] = "agree" if ok else f"MISMATCH strips={sn.get('series')} {sn['number']} mrz={mz.get('series3')}x {mz['number']}"
    _lexicon_pass(F, checks)
    # серия/номер: итог
    if sn:
        conf = sn["score"] * (1.0 if sn.get("agree") else 0.85)
        note = sn["pages"]
        if checks.get("series_number") == "agree": conf = min(1.0, conf + 0.05); note += "; = МРЗ"
        elif str(checks.get("series_number", "")).startswith("MISMATCH"): note += "; РАСХОЖДЕНИЕ с МРЗ"; conf *= 0.7
        ok = bool(sn["series"]) and bool(re.fullmatch(r"\d{2} \d{2}", sn["series"]))
        F["series"] = Field(sn["series"], conf, note, valid=ok) if sn["series"] else Field(None, 0.0, "серия не прочитана", valid=False)
        F["number"] = Field(sn["number"], conf, note, valid=len(sn["number"]) == 6)
    elif mz and mz.get("number"):
        F["number"] = Field(mz["number"], 0.5, "из МРЗ (красный номер не найден)", valid=True)
        if mz.get("series"): F["series"] = Field(mz["series"], 0.45, "из МРЗ (4-я цифра — из доп. данных)", valid=True)
    # ---------- валидаторы дат
    cur_year = date.today().year
    bdt = valid_date(F["birth_date"].value, 1850) if "birth_date" in F else None
    idt = valid_date(F["issue_date"].value, 1992) if "issue_date" in F else None
    if bdt and bdt.year < 1900: F["birth_date"].note = ((F["birth_date"].note or "") + "; год рождения <1900 — нереалистично (образец?)").strip("; "); F["birth_date"].valid = False
    if bdt and idt and (idt.year - bdt.year - ((idt.month, idt.day) < (bdt.month, bdt.day))) < 14:
        F["issue_date"].note = ((F["issue_date"].note or "") + "; выдан до 14 лет — проверьте").strip("; ")
    if "issue_date" in F and idt is None: F["issue_date"].valid = False
    if "dept_code" in F: F["dept_code"].valid = bool(re.fullmatch(r"\d{3}-\d{3}", F["dept_code"].value))
    if "series" in F and F["series"].value and "issue_date" in F and idt:
        yy = int(F["series"].value.split()[1]); 
        if abs((idt.year % 100) - yy) > 1: F["series"].note = ((F["series"].note or "") + f"; год в серии ({yy:02d}) далёк от года выдачи").strip("; ")
    # финальный confidence: страховка по OCR
    for f in F.values(): f.confidence = float(max(0.0, min(1.0, f.confidence)))
    for k_, f_ in F.items():     # ни одно обычное поле не содержит МРЗ
        if isinstance(f_.value, str) and k_ not in ("series", "number", "dept_code", "issue_date", "birth_date"):
            cl = strip_mrz_text(f_.value)
            if cl != f_.value: f_.note = ((f_.note + "; ") if f_.note else "") + "отрезан текст МРЗ из значения"; f_.value = cl
    extra = {"mrz": _mrz_block(mz, mrz_lines), "mrz_cross_check": checks, "orientation": notes, "series_number_strips": strips}
    # фото: лицо -> рамка фотографии
    photo_box = None
    f = pipe._detect_face(m)
    if f is not None:
        fx, fy, fw, fh = [float(v) for v in f[:4]]
        x0, x1 = fx - 0.55 * fw, fx + 1.55 * fw; y0, y1 = fy - 0.75 * fh, fy + 1.95 * fh
        photo_box = [max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))]
        extra["photo"] = {"label_ru": "Фото владельца", "face_detected": True, "face_confidence": round(float(f[-1]), 3)}
    else:
        extra["photo"] = {"label_ru": "Фото владельца", "face_detected": False}
    return {"fields": F, "extra": extra, "image": m, "photo_box": photo_box, "lines": lines, "angle": angle}
