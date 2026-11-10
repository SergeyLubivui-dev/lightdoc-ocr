"""Нормализаторы и валидаторы полей СТС/ПТС: госномер, VIN, номер ПТС/СТС, год, категория, мощность, масса, цвет, экологический класс."""
from __future__ import annotations
import re
from datetime import date
from difflib import get_close_matches
from .textutils import _lev

PLATE_LET = "АВЕКМНОРСТУХ"
_LAT2CYR = {"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х", "0": "О", "I": "Е", "1": "Е"}
_CYR2LAT = {v: k for k, v in {"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х"}.items()}
_DIG = {"O": "0", "О": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "З": "3", "S": "5", "B": "8", "В": "8", "G": "6", "Б": "6"}


def norm_plate(raw: str):
    """-> (value 'А123ВС77', valid, note). Позиционный разбор: буква ЦИФРЫ(3) буквы(2) регион(2-3)."""
    s = re.sub(r"[\s\-._]", "", raw.upper())
    if not (8 <= len(s) <= 9): return None, False, None
    out = []
    for i, ch in enumerate(s):
        if i in (0, 4, 5):
            c = _LAT2CYR.get(ch, ch)
            out.append(c)
        else:
            out.append(_DIG.get(ch, ch))
    v = "".join(out)
    ok = bool(re.fullmatch(rf"[{PLATE_LET}]\d{{3}}[{PLATE_LET}]{{2}}\d{{2,3}}", v))
    return v, ok, "".join(_CYR2LAT.get(c, c) for c in v) if ok else None


_VIN_FIX = {"O": "0", "Q": "0", "I": "1", "О": "0", "Х": "X", "С": "C", "Т": "T", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "Р": "P", "У": "Y"}
_VIN_W = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
_VIN_T = {**{str(i): i for i in range(10)}, **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))), "P": 7, "R": 9, **dict(zip("STUVWXYZ", range(2, 10)))}


def norm_vin(raw: str):
    s = re.sub(r"[^0-9A-Za-zА-Яа-я]", "", raw).upper()
    s = "".join(_VIN_FIX.get(c, c) for c in s)
    if len(s) not in (16, 17, 18): 
        return (s or None), False, f"длина {len(s)}"
    if len(s) != 17: return s, False, f"длина {len(s)}"
    ok = bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", s))
    note = None
    if ok:
        tot = sum(_VIN_T.get(c, 0) * w for c, w in zip(s, _VIN_W)) % 11
        chk = "X" if tot == 10 else str(tot)
        note = f"контрольная цифра (поз.9) {'совпала' if s[8] == chk else 'не совпала'} — информативно, для РФ/ЕС не обязательна"
    return s, ok, note


def norm_pts(raw: str):
    """ПТС: 2 цифры + 2 кириллические буквы + 6 цифр ('77УТ123456'), латинские двойники и 0/О исправляем позиционно."""
    s = re.sub(r"[^0-9A-Za-zА-Яа-я]", "", raw.upper())
    s = re.sub(r"^ПТС", "", s)
    if len(s) != 10: return None, False
    out = []
    for i, ch in enumerate(s):
        if i in (2, 3): out.append(_LAT2CYR.get(ch, ch) if ch not in "01" else {"0": "О", "1": "Е"}[ch])
        else: out.append(_DIG.get(ch, ch))
    v = "".join(out)
    return v, bool(re.fullmatch(r"\d{2}[А-Я]{2}\d{6}", v))


def norm_sts_number(raw: str):
    d = re.sub(r"[^0-9OО]", "", raw.upper().replace("О", "0").replace("O", "0"))
    if len(d) != 10: return None, False
    return f"{d[:2]} {d[2:4]} {d[4:]}", True


def norm_year(raw: str):
    d = re.sub(r"[^0-9OО]", "", raw.upper().replace("О", "0").replace("O", "0"))
    if len(d) < 4: return None, False
    d = d[:4] if len(d) >= 4 else d
    y = int(d)
    return d, 1900 <= y <= date.today().year + 1


_CAT = re.compile(r"^(A1?|B1?|C1?E?|D1?E?|BE|M|Tb|Tm|[ABCDEM])$")


def norm_category(raw: str):
    """«C/N3» -> {'category': 'C', 'class': 'N3'}; допустимо только 'C' или только 'N3'."""
    s = raw.upper().replace(" ", "").replace("\\", "/").replace("|", "/")
    s = "".join({"А": "A", "В": "B", "С": "C", "Е": "E", "М": "M", "О": "O", "Д": "D", "З": "3", "Ч": "4"}.get(c, c) for c in s)
    s = re.sub(r"[^A-Z0-9/]", "", s)
    s = re.sub(r"\bN(?=[0-9]\b)", "N", s)
    m = re.fullmatch(r"([ABCDEM]\d?E?)?/?(([LMNOT])(\d[A-C]?)?)?", s)
    if not m or not s: return None, False
    cat, cls = m.group(1), m.group(2)
    if cat and not _CAT.match(cat.title() if cat in ("TB", "TM") else cat): cat = None
    if not cat and not cls: return None, False
    return {"category": cat, "class": cls}, True


def norm_power(raw: str):
    """'295/401' кВт / л.с. (допускаем '.'/',' и хвостовой знак)."""
    t = raw.replace(",", ".").replace(" ", "")
    m = re.search(r"(\d{1,3}(?:\.\d)?)/(\d{1,3}(?:\.\d)?)", t)
    if not m:
        m1 = re.search(r"(\d{2,3}(?:\.\d)?)", t)
        return ({"kw": float(m1.group(1))}, False, "только одно число — вероятно срезано краем") if m1 else (None, False, None)
    kw, hp = float(m.group(1)), float(m.group(2))
    ok = abs(kw * 1.35962 - hp) <= max(1.5, 0.03 * hp)
    return {"kw": kw, "hp": hp}, ok, None if ok else f"{kw} кВт ≠ {hp} л.с. (ожидалось ≈{kw * 1.35962:.0f})"


def norm_mass(raw: str):
    d = re.sub(r"[^0-9]", "", raw.translate(str.maketrans("OОoо", "0000")).replace("S", "5").replace("B", "8"))
    if not d: return None, False
    return int(d), 100 <= int(d) <= 60000


COLORS = ["белый", "чёрный", "серый", "серебристый", "синий", "голубой", "красный", "бордовый", "зелёный", "жёлтый", "оранжевый", "коричневый",
          "бежевый", "фиолетовый", "розовый", "золотистый", "тёмно-синий", "тёмно-серый", "светло-серый", "светло-синий", "тёмно-зелёный", "серо-голубой", "вишнёвый", "салатовый", "бирюзовый"]
ECO = ["нулевой", "первый", "второй", "третий", "четвёртый", "пятый", "шестой"]
VTYPES = ["легковой", "грузовой", "автобус", "тягач", "седельный", "прицеп", "полуприцеп", "фургон", "самосвал", "бортовой", "цистерна", "мотоцикл", "мопед", "специализированный", "рефрижератор", "универсал", "седан", "хэтчбек", "внедорожник", "кроссовер", "пикап", "микроавтобус", "комби", "купе", "кабриолет", "лимузин", "трактор", "самоходная", "машина"]


def _snap(word: str, vocab: list[str], maxd: int = 2):
    w = word.lower().replace("ё", "е")
    voc = {v.replace("ё", "е"): v for v in vocab}
    if w in voc: return voc[w], True
    best = None
    for k, v in voc.items():
        d = _lev(w, k)
        if d <= maxd and d <= max(1, len(k) // 4 + 1) and (best is None or d < best[0]): best = (d, v)
    return (best[1], True) if best else (word.lower(), False)


def norm_color(raw: str):
    t = re.sub(r"[^А-Яа-яЁё\- ]", "", raw).strip()
    if not t: return None, False
    parts = [_snap(p, COLORS)[0] for p in t.split()]
    v = " ".join(parts); ok = v in COLORS or all(p in COLORS for p in v.split("-"))
    return v.capitalize(), ok


def norm_eco(raw: str):
    t = re.sub(r"[^А-Яа-яЁё]", "", raw)
    if not t: return None, False
    v, ok = _snap(t, ECO, 2)
    return v, ok


def norm_vehicle_type(raw: str):
    t = re.sub(r"[^А-Яа-яЁё\- ]", " ", raw)
    words = [w for w in t.split() if len(w) > 1]
    if not words: return None, False
    sn = [_snap(w, VTYPES, 2) for w in words]
    out = " ".join(s[0] for s in sn)
    return out.capitalize(), sum(s[1] for s in sn) >= max(1, len(sn) // 2)
