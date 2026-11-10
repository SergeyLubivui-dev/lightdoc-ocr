"""Нормализаторы и валидаторы полей СТС/ПТС: госномер, VIN, номер ПТС/СТС, год, категория, мощность, масса, цвет, экологический класс."""
from __future__ import annotations
import re
from datetime import date
from difflib import get_close_matches
from .textutils import _lev
from . import lexicon as LX

PLATE_LET = "АВЕКМНОРСТУХ"
_LAT2CYR = {"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х", "0": "О", "I": "Е", "1": "Е"}
_CYR2LAT = {v: k for k, v in {"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х"}.items()}
_DIG = {"O": "0", "О": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "З": "3", "S": "5", "B": "8", "В": "8", "G": "6", "Б": "6"}


def _plate_try(s: str, fmt: str):
    """fmt 'car': Б ЦЦЦ ББ РР(Р); 'trailer': ББ ЦЦЦЦ РР(Р) (прицепы; тип 4 по ГОСТ Р 50577)."""
    let = (0, 4, 5) if fmt == "car" else (0, 1)
    v = "".join((_LAT2CYR.get(ch, ch) if i in let else _DIG.get(ch, ch)) for i, ch in enumerate(s))
    rx = rf"[{PLATE_LET}]\d{{3}}[{PLATE_LET}]{{2}}\d{{2,3}}" if fmt == "car" else rf"[{PLATE_LET}]{{2}}\d{{4}}\d{{2,3}}"
    return v, bool(re.fullmatch(rx, v))


def norm_plate(raw: str):
    """-> (value 'А123ВС77', valid, note). Позиционный разбор: легковой/грузовой «буква ЦИФРЫ(3) буквы(2) регион(2-3)»
    или прицеп «буквы(2) ЦИФРЫ(4) регион(2-3)»; выбирается формат, при котором строка валидна."""
    s = re.sub(r"[\s\-._]", "", raw.upper())
    if not (8 <= len(s) <= 9): return None, False, None
    vc, okc = _plate_try(s, "car")
    if okc: return vc, True, "".join(_CYR2LAT.get(c, c) for c in vc)
    vt, okt = _plate_try(s, "trailer")
    if okt: return vt, True, "".join(_CYR2LAT.get(c, c) for c in vt) + " (прицеп)"
    return vc, False, None


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
    """ПТС: 2 цифры + 2 кириллические буквы + 6 цифр ('77АА123456'), латинские двойники и 0/О исправляем позиционно."""
    s = re.sub(r"[^0-9A-Za-zА-Яа-я]", "", raw.upper())
    s = re.sub(r"^(ЭПТС|ПТС)", "", s)
    d15 = "".join(_DIG.get(c, c) for c in s)
    if len(s) == 15 and d15.isdigit(): return d15, True   # электронный ПТС (ЭПТС): 15 цифр
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
    s = re.sub(r"(ПОЛУ)?ПРИЦЕП", "", raw.upper()).replace(" ", "").replace("\\", "/").replace("|", "/").strip("/")
    s = "".join({"А": "A", "В": "B", "С": "C", "Е": "E", "М": "M", "О": "O", "Д": "D", "З": "3", "Ч": "4"}.get(c, c) for c in s)
    s = re.sub(r"[^A-Z0-9/]", "", s)
    s = re.sub(r"(?:(?<=/)|^)0(?=\d$)", "O", s) if re.search(r"ПРИЦЕП", raw.upper()) else s   # «04» вместо «O4»
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
    return int(d), 100 <= int(d) <= 99999


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


# ---- привязка к лексиконам (docrec/lexicons/sts/*.txt.gz): фразы целиком, затем запасной пословный разбор выше
def _best_window(text: str, lex: str, thr: float, min_words: int = 1):
    """Лучшее совпадение среди непрерывных окон слов текста (лишние слова по краям — шум OCR). -> snap-dict или None."""
    words = text.split(); best = None
    for n in range(len(words), min_words - 1, -1):
        for i in range(0, len(words) - n + 1):
            s = LX.snap(" ".join(words[i:i + n]), lex, thr=thr)
            if s["status"] in ("exact", "snapped"):
                sc = (s["norm"] or 0.0) + 0.04 * (len(words) - n)   # потеря слов чуть штрафуется
                if best is None or sc < best[0]: best = (sc, s)
        if best is not None and n == len(words): break
    return best[1] if best else None


def norm_vehicle_type_lex(raw: str):
    t = " ".join(re.sub(r"[^А-Яа-яЁёA-Za-z\- ]", " ", raw).split())
    if len(t.replace(" ", "")) >= 5:
        s = _best_window(t, "sts/vehicle_type", 0.22, 2 if len(t.split()) > 1 else 1)
        if s: return s["value"], True, LX.note_for(s, "тип ТС")
    v, ok = norm_vehicle_type(raw)
    return v, ok, None


def norm_color_lex(raw: str):
    t = " ".join(re.sub(r"[^А-Яа-яЁё\- ]", "", raw).split())
    if len(t) >= 3:
        s = _best_window(t, "sts/color", 0.3)
        if s: return s["value"], True, LX.note_for(s, "цвет")
    v, ok = norm_color(raw)
    return v, ok, None


def _eco_prefix(t: str):
    """Значение у края/под бликом обрезано («НЕ УСТАНОВ…»): достраиваем до словарной формы, но помечаем как обрезанное."""
    from .textutils import _lev, fold
    f = fold(t.replace(" ", ""))
    if len(f) < 6: return None
    for full in ("не установлено", "не установлен"):
        ff = fold(full.replace(" ", ""))
        if len(f) < len(ff) and _lev(f, ff[:len(f)]) <= max(1, len(f) // 8): return full
    return None


def norm_eco_lex(raw: str):
    t = re.sub(r"[^А-Яа-яЁё ]", "", raw).strip()
    pre = _eco_prefix(t)
    if pre: return pre, True, "значение обрезано/закрыто бликом: достроено по словарю («не установлено»)"
    if len(t) >= 3:
        s = LX.snap(t.split()[0] if len(t.split()) == 1 else t, "sts/eco_class", thr=0.3)
        if s["status"] in ("exact", "snapped"): return s["value"].split("|")[0], True, LX.note_for(s, "экокласс")
    v, ok = norm_eco(raw)
    return v, ok, None
