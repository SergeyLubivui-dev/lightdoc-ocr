"""Универсальная постобработка текста OCR для русских документов:
 - исправление «двойников» кириллица↔латиница и цифр-двойников по контексту;
 - словарь ключевых слов реквизитов (ИНН, КПП, БИК, ОГРН, СНИЛС, ООО, ПАО Сбербанк …);
 - нормализация регистра («ПолУЧатеЛЬ» → «Получатель»);
 - порядок чтения с учётом колонок (XY-cut);
 - фильтр шума: штампы/печати (синие чернила + низкая уверенность), одиночные символы-каракули."""
from __future__ import annotations
import re
import numpy as np
import cv2
from .ocr import Line
from .textutils import _L2C, _C2L, fold, _lev

# ---------------------------------------------------------------- словарь
KEYWORDS_NUM = ["ИНН", "КПП", "БИК", "ОГРН", "ОГРНИП", "СНИЛС", "ОКПО", "ОКТМО", "ОКВЭД", "БИК"]
KEYWORDS = ["ИНН", "КПП", "БИК", "ОГРН", "ОГРНИП", "СНИЛС", "ОКПО", "ОКТМО", "ООО", "ПАО", "АО", "ОАО", "ЗАО", "ИП", "Сбербанк", "СБЕРБАНК",
            "ЗАКАЗЧИК", "ИСПОЛНИТЕЛЬ", "ПОКУПАТЕЛЬ", "ПОСТАВЩИК", "ПРОДАВЕЦ", "ПОДРЯДЧИК", "АРЕНДАТОР", "АРЕНДОДАТЕЛЬ", "ПОЛУЧАТЕЛЬ", "ПЛАТЕЛЬЩИК",
            "Получатель", "Покупатель", "Поставщик", "Заказчик", "Исполнитель", "Директор", "Адрес", "адрес", "Юридический", "Телефон", "Банк", "банка",
            "Расчетный", "Расчётный", "счет", "счёт", "Корр", "Паспорт", "Паспортные", "данные", "выдан", "Тел", "Договор"]
_ACR = {"ИНН", "КПП", "БИК", "ОГРН", "ОГРНИП", "СНИЛС", "ОКПО", "ОКТМО", "ООО", "ПАО", "АО", "ОАО", "ЗАО", "ИП"}
_KW_FOLD = {}
for _k in KEYWORDS:
    _KW_FOLD.setdefault(fold(_k), _k)

_PARTY_HEADERS = {"ЗАКАЗЧИК", "ИСПОЛНИТЕЛЬ", "ПОКУПАТЕЛЬ", "ПОСТАВЩИК", "ПРОДАВЕЦ", "ПОДРЯДЧИК", "АРЕНДАТОР", "АРЕНДОДАТЕЛЬ", "ЗАЙМОДАВЕЦ",
                  "ЗАЕМЩИК", "ПОДРЯДЧИК", "СУБПОДРЯДЧИК", "ЛИЦЕНЗИАР", "ЛИЦЕНЗИАТ", "ПРИНЦИПАЛ", "АГЕНТ", "КОМИССИОНЕР", "ПОКУПАТЕЛЬ", "НАНИМАТЕЛЬ", "РАБОТОДАТЕЛЬ", "РАБОТНИК", "ХРАНИТЕЛЬ", "ПОКЛАДОДАТЕЛЬ"}
PARTY_HEADERS = _PARTY_HEADERS

_LET = r"A-Za-zА-Яа-яЁё"
_WORD = re.compile(rf"[{_LET}]+")
_LAT_SET = set("ABCEHKMOPTXYabcehkmoprtxy")
_C2L_ALL = dict(_C2L)


def _letters_script(w: str):
    c = sum(("А" <= ch <= "я") or ch in "Ёё" for ch in w); l = sum(ch.isascii() and ch.isalpha() for ch in w)
    return c, l


def _fix_word_scripts(w: str, ctx_cyr: bool) -> str:
    """Слово из смеси скриптов приводим к доминирующему; если слово целиком из латинских «двойников» в кириллической строке — в кириллицу."""
    c, l = _letters_script(w)
    if c and l:
        # смесь: однозначные буквы решают, иначе большинство
        cyr_only = sum(ch in "БГДЖЗИЙЛПФЦЧШЩЪЫЬЭЮЯбгджзийлпфцчшщъыьэюяЁё" for ch in w)
        lat_only = sum(ch in "DFGIJLNQRSUVWZdfgijlnqrsuvwz" for ch in w)
        if lat_only > cyr_only or (lat_only == cyr_only and not ctx_cyr):
            return "".join(_C2L_ALL.get(ch, ch) for ch in w)
        return "".join(_L2C.get(ch, ch) for ch in w)
    if l and not c and ctx_cyr and len(w) >= 2 and all(ch in _LAT_SET for ch in w):
        return "".join(_L2C.get(ch, ch) for ch in w)
    return w


_DIGIT_LOOK = {"O": "0", "О": "0", "o": "0", "о": "0", "l": "1", "I": "1", "|": "1", "З": "3", "S": "5", "B": "8", "В": "8", "Б": "6", "б": "6", "Z": "2"}


def _fix_digit_token(tok: str) -> str:
    """Токен, где ≥70% цифр и есть двойники букв-цифр («407О28», «З0101») — приводим к цифрам."""
    core = re.sub(r"[\s.,:;/\\-]", "", tok)
    if len(core) < 5: return tok
    dig = sum(ch.isdigit() for ch in core)
    if dig / len(core) >= 0.7 and all(ch.isdigit() or ch in _DIGIT_LOOK for ch in core):
        return "".join(_DIGIT_LOOK.get(ch, ch) for ch in tok)
    return tok


def normalize_case(w: str) -> str:
    if len(w) < 3: return w
    letters = [ch for ch in w if ch.isalpha()]
    if len(letters) < 3: return w
    up = [ch.isupper() for ch in letters]
    if all(up) or not any(up): return w
    if up[0] and not any(up[1:]): return w          # Capitalized
    # смешанный регистр внутри слова: «ПолУЧатеЛЬ» -> «Получатель», «нИКОЛАЙ» -> «Николай»
    if sum(up) >= 0.75 * len(up): return w.upper()
    if up[0]: return w[0].upper() + w[1:].lower()
    return (w[0].upper() if up[0] else w[0].lower()) + w[1:].lower()


def snap_keyword(w: str, next_is_number: bool = False) -> str:
    f = fold(w)
    if not f: return w
    if f in _KW_FOLD:
        k = _KW_FOLD[f]
        return k if k in _ACR else w   # аббревиатуры (ИНН, КПП, ООО) — всегда капсом
    if next_is_number and 2 <= len(f) <= 6:
        # «HHH 1234567890», «KIII 23530», «BИK 0403» — искажённая метка реквизита перед числом
        best = min(KEYWORDS_NUM, key=lambda k: _lev(fold(k), f))
        if _lev(fold(best), f) <= (1 if len(f) <= 3 else 2) and (abs(len(best) - len(f)) <= 1):
            return best
        # частые искажения: ИНН→HHH/ННН/ИHH, КПП→KIII/КIII/KПП
        conf = {"HHH": "ИНН", "ННН": "ИНН", "ИНН": "ИНН", "KIII": "КПП", "КIII": "КПП", "КШ": "КПП", "ККП": "КПП", "KПП": "КПП"}
        if f in conf or w in conf: return conf.get(f) or conf[w]
    return w


_NUM_NEXT = re.compile(r"^[\s:№N#.,-]*\d{6,}")


def fix_text(text: str) -> str:
    """Строка OCR -> исправленная строка (двойники, ключевые слова, регистр)."""
    c, l = _letters_script(text)
    ctx_cyr = c >= l
    # 1. цифровые токены с буквами-двойниками
    parts = re.split(r"(\s+)", text)
    parts = [_fix_digit_token(p) if p.strip() else p for p in parts]
    text = "".join(parts)
    # 2. слова: скрипты, регистр, ключевые слова
    out = []; pos = 0
    for m in _WORD.finditer(text):
        out.append(text[pos:m.start()]); w = m.group(0)
        w = _fix_word_scripts(w, ctx_cyr)
        w = normalize_case(w)
        nxt = text[m.end():m.end() + 24]
        w = snap_keyword(w, bool(_NUM_NEXT.match(nxt)))
        out.append(w); pos = m.end()
    out.append(text[pos:])
    t = "".join(out)
    t = re.sub(r"(?<![\wА-Яа-я])[0OО]{3}(?=\s*[«\"“])", "ООО", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fix_lines(lines: list[Line]) -> list[Line]:
    for l in lines:
        l.raw_text = getattr(l, "raw_text", l.text)
        l.text = fix_text(l.text)
    return lines


# ---------------------------------------------------------------- шум
def blue_ink_fraction(img: np.ndarray, box) -> float:
    xs0, ys0 = int(max(0, box[0][0])), int(max(0, box[0][1])); xs1, ys1 = int(box[2][0]), int(box[2][1])
    roi = img[ys0:ys1, xs0:xs1]
    if roi.size == 0: return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 100) & (h <= 130) & (s >= 110) & (v >= 70)).mean())


def filter_noise(lines: list[Line], img: np.ndarray | None = None, drop_log: list | None = None) -> list[Line]:
    keep = []
    for l in lines:
        t = l.text.strip(); alnum = re.sub(r"[\W_]", "", t)
        reason = None
        if len(alnum) == 0: reason = "no_alnum"
        elif len(alnum) == 1 and not t.endswith((".", ")")) and not re.match(r"^\d$", t) is None and False: reason = None
        if reason is None and len(alnum) == 1 and not re.search(r"[.)]$", t): reason = "single_char"
        if reason is None and l.score < 0.35: reason = "low_conf"
        if reason is None and l.score < 0.6 and len(alnum) <= 8: reason = "short_low_conf"
        if reason is None and img is not None and l.score < 0.8:
            bf = blue_ink_fraction(img, l.box)
            if bf > 0.04 and (l.score < 0.65 or len(alnum) <= 6): reason = f"stamp_ink({bf:.2f})"
        if reason:
            if drop_log is not None: drop_log.append({"text": t, "conf": round(l.score, 2), "reason": reason})
            continue
        keep.append(l)
    return keep


# ---------------------------------------------------------------- порядок чтения (XY-cut с «перемычками»)
def _vgap(segs, aspect=1.0, bins=320):
    n = len(segs)
    if n < 5: return None
    x0 = min(s.x0 for s in segs); x1 = max(s.x1 for s in segs)
    if x1 - x0 < 1e-6: return None
    mh = float(np.median([s.y1 - s.y0 for s in segs])) or 0.01
    cnt = np.zeros(bins, int)
    for s in segs:
        a = int((s.x0 - x0) / (x1 - x0) * (bins - 1)); b = int((s.x1 - x0) / (x1 - x0) * (bins - 1))
        cnt[a:b + 1] += 1
    thr = max(1, int(0.12 * n))   # допускаем «перемычки» (заголовки на всю ширину)
    best = None; i = 0
    while i < bins:
        if cnt[i] <= thr:
            j = i
            while j < bins and cnt[j] <= thr: j += 1
            if i > 0 and j < bins:
                m = cnt[i:j].min(); k = i
                while k < j:                      # самый широкий участок с минимальным покрытием
                    if cnt[k] == m:
                        e = k
                        while e < j and cnt[e] == m: e += 1
                        wid = (e - k) / (bins - 1) * (x1 - x0)
                        if wid * aspect >= 2.2 * mh and wid >= 0.015:
                            xm0 = x0 + k / (bins - 1) * (x1 - x0); xm1 = x0 + e / (bins - 1) * (x1 - x0); xm = (xm0 + xm1) / 2
                            left = [s for s in segs if s.x1 <= xm1 and s.cx < xm]; right = [s for s in segs if s.x0 >= xm0 and s.cx >= xm]
                            if len(left) >= 2 and len(right) >= 2 and (best is None or wid > best[0]):
                                best = (wid, xm, xm0, xm1)
                        k = e
                    else: k += 1
            i = j
        else: i += 1
    return best


def xy_order(segs: list[Line], aspect: float = 1.0, depth: int = 0) -> list[Line]:
    if len(segs) <= 1 or depth > 6: return sorted(segs, key=lambda s: (round(s.cy, 3), s.x0))
    mh = float(np.median([s.y1 - s.y0 for s in segs])) or 0.01
    g = _vgap(segs, aspect)
    if g:
        _, xm, xm0, xm1 = g
        bridges = sorted([s for s in segs if s.x0 < xm < s.x1], key=lambda s: s.cy)
        rest = [s for s in segs if s not in bridges]
        out = []; bands = []
        edges = [b.cy for b in bridges]
        # полосы между перемычками
        cuts = [-1.0] + edges + [9.0]
        for bi in range(len(cuts) - 1):
            band = [s for s in rest if cuts[bi] <= s.cy < cuts[bi + 1] and not (bi < len(edges) and False)]
            L = [s for s in band if s.cx < xm]; R = [s for s in band if s.cx >= xm]
            out += xy_order(L, aspect, depth + 1) + xy_order(R, aspect, depth + 1)
            if bi < len(bridges): out.append(bridges[bi])
        return out
    # горизонтальные разрезы
    ss = sorted(segs, key=lambda s: s.y0)
    bands = [[ss[0]]]; ymax = ss[0].y1
    for s in ss[1:]:
        if s.y0 - ymax >= 2.0 * mh: bands.append([s])
        else: bands[-1].append(s)
        ymax = max(ymax, s.y1)
    if len(bands) > 1:
        out = []
        for b in bands: out += xy_order(b, aspect, depth + 1)
        return out
    # лист: строки по y, внутри строки по x
    rows = []
    for s in sorted(segs, key=lambda s: s.cy):
        if rows and abs(s.cy - np.mean([r.cy for r in rows[-1]])) < 0.6 * mh: rows[-1].append(s)
        else: rows.append([s])
    return [s for r in rows for s in sorted(r, key=lambda s: s.x0)]
