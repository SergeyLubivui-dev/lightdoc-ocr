from __future__ import annotations
import re
from difflib import SequenceMatcher

# Латинские буквы, визуально неотличимые от кириллических (и обратно)
_L2C = dict(zip("ABCEHKMOPTXYabcehkmoprtxy", "АВСЕНКМОРТХУавсенкморртху"))
_C2L = {v: k for k, v in _L2C.items() if k.isupper()}
_C2L.update({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y"})
CYR_ONLY = set("БГДЖЗИЙЛПФЦЧШЩЪЫЬЭЮЯбгджзийлпфцчшщъыьэюяЁё")
LAT_ONLY = set("DFGIJLNQRSUVWZdfgijlnqrsuvwz")
_D = {"O": "0", "О": "0", "o": "0", "о": "0", "З": "3", "з": "3", "Б": "6", "б": "6", "l": "1", "I": "1", "|": "1", "В": "8", "S": "5", "Z": "2"}


def script_of(s: str) -> str:
    """'ru' / 'lat' / 'mixed' / 'none' — по однозначным буквам, потом по большинству."""
    c = sum(ch in CYR_ONLY for ch in s); l = sum(ch in LAT_ONLY for ch in s)
    if c > l: return "ru"
    if l > c: return "lat"
    cc = sum("а" <= ch.lower() <= "я" for ch in s); ll = sum(ch.isascii() and ch.isalpha() for ch in s)
    if cc == ll == 0: return "none"
    if c == l == 0:
        return "ru" if cc > ll else ("lat" if ll > cc else "mixed")
    return "mixed"


def to_cyr(s: str) -> str:
    """Заменить латинские «двойники» на кириллицу в словах, где уже есть кириллица."""
    out = []
    cyr_n = sum(("А" <= ch <= "я") or ch in "Ёё" for ch in s); lat_n = sum(ch.isascii() and ch.isalpha() for ch in s)
    line_cyr = cyr_n > lat_n
    for w in re.split(r"(\s+)", s):
        core = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", w)
        if any("А" <= ch <= "я" or ch in "Ёё" for ch in w):
            w = "".join(_L2C.get(ch, ch) for ch in w)
        elif line_cyr and len(core) >= 2 and all(ch in _L2C for ch in core):   # «KP.» в кириллической строке -> «КР.»
            w = "".join(_L2C.get(ch, ch) for ch in w)
        out.append(w)
    return "".join(out)


def to_lat(s: str) -> str:
    out = []
    for w in re.split(r"(\s+)", s):
        if any(ch.isascii() and ch.isalpha() for ch in w):
            w = "".join(_C2L.get(ch, ch) for ch in w)
        out.append(w)
    return "".join(out)


def digits_fix(s: str) -> str:
    return "".join(_D.get(ch, ch) for ch in s)


def fold(s: str) -> str:
    """Для нечёткого сравнения: верхний регистр, всё в кириллицу-двойники, без пробелов/знаков."""
    s = "".join(_L2C.get(ch, ch) for ch in s.upper())
    return re.sub(r"[^0-9A-ZА-ЯЁ]", "", s)


def partial_ratio(phrase: str, text: str) -> float:
    p, t = fold(phrase), fold(text)
    if not p or not t:
        return 0.0
    if len(t) <= len(p):
        return SequenceMatcher(None, p, t).ratio()
    best = 0.0
    for n in (len(p) - 1, len(p), len(p) + 1):
        for i in range(0, len(t) - n + 1):
            best = max(best, SequenceMatcher(None, p, t[i:i + n]).ratio())
            if best > 0.98: return best
    return best


_ICAO = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E", "Ж": "ZH", "З": "Z", "И": "I", "Й": "I", "К": "K", "Л": "L",
         "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "KH", "Ц": "TS", "Ч": "CH",
         "Ш": "SH", "Щ": "SHCH", "Ъ": "IE", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "IU", "Я": "IA"}


def translit(s: str) -> str:
    return "".join(_ICAO.get(ch, ch) for ch in s.upper())


def translit_match(ru: str, lat: str) -> float:
    """0..1: насколько латинская строка похожа на транслитерацию кириллической."""
    a = re.sub(r"[^A-Z]", "", translit(ru)); b = re.sub(r"[^A-Z]", "", lat.upper())
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ---------- сверка кириллица ↔ латиница (транслитерация ICAO в ВУ/паспорте) ----------
def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_lat_to_ru(lat: str, ru: str, max_dist: int = 2) -> tuple[str, bool]:
    """Каждое латинское слово, близкое (<=max_dist правок) к транслитерации кириллического слова, заменяем транслитерацией.
    «KAZANSKIL» -> «KAZANSKII». Слова короче 5 букв не трогаем (риск испортить)."""
    ru_words = [w for w in re.split(r"[\s]+", ru.upper()) if w]
    tr = {w: re.sub(r"[^A-Z]", "", translit(w)) for w in ru_words}
    out, changed = [], False
    for w in re.split(r"(\s+)", lat):
        wu = re.sub(r"[^A-Z]", "", w.upper())
        if len(wu) >= 5 and not w.isspace():
            best = min(tr.values(), key=lambda t: _lev(t, wu), default=None)
            if best and best != wu and _lev(best, wu) <= max_dist and abs(len(best) - len(wu)) <= max_dist:
                m = re.match(r"^(\W*)(.*?)(\W*)$", w)
                w = m.group(1) + best + m.group(3); changed = True
        out.append(w)
    return "".join(out), changed


def split_given_names(ru: str, lat: str | None) -> str:
    """«ИВАНПЕТРОВИЧ» -> «ИВАН ПЕТРОВИЧ»: ищем разрез, при котором транслитерация
    первой части совпадает с первым латинским словом (если латиницы нет — по суффиксу отчества)."""
    if " " in ru.strip() or len(ru) < 8:
        return ru
    cand = []
    for i in range(3, len(ru) - 3):
        a, b = ru[:i], ru[i:]
        score = 0.0
        if lat and " " in lat.strip():
            la = lat.split()
            score = translit_match(a, la[0]) + translit_match(b, " ".join(la[1:]))
        if re.search(r"(ОВИЧ|ЕВИЧ|ИЧ|ОВНА|ЕВНА|ИЧНА|ИНИЧНА|ОГЛЫ|КЫЗЫ|ОГЛУ)$", b):
            score += 0.5
        if a.upper() in COMMON_NAMES:
            score += 0.8
        cand.append((score, i))
    if not cand: return ru
    score, i = max(cand)
    return f"{ru[:i]} {ru[i:]}" if score >= 1.2 else ru


COMMON_NAMES = set("""АЛЕКСАНДР АЛЕКСЕЙ АНАТОЛИЙ АНДРЕЙ АНТОН АРТЕМ АРТУР БОРИС ВАДИМ ВАЛЕНТИН ВАЛЕРИЙ ВАСИЛИЙ ВИКТОР ВИТАЛИЙ ВЛАДИМИР ВЛАДИСЛАВ ВЯЧЕСЛАВ ГЕННАДИЙ ГЕОРГИЙ ГРИГОРИЙ ДАНИИЛ ДАНИЛ ДЕНИС ДМИТРИЙ ЕВГЕНИЙ ЕГОР ИВАН ИГОРЬ ИЛЬЯ КИРИЛЛ КОНСТАНТИН ЛЕОНИД МАКСИМ МИХАИЛ НИКИТА НИКОЛАЙ ОЛЕГ ПАВЕЛ ПЕТР ПЁТР РОМАН РУСЛАН СЕРГЕЙ СТАНИСЛАВ СТЕПАН ТИМУР ФЕДОР ЮРИЙ ЯРОСЛАВ
АЛЕКСАНДРА АЛИНА АЛЛА АНАСТАСИЯ АННА ВАЛЕНТИНА ВАЛЕРИЯ ВЕРА ВИКТОРИЯ ГАЛИНА ДАРЬЯ ЕВГЕНИЯ ЕКАТЕРИНА ЕЛЕНА ЕЛИЗАВЕТА ИРИНА КСЕНИЯ ЛАРИСА ЛИДИЯ ЛЮБОВЬ ЛЮДМИЛА МАРИНА МАРИЯ НАДЕЖДА НАТАЛЬЯ НИНА ОКСАНА ОЛЬГА СВЕТЛАНА СОФИЯ ТАМАРА ТАТЬЯНА ЮЛИЯ""".split())


_CYR = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


def repair_ru_from_lat(ru: str, lat: str) -> tuple[str, bool]:
    """Кириллическое слово «потеряло» букву («ПЕТРО» при латинском «PETROV»): подбираем единственную правку
    (вставка / замена / удаление одной буквы), транслитерация которой в точности равна латинскому слову."""
    rw, lw = ru.split(), lat.split()
    if len(rw) != len(lw): return ru, False
    out, changed = [], False
    for r, l in zip(rw, lw):
        T = re.sub(r"[^A-Z]", "", l.upper()); core = re.sub(r"[^А-ЯЁ]", "", r.upper())
        if len(T) < 3 or not core or re.sub(r"[^A-Z]", "", translit(core)) == T:
            out.append(r); continue
        cands = set()
        for i in range(len(core) + 1):
            for ch in _CYR:
                c = core[:i] + ch + core[i:]
                if re.sub(r"[^A-Z]", "", translit(c)) == T: cands.add(c)
                if i < len(core):
                    c = core[:i] + ch + core[i + 1:]
                    if ch != core[i] and re.sub(r"[^A-Z]", "", translit(c)) == T: cands.add(c)
            if i < len(core):
                c = core[:i] + core[i + 1:]
                if re.sub(r"[^A-Z]", "", translit(c)) == T: cands.add(c)
        if len(cands) == 1:
            out.append(cands.pop()); changed = True
        else:
            out.append(r)
    return " ".join(out), changed
