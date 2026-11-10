"""МРЗ внутреннего паспорта РФ (2 строки по 44 знака): транслитерация по российской таблице, контрольные цифры (веса 7-3-1),
разбор и «ремонт» цифровых полей по контрольным суммам.
Строка 2: [0:9] серия(3)+номер(6), [9] кц, [10:13] RUS, [13:19] дата рожд. ГГММДД, [19] кц, [20] пол, [21:28] '<', [28] 4-я цифра серии,
[29:35] дата выдачи ГГММДД, [35:41] код подразделения, [41] '<', [42] кц доп. данных, [43] итоговая кц."""
from __future__ import annotations
import re
from datetime import date

CYR2LAT = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2", "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M", "Н": "N",
           "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9",
           "Э": "6", "Ю": "7", "Я": "8"}
LAT2CYR = {v: k for k, v in CYR2LAT.items()}
_CYR_LOOK = {"А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X"}
D2L = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"}      # цифра, прочитанная вместо буквы (в числовых полях наоборот)
L2D = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}


def translit_ru(s: str) -> str:
    return "".join(CYR2LAT.get(c, c if c in " -" else "") if c != " " else "<" for c in s.upper())


def decode_name(lat: str) -> str:
    """Латинский МРЗ-фрагмент -> кириллица (таблица взаимно однозначна)."""
    return "".join(LAT2CYR.get(c, "?") for c in lat if c != "<")


def _val(c: str) -> int:
    if c.isdigit(): return int(c)
    if "A" <= c <= "Z": return ord(c) - 55
    return 0   # '<'


def check_digit(s: str) -> int:
    return sum(_val(c) * (7, 3, 1)[i % 3] for i, c in enumerate(s)) % 10


def _norm_line(t: str) -> str:
    t = t.upper().replace(" ", "").replace("«", "<").replace("(", "<").replace("{", "<").replace("[", "<").replace("К<", "K<")
    t = "".join(_CYR_LOOK.get(c, c) for c in t)
    t = re.sub(r"[^A-Z0-9<]", "<", t)
    return t


def _fix_digits(s: str) -> str:
    return "".join(L2D.get(c, c) if not c.isdigit() and c != "<" else c for c in s)


def _fit(t: str, n: int = 44) -> str:
    return (t + "<" * n)[:n]


def parse_line2(t: str) -> dict:
    """Разбор строки 2 с восстановлением по контрольным цифрам. -> dict с полями и флагами check_*"""
    raw = _norm_line(t)
    m = re.search(r"RUS", raw[6:16] or "")
    if not m and "RUS" not in raw: return {"ok": False}
    i = raw.find("RUS", 6) if raw.find("RUS", 6) >= 0 else raw.find("RUS")
    # выравнивание: 'RUS' должен стоять на позиции 10
    if i > 10: raw = raw[i - 10:]
    elif i < 10: raw = raw[:i] + "<" * (10 - i) + raw[i:] if False else raw
    s = _fit(raw)
    out = {"line": s, "ok": True}
    doc, ck = _fix_digits(s[0:9]), s[9]
    ck = L2D.get(ck, ck)
    out["doc_no"] = doc; out["check_doc"] = ck.isdigit() and check_digit(doc) == int(ck)
    birth, bk = _fix_digits(s[13:19]), L2D.get(s[19], s[19])
    out["birth"] = birth; out["check_birth"] = bk.isdigit() and check_digit(birth) == int(bk)
    out["sex"] = {"M": "M", "F": "F", "Ж": "F", "H": "M"}.get(s[20], None)
    # опциональная часть — правый якорь (последние 16 знаков: 13 данных + '<' + 2 кц)
    tail = s[28:44]
    opt = _fix_digits(tail[0:7] + tail[7:13])
    out["series4"] = opt[0] if opt[0].isdigit() else None
    out["issue"] = opt[1:7]; out["dept"] = opt[7:13]
    out["check_opt"] = tail[14].isdigit() and check_digit(s[28:42]) == int(tail[14])
    comp = s[0:10] + s[13:20] + s[21:43]
    out["check_all"] = tail[15].isdigit() and check_digit(comp) == int(tail[15])
    out["check_ok"] = out["check_doc"] and out["check_birth"]
    return out


def _ymd(s: str, kind: str):
    if not re.fullmatch(r"\d{6}", s or ""): return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    cur = date.today().year % 100
    if kind == "birth": y = 1900 + yy if yy > cur else 2000 + yy
    else: y = 2000 + yy if yy <= cur + 1 else 1900 + yy
    try: d = date(y, mm, dd)
    except ValueError: return None
    return d


def parse(lines: list[str]) -> dict | None:
    """lines — кандидаты (строки МРЗ, возможно разбитые/склеенные). -> поля с флагами проверки или None"""
    l1 = l2 = None
    for t in lines:
        n = _norm_line(t)
        if "RUS" in n[5:16] and sum(c.isdigit() for c in n) >= 18:
            p = parse_line2(n)
            if p.get("ok") and (l2 is None or (p["check_doc"] + p["check_birth"] + p["check_opt"] + p["check_all"]) > (l2["check_doc"] + l2["check_birth"] + l2["check_opt"] + l2["check_all"])):
                l2 = p
        elif re.match(r"^P.{0,2}RUS", n[:8]) or ("<<" in n and n[:2] in ("PN", "P<", "PM", "PH", "PV")):
            l1 = n
    res = {"line1_found": l1 is not None, "line2_found": l2 is not None}
    if l1:
        m = re.search(r"RUS", l1)
        body = l1[m.end():] if m else l1[5:]
        parts = re.split(r"<{2,}", body.strip("<"), maxsplit=1)
        sur = parts[0].replace("<", "")
        rest = parts[1] if len(parts) > 1 else ""
        names = [x for x in re.split(r"<", rest) if x]
        res.update({"surname_lat": sur, "surname": decode_name(sur) if sur else None,
                    "name": decode_name(names[0]) if names else None, "patronymic": decode_name("".join(names[1:2])) if len(names) > 1 else None,
                    "name_lat": names[0] if names else None, "patronymic_lat": names[1] if len(names) > 1 else None, "line1": l1})
    if l2:
        res["line2"] = l2["line"]
        res["checks"] = {k: l2[k] for k in ("check_doc", "check_birth", "check_opt", "check_all")}
        res["sex"] = {"M": "МУЖ.", "F": "ЖЕН."}.get(l2["sex"])
        bd = _ymd(l2["birth"], "birth"); res["birth_date"] = bd.strftime("%d.%m.%Y") if bd else None
        idt = _ymd(l2["issue"], "issue"); res["issue_date"] = idt.strftime("%d.%m.%Y") if idt else None
        dep = l2["dept"]; res["dept_code"] = f"{dep[:3]}-{dep[3:]}" if re.fullmatch(r"\d{6}", dep) else None
        d = l2["doc_no"]
        if re.fullmatch(r"\d{9}", d):
            s4 = l2["series4"] or ""
            res["series3"] = d[:3]; res["number"] = d[3:]
            res["series"] = (d[:2] + " " + d[2] + s4) if s4 else None
    return res
