"""Извлечение и валидация реквизитов из произвольного русского текста.
Возвращает список реквизитов: {type, value, valid, confidence, party, note?}."""
from __future__ import annotations
import re
from datetime import date

# ------------------------------------------------------------------ контрольные суммы
def inn_valid(s: str) -> bool:
    if not re.fullmatch(r"\d{10}|\d{12}", s): return False
    d = [int(c) for c in s]
    if len(d) == 10:
        return d[9] == sum(a * b for a, b in zip(d, (2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
    c1 = sum(a * b for a, b in zip(d, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
    c2 = sum(a * b for a, b in zip(d, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
    return d[10] == c1 and d[11] == c2


def kpp_valid(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}[\dA-Z]{2}\d{3}", s))


def bik_valid(s: str) -> bool:
    return bool(re.fullmatch(r"04\d{7}", s))   # БИК РФ: 04 + код территории + ...


def _acc_sum(prefix3: str, acc: str) -> bool:
    if not re.fullmatch(r"\d{20}", acc) or not re.fullmatch(r"\d{3}", prefix3): return False
    s = prefix3 + acc
    w = (7, 1, 3) * 8
    return sum(int(c) * w[i] % 10 for i, c in enumerate(s)) % 10 == 0


def rs_valid(acc: str, bik: str | None) -> bool | None:
    """Р/с: ключ по последним 3 цифрам БИК. None — нельзя проверить (нет БИК)."""
    if not re.fullmatch(r"\d{20}", acc): return False
    if not bik or not re.fullmatch(r"\d{9}", bik): return None
    return _acc_sum(bik[-3:], acc)


def ks_valid(acc: str, bik: str | None) -> bool | None:
    """К/с: «0» + 5-6 цифры БИК."""
    if not re.fullmatch(r"\d{20}", acc): return False
    if not bik or not re.fullmatch(r"\d{9}", bik): return None
    return _acc_sum("0" + bik[4:6], acc)


def snils_valid(s: str) -> bool:
    d = re.sub(r"\D", "", s)
    if len(d) != 11: return False
    n = int(d[:9]); ctrl = int(d[9:])
    if n <= 1001998: return True
    t = sum(int(c) * (9 - i) for i, c in enumerate(d[:9]))
    if t < 100: c = t
    elif t in (100, 101): c = 0
    else:
        c = t % 101
        if c == 100: c = 0
    return c == ctrl


def ogrn_valid(s: str) -> bool:
    if re.fullmatch(r"\d{13}", s): return int(s[:12]) % 11 % 10 == int(s[12])
    if re.fullmatch(r"\d{15}", s): return int(s[:14]) % 13 % 10 == int(s[14])
    return False


# ------------------------------------------------------------------ перебор одиночных OCR-ошибок цифр
_CONF = {"0": "86", "1": "74", "2": "7", "3": "8", "4": "9", "5": "68", "6": "580", "7": "12", "8": "03659", "9": "4"}


def autocorrect(value: str, check) -> tuple[str, bool]:
    """Если контрольная сумма не сходится — ищем единственную правку одной цифры среди типичных путаниц OCR."""
    if check(value): return value, False
    cands = []
    for i, ch in enumerate(value):
        for r in _CONF.get(ch, ""):
            v = value[:i] + r + value[i + 1:]
            if check(v): cands.append(v)
    return (cands[0], True) if len(cands) == 1 else (value, False)


# ------------------------------------------------------------------ регулярки
_D = r"[\d\sOоОlI|]"      # цифра с возможным OCR-двойником и пробелом
_SEP = r"[\s:№N#.,;\-–—/]*"


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s.translate(str.maketrans("OоОlI|", "000111")))


def _num_after(label_re: str, text: str, nd: tuple[int, ...], span: int = 40):
    out = []
    for m in re.finditer(label_re, text, re.I):
        seg = text[m.end():m.end() + span]
        mm = re.match(rf"{_SEP}((?:{_D}){{{min(nd)},{max(nd) + 8}}})", seg)
        if not mm: continue
        raw = mm.group(1)
        d = _digits(raw)
        # берём ровно нужное число цифр: сначала пытаемся «как есть» (по границе пробела), иначе по длине
        best = None
        for n in sorted(nd, reverse=True):
            if len(d) >= n:
                best = d[:n]; break
        if best is None: continue
        out.append((best, m.start()))
    return out


def _date_ok(s: str):
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if not m: return None
    try: return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError: return None


def extract(text: str, conf: float = 0.9, party: str | None = None, context_bik: str | None = None, log: list | None = None) -> list[dict]:
    """text — склеенный текст одной стороны/документа; conf — средняя уверенность OCR по этому тексту."""
    R: list[dict] = []

    def add(typ, value, valid, note=None, c=None):
        item = {"type": typ, "value": value, "valid": valid, "confidence": round(float(c if c is not None else conf) * (1.0 if valid is not False else 0.7), 3), "party": party}
        if note: item["note"] = note
        if not any(r["type"] == typ and r["value"] == value and r["party"] == party for r in R): R.append(item)

    # ИНН
    for v, _ in _num_after(r"\bИНН", text, (10, 12)):
        v2, ch = autocorrect(v, inn_valid)
        # ИНН 12 знаков, приклеенный к КПП: «ИНН 1234567890, КПП …» — берём 10, если 12 не сходится, а 10 сходится
        if not inn_valid(v2) and len(v) == 12 and inn_valid(v[:10]): v2, ch = v[:10], False
        add("inn", v2, inn_valid(v2), "исправлено по контрольной сумме" if ch else None)
    # КПП
    for v, _ in _num_after(r"\bКПП", text, (9,)):
        add("kpp", v, kpp_valid(v))
    # БИК
    biks = []
    for v, _ in _num_after(r"\bБИК", text, (9,)):
        v2, ch = autocorrect(v, bik_valid)
        add("bik", v2, bik_valid(v2), "исправлено по маске" if ch else None); biks.append(v2)
    bik = biks[0] if biks else context_bik
    # ОГРН
    for v, _ in _num_after(r"\bОГРН(?:ИП)?", text, (15, 13)):
        add("ogrn", v, ogrn_valid(v))
    # СНИЛС
    for m in re.finditer(r"СНИЛС" + _SEP + r"((?:\d{3}[\s-]?){3}\d{2})", text, re.I):
        d = _digits(m.group(1))
        v2, ch = autocorrect(d, snils_valid)
        add("snils", f"{v2[:3]}-{v2[3:6]}-{v2[6:9]} {v2[9:]}", snils_valid(v2), "исправлено по контрольной сумме" if ch else None)
    # р/с и к/с: определяем по метке, 20 цифр
    for lab, typ in ((r"(?:корр(?:еспондентский)?\.?\s*(?:счет|счёт|с\b)\.?|к/с)", "corr_account"),
                     (r"(?:р/с|р\\с|расч[её]тный\s*(?:счет|счёт)|расч\.?\s*счет|номер\s*сч[её]та|сч[её]т(?:\s*получателя)?|л/с)", "settlement_account")):
        for v, pos in _num_after(lab, text, (20,), span=44):
            if typ == "settlement_account" and any(r["type"] == "corr_account" and r["value"] == v for r in R): continue
            chk = (lambda x: bool(ks_valid(x, bik))) if typ == "corr_account" else (lambda x: bool(rs_valid(x, bik)))
            v2, ch = (autocorrect(v, chk) if bik else (v, False))
            res = (ks_valid(v2, bik) if typ == "corr_account" else rs_valid(v2, bik))
            note = "исправлено по ключу БИК" if ch else ("нет БИК для проверки ключа" if res is None else None)
            add(typ, v2, True if res is None and len(v2) == 20 else bool(res), note, c=conf * (0.85 if res is None else 1))
    # 20-значные без метки, не совпавшие ни с чем
    # паспорт РФ
    for m in re.finditer(r"паспорт\w*[^\d]{0,60}?(\d{2}\s?\d{2})\s*(?:№|N|номер|ном\.?)?\s*(\d{6})\b", text, re.I | re.S):
        ser, num = re.sub(r"\s", "", m.group(1)), m.group(2)
        add("passport", f"{ser[:2]} {ser[2:]} {num}", True)
    for m in re.finditer(r"серия\s*(\d{2}\s?\d{2})\s*(?:№|N|номер)?\s*(\d{6})", text, re.I):
        ser = re.sub(r"\s", "", m.group(1)); add("passport", f"{ser[:2]} {ser[2:]} {m.group(2)}", True)
    # даты
    for m in re.finditer(r"\b(\d{2}\.\d{2}\.\d{4})\b", text):
        add("date", m.group(1), _date_ok(m.group(1)) is not None)
    # телефоны
    for m in re.finditer(r"(?:тел(?:ефон)?\.?|т\.|факс|моб\.?)?\s*[:/]?\s*(?<![\d])((?:\+7|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2})(?!\d)", text, re.I):
        d = re.sub(r"\D", "", m.group(1))
        if len(d) == 11: add("phone", "+7" + d[1:], True)
    for m in re.finditer(r"[Тт]ел\.?\s*[:/]?\s*[/(]?\s*(\d{3,5})\s*[/)]\s*([\d\-\s]{5,10})", text):
        d = re.sub(r"\D", "", m.group(1) + m.group(2))
        add("phone", f"({m.group(1)}) {m.group(2).strip()}", len(d) in (10, 11), "городской номер с кодом, формат нестандартный" if len(d) not in (10, 11) else None)
    # e-mail
    for m in re.finditer(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text):
        add("email", m.group(0).lower(), True)
    # организации (ООО «Ромашка»)
    for m in re.finditer(r"\b(ООО|ОАО|ЗАО|ПАО|АО|НКО|ИП)\s*[«\"“]\s*([^»\"”]{2,60}?)\s*[»\"”]", text):
        add("organization", f"{m.group(1)} «{m.group(2)}»", True)
    # ФИО
    for m in re.finditer(r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:вич|вна|ична|ич|евна|овна|оглы|кызы))\b", text):
        add("person", " ".join(m.groups()), True)
    for m in re.finditer(r"\b([А-ЯЁ]{3,}(?:-[А-ЯЁ]+)?)\s+([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,}(?:ВИЧ|ВНА|ИЧНА|ИЧ|ЕВНА|ОВНА|ОГЛЫ|КЫЗЫ))\b", text):
        add("person", " ".join(w.capitalize() for w in m.groups()), True)
    for m in re.finditer(r"\b([А-ЯЁ]\.\s?[А-ЯЁ]\.)\s*([А-ЯЁ][а-яё]{2,})\b|\b([А-ЯЁ][а-яё]{2,})\s+([А-ЯЁ]\.\s?[А-ЯЁ]\.)", text):
        add("person_short", re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g)), True)
    return R


# ------------------------------------------------------------------ сводка проверок по сторонам
def cross_check(requisites: list[dict]) -> list[str]:
    notes = []
    for r in requisites:
        if r["valid"] is False:
            notes.append(f"{r['party'] or 'doc'}: {r['type']} {r['value']} — не проходит контрольную сумму/формат")
    return notes
