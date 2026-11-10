"""Лексиконы допустимых значений полей (docrec/lexicons/<doctype>/<field>.txt.gz, shared/…) и «привязка» OCR-вывода к ближайшей записи.
Расстояние — взвешенный Левенштейн со стоимостями визуальных путаниц (кириллица↔латиница-двойники дёшево, похожие буквы — дешевле полной замены);
привязываем только выше порога схожести, иначе значение остаётся как есть (с пометкой). Файлы загружаются лениво и кэшируются (≈1–3 МБ RAM на всё).
Строка файла: «каноническое[|синоним|синоним]»."""
from __future__ import annotations
import gzip, re
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).parent / "lexicons"


def fold(s: str) -> str:
    return s.lower().replace("ё", "е")


# --- стоимости замен (по свёрнутым строчным символам)
_HOMO = ["aа", "bв", "cс", "eе", "hн", "kк", "mм", "oо", "pр", "tт", "xх", "yу", "uи", "3з", "0о", "6б", "4ч", "1і", "iі", "zз", "sѕ", "wш", "nп", "rг"]
_SIM = ["ос", "ое", "се", "гр", "гт", "пи", "пл", "лд", "вь", "вб", "вз", "зэ", "эе", "ий", "ин", "нм", "шщ", "цщ", "чу", "чн", "ьы", "ъь", "бь", "ад", "аб", "ео", "ев", "ер", "рз", "ов",
        "ло", "лр", "рв", "ур", "уз", "ку", "дл", "дц", "тн", "тг", "пн", "нп", "ип", "иы", "ыи", "ьъ", "ея", "яа", "яя", "жм", "ж*", "юо", "юк", "хж", "хк", "км", "кн", "кх", "ця", "цш", "шц",
        "зв", "зр", "бо", "бс", "бе", "бр", "гз", "гк", "гд", "тр", "рп", "ро", "ре", "рс", "ое", "ое", "ая", "аэ", "еэ", "ыь", "иь", "уч", "уц", "мп", "мн", "мл", "лп", "лн", "лм", "дд"]
_COST: dict[tuple[str, str], float] = {}
for p in _HOMO:
    _COST[(p[0], p[1])] = _COST[(p[1], p[0])] = 0.12
for p in _SIM:
    _COST.setdefault((p[0], p[1]), 0.5); _COST.setdefault((p[1], p[0]), 0.5)


# цифры/латинские буквы, которые путаются на матричной печати и размытых снимках (0/8, 5/S/8, 3/8, 6/8, 9/8 …)
for a_, b_, c_ in [("5", "s", 0.35), ("8", "s", 0.45), ("0", "8", 0.4), ("8", "b", 0.4), ("3", "8", 0.5), ("6", "8", 0.5), ("9", "8", 0.5), ("5", "6", 0.6),
                   ("0", "o", 0.12), ("0", "d", 0.5), ("0", "q", 0.5), ("1", "l", 0.12), ("1", "i", 0.2), ("1", "7", 0.6), ("5", "9", 0.6), ("2", "z", 0.35), ("9", "s", 0.6)]:
    _COST[(a_, b_)] = _COST[(b_, a_)] = c_


def sub_cost(a: str, b: str) -> float:
    return 0.0 if a == b else _COST.get((a, b), 1.0)


def wdist(a: str, b: str, cutoff: float = 1e9) -> float:
    """Взвешенный Левенштейн (замены по матрице визуальных путаниц; пробел вставляется/удаляется за 0.3)."""
    n, m = len(a), len(b)
    if abs(n - m) > cutoff: return cutoff + 1
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1): prev[j] = prev[j - 1] + (0.3 if b[j - 1] == " " else 1.0)
    for i in range(1, n + 1):
        ca = a[i - 1]; cur = [prev[0] + (0.3 if ca == " " else 1.0)] + [0.0] * m
        best = cur[0]
        for j in range(1, m + 1):
            cb = b[j - 1]
            d = min(prev[j - 1] + sub_cost(ca, cb), prev[j] + (0.3 if ca == " " else 1.0), cur[j - 1] + (0.3 if cb == " " else 1.0))
            cur[j] = d
            if d < best: best = d
        if best > cutoff: return cutoff + 1
        prev = cur
    return prev[m]


class Lexicon:
    def __init__(self, name: str):
        self.name = name
        path = DIR / f"{name}.txt.gz"
        self.canon: list[str] = []; self.keys: dict[str, str] = {}
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    self.canon.append(parts[0])
                    for p in parts: self.keys.setdefault(fold(p), parts[0])
        self._by_len: dict[int, list[str]] | None = None

    def __len__(self): return len(self.canon)

    def __contains__(self, raw: str) -> bool:
        return fold(raw.strip()) in self.keys

    def _buckets(self):
        if self._by_len is None:
            b: dict[int, list[str]] = {}
            for k in self.keys: b.setdefault(len(k.replace(" ", "")), []).append(k)
            self._by_len = b
        return self._by_len


@lru_cache(maxsize=64)
def get(name: str) -> Lexicon:
    return Lexicon(name)


def snap(raw: str, name: str, thr: float = 0.25, margin: float = 0.03, max_alts: int = 3):
    """Ближайшая запись лексикона. -> dict(value, raw, dist, norm, status, alts).
    status: exact | snapped (norm ≤ thr и отрыв от второго ≥ margin) | ambiguous | kept (нет близкой записи). value — каноническая запись или raw."""
    lex = get(name); r = (raw or "").strip()
    out = dict(value=r, raw=r, dist=None, norm=None, status="kept", alts=[])
    if not r or not len(lex): return out
    fr = fold(r)
    if fr in lex.keys:
        out.update(value=lex.keys[fr], dist=0.0, norm=0.0, status="exact"); return out
    nr = len(fr.replace(" ", ""))
    cands: list[tuple[float, str, str]] = []
    lim = thr * max(nr, 1) * 1.6 + 0.5
    for L, keys in lex._buckets().items():
        if abs(L - nr) > max(2, int(0.3 * nr)): continue
        for k in keys:
            if nr == L and nr >= 4 and sum(1 for x, y in zip(fr.replace(" ", ""), k.replace(" ", "")) if x == y or sub_cost(x, y) < 1) < 0.45 * nr: continue
            d = wdist(fr, k, lim)
            if d <= lim: cands.append((d, k, lex.keys[k]))
    if not cands: return out
    cands.sort(key=lambda x: x[0])
    best = cands[0]; norm = best[0] / max(nr, len(best[1].replace(" ", "")), 1)
    seen = []; alts = []
    for d, k, c in cands:
        if c in seen: continue
        seen.append(c); alts.append((c, round(d / max(nr, len(k.replace(" ", "")), 1), 3)))
        if len(alts) > max_alts: break
    out.update(dist=round(best[0], 2), norm=round(norm, 3), alts=alts[:max_alts])
    second = alts[1][1] if len(alts) > 1 else None
    if norm <= thr and (second is None or second - norm >= margin):
        out.update(value=best[2], status="snapped")
    elif norm <= thr:
        out.update(status="ambiguous")
    return out


def note_for(s: dict, label: str = "") -> str | None:
    """Строка для поля note: исходное значение, привязка, расстояние."""
    if s["status"] == "exact" or s["raw"] == "": return None
    if s["status"] == "snapped": return f"лексикон{(' ' + label) if label else ''}: «{s['raw']}» → «{s['value']}» (расстояние {s['norm']})"
    if s["status"] == "ambiguous": return f"лексикон{(' ' + label) if label else ''}: неоднозначно, варианты " + ", ".join(f"«{a}» ({n})" for a, n in s["alts"])
    return None


def snap_words(text: str, name: str, thr: float = 0.2, min_len: int = 4):
    """Пословная привязка (реквизиты, места): слово известно -> как есть; иначе ближайшее слово лексикона при норме ≤ thr.
    -> (новый текст, [(raw, value, norm)])."""
    changes = []; out = []
    for tok in re.split(r"(\s+)", text):
        if not tok.strip() or len(re.sub(r"\W", "", tok)) < min_len: out.append(tok); continue
        core = re.sub(r"^[\W_]+|[\W_]+$", "", tok); pre = tok[:tok.index(core)] if core in tok else ""; post = tok[len(pre) + len(core):]
        s = snap(core, name, thr=thr)
        if s["status"] == "snapped":
            val = s["value"]
            if core.isupper(): val = val.upper()
            elif core[:1].isupper(): val = val[:1].upper() + val[1:]
            changes.append((core, val, s["norm"])); out.append(pre + val + post)
        else: out.append(tok)
    return "".join(out), changes


def snap_multi(text: str, rules, min_len: int = 4):
    """Пословная привязка по нескольким лексиконам. rules = [(имя, порог, min_len)…]; слово, известное любому из лексиконов, не трогаем.
    -> (новый текст, [(raw, value, norm)…])."""
    changes = []; out = []
    for tok in re.split(r"([\s.,]+)", text):
        core = re.sub(r"^[\W_]+|[\W_]+$", "", tok)
        if not core or len(re.sub(r"\W", "", core)) < min_len or any(ch.isdigit() for ch in core): out.append(tok); continue
        if any(core in get(n) for n, _, _ in rules): out.append(tok); continue
        pre = tok[:tok.index(core)]; post = tok[len(pre) + len(core):]; done = False
        for n, thr, ml in rules:
            if len(core) < ml: continue
            sn = snap(core, n, thr=thr)
            if sn["status"] == "snapped":
                val = sn["value"]
                if core.isupper(): val = val.upper()
                elif core[:1].isupper(): val = val[:1].upper() + val[1:]
                changes.append((core, val, sn["norm"])); out.append(pre + val + post); done = True; break
        if not done: out.append(tok)
    return "".join(out), changes


def total_size() -> int:
    return sum(p.stat().st_size for p in DIR.rglob("*.gz"))
