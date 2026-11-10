"""Марка/модель ТС в СТС: две плотно стоящие строки (кириллица и латиница) читаются отдельно и сверяются.
Шаги: разрезать блок значений по профилю проекции -> перечитать каждую строку в нескольких вариантах кропа (Paddle + Tesseract, разный масштаб/слой)
-> разобрать «МАРКА 1234-S5» -> общее голосование: марка по парам КАМАЗ<->KAMAZ, цифры модели, суффикс (путаницы S/5, O/0, B/8, Z/2).
Модель в обеих строках одинакова, поэтому кандидаты обеих строк голосуют вместе; согласие строк повышает уверенность."""
from __future__ import annotations
import re
from collections import defaultdict
import cv2
import numpy as np

# марки, у которых на бланке есть кириллическая и латинская строки (пары «как печатают в ГИБДД»)
PAIRS = [("КАМАЗ", "KAMAZ"), ("ГАЗ", "GAZ"), ("ЗИЛ", "ZIL"), ("УАЗ", "UAZ"), ("ВАЗ", "VAZ"), ("ЛАДА", "LADA"), ("МАЗ", "MAZ"), ("УРАЛ", "URAL"),
         ("ПАЗ", "PAZ"), ("ЛИАЗ", "LIAZ"), ("НЕФАЗ", "NEFAZ"), ("ГАЗЕЛЬ", "GAZEL"), ("ЛУАЗ", "LUAZ"), ("ЗАЗ", "ZAZ"), ("ИЖ", "IZH"), ("КРАЗ", "KRAZ"),
         ("МОСКВИЧ", "MOSKVICH"), ("ТАТРА", "TATRA"), ("ЛАЗ", "LAZ"), ("ГАЗЕЛЬ NEXT", "GAZELLE NEXT")]
TR = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯЬЪ", ["A", "B", "V", "G", "D", "E", "E", "ZH", "Z", "I", "Y", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "F", "KH", "TS", "CH", "SH", "SCH", "Y", "E", "YU", "YA", "", ""]))
LAT2CYR = {"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У"}
CYR2LAT = {v: k for k, v in LAT2CYR.items()}
DIG2CYR = {"3": "З", "0": "О", "8": "В", "6": "Б", "4": "Ч", "1": "І"}
DIG2LAT = {"7": "Z", "2": "Z", "0": "O", "8": "B", "5": "S", "1": "I"}
NUMFIX = {"O": "0", "О": "0", "o": "0", "S": "5", "B": "8", "В": "8", "Z": "2", "I": "1", "l": "1", "|": "1"}
AMB = [("S", "5"), ("O", "0"), ("B", "8"), ("Z", "2"), ("I", "1")]
ALNUM = r"A-Za-zА-Яа-я0-9"
PARSE = re.compile(rf"([{ALNUM}]{{3,9}})\.?\s+([0-9OОoSBВZIl]{{3,5}})\s*[-–—]\s*([{ALNUM}]{{1,2}})")
PARSE_GLUED = re.compile(rf"([A-Za-zА-Яа-я]{{2,8}})\.?\s*([0-9]{{3,5}})\s*[-–—]\s*([{ALNUM}]{{1,2}})")
PARSE_NOSUF = re.compile(rf"([{ALNUM}]{{3,9}})\.?\s+([0-9OОoSBВZIl]{{3,5}})(?![{ALNUM}])")


def lk(s: str) -> str:
    return "".join(TR.get(c, c) for c in s.upper())


def _lev(a: str, b: str) -> int:
    from .textutils import _lev as f
    return f(a, b)


def _norm_make_lang(w: str, lang: str) -> str:
    w = w.upper()
    if lang == "ru":
        return "".join(LAT2CYR.get(c, DIG2CYR.get(c, c)) for c in w)
    return "".join(CYR2LAT.get(c, DIG2LAT.get(c, c)) for c in w)


def _suffix_canon(c: str) -> str:
    c = c.upper()
    return CYR2LAT.get(c, c)


def parse_candidate(text: str, confs: list[float], lang: str):
    """«KAMA3 5490-S5» -> dict(make, num, suf, sconf=[...], mconf) или None."""
    t = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    m = PARSE.search(t) or PARSE_GLUED.search(t)
    if not m:
        m2 = PARSE_NOSUF.search(t) or re.search(rf"([A-Za-zА-Яа-я]{{2,8}})\s*([0-9]{{3,5}})(?![{ALNUM}])", t)
        if not m2: return None
        suf, sc = "", []
        make_raw, num_raw = m2.group(1), m2.group(2)
        sp = (m2.start(), m2.end())
    else:
        make_raw, num_raw, suf = m.group(1), m.group(2), m.group(3)
        sp = (m.start(), m.end())
        # уверенность символов суффикса: последние len(suf) символов совпадения
        cs = confs if len(confs) == len(text) else None
        sc = []
        if cs is not None and t == text.strip():
            off = len(text) - len(text.lstrip())
            sc = [cs[off + sp[1] - len(suf) + i] for i in range(len(suf))]
        if not sc: sc = [0.6] * len(suf)
    num = "".join(NUMFIX.get(c, c) for c in num_raw)
    if not num.isdigit(): return None
    return dict(make=_norm_make_lang(make_raw, lang), num=num, suf=suf.upper(), sconf=sc)


def snap_make(make: str):
    """-> (индекс пары, расстояние) по латинизированной форме; None если марка неизвестна."""
    key = lk(make)
    best = None
    for i, (ru, la) in enumerate(PAIRS):
        d = min(_lev(key, la), _lev(key, lk(ru)))
        tol = 1 if len(la) >= 3 else 0
        if d <= tol and (best is None or d < best[1]): best = (i, d)
    return best


# ---------------------------------------------------------------- геометрия
def find_value_block(V: np.ndarray, y_lab: float, lh: float = 0.0235):
    """Блок значений марки (одна или две строки) в слое значений -> (x0, x1, top, bottom, [границы строк]) в пикселях."""
    H, W = V.shape[:2]
    x0, x1 = int(0.28 * W), int(0.99 * W)
    ya, yb = int(max(0, (y_lab - 0.017) * H)), int(min(H, (y_lab + 0.066) * H))
    reg = V[ya:yb, x0:x1]
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = np.clip(np.median(g) - g, 0, None)
    prof = np.convolve(ink.mean(1), np.ones(3) / 3, "same")
    if prof.max() < 2: return None
    rows = prof > 0.33 * prof.max()
    runs, s = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and s is None: s = i
        if not v and s is not None: runs.append([s, i - 1]); s = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 3: merged[-1][1] = r[1]
        else: merged.append(r)
    if not merged: return None
    # полосы чернил: значимые, не упирающиеся в верх окна (это хвост строки выше — VIN)
    mx = max(prof[q[0]:q[1] + 1].sum() for q in merged)
    runs_ = [(ya + q[0] - 2, ya + q[1] + 1) for q in merged if q[0] > 1 and prof[q[0]:q[1] + 1].sum() >= 0.25 * mx]
    if not runs_: return None
    return x0, x1, runs_


def _prep(c: np.ndarray, h: int) -> np.ndarray:
    f = h / c.shape[0]
    c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(c, 12, 12, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _trim_cols(c: np.ndarray, full_w: int | None = None) -> np.ndarray:
    """Обрезка по колонкам чернил: берём самую «тяжёлую» группу колонок (разрывы до ~3% ширины карточки — это пробелы между словами)."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = np.clip(np.median(g) - g, 0, None).mean(0)
    ink = np.convolve(ink, np.ones(5) / 5, "same")
    on = ink > 0.3 * ink.max()
    gap = max(8, int(0.032 * (full_w or c.shape[1] / 0.55)))
    runs, st = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and st is None: st = i
        if not v and st is not None: runs.append([st, i - 1]); st = None
    if not runs: return c
    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] <= gap: merged[-1][1] = r[1]
        else: merged.append(r)
    r = max(merged, key=lambda r: ink[r[0]:r[1] + 1].sum())
    if r[1] - r[0] < 8: return c
    return c[:, max(0, r[0] - 8):min(c.shape[1], r[1] + 9)]


# ---------------------------------------------------------------- чтение и голосование
def read_make_model(card: np.ndarray, V: np.ndarray, y_lab: float, paddle, cell=None):
    """-> dict(ru, lat, combined, conf, valid, notes, agree) или None. y_lab — нормированный y подписи «Марка, модель»."""
    if paddle is None or not hasattr(paddle, "recognize_chars"): return None
    H, W = V.shape[:2]
    blk = find_value_block(V, y_lab)
    if blk is None: return None
    x0, x1, runs_ = blk
    top, bot = runs_[0]
    h = bot - top + 1

    def rd(a, b, hh=64, src=V):
        crop = _trim_cols(src[max(0, a):min(H, b + 1), x0:x1], W)
        if crop.shape[0] < 6: return None
        t, sc, cf = paddle.recognize_chars(_prep(crop, hh))
        return t, sc, cf

    # строки значения: первая полоса — одна строка или две вплотную (разрез подбирается по тому, при каком обе половины читаются лучше);
    # если первая полоса — одна строка, вторая полоса ниже (зазор небольшой), читаемая как «МАРКА 1234-XX», — это вторая строка
    one = rd(top, bot)
    p1 = parse_candidate(one[0], one[2], "lat") if one else None
    one_ok = bool(p1 and one[1] >= 0.6)
    spans = [(top, bot)]
    if h >= 0.014 * H:
        best = None; step = max(1, int(round(h / 14)))
        for c in range(top + int(0.3 * h), top + int(0.7 * h) + 1, step):
            tot = 0.0; both = True
            for (a, b) in ((top, c + 1), (c - 1, bot)):
                r_ = rd(a, b)
                if r_ is None or not parse_candidate(r_[0], r_[2], "lat"): both = False; break
                tot += r_[1]
            if both and (best is None or tot > best[0]): best = (tot, c)
        if best and (h >= 0.0235 * H * 1.45 or not one_ok or best[0] / 2 >= one[1] - 0.05):
            spans = [(top, best[1] + 1), (best[1] - 1, bot)]
    if len(spans) == 1 and len(runs_) > 1:
        a2, b2 = runs_[1]
        if a2 - bot <= 0.02 * H:
            r2 = rd(a2, b2)
            if one_ok and r2 and parse_candidate(r2[0], r2[2], "lat") and r2[1] >= 0.6:
                spans = [(top, bot), (a2, b2)]
    nlines = len(spans)
    cuts = [spans[0][0], spans[0][1], spans[-1][1]]
    cands = []   # (lang_guess, parsed, weight, engine)
    srcs = (("V", V), ("card", card))
    for li, (a, b) in enumerate(spans):
        offs = [(-2, 0), (0, 1), (-1, 2)] if nlines == 2 else [(-1, 1), (0, 0)]
        for da, db in offs:
            ya = max(0, a + da); yb = min(H, b + db)
            for sn, img in srcs:
                crop = _trim_cols(img[ya:yb, x0:x1], W)
                if crop.shape[0] < 6: continue
                for hh in (64, 96):
                    t, sc, cf = paddle.recognize_chars(_prep(crop, hh))
                    cands.append((li, t, cf, sc, "paddle"))
                if cell is not None and sn == "card":
                    try:
                        t, sc = cell(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", 7, 3)
                        if t: cands.append((li, t, [], sc, "tesseract"))
                    except Exception:
                        pass
    parsed = []
    for li, t, cf, sc, eng in cands:
        for lang in ("ru", "lat"):
            p = parse_candidate(t, cf, lang)
            if p is None: continue
            # какой язык у строки: по символам марки (кириллица/латиница; Z/З, K/К не различаются — решает порядок строк и пара)
            letters = re.sub(r"[^A-Za-zА-Яа-я]", "", t.split()[0] if t.split() else "")
            cyr = sum(ch in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯЬЪабвгдеёжзийклмнопрстуфхцчшщыэюяьъ" for ch in letters)
            lat = sum(ch.isascii() and ch.isalpha() for ch in letters)
            parsed.append(dict(line=li, lang_guess=("ru" if cyr > lat else "lat" if lat > cyr else None), t=t, p=p, sc=sc, eng=eng, cf=cf))
            break  # одного разбора достаточно (make нормируется ниже по языку строки)
    if not parsed: return None
    # язык строки: при двух строках верхняя — кириллица, нижняя — латиница (порядок на бланке); при одной — по символам
    langs = {}
    if nlines == 2: langs = {0: "ru", 1: "lat"}
    else:
        votes = defaultdict(int)
        for d in parsed:
            if d["lang_guess"]: votes[d["lang_guess"]] += 1
        langs = {0: max(votes, key=votes.get) if votes else "lat"}
    items = []
    for d in parsed:
        lang = langs.get(d["line"], "lat")
        p = parse_candidate(d["t"], d["cf"], lang)
        if p is None: continue
        w = d["sc"] * (0.7 if d["eng"] == "tesseract" else 1.0)
        if d["eng"] == "tesseract": p["sconf"] = [min(0.6, d["sc"])] * len(p["suf"])
        items.append(dict(lang=lang, p=p, w=max(w, 0.05), line=d["line"]))
    if not items: return None
    notes = []
    # --- марка: голосование по парам
    pv = defaultdict(float); unknown = defaultdict(float)
    for it in items:
        s = snap_make(it["p"]["make"])
        if s: pv[s[0]] += it["w"] * (1.0 - 0.3 * s[1])
        else: unknown[lk(it["p"]["make"])] += it["w"]
    if pv and max(pv.values()) >= 0.5 * (sum(pv.values()) + sum(unknown.values())):
        pi = max(pv, key=pv.get); ru_mk, lat_mk = PAIRS[pi]; known = True
        support = pv[pi] / (sum(pv.values()) + sum(unknown.values()))
    else:
        known = False; support = 0.5
        best = {}
        for it in items:
            k = (it["lang"], it["p"]["make"])
            best[k] = best.get(k, 0) + it["w"]
        ru_mk = max(((m, w) for (l, m), w in best.items() if l == "ru"), key=lambda x: x[1], default=(None, 0))[0]
        lat_mk = max(((m, w) for (l, m), w in best.items() if l == "lat"), key=lambda x: x[1], default=(None, 0))[0]
        if ru_mk and lat_mk and lk(ru_mk) != lat_mk: notes.append("марка в строках не совпала после транслитерации")
        notes.append("марка не из словаря пар")
    # --- число модели: голосование по строке цифр
    nv = defaultdict(float)
    for it in items: nv[it["p"]["num"]] += it["w"]
    num = max(nv, key=nv.get); num_support = nv[num] / sum(nv.values())
    # --- суффикс: длина, затем посимвольно с учётом путаниц
    sl = defaultdict(float)
    for it in items: sl[len(it["p"]["suf"])] += it["w"]
    L = max(sl, key=sl.get)
    suf = ""; suf_notes = []; suf_support = 1.0
    ref = [c for it in items for c in it["p"]["sconf"][-1:] if it["p"]["suf"][-1:].isdigit()]
    clean_digit = float(np.median(ref)) if ref else 0.99    # как уверенно читается заведомо чёткая цифра этой строки
    for pos in range(L):
        tally = defaultdict(float); confs = defaultdict(list)
        for it in items:
            s = it["p"]["suf"]
            if len(s) != L: continue
            ch = _suffix_canon(s[pos]); tally[ch] += it["w"]; confs[ch].append(it["p"]["sconf"][pos])
        tot = sum(tally.values()); topc = max(tally, key=tally.get)
        for letter, digit in AMB:
            if topc in (letter, digit):
                alt = letter if topc == digit else digit
                share_alt = tally.get(alt, 0) / tot
                if topc == digit and share_alt >= 0.12 and float(np.mean(confs[digit])) < clean_digit - 0.15 and float(np.mean(confs[alt])) >= 0.45:
                    suf_notes.append(f"{digit}↔{letter}: символ читается нечётко как цифра (уверенность {np.mean(confs[digit]):.2f} против {clean_digit:.2f} у чёткой цифры), в другом чтении/строке — {alt}; выбрано {alt}")
                    topc = alt
                elif share_alt >= 0.3:
                    suf_notes.append(f"{digit}↔{letter}: неоднозначно ({topc} {tally[topc]/tot:.0%} / {alt} {share_alt:.0%})")
                break
        suf += topc; suf_support = min(suf_support, tally[topc] / tot)
    notes += suf_notes
    model = num + (f"-{suf}" if suf else "")
    # согласие строк: модель, прочитанная только по строке ru и только по lat
    agree = None
    if nlines == 2:
        per = {}
        for lang in ("ru", "lat"):
            nn = defaultdict(float)
            for it in items:
                if it["lang"] == lang: nn[it["p"]["num"] + "-" + it["p"]["suf"]] += it["w"]
            per[lang] = max(nn, key=nn.get) if nn else None
        agree = per["ru"] == per["lat"] if per["ru"] and per["lat"] else None
    ru = f"{ru_mk} {model}" if ru_mk and nlines == 2 or (ru_mk and langs.get(0) == "ru") else None
    lat = f"{lat_mk} {model}" if lat_mk and (nlines == 2 or langs.get(0) == "lat") else None
    if known and nlines == 1:   # одна строка — пару не достраиваем
        ru = ru if langs.get(0) == "ru" else None
        lat = lat if langs.get(0) == "lat" else None
    mean_sc = float(np.mean([it["w"] for it in items]))
    conf = float(np.clip(0.35 * mean_sc + 0.25 * support + 0.2 * num_support + 0.2 * suf_support + (0.08 if agree else 0) - (0.1 if agree is False else 0), 0, 0.99))
    if not suf: conf *= 0.9
    return dict(ru=ru, lat=lat, combined=" / ".join(x for x in (ru, lat) if x), conf=conf, valid=bool(known and num and suf is not None), notes=notes, agree=agree,
                lines=nlines, y_lab=round(float(y_lab), 4), block=[int(top), int(bot)], cuts=cuts, candidates=len(items))


# ---------------------------------------------------------------- цвет: чтение нескольких окон/слоёв + словарь
def read_color(card: np.ndarray, V: np.ndarray, y_center: float, paddle, norm_color):
    """Цвет: окна вокруг расчётной позиции значения, разные слои/масштабы; голосуем по словарному совпадению.
    -> dict(value, conf, n, share) или None."""
    if paddle is None or not hasattr(paddle, "recognize_chars"): return None
    H, W = V.shape[:2]
    votes = defaultdict(list); total = 0
    for k in range(-4, 8, 2):
        yc = y_center + 0.004 * k
        a, b = int(max(0, (yc - 0.012) * H)), int(min(H, (yc + 0.018) * H))
        if b - a < 8: continue
        for img in (V, card):
            crop = img[a:b, int(0.28 * W):int(0.75 * W)]
            for hh in (64, 96):
                t, sc, cf = paddle.recognize_chars(_prep(crop, hh))
                if not t.strip(): continue
                total += 1
                val, ok = norm_color(t)
                if val and ok: votes[val].append(sc)
    if not votes: return None
    best = max(votes, key=lambda v: sum(votes[v]))
    n = len(votes[best]); sc = float(np.mean(sorted(votes[best])[-max(1, n // 2):]))
    share = n / max(1, sum(len(x) for x in votes.values()))
    conf = float(np.clip(0.55 * sc + 0.25 * min(1.0, n / 4) + 0.2 * share, 0, 0.95))
    return dict(value=best, conf=conf, n=n, share=share, scores=sc)
