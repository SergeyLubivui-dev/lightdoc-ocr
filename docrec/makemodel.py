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
    for i, (ru, la) in enumerate(pairs_list()):
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
def _locate(V, y_lab, rd):
    """Строки значения: перебираем полосы чернил в окне (первая могла быть хвостом VIN) и берём первую, которая читается как «МАРКА 1234-XX».
    -> (x0, x1, spans[(top,bot),...], one_text) или None. Полоса из двух строк вплотную режется по тому разрезу, где обе половины читаются лучше."""
    H, W = V.shape[:2]
    blk = find_value_block(V, y_lab)
    if blk is None: return None
    x0, x1, runs_ = blk
    for si in range(min(3, len(runs_))):
        top, bot = runs_[si]; h = bot - top + 1
        one = rd(top, bot)
        p1 = parse_candidate(one[0], one[2], "lat") if one else None
        one_ok = bool(p1 and one[1] >= 0.6)
        spans = [(top, bot)]; best = None
        if h >= 0.014 * H:
            step = max(1, int(round(h / 14)))
            g = cv2.cvtColor(V[top:bot + 1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            prof = np.convolve(np.clip(255 - g, 0, None).mean(axis=1), np.ones(3) / 3, mode="same")
            lo, hi = int(0.3 * h), int(0.7 * h) + 1
            valley = top + lo + int(np.argmin(prof[lo:hi])) if hi > lo else top + h // 2   # провал чернил между строками
            for c in range(max(top + lo, valley - max(2, int(0.14 * h))), min(top + hi, valley + max(2, int(0.14 * h))) + 1, 1 if h < 60 else step):
                tot = 0.0; both = True
                for (a, b) in ((top, c + 1), (c - 1, bot)):
                    r_ = rd(a, b)
                    if r_ is None or not parse_candidate(r_[0], r_[2], "lat"): both = False; break
                    tot += r_[1]
                if both and (best is None or tot > best[0]): best = (tot, c)
            if best and (h >= 0.0235 * H * 1.45 or not one_ok or best[0] / 2 >= one[1] - 0.05):
                spans = [(top, best[1] + 1), (best[1] - 1, bot)]
        if len(spans) == 1 and si + 1 < len(runs_):
            a2, b2 = runs_[si + 1]
            if a2 - bot <= 0.02 * H and one_ok:
                r2 = rd(a2, b2)
                if r2 and parse_candidate(r2[0], r2[2], "lat") and r2[1] >= 0.6: spans = [(top, bot), (a2, b2)]
        if len(spans) == 2 or one_ok:
            return x0, x1, spans, one
    return None


def _enhance(c: np.ndarray, mode: str) -> np.ndarray:
    """Дополнительные «взгляды» на кроп для вторичного чтения плохих снимков."""
    if mode == "clahe":
        lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB); l = cv2.createCLAHE(3.0, (4, 4)).apply(lab[..., 0]); lab[..., 0] = l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if mode == "sharp":
        b = cv2.GaussianBlur(c, (0, 0), 1.2); return cv2.addWeighted(c, 1.8, b, -0.8, 0)
    if mode == "gamma":
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0; g = np.clip((g - g.min()) / max(1e-3, g.max() - g.min()), 0, 1) ** 1.6
        return cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return c


def _collect(spans, x0, x1, V, card, paddle, cell, W, H, extra=False):
    """Кандидаты чтения каждой строки: [(строка, текст, conf по символам, score, движок, число-только?)]."""
    H_, W_ = H, W
    cands = []; nlines = len(spans)
    for li, (a, b) in enumerate(spans):
        offs = [(-2, 0), (0, 1), (-1, 2)] if nlines == 2 else [(-1, 1), (0, 0)]
        if extra: offs = offs + ([(-3, 1), (1, 3), (-2, 2)] if nlines == 2 else [(-2, 2), (-1, 0), (0, 1)])
        srcs = [("V", V), ("card", card)]
        if extra: srcs += [("clahe", _enhance(card, "clahe")), ("sharp", _enhance(card, "sharp")), ("gamma", _enhance(V, "gamma"))]
        for da, db in offs:
            ya = max(0, a + da); yb = min(H_, b + db)
            for sn, img in srcs:
                crop = _trim_cols(img[ya:yb, x0:x1], W_)
                if crop.shape[0] < 6: continue
                for hh in ((64, 96, 48, 128) if extra else (64, 96)):
                    P = _prep(crop, hh)
                    t, sc, cf, pos = paddle.recognize_chars(P, with_pos=True)
                    cands.append((li, t, cf, sc, "paddle", None))
                    # только цифровая часть модели: Tesseract с белым списком цифр (digit-favoured re-OCR)
                    if cell is not None and sn in ("card", "clahe") and hh == 96:
                        d = _digits_crop(P, t, pos)
                        if d is not None:
                            try:
                                dt, dsc = cell(d, "0123456789", 7, 2)
                                dt = re.sub(r"\D", "", dt)
                                if 3 <= len(dt) <= 5: cands.append((li, dt, [], dsc, "tess-digits", dt))
                            except Exception:
                                pass
                if cell is not None and sn == "card":
                    try:
                        t, sc = cell(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", 7, 3)
                        if t: cands.append((li, t, [], sc, "tesseract", None))
                    except Exception:
                        pass
    return cands


def _digits_crop(P: np.ndarray, text: str, pos):
    """Участок подготовленного кропа с цифрами номера модели (между пробелом после марки и дефисом) по приблизительным позициям символов."""
    try:
        idx = [i for i, ch in enumerate(text) if ch.isdigit() or ch in "OОSB"]
        m = re.search(r"\s(\d{3,5})\s*[-–—]", text)
        if not m or pos is None: return None
        i0, i1 = m.start(1), m.end(1) - 1
        xs = [pos[i] for i in range(i0, i1 + 1) if pos[i] is not None]
        if len(xs) < 3: return None
        pitch = np.median(np.diff(xs)); a, b = int(max(0, xs[0] - 0.7 * pitch)), int(min(P.shape[1], xs[-1] + 0.7 * pitch))
        return P[:, a:b] if b - a > 20 else None
    except Exception:
        return None


def _models_for(make_lat: str, which: str):
    """Известные числовые модели/суффиксы марки из лексиконов sts/make_model и sts/model_suffix."""
    from . import lexicon as LX
    if which == "num":
        lex = LX.get("sts/make_model"); out = set()
        for k in lex.keys:
            parts = k.upper().split()
            if len(parts) >= 2 and parts[0] == make_lat.upper() and parts[-1].isdigit(): out.add(parts[-1])
        return out
    return set()


def _suffixes_for(make_lat: str, num: str = None):
    """Допустимые суффиксы марки; если для модели есть своя «семья» комплектаций (sts/model_family) — только они."""
    import gzip
    from . import lexicon as LX
    out = set(); fam = set()
    try:
        with gzip.open(LX.DIR / "sts" / "model_family.txt.gz", "rt", encoding="utf-8") as f:
            for line in f:
                p = [x.strip().upper() for x in line.split("|")]
                if len(p) >= 4 and p[1] == make_lat.upper() and p[2] == str(num): fam.add(p[3])
    except Exception:
        pass
    if fam: return fam
    try:
        with gzip.open(LX.DIR / "sts" / "model_suffix.txt.gz", "rt", encoding="utf-8") as f:
            for line in f:
                p = [x.strip().upper() for x in line.split("|")]
                if len(p) >= 3 and p[1] == make_lat.upper(): out.add(p[2])
    except Exception:
        pass
    return out


def _decide(items, nlines, langs, known_pairs=True):
    """Голосование по кандидатам обеих строк. Каждый язык получает равный суммарный вес (чтобы латинская строка в одиночку не перевешивала кириллическую)."""
    notes = []
    tot = defaultdict(float)
    for it in items: tot[it["lang"]] += it["w"]
    nl = len(tot) or 1
    for it in items: it["wn"] = it["w"] / tot[it["lang"]] / nl      # сумма по каждому языку = 1/nl
    # --- марка
    pv = defaultdict(float); unknown = defaultdict(float)
    for it in items:
        if it.get("digits_only"): continue
        sm = snap_make(it["p"]["make"])
        if sm: pv[sm[0]] += it["wn"] * (1.0 - 0.3 * sm[1])
        else: unknown[lk(it["p"]["make"])] += it["wn"]
    mk_items = [it for it in items if not it.get("digits_only")]
    pairs = pairs_list()
    if pv and max(pv.values()) >= 0.5 * (sum(pv.values()) + sum(unknown.values())):
        pi = max(pv, key=pv.get); ru_mk, lat_mk = pairs[pi]; known = True
        support = pv[pi] / (sum(pv.values()) + sum(unknown.values()))
    else:
        known = False; support = 0.5; best = {}
        for it in mk_items:
            k = (it["lang"], it["p"]["make"]); best[k] = best.get(k, 0) + it["wn"]
        ru_mk = max(((m, w) for (l, m), w in best.items() if l == "ru"), key=lambda x: x[1], default=(None, 0))[0]
        lat_mk = max(((m, w) for (l, m), w in best.items() if l == "lat"), key=lambda x: x[1], default=(None, 0))[0]
        if ru_mk and lat_mk and lk(ru_mk) != lat_mk: notes.append("марка в строках не совпала после транслитерации")
        notes.append("марка не из словаря пар")
    # --- число модели
    nv = defaultdict(float); per_lang_num = defaultdict(lambda: defaultdict(float))
    for it in items:
        nv[it["p"]["num"]] += it["wn"]; per_lang_num[it["lang"]][it["p"]["num"]] += it["wn"]
    num = max(nv, key=nv.get); num_support = nv[num] / sum(nv.values())
    if known:
        known_nums = _models_for(lat_mk, "num")
        if known_nums and num not in known_nums:
            from . import lexicon as LX
            cand = [(n, w) for n, w in sorted(nv.items(), key=lambda x: -x[1]) if n in known_nums and w >= 0.08 * sum(nv.values())]
            if cand:
                notes.append(f"число модели {num} нет в списке моделей {lat_mk}; выбрано {cand[0][0]} (доля чтений {cand[0][1] / sum(nv.values()):.0%})"); num = cand[0][0]
            else:
                near = sorted(((LX.wdist(num, k), k) for k in known_nums if len(k) == len(num)))
                if near and near[0][0] <= 0.65 and (len(near) == 1 or near[1][0] - near[0][0] >= 0.3):
                    notes.append(f"число модели {num} нет в списке моделей {lat_mk}: визуально ближайшая известная {near[0][1]} (0↔8/3↔8 и т.п.) — исправлено; чтения: " + ", ".join(f"{n} {w / sum(nv.values()):.0%}" for n, w in sorted(nv.items(), key=lambda x: -x[1])[:3]))
                    num = near[0][1]
                else:
                    notes.append(f"число модели {num} нет в списке моделей {lat_mk}; альтернативы: " + ", ".join(f"{n} {w / sum(nv.values()):.0%}" for n, w in sorted(nv.items(), key=lambda x: -x[1])[:3]))
        num_support = nv.get(num, 0) / sum(nv.values())
    # --- суффикс
    sitems = [it for it in items if not it.get("digits_only")]
    sl = defaultdict(float)
    for it in sitems: sl[len(it["p"]["suf"])] += it["wn"]
    L = max(sl, key=sl.get) if sl else 0
    suf = ""; suf_notes = []; suf_support = 1.0
    ref = [c for it in sitems for c in it["p"]["sconf"][-1:] if it["p"]["suf"][-1:].isdigit()]
    clean_digit = float(np.median(ref)) if ref else 0.99
    pos_tallies = []
    for pos in range(L):
        tally = defaultdict(float); confs = defaultdict(list)
        for it in sitems:
            sx = it["p"]["suf"]
            if len(sx) != L: continue
            ch = _suffix_canon(sx[pos]); tally[ch] += it["wn"]; confs[ch].append(it["p"]["sconf"][pos])
        tt = sum(tally.values()); topc = max(tally, key=tally.get)
        for letter, digit in AMB:
            if topc in (letter, digit):
                alt = letter if topc == digit else digit
                share_alt = tally.get(alt, 0) / tt
                if topc == digit and share_alt >= 0.12 and float(np.mean(confs[digit])) < clean_digit - 0.15 and float(np.mean(confs[alt])) >= 0.45:
                    suf_notes.append(f"{digit}↔{letter}: символ читается нечётко как цифра (уверенность {np.mean(confs[digit]):.2f} против {clean_digit:.2f} у чёткой цифры), в другом чтении/строке — {alt}; выбрано {alt}")
                    topc = alt
                elif share_alt >= 0.3:
                    suf_notes.append(f"{digit}↔{letter}: неоднозначно ({topc} {tally[topc] / tt:.0%} / {alt} {share_alt:.0%})")
                break
        pos_tallies.append(dict(tally)); suf += topc; suf_support = min(suf_support, tally[topc] / tt)
    # суффикс против списка известных суффиксов марки: S/8/5 и 0/8 (правила путаниц через матрицу стоимостей)
    if known and suf:
        ks = _suffixes_for(lat_mk, num)
        if ks and suf not in ks:
            from . import lexicon as LX
            best = None
            for k in ks:
                if len(k) != len(suf): continue
                share = 1.0
                for i, ch in enumerate(k):
                    tl = pos_tallies[i]; share *= (sum(w for c, w in tl.items() if c == ch) / max(sum(tl.values()), 1e-6)) + 0.04
                d = LX.wdist(suf.lower(), k.lower())
                if d <= 1.1 and (best is None or (share, -d) > (best[0], -best[1])): best = (share, d, k)
            if best and best[0] >= 0.02:
                suf_notes.append(f"суффикс {suf} не бывает у {lat_mk} {num} (допустимые: {', '.join(sorted(ks))}); выбран ближайший по форме и чтениям {best[2]} (S/8/5, 0/8)")
                suf = best[2]; suf_support = min(suf_support, 0.6)
            else:
                suf_notes.append(f"суффикс {suf} не в списке известных для {lat_mk} {num}")
    notes += suf_notes
    model = num + (f"-{suf}" if suf else "")
    # согласие строк
    agree = None; alts = []
    if nlines == 2:
        per = {}
        for lang in ("ru", "lat"):
            nn = defaultdict(float)
            for it in sitems:
                if it["lang"] == lang: nn[it["p"]["num"] + "-" + it["p"]["suf"]] += it["wn"]
            per[lang] = (max(nn, key=nn.get) if nn else None, dict(nn))
        a_, b_ = per["ru"][0], per["lat"][0]
        norm_ = lambda x: x and x.replace("S", "5").replace("O", "0").replace("B", "8")
        agree = (a_ == b_) if (a_ and b_) else None
        if a_ and b_ and a_ != b_ and norm_(a_) == norm_(b_): agree = None   # различаются только путаницей S/5/O/0 — решено выше
        if agree is False:
            alts = [f"ru: {a_}", f"lat: {b_}"]
            notes.append(f"строки не согласованы: кириллица читается {a_}, латиница {b_}; выбрано {model}")
    return dict(ru_mk=ru_mk, lat_mk=lat_mk, known=known, support=support, num=num, num_support=num_support, suf=suf, suf_support=suf_support, model=model, notes=notes, agree=agree)


def pairs_list():
    """Пары (кириллица, латиница) из лексикона sts/make; запасной список — PAIRS."""
    global _PAIRS
    if _PAIRS is None:
        from . import lexicon as LX
        pl = [tuple(l.split("|")[:2]) for l in LX.get("sts/make").canon] if False else []
        import gzip
        try:
            with gzip.open(LX.DIR / "sts" / "make.txt.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    p = [x.strip() for x in line.split("|") if x.strip()]
                    if len(p) == 2: pl.append((p[0].upper(), p[1].upper()))
        except Exception:
            pass
        _PAIRS = pl or list(PAIRS)
    return _PAIRS


_PAIRS = None


def read_make_model(card: np.ndarray, V: np.ndarray, y_lab: float, paddle, cell=None):
    """-> dict(ru, lat, combined, conf, valid, notes, agree) или None. y_lab — нормированный y подписи «Марка, модель»."""
    if paddle is None or not hasattr(paddle, "recognize_chars"): return None
    H, W = V.shape[:2]

    def rd(a, b, hh=64, src=V):
        crop = _trim_cols(src[max(0, a):min(H, b + 1), int(0.28 * W):int(0.99 * W)], W)
        if crop.shape[0] < 6: return None
        t, sc, cf = paddle.recognize_chars(_prep(crop, hh))
        return t, sc, cf

    loc = _locate(V, y_lab, lambda a, b: rd(a, b))
    if loc is None: return None
    x0, x1, spans, _ = loc
    nlines = len(spans); top, bot = spans[0][0], spans[-1][1]
    cuts = [spans[0][0], spans[0][1], spans[-1][1]]
    items = []; langs = {}

    def build(cands):
        parsed = []
        for li, t, cf, sc, eng, dig in cands:
            if eng == "tess-digits":
                parsed.append(dict(line=li, lang_guess=None, t=t, p=dict(make="", num=dt_ if (dt_ := t) else "", suf="", sconf=[]), sc=sc, eng=eng, cf=cf, digits_only=True)); continue
            for lang in ("ru", "lat"):
                p = parse_candidate(t, cf, lang)
                if p is None: continue
                letters = re.sub(r"[^A-Za-zА-Яа-я]", "", t.split()[0] if t.split() else "")
                cyr = sum(ch in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯЬЪабвгдеёжзийклмнопрстуфхцчшщыэюяьъ" for ch in letters)
                lat = sum(ch.isascii() and ch.isalpha() for ch in letters)
                parsed.append(dict(line=li, lang_guess=("ru" if cyr > lat else "lat" if lat > cyr else None), t=t, p=p, sc=sc, eng=eng, cf=cf)); break
        return parsed

    parsed_all = build(_collect(spans, x0, x1, V, card, paddle, cell, W, H))
    if not [d for d in parsed_all if not d.get("digits_only")]: return None

    def make_items(parsed):
        if nlines == 2: lg = {0: "ru", 1: "lat"}
        else:
            votes = defaultdict(int)
            for d in parsed:
                if d.get("lang_guess"): votes[d["lang_guess"]] += 1
            lg = {0: max(votes, key=votes.get) if votes else "lat"}
        its = []
        for d in parsed:
            lang = lg.get(d["line"], "lat")
            if d.get("digits_only"):
                its.append(dict(lang=lang, p=dict(make="", num=d["p"]["num"], suf="", sconf=[]), w=max(0.75 * d["sc"], 0.05), line=d["line"], digits_only=True)); continue
            p = parse_candidate(d["t"], d["cf"], lang)
            if p is None: continue
            w = d["sc"] * (0.7 if d["eng"] == "tesseract" else 1.0)
            if d["eng"] == "tesseract": p["sconf"] = [min(0.6, d["sc"])] * len(p["suf"])
            its.append(dict(lang=lang, p=p, w=max(w, 0.05), line=d["line"]))
        return its, lg

    items, langs = make_items(parsed_all)
    dec = _decide(items, nlines, langs)
    conf_of = lambda d, its: float(np.clip(0.35 * float(np.mean([it["w"] for it in its])) + 0.25 * d["support"] + 0.2 * d["num_support"] + 0.2 * d["suf_support"]
                                           + (0.08 if d["agree"] else 0) - (0.1 if d["agree"] is False else 0), 0, 0.99))
    conf = conf_of(dec, items)
    escalated = False
    if conf < 0.8 or dec["agree"] is not True:      # плохой снимок: больше вариантов кропа/обработки, затем общий пересчёт
        parsed_all += build(_collect(spans, x0, x1, V, card, paddle, cell, W, H, extra=True))
        items, langs = make_items(parsed_all)
        dec = _decide(items, nlines, langs); conf = conf_of(dec, items); escalated = True
    if not dec["suf"]: conf *= 0.9
    ru_mk, lat_mk, model = dec["ru_mk"], dec["lat_mk"], dec["model"]
    ru = f"{ru_mk} {model}" if ru_mk and (nlines == 2 or langs.get(0) == "ru") else None
    lat = f"{lat_mk} {model}" if lat_mk and (nlines == 2 or langs.get(0) == "lat") else None
    if dec["known"] and nlines == 1:
        ru = ru if langs.get(0) == "ru" else None
        lat = lat if langs.get(0) == "lat" else None
    notes = dec["notes"] + (["вторичное чтение (CLAHE/резкость/масштабы) из-за низкой уверенности или расхождения строк"] if escalated else [])
    return dict(ru=ru, lat=lat, combined=" / ".join(x for x in (ru, lat) if x), conf=conf, valid=bool(dec["known"] and dec["num"]), notes=notes, agree=dec["agree"],
                lines=nlines, y_lab=round(float(y_lab), 4), block=[int(top), int(bot)], cuts=cuts, candidates=len(items), escalated=escalated)


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
                val, ok = norm_color(t)[:2]
                if val and ok: votes[val].append(sc)
    if not votes: return None
    best = max(votes, key=lambda v: sum(votes[v]))
    n = len(votes[best]); sc = float(np.mean(sorted(votes[best])[-max(1, n // 2):]))
    share = n / max(1, sum(len(x) for x in votes.values()))
    conf = float(np.clip(0.55 * sc + 0.25 * min(1.0, n / 4) + 0.2 * share, 0, 0.95))
    return dict(value=best, conf=conf, n=n, share=share, scores=sc)
