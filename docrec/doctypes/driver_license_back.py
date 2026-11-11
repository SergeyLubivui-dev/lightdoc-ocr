"""Водительское удостоверение РФ, ОБОРОТНАЯ сторона: таблица категорий (колонки 9 / 10 / 11 / 12 — категория, дата выдачи, дата окончания, особые отметки),
поле 14 (год), красный номер. Таблица разбирается по линиям сетки (горизонтальные/вертикальные), строки сопоставляются с фиксированным порядком категорий —
буквы категорий OCR не читаем. Строка «открыта», если в ней есть даты (колонка пиктограмм ненадёжна). Даты читаются по ячейкам (увеличение Lanczos + резкость,
голосование Paddle + Tesseract) и приводятся к согласованному виду: дата окончания обычно одна на все строки (= срок действия), дата выдачи ≤ дате окончания."""
from __future__ import annotations
import re
from collections import Counter
from datetime import date
import cv2
import numpy as np
from ..ocr import Line
from .base import DocType, Extraction, Field, register

CATS = ["A", "A1", "B", "B1", "C", "C1", "D", "D1", "BE", "CE", "C1E", "DE", "D1E", "M", "Tm", "Tb"]
CAT_TOKENS = {c.upper() for c in CATS}
NUM_RE = re.compile(r"(?<!\d)(\d{2})\s*(\d{2})\s*(\d{6})(?!\d)")
CONF = {"0": "8", "8": "0", "1": "7", "7": "1", "3": "8", "5": "6", "6": "5", "2": "7", "9": "8", "4": "9"}   # частые путаницы цифр (в обе стороны ищем допустимую дату)


def _tok(t: str) -> str:
    t = re.sub(r"[^A-Za-zА-Яа-я0-9]", "", t).upper()
    return t.translate(str.maketrans("АВСЕМНОРТХ", "ABCEMHOPTX")).replace("TM", "TM")


def _valid(d: str, lo=1950, hi=2060):
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", d or "")
    if not m: return None
    try: dt = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError: return None
    return dt if lo <= dt.year <= hi else None


def parse_dates(text: str):
    """Все варианты даты в строке OCR: 'ДД.ММ.ГГГГ' и слитные 8 цифр."""
    t = text.replace("O", "0").replace("О", "0").replace("o", "0").replace("l", "1").replace("I", "1").replace("S", "5")
    out = []
    for m in re.finditer(r"(\d{1,2})\s*[.,:;\-/ ]\s*(\d{1,2})\s*[.,:;\-/ ]\s*(\d{4})", t): out.append(f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}")
    d = re.sub(r"\D", "", t)
    if not out and len(d) == 8: out.append(f"{d[:2]}.{d[2:4]}.{d[4:]}")
    return out


# ---------- сетка таблицы ----------
def estimate_skew(card: np.ndarray):
    """Остаточный наклон карточки после выпрямления (градусы): медиана углов длинных почти горизонтальных отрезков в правой части (таблица)."""
    H, W = card.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    e = cv2.Canny(g, 40, 110); e[:, :int(0.30 * W)] = 0
    segs = cv2.HoughLinesP(e, 1, np.pi / 720, threshold=40, minLineLength=int(0.08 * W), maxLineGap=int(0.02 * W))
    if segs is None: return 0.0
    ang = [np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) for x1, y1, x2, y2 in segs.reshape(-1, 4)]
    ang = [v for v in ang if abs(v) < 8]
    return float(np.median(ang)) if len(ang) >= 6 else 0.0


def deskew(card: np.ndarray):
    a = estimate_skew(card)
    if abs(a) < 0.25: return card, 0.0
    H, W = card.shape[:2]; M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
    return cv2.warpAffine(card, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), a

def _runs(mask1d: np.ndarray, min_len=1):
    idx = np.where(mask1d)[0]
    if not len(idx): return []
    out = []; s = p = idx[0]
    for i in idx[1:]:
        if i != p + 1: out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return [r for r in out if r[1] - r[0] + 1 >= min_len]


def find_grid(card: np.ndarray, close: bool | None = None):
    """-> dict(ys=[...], xs=[...], x0, x1, rows_n, confidence, method) | None. Линии — морфологическое открытие бинарного изображения длинными ядрами."""
    if close is None:     # сначала по сплошным линиям; пунктирные (алиасинг мелкого снимка) — со склейкой штрихов
        r = find_grid(card, False)
        if r is not None and len(r["ys"]) >= 16: return r
        r2 = find_grid(card, True)
        return r2 if r2 is not None and (r is None or len(r2["ys"]) > len(r["ys"])) else r
    H, W = card.shape[:2]
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    bwh = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))) if close else bw    # пунктир от алиасинга мелкого снимка -> сплошная линия
    hl = cv2.morphologyEx(bwh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.10 * W), 1)))
    # столбцы таблицы: там, где много горизонтальных линий
    col = (hl > 0).sum(0).astype(np.float32); colmax = col.max()
    col[:int(0.30 * W)] = 0        # слева — штрихкод (PDF417): горизонтальных линий таблицы там нет
    colmax = col.max()
    if colmax < 5: return None
    cm = cv2.morphologyEx((col >= 0.5 * colmax).astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)).ravel() > 0
    rr = _runs(cm)
    if not rr: return None
    x0, x1 = (int(v) for v in max(rr, key=lambda r: r[1] - r[0]))
    if x1 - x0 < 0.25 * W: return None
    prof = (hl[:, x0:x1] > 0).sum(1).astype(np.float32)
    lines = _runs(prof > 0.55 * (x1 - x0))
    ys = [float((a + b) / 2) for a, b in lines]
    # слить слишком близкие (двойная линия)
    ys2 = []
    for y in ys:
        if ys2 and y - ys2[-1] < 0.012 * H: ys2[-1] = (ys2[-1] + y) / 2
        else: ys2.append(y)
    ys = ys2
    if len(ys) < 8: return None
    # выбрать подпоследовательность из ≤18 линий с самым ровным шагом (таблица: шапка + 16 строк + низ)
    diffs = np.diff(ys); med = float(np.median(diffs)); keep = [ys[0]]
    best = None
    for i in range(len(ys)):
        run = [ys[i]]
        for y in ys[i + 1:]:
            if abs((y - run[-1]) - med) <= 0.35 * med or abs((y - run[-1]) - 2 * med) <= 0.2 * med: run.append(y)
        if best is None or len(run) > len(best): best = run
    ys = [y for y in best if 0.03 * H < y < 0.97 * H] if len(best) > 3 else best      # рамка самой карточки — не линия таблицы
    # вертикальные линии внутри таблицы
    bwv = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))) if close else bw
    vl = cv2.morphologyEx(bwv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.25 * H))))
    ya, yb = int(ys[0]), int(ys[-1])
    cp = (vl[ya:yb, :] > 0).sum(0).astype(np.float32)
    xr = _runs(cp > 0.5 * (yb - ya)); xs = [float((a + b) / 2) for a, b in xr if x0 - 0.012 * W <= (a + b) / 2 <= x1 + 0.012 * W]
    xs2 = []
    for x in xs:
        if xs2 and x - xs2[-1] < 0.015 * W: xs2[-1] = (xs2[-1] + x) / 2
        else: xs2.append(x)
    return dict(ys=ys, xs=xs2, x0=x0, x1=x1, hl=hl)


def rows_from_grid(ys: list, n_cats=16):
    """Линии -> границы 16 строк категорий. Шапка (если есть) отбрасывается; при нехватке линий (выцвели) — экстраполяция по шагу, confidence ниже.
    -> (list[(top,bot)], conf, note)"""
    n = len(ys) - 1
    pitch = float(np.median(np.diff(ys)))
    if n == n_cats: rows = list(zip(ys[:-1], ys[1:])); return rows, 0.95, "16 строк по линиям сетки"
    if n == n_cats + 1: rows = list(zip(ys[1:-1], ys[2:])); return rows, 0.95, "шапка + 16 строк по линиям сетки"
    if n > n_cats + 1:      # лишние линии — берём последние 16 строк (шапка вверху)
        yy = ys[-(n_cats + 1):]; return list(zip(yy[:-1], yy[1:])), 0.6, f"найдено {n} строк, взяты нижние {n_cats}"
    # меньше: экстраполируем вниз/вверх по шагу, ориентируясь на то, что низ таблицы ближе к нижнему краю карточки
    missing = n_cats - n
    yy = list(ys) + [ys[-1] + pitch * (i + 1) for i in range(missing)]
    return list(zip(yy[:-1], yy[1:])), 0.4, f"линий найдено {n + 1} из {n_cats + 1}: строки достроены по шагу {pitch:.1f}px"


# ---------- ячейки ----------
def _cell_ink(card_gray: np.ndarray, x0, x1, y0, y1):
    """(доля чернил, число связных компонент символов, кроп). Фон ячейки — медиана; чернила — заметно темнее."""
    c = card_gray[int(y0) + 3:int(y1) - 3, int(x0) + 3:int(x1) - 3]
    if c.size == 0: return 0.0, 0, c
    bg = float(np.median(c)); m = ((bg - c.astype(np.float32)) > 45).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    cc = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 6 and st[i, cv2.CC_STAT_HEIGHT] >= 0.25 * c.shape[0])
    return float(m.mean()), cc, c


def _prep(c: np.ndarray, scale=4, mode="clahe"):
    g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    if mode == "clahe": g = cv2.createCLAHE(2.5, (4, 4)).apply(g)
    big = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    if mode in ("clahe", "sharp"):
        bl = cv2.GaussianBlur(big, (0, 0), 2.0); big = cv2.addWeighted(big, 1.8, bl, -0.8, 0)
    if mode == "bin":
        _, big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(big, 18, 18, 24, 24, cv2.BORDER_CONSTANT, value=255 if mode != "bin" else 255)


def read_date_cell(crop: np.ndarray, paddle, cell):
    """Несколько чтений ячейки (режимы усиления × масштабы × движки) -> list[(строка даты, score, источник)]."""
    reads = []
    for mode in ("clahe", "sharp", "bin"):
        for sc in (3, 5):
            im = cv2.cvtColor(_prep(crop, sc, mode), cv2.COLOR_GRAY2BGR)
            if paddle is not None and hasattr(paddle, "recognize_crop"):
                try:
                    t, s = paddle.recognize_crop(im)
                    for d in parse_dates(t): reads.append((d, float(s), f"paddle/{mode}/{sc}"))
                except Exception: pass
            if cell is not None:
                try:
                    t, s = cell(im, "0123456789.", 7, 1)
                    for d in parse_dates(t): reads.append((d, float(s / 100 if s > 1 else s) * 0.9, f"tess/{mode}/{sc}"))
                except Exception: pass
    return reads


def _dist(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def _repair(cands: list, ok) -> tuple[str | None, str | None]:
    """cands — строки-даты с ошибками; ищем ближайшую, удовлетворяющую ok(), подстановкой одной цифры по таблице путаниц."""
    for c in cands:
        for i, ch in enumerate(c):
            if ch in CONF:
                alt = c[:i] + CONF[ch] + c[i + 1:]
                if ok(alt): return alt, f"{c} → {alt} (цифра {ch}→{CONF[ch]}, по согласованности дат)"
    return None, None


def _below_table(card):
    """нормированный y нижней линии таблицы категорий: красный номер и поле 14 лежат сразу под ней (а не на фиксированной высоте карточки)"""
    try:
        g = find_grid(card)
        if g and g.get("ys"): return float(max(g["ys"])) / card.shape[0]
    except Exception: pass
    return None


@register
class DriverLicenseBack(DocType):
    name = "driver_license_back"; title_ru = "Водительское удостоверение РФ, оборотная сторона"
    card_size = (1026, 648); aspect = 1026 / 648
    keywords = []
    implemented = True; hybrid_internal = True
    expected_fields = ["categories", "licence_expiry", "number", "field_14"]

    def classify(self, lines: list[Line]) -> float:
        """Оборот: категории стоят отдельными строками (A1, B1, C1E, Tm …) + заголовки колонок «10.», «11.», «12.» (на лице категории идут одной строкой)."""
        toks = {_tok(l.text) for l in lines if len(re.sub(r"[^A-Za-zА-Яа-я0-9]", "", l.text)) <= 4 and l.x0 > 0.3}
        singles = len((toks & CAT_TOKENS) - {"A", "B", "C", "D", "E", "M"})
        heads = sum(1 for n in ("10", "11", "12") if any(re.fullmatch(rf"\W*{n}\W*", l.text.strip()) for l in lines))
        sc = min(1.0, singles / 6) * 0.75 + min(1.0, heads / 2) * 0.25
        return float(sc)

    def grid_score(self, card: np.ndarray) -> float:
        """Структурный признак оборота: таблица из ~17 горизонтальных линий и ≥4 вертикальных (на лицевой стороне такой сетки нет)."""
        try: g = find_grid(deskew(card)[0])
        except Exception: return 0.0
        if g is None: return 0.0
        n = len(g["ys"]) - 1; d = np.diff(g["ys"])
        regular = len(d) >= 8 and float(np.std(d) / max(np.mean(d), 1e-6)) < 0.10      # строки таблицы — равный шаг (текст/фото формы дают рваный)
        wide = (g["x1"] - g["x0"]) >= 0.4 * card.shape[1] and 4 <= len(g["xs"]) <= 7
        if not (regular and wide): return 0.0
        return 0.9 if n in (16, 17) else 0.7 if n in (15, 18) else 0.0

    def extract(self, card, lines, rec=None, cell=None, ocr=None, paddle=None) -> Extraction:
        ex = Extraction(); F = ex.fields; ex.extra["side"] = "back"
        # ориентация: таблица одинакова вверх ногами — решаем по тексту (категории/заголовки); при необходимости поворачиваем на 180°
        if self.classify(lines) < 0.3 and (ocr is not None or paddle is not None):
            try:
                c2 = cv2.rotate(card, cv2.ROTATE_180); l2 = paddle(c2, merge=False) if paddle is not None else ocr(c2)
                if self.classify(l2) > self.classify(lines): card, lines = c2, l2; ex.extra["rotated_180"] = True
            except Exception: pass
        card, skew = deskew(card)
        if skew: ex.extra["deskew_deg"] = round(skew, 2)
        H, W = card.shape[:2]; gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        grid = find_grid(card)
        if grid is None:
            ex.extra["warnings_back"] = ["grid_not_found: линии таблицы категорий не найдены"]
            F["categories"] = Field(None, 0.0, "таблица категорий не найдена", valid=False)
            self._number_and_14(F, lines, card, cell, paddle)
            return ex
        rows, rconf, rnote = rows_from_grid(grid["ys"])
        xs = grid["xs"]
        # колонки: справа налево — особые отметки (12), дата окончания (11), дата выдачи (10)
        if len(xs) >= 4: cx = xs[2:6] if len(xs) == 6 else xs[-4:]; cnote = "колонки по вертикальным линиям"
        else:
            x0, x1 = grid["x0"], grid["x1"]; wd = x1 - x0
            cx = self._cols_from_ink(gray, grid, rows, x0, x1); cnote = "вертикальные линии не найдены — колонки по кластерам текста в строках"; rconf *= 0.8
        ex.extra["table"] = {"rows": len(rows), "grid_lines_y": len(grid["ys"]), "grid_lines_x": len(xs), "note": rnote + "; " + cnote, "box_norm": [round(grid["x0"] / W, 3), round(grid["ys"][0] / H, 3), round(grid["x1"] / W, 3), round(grid["ys"][-1] / H, 3)]}
        info = []
        for (top, bot), cat in zip(rows, CATS):
            a = _cell_ink(gray, cx[0], cx[1], top, bot); b = _cell_ink(gray, cx[1], cx[2], top, bot); n12 = _cell_ink(gray, cx[2], cx[3], top, bot)
            info.append(dict(category=cat, top=top, bot=bot, ink_from=a, ink_to=b, ink_notes=n12))
        # порог «есть текст»: ячейки с >=6 символьных компонент (дата = 8 цифр + точки)
        for r in info:
            r["open_from"] = r["ink_from"][1] >= 5; r["open_to"] = r["ink_to"][1] >= 5
        reads = {}
        for r in info:
            for col, key in (("from", "ink_from"), ("to", "ink_to")):
                if r[f"open_{col}"]:
                    c = r[key][2]
                    reads[(r["category"], col)] = read_date_cell(c, paddle, cell)
        # --- согласование: срок окончания (мода по строкам) и date_from ≤ date_to
        def best_by_votes(rs, ok=lambda d: _valid(d) is not None):
            v = Counter()
            for d, s, _ in rs:
                if ok(d): v[d] += s
            return v
        to_votes = Counter()
        for (cat, col), rs in reads.items():
            if col == "to":
                for d, s in best_by_votes(rs).items(): to_votes[d] += s
        expiry = to_votes.most_common(1)[0][0] if to_votes else None
        exp_dt = _valid(expiry) if expiry else None
        out_rows = []; notes_all = []
        for r in info:
            cat = r["category"]; opened = bool(r["open_from"] or r["open_to"])
            row = dict(category=cat, open=opened, date_from=None, date_to=None, notes=None, confidence=0.9 if rconf >= 0.9 else rconf)
            if opened:
                rs_to = reads.get((cat, "to"), []); rs_fr = reads.get((cat, "from"), [])
                dt_to, c_to, n_to = self._pick(rs_to, expiry, None, role="to")
                hi = _valid(dt_to) if dt_to else exp_dt
                dt_fr, c_fr, n_fr = self._pick(rs_fr, None, hi, role="from")
                row.update(date_from=dt_fr, date_to=dt_to, confidence=round(float(min(rconf, 0.5 * (c_fr + c_to) if (dt_fr and dt_to) else 0.3 * max(c_fr, c_to))), 3))
                nn = [x for x in (n_fr and "date_from: " + n_fr, n_to and "date_to: " + n_to) if x]
                if not dt_fr or not dt_to: nn.append("дата прочитана не полностью")
                if nn: row["note"] = "; ".join(nn)
                if r["ink_notes"][1] >= 3: row["notes"] = None; row["note"] = ((row.get("note") + "; ") if row.get("note") else "") + "в колонке 12 есть текст (не распознаётся)"
            out_rows.append(row)
        ex.extra["categories_rows"] = len(out_rows)
        mean_c = float(np.mean([r["confidence"] for r in out_rows if r["open"]])) if any(r["open"] for r in out_rows) else rconf
        F["categories"] = Field(out_rows, round(mean_c, 3), f"{rnote}; {cnote}", valid=len(rows) == 16)
        F["categories_open"] = Field([r["category"] for r in out_rows if r["open"]], round(mean_c, 3), "строки с датами (колонка пиктограмм не используется)", valid=True)
        if expiry:
            F["licence_expiry"] = Field(expiry, round(min(0.95, 0.3 + 0.2 * sum(1 for r in out_rows if r["date_to"] == expiry)), 3), f"дата окончания, общая для {sum(1 for r in out_rows if r['date_to'] == expiry)} строк (колонка 11)", valid=True)
        self._number_and_14(F, lines, card, cell, paddle)
        self._field14_extras(F, card, paddle); self._expired_flag(F)
        return ex

    @staticmethod
    def _field14_extras(F, card, paddle):
        """Строка поля 14 («14. стаж с 1992г. ОЧКИ ИЛИ К.ЛИНЗЫ»): год начала стажа и ограничение (очки/линзы) — отдельными полями."""
        if paddle is None or not hasattr(paddle, "recognize_crop"): return
        H, W = card.shape[:2]; best = None
        if "field_14" not in F:        # поле 14 не прочитано общим OCR (мелкая карточка) — отдельный проход Paddle по левой нижней части
            try:
                sub = card[int(0.55 * H):, :int(0.48 * W)]; sl = paddle(sub, merge=False)
                for l in sl:
                    y = re.fullmatch(r"\W*(?:14\W*)?\s*(19[5-9]\d|20[0-3]\d)\W*", l.text.strip())
                    if y: F["field_14"] = Field(y.group(1), float(l.score) * 0.85, "поле 14 (год), отдельный проход по левой нижней части", valid=True); break
            except Exception: pass
        tb = _below_table(card)
        wins = ((0.79, 0.885), (0.77, 0.9), (0.8, 0.87))
        if tb is not None and tb < 0.93: wins = ((tb + 0.004, min(1.0, tb + 0.085)), (tb - 0.01, min(1.0, tb + 0.10))) + wins
        for y0, y1 in wins:
            crop = cv2.cvtColor(card[int(y0 * H):int(y1 * H), int(0.10 * W):int(0.62 * W)], cv2.COLOR_BGR2GRAY)
            for sc in (2, 3):
                try: t, cf = paddle.recognize_crop(cv2.cvtColor(cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR))
                except Exception: continue
                tt = t.upper()
                if re.search(r"СТАЖ|С\s*(19|20)\d\d", tt) and (best is None or cf > best[1]): best = (t, float(cf))
        if not best: return
        t, cf = best; tu = t.upper()
        m = re.search(r"С\s*((?:19[5-9]|20[0-3])\d)", tu)
        if m: F["experience_since"] = Field(int(m.group(1)), round(cf * 0.9, 3), f"поле 14: «{t.strip()}»", valid=True)
        tail = tu[m.end():] if m else tu
        tail = re.sub(r"^\W*Г\.?", "", tail).strip(" .,:;")
        if tail and len(re.sub(r"\W", "", tail)) >= 3:
            has_o, has_l = ("ОЧК" in tail or "ОЧИ" in tail), bool(re.search(r"ЛИН", tail))
            norm = "Очки или контактные линзы" if (has_o and has_l) else ("Контактные линзы" if has_l else ("Очки" if has_o else None))
            F["restrictions"] = Field(norm or tail.capitalize(), round(cf * (0.9 if norm else 0.5), 3), f"поле 14, исходный текст: «{tail}»", valid=bool(norm))

    @staticmethod
    def _expired_flag(F):
        e = F.get("licence_expiry")
        if e is not None and e.value:
            try:
                d, mth, y = (int(x) for x in e.value.split("."))
                dt = date(y, mth, d); today = date.today()
                F["expired"] = Field(bool(dt < today), e.confidence, f"срок действия до {e.value}; сегодня {today.strftime('%d.%m.%Y')}", valid=True)
            except Exception: pass

    @staticmethod
    def _cols_from_ink(gray, grid, rows, x0, x1):
        """Колонки 10/11/12 без вертикальных линий: профиль чернил по строкам таблицы правее колонки пиктограмм; два самых широких кластера = даты, остаток = колонка 12."""
        ya, yb = int(rows[1][0]), int(rows[-1][1]); wd = x1 - x0
        sub = gray[ya:yb, int(x0 + 0.20 * wd):int(x1)]
        bg = float(np.median(sub)); ink = ((bg - sub.astype(np.float32)) > 45).astype(np.float32).sum(0)
        m = cv2.morphologyEx((ink > 0.4).astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, int(0.03 * wd)), np.uint8)).ravel() > 0
        rr = [r for r in _runs(m) if r[1] - r[0] > 0.08 * wd]
        off = int(x0 + 0.20 * wd)
        if len(rr) >= 2:
            (a1, b1), (a2, b2) = rr[0], rr[1]
            return [off + a1 - 8, off + (b1 + a2) / 2, off + b2 + 8, x1]
        return [x0 + 0.296 * wd, x0 + 0.558 * wd, x0 + 0.82 * wd, x1]

    @staticmethod
    def _pick(rs, expiry, hi, role):
        """Выбор даты по чтениям. role='to': согласовать с общим сроком окончания; role='from': допустимая дата ≤ hi."""
        if role == "to":
            ok = lambda d: _valid(d) is not None
        else:
            ok = lambda d: _valid(d) is not None and (hi is None or _valid(d) <= hi)
        v = Counter(); tot = sum(s for _, s, _ in rs) or 1.0
        for d, s, _ in rs:
            if ok(d): v[d] += s
        note = None
        if role == "to" and expiry:
            # близкое к сроку окончания чтение — приводим к нему (одна-две цифры)
            near = [(d, s) for d, s, _ in rs if _dist(d, expiry) <= 2]
            if v.get(expiry) or near:
                sc = sum(s for d, s in near) / tot
                if not v.get(expiry): note = f"приведено к общему сроку окончания {expiry}"
                return expiry, round(min(0.95, 0.4 + sc * 0.6), 3), note
        if v:
            d, s = v.most_common(1)[0]
            return d, round(min(0.95, 0.25 + 0.7 * s / tot), 3), None
        # нет допустимых чтений: чиним подстановкой цифр
        cands = [d for d, _, _ in rs] if rs else []
        alt, why = _repair(cands, ok)
        if alt: return alt, 0.4, why
        return None, 0.0, None

    @staticmethod
    def _number_and_14(F, lines, card, cell, paddle):
        H, W = card.shape[:2]
        best = None
        for l in lines:
            if l.cy > 0.8 and l.cx > 0.4:
                m = NUM_RE.search(l.text.replace("O", "0").replace("О", "0"))
                if m and (best is None or l.score > best[1]): best = (f"{m.group(1)} {m.group(2)} {m.group(3)}", float(l.score))
        if best is None and paddle is not None and hasattr(paddle, "recognize_crop"):    # красный номер: разность каналов R−G убирает розовый фон, строка читается как одна
            votes = Counter(); sc_ = {}
            tb = _below_table(card)
            wins = ((0.83, 0.94), (0.81, 0.95), (0.84, 0.93))
            if tb is not None and tb < 0.93: wins = ((tb + 0.004, min(1.0, tb + 0.095)), (tb - 0.01, min(1.0, tb + 0.11))) + wins
            for y0, y1 in wins:
                crop = card[int(y0 * H):int(y1 * H), int(0.42 * W):int(0.99 * W)]
                bb, gg, rr = [crop[:, :, i].astype(np.float32) for i in range(3)]
                red = np.clip(255 - np.clip((rr - gg) * 3.0, 0, 255), 0, 255).astype(np.uint8)
                for sc in (2, 3):
                    try: t, cf = paddle.recognize_crop(cv2.cvtColor(cv2.resize(red, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR))
                    except Exception: continue
                    m = NUM_RE.search(t.replace("O", "0").replace("О", "0"))
                    if m:
                        v = f"{m.group(1)} {m.group(2)} {m.group(3)}"; votes[v] += 1; sc_[v] = max(sc_.get(v, 0), float(cf))
            if votes:
                v, n = votes.most_common(1)[0]; best = (v, min(0.95, sc_[v] * (0.7 + 0.1 * min(3, n))))
        if best is None and cell is not None:     # перечитать красную полосу внизу справа
            crop = card[int(0.88 * H):, int(0.4 * W):]
            for mode in ("clahe", "sharp"):
                try:
                    t, s = cell(cv2.cvtColor(_prep(crop, 2, mode), cv2.COLOR_GRAY2BGR), "0123456789 ", 7, 1)
                    m = NUM_RE.search(t)
                    if m: best = (f"{m.group(1)} {m.group(2)} {m.group(3)}", float(s / 100 if s > 1 else s) * 0.8); break
                except Exception: pass
        if best: F["number"] = Field(best[0], best[1], "красный номер внизу справа", valid=True)
        # поле 14: «14.» + год рядом/ниже слева
        for l in lines:
            if l.cx < 0.5 and re.search(r"(?:^|\s)14\s*[.,]?(?:\s|$)", l.text) or (l.cx < 0.5 and re.match(r"^\W*14\W", l.text)):
                y = re.search(r"(19[5-9]\d|20[0-3]\d)", l.text)
                if y: F["field_14"] = Field(y.group(1), float(l.score), "поле 14 (год)", valid=True); return
                near = [m for m in lines if m is not l and abs(m.cy - l.cy) < 0.05 and m.x0 >= l.x0 and re.search(r"(19[5-9]\d|20[0-3]\d)", m.text)]
                if near:
                    m = near[0]; F["field_14"] = Field(re.search(r"(19[5-9]\d|20[0-3]\d)", m.text).group(1), float(m.score), "поле 14 (год)", valid=True); return
        for l in lines:
            if l.cx < 0.35 and l.cy > 0.55 and re.fullmatch(r"\W*(19[5-9]\d|20[0-3]\d)\W*", l.text.strip()):
                F["field_14"] = Field(re.search(r"\d{4}", l.text).group(0), float(l.score) * 0.8, "год слева внизу (метка «14.» не прочитана)", valid=True); return
