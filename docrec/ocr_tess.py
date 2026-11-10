"""Движок Tesseract 5 (через tesserocr: движок встроен в wheel, отдельная установка не нужна).
Модели tessdata_best (rus+eng) — лучшая на наших тестах точность по кириллице при ~170 МБ RAM."""
from __future__ import annotations
import os, re
from contextlib import contextmanager
from pathlib import Path
import cv2
import numpy as np
from .ocr import Line

MODELS = Path(__file__).parent / "models"
_LABEL_WORD = re.compile(r"^\d{1,2}\s*[abcабс]?\s*[.)]$", re.I)


@contextmanager
def _quiet_stderr():
    """Leptonica пишет в stderr из C-кода («Error in pixScanForForeground»); глушим на время вызова."""
    try:
        fd = os.dup(2); null = os.open(os.devnull, os.O_WRONLY); os.dup2(null, 2)
    except OSError:
        yield; return
    try:
        yield
    finally:
        os.dup2(fd, 2); os.close(fd); os.close(null)


class TessEngine:
    name = "tesseract"

    def __init__(self, fast: bool = False, lang: str = "rus+eng", psm: int = 3):
        import tesserocr
        self.tr = tesserocr
        self.path = str(MODELS / ("tessdata_fast" if fast else "tessdata"))
        self.fast = fast
        self.api = tesserocr.PyTessBaseAPI(path=self.path, lang=lang, psm=psm)
        self._cell = None
        self._osd = None

    @staticmethod
    def _pil(bgr):
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # ---------- полноценное чтение страницы/карточки -> строки с координатами
    def __call__(self, img: np.ndarray, split_gap: float | None = None, psm: int | None = None) -> list[Line]:
        H, W = img.shape[:2]
        tr = self.tr
        if psm is not None: self.api.SetPageSegMode(psm)
        with _quiet_stderr():
            self.api.SetImage(self._pil(img)); self.api.Recognize()
            it = self.api.GetIterator()
            words = []  # (text, conf, x0,y0,x1,y1, line_idx, block_idx)
            line_idx = -1; block_idx = -1
            while it:
                if it.IsAtBeginningOf(tr.RIL.BLOCK): block_idx += 1
                if it.IsAtBeginningOf(tr.RIL.TEXTLINE): line_idx += 1
                try: t = it.GetUTF8Text(tr.RIL.WORD)
                except RuntimeError: t = None
                b = it.BoundingBox(tr.RIL.WORD); c = it.Confidence(tr.RIL.WORD)
                if t and t.strip() and b:
                    words.append((t.strip(), c / 100.0, b[0], b[1], b[2], b[3], line_idx, block_idx))
                if not it.Next(tr.RIL.WORD): break
        if psm is not None: self.api.SetPageSegMode(3)
        # слова -> строки (а в карточке — ещё и сегменты по большим разрывам, подписи полей отдельно)
        out: list[Line] = []; cur: list = []; 
        def flush():
            if not cur: return
            xs0 = min(w[2] for w in cur); ys0 = min(w[3] for w in cur); xs1 = max(w[4] for w in cur); ys1 = max(w[5] for w in cur)
            n = np.array([max(1, len(w[0])) for w in cur], float)
            ln = Line(text=" ".join(w[0] for w in cur), score=float(np.average([w[1] for w in cur], weights=n)),
                      box=np.float32([[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]]))
            ln.x0, ln.x1, ln.y0, ln.y1 = xs0 / W, xs1 / W, ys0 / H, ys1 / H
            ln.cx, ln.cy = (ln.x0 + ln.x1) / 2, (ln.y0 + ln.y1) / 2
            ln.block = cur[0][7]
            out.append(ln); cur.clear()
        prev = None
        for w in words:
            if prev is not None and (w[6] != prev[6]):
                flush()
            elif prev is not None and split_gap is not None:
                gap = (w[2] - prev[4]) / W
                if gap > split_gap or (_LABEL_WORD.match(w[0]) and cur) or (cur and len(cur) == 1 and _LABEL_WORD.match(cur[0][0])):
                    flush()
            elif prev is None:
                pass
            cur.append(w); prev = w
        flush()
        return out

    # ---------- одно слово/ячейка (категории ВУ и т.п.)
    def read_cell(self, bgr: np.ndarray, whitelist: str | None = None, psm: int = 8, scale: int = 3) -> tuple[str, float]:
        if self._cell is None:
            self._cell = self.tr.PyTessBaseAPI(path=self.path, lang="eng", psm=psm)
        api = self._cell; api.SetPageSegMode(psm)
        api.SetVariable("tessedit_char_whitelist", whitelist or "")
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if scale != 1: g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
        with _quiet_stderr():
            from PIL import Image
            api.SetImage(Image.fromarray(g)); t = api.GetUTF8Text().strip(); c = api.MeanTextConf() / 100.0
        return t, c

    def read_latin_line(self, bgr: np.ndarray) -> tuple[str, float]:
        """Одна строка латиницы (ВУ/паспорт: транслитерация) — белый список A-Z . - ' пробел."""
        return self.read_cell(bgr, "ABCDEFGHIJKLMNOPQRSTUVWXYZ .-'", 7, scale=1)

    # ---------- ориентация страницы: угол, на который надо повернуть изображение ПРОТИВ часовой стрелки
    def detect_rotation(self, bgr: np.ndarray, max_side: int = 1400) -> tuple[int, float]:
        if self._osd is None:
            self._osd = self.tr.PyTessBaseAPI(path=self.path, lang="osd", psm=self.tr.PSM.OSD_ONLY)
        h, w = bgr.shape[:2]; s = min(1.0, max_side / max(h, w))
        im = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
        with _quiet_stderr():
            self._osd.SetImage(self._pil(im))
            try:
                r = self._osd.DetectOrientationScript()
            except Exception:
                r = None
        if not r: return 0, 0.0
        return int(r["orient_deg"]), float(r["orient_conf"])

    def close(self):
        for a in (self.api, self._cell, self._osd):
            if a is not None:
                try: a.End()
                except Exception: pass


def rotate_ccw(img: np.ndarray, deg: int) -> np.ndarray:
    return {0: img, 90: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 180: cv2.rotate(img, cv2.ROTATE_180), 270: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)}[deg % 360]
