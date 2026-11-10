"""Обёртка над RapidOCR (PP-OCRv5 mobile, кириллица, onnxruntime)."""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Line:
    text: str
    score: float
    box: np.ndarray  # 4x2 в пикселях карточки
    # нормированные координаты центра/границ (0..1) заполняются после OCR
    cx: float = 0.0; cy: float = 0.0
    block: int = -1
    x0: float = 0.0; y0: float = 0.0; x1: float = 0.0; y1: float = 0.0


class OCREngine:
    def __init__(self, threads: int = 0, lang: str = "cyrillic"):
        from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
        params = {
            "Global.use_cls": False,
            "Global.log_level": "error",
            "Det.engine_type": EngineType.ONNXRUNTIME, "Det.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.model_type": ModelType.MOBILE, "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": {"cyrillic": LangRec.CYRILLIC, "eslav": LangRec.ESLAV}[lang],
        }
        mdir = Path(__file__).parent / "models"   # локальные копии моделей -> работа полностью офлайн
        det_p, rec_p = mdir / "ch_PP-OCRv5_det_mobile.onnx", mdir / "cyrillic_PP-OCRv5_rec_mobile.onnx"
        if lang == "cyrillic" and det_p.exists() and rec_p.exists():
            params["Det.model_path"] = str(det_p); params["Rec.model_path"] = str(rec_p)
        if threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        self.engine = RapidOCR(params=params)

    def __call__(self, card_bgr: np.ndarray, merge: bool = True) -> list[Line]:
        H, W = card_bgr.shape[:2]
        r = self.engine(card_bgr, use_det=True, use_cls=False, use_rec=True)  # явно: recognize_crop меняет режим движка
        lines: list[Line] = []
        if r is None or r.txts is None:
            return lines
        for b, t, s in zip(r.boxes, r.txts, r.scores):
            b = np.asarray(b, np.float32)
            ln = Line(text=t.strip(), score=float(s), box=b)
            ln.x0, ln.x1 = float(b[:, 0].min()) / W, float(b[:, 0].max()) / W
            ln.y0, ln.y1 = float(b[:, 1].min()) / H, float(b[:, 1].max()) / H
            ln.cx, ln.cy = (ln.x0 + ln.x1) / 2, (ln.y0 + ln.y1) / 2
            if ln.text:
                lines.append(ln)
        return merge_row_fragments(lines) if merge else sorted(lines, key=lambda l: (l.cy, l.x0))

    def recognize_crop(self, crop_bgr: np.ndarray):
        """Только распознавание (без детектора) для готового кропа одной строки. -> (text, score)"""
        r = self.engine(crop_bgr, use_det=False, use_cls=False, use_rec=True)
        if r is None or not r.txts:
            return "", 0.0
        return r.txts[0].strip(), float(r.scores[0])


_LABEL = re.compile(r"^\s*\d{1,2}\s*[abcабс]?\s*[.)]?\s*$", re.I)


def _is_label(l: Line) -> bool:
    return bool(_LABEL.match(l.text)) and l.x0 < 0.40


def merge_row_fragments(lines: list[Line], max_gap: float = 0.035) -> list[Line]:
    """Детектор иногда режет одну строку на несколько боксов — склеиваем соседей по строке."""
    lines = sorted(lines, key=lambda l: (l.cy, l.x0))
    rows: list[list[Line]] = []
    for l in lines:
        h = l.y1 - l.y0
        for row in rows:
            r = row[-1]; rh = r.y1 - r.y0
            overlap = min(l.y1, r.y1) - max(l.y0, r.y0)
            if overlap > 0.6 * min(h, rh):
                row.append(l); break
        else:
            rows.append([l])
    out: list[Line] = []
    for row in rows:
        row.sort(key=lambda l: l.x0)
        cur = [row[0]]
        for l in row[1:]:
            if l.x0 - cur[-1].x1 <= max_gap and not (_is_label(l) or _is_label(cur[-1])):
                cur.append(l)
            else:
                out.append(_join(cur)); cur = [l]
        out.append(_join(cur))
    out.sort(key=lambda l: (round(l.cy, 2), l.cx))
    return out


def _join(parts: list[Line]) -> Line:
    if len(parts) == 1:
        return parts[0]
    box_all = np.concatenate([p.box for p in parts])
    x0, y0 = box_all.min(0); x1, y1 = box_all.max(0)
    box = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    n = np.array([max(1, len(p.text)) for p in parts], float)
    ln = Line(text=" ".join(p.text for p in parts), score=float(np.average([p.score for p in parts], weights=n)), box=box)
    ln.x0, ln.x1 = min(p.x0 for p in parts), max(p.x1 for p in parts)
    ln.y0, ln.y1 = min(p.y0 for p in parts), max(p.y1 for p in parts)
    ln.cx, ln.cy = (ln.x0 + ln.x1) / 2, (ln.y0 + ln.y1) / 2
    return ln
