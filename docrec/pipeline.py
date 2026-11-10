from __future__ import annotations
import json, time, os
from .sysutil import peak_rss_mb
from pathlib import Path
import cv2
import numpy as np
from .geometry import find_card_quad, warp_card, order_corners
from .doctypes import REGISTRY
from .doctypes import free_text
from . import postproc

DATA = Path(__file__).parent / "data"


def imread(path) -> np.ndarray:
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не удалось прочитать изображение: {path}")
    return img


def imwrite(path, img, q=92):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    buf.tofile(str(path))


class _CardOCR:
    """Единый вызов OCR для карточки: Tesseract (по умолчанию) или PaddleOCR (опционально)."""
    def __init__(self, pipe): self.p = pipe
    def __call__(self, img):
        if self.p.engine == "tess":
            ls = self.p.tess(img, split_gap=0.04)
        else:
            ls = self.p.paddle(img)
        return postproc.filter_noise(ls, None)


class DocumentPipeline:
    """engine: 'tess' (Tesseract 5 best, по умолчанию, ~170 МБ RAM) | 'paddle' (RapidOCR PP-OCRv5, опционально).
    mode: 'auto' | 'card' (удостоверение) | 'page' (лист А4 / произвольный текст + реквизиты)."""
    def __init__(self, threads: int = 0, min_type_conf: float = 0.35, engine: str = "tess", fast: bool = False, mode: str = "auto"):
        self.engine, self.mode, self.threads, self.fast = engine, mode, threads, fast
        self._tess = None; self._paddle = None
        self.min_type_conf = min_type_conf
        self._face = None
        self.ocr = _CardOCR(self)

    @property
    def tess(self):
        if self._tess is None:
            from .ocr_tess import TessEngine
            self._tess = TessEngine(fast=self.fast)
        return self._tess

    @property
    def paddle(self):
        if self._paddle is None:
            from .ocr import OCREngine
            self._paddle = OCREngine(threads=self.threads)
        return self._paddle

    def _rec_fn(self):
        """Перечитывание мелкой латинской строки (кроп -> (текст, уверенность))."""
        if self.engine == "paddle": return self.paddle.recognize_crop
        try: return self.tess.read_latin_line
        except Exception: return None

    def _cell_fn(self):
        try: return self.tess.read_cell
        except Exception: return None

    # --- лицо (YuNet, 230 КБ, ~5 мс) — только для проверки зоны фото
    def _detect_face(self, img):
        if self._face is None:
            self._face = cv2.FaceDetectorYN.create(str(DATA / "yunet.onnx"), "", (320, 320), 0.6)
        h, w = img.shape[:2]
        self._face.setInputSize((w, h))
        _, f = self._face.detect(img)
        return None if f is None else f[np.argmax(f[:, -1])]

    def _classify(self, lines):
        scores = {n: t.classify(lines) for n, t in REGISTRY.items()}
        best = max(scores, key=scores.get)
        return best, scores

    def process(self, path, out_dir) -> dict:
        t0 = time.perf_counter()
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        img = imread(path)
        if self.mode == "page":
            return self._process_page(img, path, out, t0, reason="mode=page")
        res = self._process_card(img, path, out, t0, allow_page=(self.mode == "auto"))
        if res["doc_type"] == "free_text":
            return res
        # слабый результат Tesseract (мелкая латиница, шумный фон) -> добираем PaddleOCR, если он установлен
        if self.engine == "tess" and self._weak(res) and self._paddle_available():
            alt_out = out / "_paddle"; alt_out.mkdir(parents=True, exist_ok=True)
            self.engine = "paddle"
            try: alt = self._process_card(img, path, alt_out, time.perf_counter(), allow_page=False)
            finally: self.engine = "tess"
            res = self._merge(res, alt, out)
        if res["doc_type"] == "unknown" and self.mode == "auto":
            return self._process_page(img, path, out, t0, reason="auto: тип карточки не опознан — читаю как произвольный текст")
        return res

    def _weak(self, res) -> bool:
        if res["doc_type"] == "unknown": return True
        if getattr(REGISTRY[res["doc_type"]], "hybrid_internal", False): return False
        exp = getattr(REGISTRY[res["doc_type"]], "expected_fields", None)
        return bool(exp) and sum(k in res["fields"] for k in exp) < 0.85 * len(exp)

    @staticmethod
    def _paddle_available() -> bool:
        try:
            import rapidocr  # noqa: F401
            return (Path(__file__).parent / "models" / "cyrillic_PP-OCRv5_rec_mobile.onnx").exists()
        except Exception:
            return False

    def _merge(self, base, alt, out):
        """Берём лучший по числу полей результат; недостающие поля добавляем из второго."""
        exp = getattr(REGISTRY.get(alt["doc_type"], None), "expected_fields", None) or []
        score = lambda r: (r["doc_type"] != "unknown", sum(k in r["fields"] for k in exp), r["confidence"])
        best, other = (alt, base) if score(alt) > score(base) else (base, alt)
        if best["doc_type"] == other["doc_type"]:
            for k, v in other["fields"].items():
                if k not in best["fields"]:
                    v = dict(v); v["note"] = (v.get("note", "") + f" [from {other['engine']}]").strip(); best["fields"][k] = v
        best["engine_fallback"] = f"{base['engine']} -> {alt['engine']}: использован {best['engine']} (+поля из {other['engine']})"
        best["timings_sec"]["total_with_fallback"] = round(base["timings_sec"]["total"] + alt["timings_sec"]["total"], 3)
        if best is alt:   # артефакты Paddle-прогона поднимаем в основную папку
            import shutil
            for k, pth in list(alt["files"].items()):
                dst = Path(out) / Path(pth).name
                try: shutil.copyfile(pth, dst); alt["files"][k] = str(dst)
                except OSError: pass
            with open(Path(out) / "result.json", "w", encoding="utf-8") as f: json.dump(alt, f, ensure_ascii=False, indent=2)
        else:
            with open(Path(out) / "result.json", "w", encoding="utf-8") as f: json.dump(base, f, ensure_ascii=False, indent=2)
        import shutil
        shutil.rmtree(Path(out) / "_paddle", ignore_errors=True)
        best["peak_rss_mb"] = round(peak_rss_mb(), 1)
        return best

    def _process_card(self, img, path, out, t0, allow_page=True) -> dict:
        T = {}; H, W = img.shape[:2]

        t = time.perf_counter()
        aspects = tuple(sorted({t_.aspect for t_ in REGISTRY.values()}))
        quad, qscore, qmethod = find_card_quad(img, target_aspects=aspects)
        quad = order_corners(quad)
        T["detect_card"] = time.perf_counter() - t
        if allow_page:
            area = cv2.contourArea(quad.astype(np.float32)) / (W * H)
            if area > 0.55 and qscore < 0.5:
                return self._process_page(img, path, out, t0, reason=f"auto: рамка карточки не найдена (area={area:.2f}, score={qscore:.2f}) — это лист", T=T)

        # размер нормализованной карточки подбираем по ближайшему аспекту (по умолчанию ID-1)
        t = time.perf_counter()
        w_ = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2
        h_ = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2
        asp = max(w_, h_) / max(1e-6, min(w_, h_))
        size = (1026, 648)  # первый проход — всегда ID-1 (85.6x54 мм), аспект по перспективе оценивать ненадёжно
        T["warp"] = time.perf_counter() - t

        def attempt(q):
            """warp -> OCR -> классификация; если не опознано — пробуем повернуть на 180°."""
            c = warp_card(img, q, size)
            t_ = time.perf_counter(); ls = self.ocr(c); b_, sc_ = self._classify(ls)
            T["ocr"] = T.get("ocr", 0) + time.perf_counter() - t_
            rot = 0
            if sc_[b_] < self.min_type_conf:
                t_ = time.perf_counter(); c2 = cv2.rotate(c, cv2.ROTATE_180); ls2 = self.ocr(c2); b2, sc2 = self._classify(ls2)
                T["ocr_rot180"] = T.get("ocr_rot180", 0) + time.perf_counter() - t_
                if sc2[b2] > sc_[b_]:
                    c, ls, b_, sc_, rot = c2, ls2, b2, sc2, 180
            return c, ls, b_, sc_, rot

        card, lines, best, scores, rotation = attempt(quad)
        if scores[best] < self.min_type_conf:
            # второй шанс: возможно, кадр и есть документ (скан/плотный кроп), а найденный «четырёхугольник» — не карточка
            full = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
            if not np.allclose(quad, full, atol=0.02 * max(W, H)):
                r2 = attempt(full)
                if r2[3][r2[2]] > scores[best]:
                    card, lines, best, scores, rotation = r2
                    quad, qmethod, qscore = full, "full_frame(second_chance)", 0.5
        doc_type = best if scores[best] >= self.min_type_conf else "unknown"
        if doc_type != "unknown" and REGISTRY[doc_type].card_size != size:
            # тип требует другой формат (напр. разворот паспорта) — перевыпрямляем и перечитываем
            size = REGISTRY[doc_type].card_size
            card = warp_card(img, quad, size)
            if rotation: card = cv2.rotate(card, cv2.ROTATE_180)
            t = time.perf_counter(); lines = self.ocr(card); T["ocr_retype"] = time.perf_counter() - t
            sc0 = REGISTRY[doc_type].classify(lines)
            if sc0 < 0.5:   # возможно, документ вверх ногами после перевыпрямления
                c2 = cv2.rotate(card, cv2.ROTATE_180); l2 = self.ocr(c2); sc2 = REGISTRY[doc_type].classify(l2)
                if sc2 > sc0 + 0.05: card, lines, sc0, rotation = c2, l2, sc2, (rotation + 180) % 360
            scores[doc_type] = max(scores[doc_type], sc0); best = doc_type

        warnings_ = []
        card_w_src = float((np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2)
        if card_w_src < 700:
            warnings_.append(f"low_resolution: ширина карточки в исходнике ~{card_w_src:.0f}px (рекомендуется >= 900px), точность чтения упадёт")
        if qscore < 0.6:
            warnings_.append("corners_low_confidence: рамка карточки найдена неуверенно")
        res = {
            "image": str(path), "image_size": [W, H],
            "engine": self.engine, "doc_type": doc_type, "doc_type_title": REGISTRY[best].title_ru if doc_type != "unknown" else None,
            "confidence": round(float(scores[best]), 3),
            "type_scores": {k: round(float(v), 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
            "corners": {"order": "TL,TR,BR,BL", "points": [[round(float(x), 1), round(float(y), 1)] for x, y in quad]},
            "corners_confidence": round(qscore, 3), "corners_method": qmethod,
            "card_rotation_applied": rotation,
            "warnings": warnings_,
            "fields": {}, "files": {},
        }
        cw, ch = card.shape[1], card.shape[0]
        photo = None
        if doc_type != "unknown":
            t = time.perf_counter()
            ex = REGISTRY[doc_type].extract(card, lines, rec=self._rec_fn(), cell=self._cell_fn(), ocr=self.ocr, paddle=(self.paddle if self._paddle_available() else None))
            res["fields"] = {k: v.as_dict() for k, v in ex.fields.items()}
            res.update({k: v for k, v in ex.extra.items()})
            T["extract"] = time.perf_counter() - t
            if ex.photo_zone:
                x0, y0, x1, y1 = ex.photo_zone
                px0, py0, px1, py1 = int(x0 * cw), int(y0 * ch), int(x1 * cw), int(y1 * ch)
                photo = card[py0:py1, px0:px1].copy()
                pad = int(0.25 * (px1 - px0))
                ctx = card[max(0, py0 - pad):py1 + pad, max(0, px0 - pad):px1 + pad]
                f = self._detect_face(ctx)
                res["photo"] = {"zone_rel": [round(v, 3) for v in ex.photo_zone], "face_detected": f is not None}
                if f is not None:
                    fx, fy, fw, fh = f[:4]; fcx = (fx + fw / 2 + max(0, px0 - pad)) / cw; fcy = (fy + fh / 2 + max(0, py0 - pad)) / ch
                    inside = x0 <= fcx <= x1 and y0 <= fcy <= y1
                    res["photo"].update({"face_confidence": round(float(f[-1]), 3), "face_center_inside_zone": bool(inside)})
        # --- сохранение
        p_card, p_photo, p_dbg, p_json = out / "card.jpg", out / "photo.jpg", out / "debug.jpg", out / "result.json"
        imwrite(p_card, card)
        res["files"]["card"] = str(p_card)
        if photo is not None and photo.size:
            imwrite(p_photo, photo); res["files"]["photo"] = str(p_photo)
        dbg = img.copy(); th = max(2, int(max(W, H) / 400))
        cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 255, 0), th, cv2.LINE_AA)
        for i, (x, y) in enumerate(quad):
            cv2.circle(dbg, (int(x), int(y)), th * 3, (0, 0, 255), -1)
            cv2.putText(dbg, ["TL", "TR", "BR", "BL"][i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, th / 2.2, (0, 255, 255), max(1, th - 1), cv2.LINE_AA)
        cv2.putText(dbg, f"{doc_type} {res['confidence']:.2f}", (12, 14 + 12 * th), cv2.FONT_HERSHEY_SIMPLEX, th / 2.0, (0, 255, 0), th, cv2.LINE_AA)
        imwrite(p_dbg, dbg); res["files"]["debug"] = str(p_dbg)
        # вспомогательная отладка: карточка с зонами и строками OCR
        dc = card.copy()
        for l in lines:
            cv2.polylines(dc, [l.box.astype(np.int32)], True, (255, 0, 0), 1)
        if doc_type == "driver_license_rf":
            from .doctypes.driver_license import ZONES, PHOTO_ZONE
            for z in list(ZONES.values()) + [PHOTO_ZONE]:
                cv2.rectangle(dc, (int(z[0] * cw), int(z[1] * ch)), (int(z[2] * cw), int(z[3] * ch)), (0, 200, 0), 1)
        imwrite(out / "debug_card.jpg", dc); res["files"]["debug_card"] = str(out / "debug_card.jpg")
        T["total"] = time.perf_counter() - t0
        res["timings_sec"] = {k: round(v, 3) for k, v in T.items()}
        res["peak_rss_mb"] = round(peak_rss_mb(), 1)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        res["files"]["json"] = str(p_json)
        return res

    # =====================================================================  страница / произвольный текст
    def _read_page(self, img):
        if self.engine == "tess":
            return self.tess(img, split_gap=0.035)
        return self.paddle(img)

    def _process_page(self, img, path, out, t0, reason="", T=None) -> dict:
        from .ocr_tess import rotate_ccw
        T = T or {}; H0, W0 = img.shape[:2]
        # 1) ориентация (OSD Tesseract), при неуверенности — проверка по уверенности OCR
        t = time.perf_counter(); deg, oconf = 0, 0.0
        try: deg, oconf = self.tess.detect_rotation(img)
        except Exception: pass
        T["osd"] = time.perf_counter() - t
        applied = deg if (deg and oconf >= 1.5) else 0
        page = rotate_ccw(img, applied) if applied else img
        t = time.perf_counter(); lines = self._read_page(page); T["ocr"] = T.get("ocr", 0) + time.perf_counter() - t
        mconf = float(np.mean([l.score for l in lines])) if lines else 0.0
        orient_note = f"OSD: повернуть против часовой на {deg}° (уверенность {oconf:.1f}); применено {applied}°"
        if (mconf < 0.6 or len(lines) < 3) and oconf < 1.5:
            best = (mconf, 0, page, lines)
            for d in (90, 180, 270):
                p2 = rotate_ccw(img, d); t = time.perf_counter(); l2 = self._read_page(p2); T["ocr_orient_probe"] = T.get("ocr_orient_probe", 0) + time.perf_counter() - t
                m2 = float(np.mean([l.score for l in l2])) if l2 else 0.0
                if m2 > best[0] + 0.05 and len(l2) >= 3: best = (m2, d, p2, l2)
            mconf, applied, page, lines = best
            orient_note += f"; перебор ориентаций по уверенности OCR -> {applied}°"
        H, W = page.shape[:2]
        # 2) постобработка
        t = time.perf_counter()
        dropped: list = []
        n_raw = len(lines)
        lines = postproc.filter_noise(lines, page, dropped)
        lines = postproc.fix_lines(lines)
        lines = postproc.xy_order(lines, aspect=W / H)
        T["postproc"] = time.perf_counter() - t
        t = time.perf_counter()
        an = free_text.analyze(lines)
        T["requisites"] = time.perf_counter() - t
        text = "\n".join(l.text for l in lines)
        conf = float(np.mean([l.score for l in lines])) if lines else 0.0
        res = {
            "image": str(path), "image_size": [W0, H0], "engine": self.engine,
            "doc_type": "free_text", "doc_type_title": "Произвольный текст / страница договора",
            "confidence": round(conf, 3), "route": reason,
            "page_rotation_applied_ccw": applied, "orientation": orient_note, "warnings": [],
            "text": text,
            "lines": [{"text": l.text, **({"raw_text": l.raw_text} if getattr(l, "raw_text", l.text) != l.text else {}),
                       "confidence": round(l.score, 3), "box_rel": [round(l.x0, 4), round(l.y0, 4), round(l.x1, 4), round(l.y1, 4)]} for l in lines],
            "parties": [{k: v for k, v in p.items() if k != "text"} for p in an["parties"]],
            "requisites": an["requisites"], "validation_issues": an["validation_issues"],
            "noise_filtered": dropped, "fields": {}, "files": {},
        }
        if conf < 0.7: res["warnings"].append("low_ocr_confidence: средняя уверенность OCR < 0.7 (плохое фото/мелкий шрифт)")
        if min(W, H) < 900 and max(W, H) < 1500: res["warnings"].append("low_resolution: для страницы желательно ≥1500 px по длинной стороне")
        # 3) файлы
        p_json, p_txt = out / "result.json", out / "text.txt"
        if applied:
            imwrite(out / "page_rotated.jpg", page); res["files"]["page_rotated"] = str(out / "page_rotated.jpg")
        dbg = page.copy(); th = max(1, int(max(W, H) / 900))
        for i, l in enumerate(lines):
            cv2.rectangle(dbg, (int(l.x0 * W), int(l.y0 * H)), (int(l.x1 * W), int(l.y1 * H)), (255, 0, 0), th)
            cv2.putText(dbg, str(i + 1), (int(l.x0 * W) - 6 * th, int(l.y1 * H)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, (0, 0, 255), th, cv2.LINE_AA)
        imwrite(out / "debug.jpg", dbg); res["files"]["debug"] = str(out / "debug.jpg")
        p_txt.write_text(text, encoding="utf-8"); res["files"]["text"] = str(p_txt)
        T["total"] = time.perf_counter() - t0
        res["timings_sec"] = {k: round(v, 3) for k, v in T.items()}
        res["peak_rss_mb"] = round(peak_rss_mb(), 1)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        res["files"]["json"] = str(p_json)
        return res
