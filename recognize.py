#!/usr/bin/env python
"""CLI: python recognize.py <image> [<image2> …] --out <dir> [--threads N]
Несколько файлов (лицо + оборот СТС/ВУ) -> {"documents":[…], "cross_check":[…]}, номер лица и оборота сверяется."""
import argparse, json, sys, logging, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
from docrec import DocumentPipeline


def main():
    ap = argparse.ArgumentParser(description="Распознавание российских документов (офлайн, CPU)")
    ap.add_argument("image", nargs="+", help="одно или несколько изображений (лицо+оборот: общий JSON и сверка номера)"); ap.add_argument("--out", default="out")
    ap.add_argument("--threads", type=int, default=0, help="потоков onnxruntime (0 = авто)")
    ap.add_argument("--engine", choices=["tess", "paddle"], default="tess", help="tess = Tesseract 5 best (по умолчанию), paddle = RapidOCR PP-OCRv5 (нужен rapidocr)")
    ap.add_argument("--fast", action="store_true", help="tessdata_fast (быстрее, чуть менее точно)")
    ap.add_argument("--mode", choices=["auto", "card", "page", "passport"], default="auto", help="card = удостоверение, page = лист/договор (текст+реквизиты), passport = паспорт РФ, auto = сам решает")
    ap.add_argument("--quiet", action="store_true", help="не печатать JSON в stdout")
    a = ap.parse_args()
    pipe = DocumentPipeline(threads=a.threads, engine=a.engine, fast=a.fast, mode=a.mode)
    if len(a.image) == 1:
        res = pipe.process(a.image[0], a.out)
    else:
        from docrec.combine import cross_check
        docs = [pipe.process(p, os.path.join(a.out, f"{i + 1}_{os.path.splitext(os.path.basename(p))[0]}")) for i, p in enumerate(a.image)]
        res = {"documents": docs, "cross_check": cross_check(docs)}
        os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "combined.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not a.quiet:
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
