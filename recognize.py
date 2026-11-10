#!/usr/bin/env python
"""CLI: python recognize.py <image> --out <dir> [--threads N] [--print-lines]"""
import argparse, json, sys, logging, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
from docrec import DocumentPipeline


def main():
    ap = argparse.ArgumentParser(description="Распознавание российских документов (офлайн, CPU)")
    ap.add_argument("image"); ap.add_argument("--out", default="out")
    ap.add_argument("--threads", type=int, default=0, help="потоков onnxruntime (0 = авто)")
    ap.add_argument("--engine", choices=["tess", "paddle"], default="tess", help="tess = Tesseract 5 best (по умолчанию), paddle = RapidOCR PP-OCRv5 (нужен rapidocr)")
    ap.add_argument("--fast", action="store_true", help="tessdata_fast (быстрее, чуть менее точно)")
    ap.add_argument("--mode", choices=["auto", "card", "page"], default="auto", help="card = удостоверение, page = лист/договор (текст+реквизиты), auto = сам решает")
    ap.add_argument("--quiet", action="store_true", help="не печатать JSON в stdout")
    a = ap.parse_args()
    res = DocumentPipeline(threads=a.threads, engine=a.engine, fast=a.fast, mode=a.mode).process(a.image, a.out)
    if not a.quiet:
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
