#!/usr/bin/env sh
# Linux/macOS: python 3.11+
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
echo "OK. Usage: .venv/bin/python recognize.py <image> --out out/"
