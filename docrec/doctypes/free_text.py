"""Универсальный тип «произвольный текст / страница договора»: упорядоченный текст + реквизиты с проверкой."""
from __future__ import annotations
import re
import numpy as np
from ..ocr import Line
from ..textutils import fold
from ..postproc import PARTY_HEADERS
from .. import requisites as rq

_HDR = {fold(h): h for h in PARTY_HEADERS}


def _is_header(text: str) -> str | None:
    t = re.sub(r"^[^\wА-Яа-я]+|[^\wА-Яа-я]+$", "", text)
    f = fold(t)
    return _HDR.get(f)


def analyze(lines: list[Line]) -> dict:
    """lines — уже очищены, исправлены и упорядочены (xy_order)."""
    groups: list[dict] = [{"role": None, "lines": []}]
    for l in lines:
        h = _is_header(l.text)
        if h and len(l.text) <= len(h) + 4:
            groups.append({"role": h.capitalize(), "lines": []})
        else:
            groups[-1]["lines"].append(l)
    parties = []; all_req: list[dict] = []
    for i, g in enumerate(groups):
        if not g["lines"]: continue
        text = " ".join(l.text for l in g["lines"])
        conf = float(np.mean([l.score for l in g["lines"]]))
        label = g["role"] or ("(без заголовка)" if len(groups) > 1 and i == 0 else None)
        reqs = rq.extract(text, conf=conf, party=g["role"])
        name = next((r["value"] for r in reqs if r["type"] == "organization"), None) or next((r["value"] for r in reqs if r["type"] == "person"), None)
        if not reqs and g["role"] is None:
            pass
        parties.append({"role": g["role"], "name": name, "text": text, "requisites": reqs})
        all_req += reqs
    return {"parties": parties, "requisites": all_req, "validation_issues": rq.cross_check(all_req)}
