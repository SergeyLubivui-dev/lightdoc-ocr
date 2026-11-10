"""Несколько изображений одного комплекта (лицо + оборот СТС / ВУ): общий JSON и сверка номера лицевой и оборотной сторон."""
from __future__ import annotations
import re

_PAIRS = [({"sts"}, {"sts_back"}, "sts_number", "sts_number", "Номер СТС"), ({"driver_license_rf"}, {"driver_license_back"}, "number", "number", "Номер ВУ")]


def _val(res, key):
    f = (res.get("fields") or {}).get(key)
    return (f or {}).get("value") if isinstance(f, dict) else None


def _norm(n):
    return re.sub(r"\D", "", str(n or ""))


def cross_check(results: list[dict]) -> list[dict]:
    """-> список сверок: {kind, front, back, match, note}. match: True | False | None (одной из сторон нет / не прочитан)."""
    out = []
    for fr_t, bk_t, fk, bk, title in _PAIRS:
        fronts = [(i, r) for i, r in enumerate(results) if r.get("doc_type") in fr_t]
        backs = [(i, r) for i, r in enumerate(results) if r.get("doc_type") in bk_t]
        for i, fr in fronts:
            for j, bk_r in backs:
                a, b = _val(fr, fk), _val(bk_r, bk)
                m = (_norm(a) == _norm(b)) if (a and b) else None
                out.append(dict(kind=title, front_index=i, back_index=j, front=a, back=b, match=m,
                                note=("совпадают" if m else "НЕ СОВПАДАЮТ — вероятно, лицо и оборот от разных документов или ошибка чтения" if m is False else "номер прочитан не на обеих сторонах")))
    return out
