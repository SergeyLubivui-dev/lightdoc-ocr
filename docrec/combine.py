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
    out += _categories_check(results)
    return out


def _categories_check(results):
    """Категории: лицевая сторона (цепочка рамок поля 9) против открытых строк таблицы на обороте. Расхождение — повод для проверки, а не ошибка:
    на лицевой печатаются только категории с действующим правом, а таблица оборота может содержать и истёкшие."""
    out = []
    for i, fr in enumerate(results):
        if fr.get("doc_type") != "driver_license_rf": continue
        fc = _val(fr, "categories")
        if not isinstance(fc, list): continue
        for j, bk in enumerate(results):
            if bk.get("doc_type") != "driver_license_back": continue
            rows = _val(bk, "categories")
            if not isinstance(rows, list): continue
            back = [r["category"] for r in rows if isinstance(r, dict) and r.get("open")]
            only_front, only_back = sorted(set(fc) - set(back)), sorted(set(back) - set(fc))
            m = not only_front and not only_back
            out.append(dict(kind="Категории", front_index=i, back_index=j, front=fc, back=back, match=m,
                            note="совпадают" if m else f"расхождение: только на лицевой {only_front}, только на обороте {only_back}"))
    return out
