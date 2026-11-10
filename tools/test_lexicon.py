"""Быстрые проверки привязки к лексиконам (без OCR): искажения вида «Геурлрой» из жалоб, омоглифы, нетронутые редкие слова. Все слова публичные/вымышленные."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec import lexicon as LX
from docrec import vehicle as V

CASES = [  # (лексикон, ввод, ожидание или None = «не трогать»)
    ("sts/vehicle_type", "Геурлрой тягач седельный", "Грузовой тягач седельный"),
    ("sts/vehicle_type", "Срузррой тягач седельный", "Грузовой тягач седельный"),
    ("sts/vehicle_type", "Грурпеой тягач седельный", "Грузовой тягач седельный"),
    ("sts/vehicle_type", "Легковой седан", "Легковой седан"),
    ("sts/vehicle_type", "Zzzzzz qqqq", None),
    ("sts/color", "Бепый", "Белый"), ("sts/color", "Чёрньй", "Чёрный"),
    ("sts/eco_class", "пятьй", "пятый"),
    ("passport/issuer_places", "САРАТОВСКОП", "САРАТОВСКОЙ"),
    ("shared/cities", "ВОЛНОВАХО", "ВОЛНОВАХА"),
    ("shared/first_names", "АЛЕКСАНДР", "АЛЕКСАНДР"),
    ("shared/patronymics", "СЕРГЕЕВИЧ", "СЕРГЕЕВИЧ"),
]
bad = 0
for lex, raw, exp in CASES:
    s = LX.snap(raw, lex, thr=0.3 if lex == "sts/color" else 0.22)
    got = s["value"] if s["status"] in ("exact", "snapped") else None
    ok = (got is not None and LX.fold(got) == LX.fold(exp)) if exp else got is None
    bad += not ok
    print(("OK  " if ok else "FAIL"), lex, repr(raw), "->", repr(got), s["status"], s["norm"])
for raw, exp in [("Геурлрой тягач седельный", "Грузовой тягач седельный")]:
    v, ok, note = V.norm_vehicle_type_lex(raw); good = v == exp and ok; bad += not good
    print("OK  " if good else "FAIL", "vehicle.norm_vehicle_type_lex", repr(raw), "->", repr(v), note)
print(f"lexicons: {LX.total_size()/1024:.0f} KB; failed {bad}")
sys.exit(1 if bad else 0)
