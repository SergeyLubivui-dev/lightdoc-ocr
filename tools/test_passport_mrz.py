"""Проверки отсечения МРЗ и мусора подписей в полях паспорта (вымышленные строки, без OCR)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec.doctypes.passport import is_mrz_like, strip_mrz_text, clean_place

MRZ = ["P<RUSTESTOV<<IVAN<PETROVICH<<<<<<<<<<<<<<<<<<<", "PIRUSTESTOVLLIVANLPETROVI3LLLLLLLLLLL", "1234567890RUS8501157M1503031690001<<12 84",
       "4507654321RUS8001018M2905123123456 70", "PNRUSTESTOV<<IVAN<PETROVICH"]
NOT = ["ГОР. ТВЕРЬ", "ТВЕРСКОЙ ОБЛАСТИ", "15.06.2011", "690-001", "ИВАН ПЕТРОВИЧ", "Паспорт выдан"]
bad = 0
for t in MRZ:
    ok = is_mrz_like(t); bad += not ok; print("OK  " if ok else "FAIL", "mrz-like", t)
for t in NOT:
    ok = not is_mrz_like(t); bad += not ok; print("OK  " if ok else "FAIL", "not mrz", t)
cases = [("АТА Д. НОВОЕ ТЕСТОВО ИВАНОВСКИЙ РАЙОН ТВЕРСКАЯ АССР 4507654321RUS8001018M2905123123456 70", "Д. НОВОЕ ТЕСТОВО ИВАНОВСКИЙ РАЙОН ТВЕРСКАЯ АССР"),
         ("ТА ГОР. ТВЕРЬ P<RUSTESTOV<<IVAN", "ГОР. ТВЕРЬ"), ("ГОР. КАЗАНЬ РЕСП. ТАТАРСТАН", "ГОР. КАЗАНЬ РЕСП. ТАТАРСТАН"), ("Д. МАЛИНОВКА", "Д. МАЛИНОВКА"), ("МЕСТО ГОР. ТВЕРЬ", "ГОР. ТВЕРЬ")]
for raw, exp in cases:
    got = clean_place(raw); ok = got == exp; bad += not ok; print("OK  " if ok else "FAIL", "place", repr(raw), "->", repr(got))
print("failed", bad); sys.exit(1 if bad else 0)
