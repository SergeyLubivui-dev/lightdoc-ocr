"""Оценка на синтетических сценах (вымышленные данные): поля + ошибка углов относительно истинных."""
import sys, json, os, glob, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from docrec import DocumentPipeline
EXPECT = {"surname": "ПЕТРОВ", "surname_lat": "PETROV", "given_names": "АЛЕКСАНДР СЕРГЕЕВИЧ", "given_names_lat": "ALEKSANDR SERGEEVICH",
          "birth_date": "14.03.1987", "birth_place": "ГОР. КАЗАНЬ", "birth_place_lat": "KAZAN", "issue_date": "20.10.2011", "expiry_date": "20.10.2021",
          "issuer": "ГИБДД 1608", "issuer_code": "1608", "number": "16 12 345678", "region": "РЕСП. ТАТАРСТАН (ТАТАРСТАН)", "region_lat": "RESP. TATARSTAN", "categories": ["B"]}
p = DocumentPipeline(threads=int(os.environ.get("THREADS", "4")), engine=os.environ.get("ENGINE", "tess"))
tot = ok_all = 0
for f in sorted(glob.glob("samples/synth[0-9].jpg")):
    r = p.process(f, f"out/eval_{os.path.basename(f)[:-4]}")
    if "corners" not in r:
        print(f"{os.path.basename(f)}: ROUTED TO PAGE ({r['doc_type']}) — карточка не опознана"); tot += len(EXPECT); continue
    gt = np.array(json.load(open(f.replace(".jpg", ".gt.json")))["corners_TL_TR_BR_BL"], np.float32)
    pr = np.array(r["corners"]["points"], np.float32)
    # сопоставляем по ближайшему циклическому сдвигу (карточка могла быть найдена с другим стартовым углом)
    errs = min(np.linalg.norm(np.roll(pr, k, 0) - gt, axis=1).mean() for k in range(4))
    diag = np.hypot(*r["image_size"])
    ok = [k for k, v in EXPECT.items() if k in r["fields"] and r["fields"][k]["value"] == v]
    bad = {k: (r["fields"].get(k, {}).get("value")) for k in EXPECT if k not in ok}
    tot += len(EXPECT); ok_all += len(ok)
    print(f"{os.path.basename(f)}: type={r['doc_type']}({r['confidence']}) corner_err={errs:.1f}px ({100*errs/diag:.2f}% diag) fields {len(ok)}/{len(EXPECT)} t={r['timings_sec']['total']}s rot={r['card_rotation_applied']}")
    if bad: print("    wrong/missing:", bad)
print(f"TOTAL fields {ok_all}/{tot} = {100*ok_all/tot:.1f}%")
