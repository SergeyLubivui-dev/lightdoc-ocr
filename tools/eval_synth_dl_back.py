"""Оценка оборота водительского удостоверения (синтетика, вымышленные данные): по строкам таблицы — open / date_from / date_to; номер, поле 14, срок действия."""
import sys, os, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec import DocumentPipeline
p = DocumentPipeline(threads=int(os.environ.get("THREADS", "4")))
tot = ok = 0
for f in sorted(glob.glob("samples/synth_dlback[0-9].jpg")):
    r = p.process(f, f"out/eval_{os.path.basename(f)[:-4]}"); gt = json.load(open(f.replace(".jpg", ".gt.json")))
    F = r["fields"]; checks = [("doc_type", r["doc_type"] == "driver_license_back")]
    rows = {c["category"]: c for c in (F.get("categories", {}).get("value") or [])}
    for g in gt["categories"]:
        q = rows.get(g["category"], {})
        checks += [(f"{g['category']}.open", q.get("open") == g["open"])]
        if g["open"]: checks += [(f"{g['category']}.from", q.get("date_from") == g["date_from"]), (f"{g['category']}.to", q.get("date_to") == g["date_to"])]
    checks += [("number", F.get("number", {}).get("value") == gt["number"]), ("field_14", F.get("field_14", {}).get("value") == gt["field_14"]), ("expiry", F.get("licence_expiry", {}).get("value") == gt["expiry"])]
    bad = [k for k, v in checks if not v]; tot += len(checks); ok += len(checks) - len(bad)
    print(f"{os.path.basename(f)}: {len(checks)-len(bad)}/{len(checks)} t={r['timings_sec']['total']}s small_card_scale={r.get('small_card', {}).get('scale')} conf_rows={[c.get('confidence') for c in rows.values() if c.get('open')]}", bad or "")
print(f"TOTAL DL back {ok}/{tot} = {100*ok/tot:.1f}%")
