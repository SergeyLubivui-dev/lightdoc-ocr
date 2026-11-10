"""Оценка страниц паспорта без основного разворота (синтетика, вымышленные данные): номера страниц, вид, штампы (тип/орган/дата), перфорация серии/номера."""
import sys, os, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec import DocumentPipeline
p = DocumentPipeline(threads=int(os.environ.get("THREADS", "4")))
tot = ok = 0
for f in sorted(glob.glob("samples/synth_pass_pages[0-9].jpg")):
    r = p.process(f, f"out/eval_{os.path.basename(f)[:-4]}"); gt = json.load(open(f.replace(".jpg", ".gt.json")))
    checks = []
    pg = r.get("pages", [])
    for i, g in enumerate(gt["pages"]):
        q = pg[i] if i < len(pg) else {}
        checks += [(f"p{i+1}.page_number", q.get("page_number") == g["page_number"]), (f"p{i+1}.kind", q.get("kind") == g["kind"]), (f"p{i+1}.stamps", q.get("stamps_n") == g["stamps"])]
        for j, t in enumerate(g.get("stamp_types", [])):
            st = (q.get("stamps") or [{}] * 9)[j] if j < len(q.get("stamps") or []) else {}
            checks += [(f"p{i+1}.stamp{j+1}.type", st.get("type_text") == t), (f"p{i+1}.stamp{j+1}.authority", "Республике Крым в Сакском районе" in ((st.get("authority") or {}).get("value") or ""))]
    perf = r.get("summary", {}).get("perforated_series_number", "")
    dg = sum(a == b for a, b in zip(perf.replace(" ", ""), gt["perforated"].replace(" ", "")))
    checks.append(("perforated(>=8/10 digits)", dg >= 8))
    bad = [k for k, v in checks if not v]; tot += len(checks); ok += len(checks) - len(bad)
    print(f"{os.path.basename(f)}: {len(checks)-len(bad)}/{len(checks)} t={r['timings_sec']['total']}s perforated={perf!r} ({dg}/10)", bad or "")
print(f"TOTAL passport pages {ok}/{tot}")
