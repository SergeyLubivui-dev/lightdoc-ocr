"""Оценка паспорта на синтетических сценах (вымышленные данные): поля сравниваются с .gt.json."""
import sys, json, os, glob, warnings, re
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec import DocumentPipeline
p = DocumentPipeline(threads=int(os.environ.get("THREADS", "4")))
canon = lambda x: None if x is None else re.sub(r"[\s,]", "", str(x))
tot = ok_all = 0
for f in sorted(glob.glob("samples/synth_pass[0-9].jpg")):
    r = p.process(f, f"out/eval_{os.path.basename(f)[:-4]}")
    gt = json.load(open(f.replace(".jpg", ".gt.json")))
    bad = {k: r["fields"].get(k, {}).get("value") for k, v in gt.items() if canon(r["fields"].get(k, {}).get("value")) != canon(v)}
    tot += len(gt); ok_all += len(gt) - len(bad)
    print(f"{os.path.basename(f)}: type={r['doc_type']} fields {len(gt)-len(bad)}/{len(gt)} t={r['timings_sec']['total']}s rss={r['peak_rss_mb']}MB", bad or "")
print(f"TOTAL passport fields {ok_all}/{tot} = {100*ok_all/tot:.1f}%")
