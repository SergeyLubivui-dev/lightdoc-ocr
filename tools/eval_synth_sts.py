"""Оценка СТС (лицо и оборот) на синтетических сценах (вымышленные данные). марка/модель — две строки бланка (ru/lat/combined)."""
import sys, json, os, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docrec import DocumentPipeline
p = DocumentPipeline(threads=int(os.environ.get("THREADS", "4")))
tot = ok_all = 0
for f in sorted(glob.glob("samples/synth_sts[0-9]*.jpg") + glob.glob("samples/synth_sts_back[0-9]*.jpg")):
    r = p.process(f, f"out/eval_{os.path.basename(f)[:-4]}")
    gt = json.load(open(f.replace(".jpg", ".gt.json")))["fields"]
    bad = {}
    for k, v in gt.items():
        got = r["fields"].get(k, {}).get("value")
        if got != v: bad[k] = got
    tot += len(gt); ok_all += len(gt) - len(bad)
    print(f"{os.path.basename(f)}: type={r['doc_type']} fields {len(gt)-len(bad)}/{len(gt)} t={r['timings_sec']['total']}s rss={r['peak_rss_mb']}MB", bad or "")
print(f"TOTAL STS fields {ok_all}/{tot} = {100*ok_all/tot:.1f}%")
