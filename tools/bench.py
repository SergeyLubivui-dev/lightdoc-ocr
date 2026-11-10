"""python tools/bench.py <image> [--threads N] [--runs 5] — холодный старт, время на изображение, пик RAM."""
import sys, time, argparse, os, warnings, statistics
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("image", nargs="+"); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--runs", type=int, default=5)
a = ap.parse_args()
t = time.perf_counter(); from docrec import DocumentPipeline; from docrec.sysutil import peak_rss_mb, n_cores; p = DocumentPipeline(threads=a.threads); init = time.perf_counter() - t
rss_init = peak_rss_mb()
times = []
for i in range(a.runs):
    for im in a.image:
        t = time.perf_counter(); p.process(im, os.path.join(__import__("tempfile").gettempdir(),"bench_out")); times.append(time.perf_counter() - t)
print(f"threads={a.threads} cores={n_cores()} init={init:.2f}s first={times[0]:.2f}s median={statistics.median(times):.2f}s min={min(times):.2f}s max={max(times):.2f}s rss_after_init={rss_init:.0f}MB peak_rss={peak_rss_mb():.0f}MB")
