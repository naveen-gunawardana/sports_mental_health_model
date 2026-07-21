"""Single-process driver: prune all corpus arms with the gate, one arm after another.
One python process (no wrapper to orphan). Resumable: skips months already written.
"""
import glob, os, time
import classify_corpus as c

ARMS = [
    ("comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched",
     "data/classified/matched_2018_2022"),
    ("comments/2023/comments_2023/filtered_subreddit_keywords/matched",
     "data/classified/matched_2023"),
    ("comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/baseline",
     "data/classified/baseline_2018_2022"),
]

c.load()
print(f"[driver] models loaded on {c.DEVICE} | thr mh>={c.MH_THR} sp>={c.SP_THR}", flush=True)
gtot = grel = 0
t0 = time.time()
for ai, (inp, out) in enumerate(ARMS, 1):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(inp, "*.csv")))
    print(f"===== ARM {ai}/{len(ARMS)}: {out}  ({len(files)} files) =====", flush=True)
    for i, f in enumerate(files, 1):
        outp = os.path.join(out, os.path.basename(f))
        if os.path.exists(outp):
            print(f"[{ai}:{i}/{len(files)}] skip {os.path.basename(f)}", flush=True)
            continue
        n, rel = c.classify_file(f, outp)
        gtot += n; grel += rel
        print(f"[{ai}:{i}/{len(files)}] {os.path.basename(f)}: {n:,} rows, {rel:,} relevant "
              f"({rel/max(n,1)*100:.1f}%) | cum {gtot:,} rows {grel:,} rel", flush=True)
print(f"===== ALL ARMS DONE: {gtot:,} rows -> {grel:,} relevant in {(time.time()-t0)/60:.1f} min =====", flush=True)
