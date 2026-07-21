"""Run the Layer-1 gate over a corpus arm and write a classified dataset.

For each month CSV in the input dir, runs both models (GPU, batched), applies the gate
  relevant = (P(mh) >= MH_THR) AND (P(sport) >= SP_THR)
and writes an output CSV = original columns + p_mh, p_sport, mh, sport, relevant.

Resumable: months whose output already exists are skipped. Writes per-month so a crash
loses at most one month.

Usage:
  python code/classify_corpus.py <input_dir> <output_dir> [--limit N]
"""
import csv, glob, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
csv.field_size_limit(2**31 - 1)

MH_THR, SP_THR = 0.50, 0.40      # current deliverable operating point (gate_status)
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")

# Drop comments posted in dedicated mental-health / non-sport subreddits: being in r/depression
# doesn't make a comment about sport, and the study wants sports-situated athlete MH.
DROP_SUBS = {
    "depression", "Anxiety", "anxiety", "mentalhealth", "MentalHealth", "SuicideWatch",
    "offmychest", "TrueOffMyChest", "confession", "confessions", "BPD", "bipolar", "BipolarReddit",
    "ADHD", "adhd", "ptsd", "PTSD", "cptsd", "socialanxiety", "lonely", "ForeverAlone",
    "mentalillness", "getting_over_it", "KindVoice", "depression_help", "SuicideBereavement",
    "selfharm", "SelfHarm", "EatingDisorders", "AnorexiaNervosa", "bulimia", "psychology",
    "Anxietyhelp", "depressionregimens", "MMFB", "sad", "GriefSupport",
}

_M = {}
def load():
    for g in ("mh", "sport"):
        p = f"models/filter_relevance_{g}"
        model = AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE).eval()
        if USE_FP16:
            model = model.half()          # fp16 inference: ~2x faster on GPU
        _M[g] = (AutoTokenizer.from_pretrained(p), model)

def probs(group, texts, tag=""):
    tok, model = _M[group]
    n = len(texts)
    out = [0.0] * n
    # length-sort so each batch pads to a similar length (short comments batch together)
    order = sorted(range(n), key=lambda i: len(texts[i]))
    t0 = time.time(); done = 0
    with torch.no_grad():
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            enc = tok([texts[i] for i in idx], truncation=True, padding=True, max_length=512,
                      return_tensors="pt").to(DEVICE)
            vals = model(**enc).logits.float().softmax(1)[:, 1].tolist()
            for i, v in zip(idx, vals):
                out[i] = v
            done += len(idx)
            if tag and (done % 6400 < BATCH or done == n):
                el = time.time() - t0
                print(f"    [{tag}] {done}/{n}  {done/el:.0f} rows/s", flush=True)
    return np.array(out) if out else np.array([])

def classify_file(inp, outp):
    rows, texts = [], []
    n_read = 0
    with open(inp, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh); fields = rd.fieldnames
        for r in rd:
            n_read += 1
            if r.get("subreddit", "") in DROP_SUBS:      # drop MH/non-sport subreddits entirely
                continue
            rows.append(r); texts.append(r.get("text", "") or "")
    if not rows:
        # still write an empty output so the month counts as done (resumable)
        with open(outp, "w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(fields) + ["p_mh", "p_sport"]).writeheader()
        return n_read, 0
    # cascade: gate is mh AND sport, so only run sport on the mh-passers.
    p_mh = probs("mh", texts, tag="mh " + os.path.basename(inp))
    keep_idx = [i for i, pm in enumerate(p_mh) if pm >= MH_THR]
    p_sp_map = {}
    if keep_idx:
        sp_vals = probs("sport", [texts[i] for i in keep_idx], tag="sport")
        p_sp_map = {i: v for i, v in zip(keep_idx, sp_vals)}
    out_fields = list(fields) + ["p_mh", "p_sport"]
    rel = 0
    # PRUNE: keep only comments the gate marks relevant (athlete mental health)
    with open(outp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields); w.writeheader()
        for i, r in enumerate(rows):
            ps = p_sp_map.get(i)
            if ps is not None and ps >= SP_THR:      # implies p_mh>=MH_THR already
                rel += 1
                r.update(p_mh=f"{p_mh[i]:.4f}", p_sport=f"{ps:.4f}")
                w.writerow(r)
    return n_read, rel      # n_read = total incl. dropped subs, so % reflects the drop

def main():
    inp_dir, out_dir = sys.argv[1], sys.argv[2]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    os.makedirs(out_dir, exist_ok=True)
    load()
    files = sorted(glob.glob(os.path.join(inp_dir, "*.csv")))
    if limit:
        files = files[:limit]
    print(f"[classify] {len(files)} files | device {DEVICE} | thr mh>={MH_THR} sp>={SP_THR}", flush=True)
    tot = totrel = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        outp = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(outp):
            print(f"[{i}/{len(files)}] skip (done) {os.path.basename(f)}", flush=True)
            continue
        ft = time.time()
        n, rel = classify_file(f, outp)
        tot += n; totrel += rel
        dt = time.time() - ft
        rate = n / dt if dt else 0
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {n:,} rows, {rel:,} relevant "
              f"({rel/max(n,1)*100:.1f}%) | {rate:.0f}/s | cum {tot:,} rows {totrel:,} rel", flush=True)
    el = time.time() - t0
    print(f"[classify] DONE {tot:,} rows, {totrel:,} relevant ({totrel/max(tot,1)*100:.1f}%) in {el/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
