"""
Precision-favored threshold grid for the AND gate on the fresh held-out set (GPU).
Prints the full precision/recall/F1 frontier over (mh_thr, sport_thr) so we can pick a
conservative operating point (Babak: favor precision). Highlights best-F1 and the highest
precision reachable at recall >= 0.6.
Usage:  .\.venv\Scripts\python.exe code\eval_holdout_grid.py
"""
import csv
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
csv.field_size_limit(2**31 - 1)
HOLD = "data/data_relevance_ratings/comments/holdout_labeled.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(group, texts):
    p = f"models/filter_relevance_{group}"
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
            out.extend(model(**enc).logits.softmax(1)[:, 1].tolist())
    return np.array(out)


def score(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
    return pr, re, f1, (tp + tn) / len(y)


texts, rel = [], []
with open(HOLD, encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        texts.append(r["text"]); rel.append(int(r["relevant"]))
rel = np.array(rel)
p_mh = predict("mh", texts); p_sp = predict("sport", texts)
print(f"held-out n={len(rel)}  relevant={int(rel.sum())} ({rel.mean():.2f})  "
      f"P(mh)max={p_mh.max():.2f} P(sport)max={p_sp.max():.2f}\n")

rows = []
for mt in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    for st in [0.40, 0.45, 0.50, 0.55]:
        gate = ((p_mh >= mt) & (p_sp >= st)).astype(int)
        pr, re, f1, ac = score(gate, rel)
        rows.append((mt, st, pr, re, f1, ac))

print(f"{'mh_thr':>7}{'sp_thr':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'acc':>7}")
for mt, st, pr, re, f1, ac in rows:
    print(f"{mt:>7.2f}{st:>7.2f}{pr:>7.2f}{re:>7.2f}{f1:>7.2f}{ac:>7.2f}")

best_f1 = max(rows, key=lambda r: r[4])
prec_at_rec = [r for r in rows if r[3] >= 0.60]
best_prec = max(prec_at_rec, key=lambda r: r[2]) if prec_at_rec else None
print(f"\nbest F1:            mh {best_f1[0]:.2f} / sp {best_f1[1]:.2f}  ->  prec {best_f1[2]:.2f} rec {best_f1[3]:.2f} F1 {best_f1[4]:.2f}")
if best_prec:
    print(f"best prec @ rec>=.60: mh {best_prec[0]:.2f} / sp {best_prec[1]:.2f}  ->  prec {best_prec[2]:.2f} rec {best_prec[3]:.2f} F1 {best_prec[4]:.2f}")

# write status for the live monitor (watch_training.py GATE panel)
with open("run_logs/gate_status.txt", "w", encoding="utf-8") as fh:
    fh.write(f"prec={best_f1[2]:.2f} rec={best_f1[3]:.2f} f1={best_f1[4]:.2f} "
             f"thr=mh{best_f1[0]:.2f}/sp{best_f1[1]:.2f} pmhmax={p_mh.max():.2f}\n")
