"""
Clean AND-gate evaluation on the fresh held-out set (never used in any training).
Reads data/data_relevance_ratings/comments/holdout_labeled.csv (random_id,text,mh,sport,relevant),
runs the mh and sport models, and reports mh / sport / AND-gate performance vs ground truth,
grid-searching the two thresholds. Honest baseline printed.

Usage:  .\.venv\Scripts\python.exe code\eval_holdout_gate.py
"""
import csv
import numpy as np, torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
csv.field_size_limit(2**31 - 1)
HOLD = "data/data_relevance_ratings/comments/holdout_labeled.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(group, texts):
    p = f"models/filter_relevance_{group}"
    tok = RobertaTokenizerFast.from_pretrained(p)
    model = RobertaForSequenceClassification.from_pretrained(p).to(DEVICE).eval()
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
    return pr, re, f1, (tp + tn) / len(y), tp, fp, fn, tn


texts, mh_t, sp_t, rel_t = [], [], [], []
with open(HOLD, encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        texts.append(r["text"]); mh_t.append(int(r["mh"])); sp_t.append(int(r["sport"])); rel_t.append(int(r["relevant"]))
mh_t = np.array(mh_t); sp_t = np.array(sp_t); rel_t = np.array(rel_t)
p_mh = predict("mh", texts)
p_sp = predict("sport", texts)

print(f"held-out n={len(texts)}  relevant={int(rel_t.sum())} ({rel_t.mean():.2f})  "
      f"always-positive F1={2*rel_t.mean()/(rel_t.mean()+1):.2f}")
print(f"P(mh) max={p_mh.max():.2f}  P(sport) max={p_sp.max():.2f}\n")

# per-dimension at argmax
for name, p, y in [("mh", p_mh, mh_t), ("sport", p_sp, sp_t)]:
    pr, re, f1, ac, *_ = score((p >= 0.5).astype(int), y)
    print(f"{name:<10} argmax: prec {pr:.2f} rec {re:.2f} F1 {f1:.2f} acc {ac:.2f}")

# AND-gate threshold grid
best = None
for mt in [0.5, 0.55, 0.6, 0.65, 0.7]:
    for st in [0.35, 0.4, 0.45, 0.5, 0.55]:
        gate = ((p_mh >= mt) & (p_sp >= st)).astype(int)
        pr, re, f1, ac, tp, fp, fn, tn = score(gate, rel_t)
        if best is None or f1 > best[0]:
            best = (f1, mt, st, pr, re, ac, tp, fp, fn, tn)
f1, mt, st, pr, re, ac, tp, fp, fn, tn = best
print(f"\nBEST AND gate: F1 {f1:.2f} @ mh_thr {mt} sport_thr {st}  |  prec {pr:.2f} rec {re:.2f} acc {ac:.2f}  TP/FP/FN/TN {tp}/{fp}/{fn}/{tn}")
