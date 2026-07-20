"""
Grid-search the two decision thresholds (mh, sport) for the AND gate, to find the best
operating point from the existing models without retraining. Computes model probabilities
once on the shared held-out set, then sweeps thresholds in memory.
Usage:  .\.venv\Scripts\python.exe code\eval_and_gate_grid.py
"""
import os, csv
import numpy as np, torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
csv.field_size_limit(2**31 - 1)
SPLIT = "models/train_relevance_data_split"


def read_text(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return [r[0] for r in csv.reader(fh) if r]


def read_labels(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return np.array([int(x.strip()) for x in fh if x.strip() != ""])


def predict(group, texts):
    p = f"models/filter_relevance_{group}"
    tok = RobertaTokenizerFast.from_pretrained(p)
    model = RobertaForSequenceClassification.from_pretrained(p).to("cpu").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], truncation=True, padding=True, max_length=512, return_tensors="pt")
            out.extend(model(**enc).logits.softmax(1)[:, 1].tolist())
    return np.array(out)


def score(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
    return pr, re, f1, (tp + tn) / len(y)


texts = read_text(os.path.join(SPLIT, "text_mh_test.csv"))
mh_true = read_labels(os.path.join(SPLIT, "label_mh_test.txt"))
sport_true = read_labels(os.path.join(SPLIT, "label_sport_test.txt"))
rel = (mh_true & sport_true).astype(int)
p_mh = predict("mh", texts)
p_sp = predict("sport", texts)

rows = []
for mt in [0.5, 0.55, 0.6, 0.65, 0.7]:
    for st in [0.35, 0.4, 0.45, 0.5, 0.55]:
        gate = ((p_mh >= mt) & (p_sp >= st)).astype(int)
        pr, re, f1, ac = score(gate, rel)
        rows.append((f1, mt, st, pr, re, ac))
rows.sort(reverse=True)
print(f"relevant base rate {rel.mean():.2f}  (always-positive F1={2*rel.mean()/(rel.mean()+1):.2f})")
print(f"{'F1':>6}{'mh_thr':>8}{'sp_thr':>8}{'prec':>7}{'rec':>7}{'acc':>7}")
for f1, mt, st, pr, re, ac in rows[:10]:
    print(f"{f1:>6.2f}{mt:>8.2f}{st:>8.2f}{pr:>7.2f}{re:>7.2f}{ac:>7.2f}")
