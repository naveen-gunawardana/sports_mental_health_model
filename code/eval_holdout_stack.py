"""
Stacking meta-classifier for the Layer-1 gate vs. the hard AND gate (GPU inference).

The hard AND gate binarizes each model at a threshold, then ANDs — throwing away the
continuous confidence and any interaction between the two signals. Here we instead feed
[p_mh, p_sport, p_mh*p_sport] into a logistic meta-learner that learns the joint boundary.

To keep the meta-learner honest on only 292 held-out rows, we score STRATIFIED 5-fold
OUT-OF-FOLD predictions: each row is predicted by a meta-learner that never saw it. The
hard AND gate's best-F1 is grid-searched on ALL 292 (i.e. slightly optimistic), so any
OOF win over it is a real win.

Usage:  .\.venv\Scripts\python.exe code\eval_holdout_stack.py
"""
import csv
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
y = np.array(rel)
p_mh = predict("mh", texts); p_sp = predict("sport", texts)
print(f"held-out n={len(y)}  relevant={int(y.sum())} ({y.mean():.2f})  "
      f"P(mh)max={p_mh.max():.2f} P(sport)max={p_sp.max():.2f}\n")

# ---- baseline: hard AND gate, grid-best F1 (optimistic: tuned on all rows) ----
best = (0, 0, 0, 0, 0, 0)
for mt in np.arange(0.30, 0.75, 0.025):
    for st in np.arange(0.30, 0.70, 0.025):
        g = ((p_mh >= mt) & (p_sp >= st)).astype(int)
        pr, re, f1, ac = score(g, y)
        if f1 > best[4]:
            best = (mt, st, pr, re, f1, ac)
print(f"HARD AND gate (grid-best on all rows): mh>={best[0]:.3f} sp>={best[1]:.3f}"
      f"  ->  prec {best[2]:.3f} rec {best[3]:.3f} F1 {best[4]:.3f} acc {best[5]:.3f}")

# ---- stacking meta-learners, honest 5-fold out-of-fold ----
FEATURES = {
    "logistic [p_mh,p_sp]":          np.column_stack([p_mh, p_sp]),
    "logistic [p_mh,p_sp,product]":  np.column_stack([p_mh, p_sp, p_mh * p_sp]),
    "logistic [logit_mh,logit_sp,product]": np.column_stack([
        np.log(np.clip(p_mh, 1e-6, 1 - 1e-6) / (1 - np.clip(p_mh, 1e-6, 1 - 1e-6))),
        np.log(np.clip(p_sp, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sp, 1e-6, 1 - 1e-6))),
        p_mh * p_sp,
    ]),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
print()
for name, X in FEATURES.items():
    oof = np.zeros(len(y))            # out-of-fold P(relevant)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    # pick the threshold on the OOF probs that maximizes F1 (this is the honest gate)
    bt = max(np.arange(0.2, 0.85, 0.01),
             key=lambda t: score((oof >= t).astype(int), y)[2])
    pr, re, f1, ac = score((oof >= bt).astype(int), y)
    print(f"{name:<40} thr={bt:.2f}  prec {pr:.3f} rec {re:.3f} F1 {f1:.3f} acc {ac:.3f}")
