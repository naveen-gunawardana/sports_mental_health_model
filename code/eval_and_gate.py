"""
Evaluate the two-model Layer-1 gate: relevant = (mh model = 1) AND (sport model = 1).

The mh and sport groups are written from the same comment order and split with the same
seed, so text_mh_test.csv and text_sport_test.csv are the SAME held-out comments. We use
those texts (held out from BOTH models), take true relevance = (mh_label AND sport_label)
from the two label files, run both models, AND their predictions, and score honestly against
the always-positive baseline.

Usage:  .\.venv\Scripts\python.exe code\eval_and_gate.py
"""
import os, csv
import numpy as np, torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
csv.field_size_limit(2**31 - 1)

SPLIT = "models/train_relevance_data_split"


def read_text(f):
    out = []
    with open(f, encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if row:
                out.append(row[0])
    return out


def read_labels(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return np.array([int(x.strip()) for x in fh if x.strip() != ""])


def predict(group, texts):
    path = f"models/filter_relevance_{group}"
    tok = RobertaTokenizerFast.from_pretrained(path)
    model = RobertaForSequenceClassification.from_pretrained(path).to("cpu").eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], truncation=True, padding=True, max_length=512, return_tensors="pt")
            probs.extend(model(**enc).logits.softmax(1)[:, 1].tolist())
    return np.array(probs)


def metrics(pred, labels):
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
    return pr, re, f1, (tp + tn) / len(labels), tp, fp, fn, tn


def baseline_f1(labels):
    p = labels.mean()
    return 2 * p * 1.0 / (p + 1.0), p


def main():
    texts = read_text(os.path.join(SPLIT, "text_mh_test.csv"))
    sport_texts = read_text(os.path.join(SPLIT, "text_sport_test.csv"))
    mh_true = read_labels(os.path.join(SPLIT, "label_mh_test.txt"))
    sport_true = read_labels(os.path.join(SPLIT, "label_sport_test.txt"))
    same = (len(texts) == len(sport_texts) and all(a == b for a, b in zip(texts, sport_texts)))
    print(f"common test n={len(texts)}   mh/sport splits identical: {same}")
    if not same:
        print("WARNING: mh and sport test splits differ — AND-gate eval is not on a common set.")
    rel_true = (mh_true & sport_true).astype(int)

    p_mh = predict("mh", texts)
    p_sp = predict("sport", texts)
    print(f"P(mh):    min={p_mh.min():.2f} med={np.median(p_mh):.2f} max={p_mh.max():.2f}")
    print(f"P(sport): min={p_sp.min():.2f} med={np.median(p_sp):.2f} max={p_sp.max():.2f}")

    for thr in (0.5, 0.45, 0.4):
        mh_pred = (p_mh >= thr).astype(int)
        sp_pred = (p_sp >= thr).astype(int)
        gate = (mh_pred & sp_pred).astype(int)
        bf, bp = baseline_f1(rel_true)
        print(f"\n=== threshold {thr} ===")
        print(f"  always-positive baseline on relevant: F1={bf:.2f} acc={bp:.2f}")
        print(f"  {'model':<12}{'prec':>7}{'rec':>7}{'F1':>7}{'acc':>7}   TP/FP/FN/TN")
        for name, pred, truth in [("mh", mh_pred, mh_true), ("sport", sp_pred, sport_true), ("AND gate", gate, rel_true)]:
            pr, re, f1, ac, tp, fp, fn, tn = metrics(pred, truth)
            print(f"  {name:<12}{pr:>7.2f}{re:>7.2f}{f1:>7.2f}{ac:>7.2f}   {tp}/{fp}/{fn}/{tn}")


if __name__ == "__main__":
    main()
