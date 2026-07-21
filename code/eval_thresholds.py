"""
Reusable evaluation for a trained relevance model: loads models/filter_relevance_<group>/
and the held-out test split, reports the P(relevant) distribution and precision/recall/F1
at several decision thresholds (argmax + sweep). Used to judge each training trial.

Usage:
  .\.venv\Scripts\python.exe code\eval_thresholds.py [group]
  (group defaults to mental_health)
"""
import os, csv, sys
import numpy as np, torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
csv.field_size_limit(2**31 - 1)

group = sys.argv[1] if len(sys.argv) > 1 else "mental_health"
MODEL = f"models/filter_relevance_{group}"
SPLIT = "models/train_relevance_data_split"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_text(f):
    out = []
    with open(f, encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if row:
                out.append(row[0])
    return out


def read_labels(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return [int(x.strip()) for x in fh if x.strip() != ""]


def main():
    texts = read_text(os.path.join(SPLIT, f"text_{group}_test.csv"))
    labels = np.array(read_labels(os.path.join(SPLIT, f"label_{group}_test.txt")))
    print(f"[{group}] loading model from {MODEL} ...")
    tok = RobertaTokenizerFast.from_pretrained(MODEL)
    model = RobertaForSequenceClassification.from_pretrained(MODEL).to(DEVICE).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
            probs.extend(model(**enc).logits.softmax(1)[:, 1].tolist())
    probs = np.array(probs)
    print(f"test n={len(labels)}  true_relevant={int(labels.sum())}  true_irrelevant={int((labels==0).sum())}")
    print(f"P(relevant): min={probs.min():.3f} p25={np.percentile(probs,25):.3f} "
          f"median={np.median(probs):.3f} p75={np.percentile(probs,75):.3f} max={probs.max():.3f}")

    def m(pred):
        tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * re / (pr + re) if pr + re else 0.0
        return pr, re, f1, (tp + tn) / len(labels), tp, fp, fn, tn

    print(f"\n{'setting':<12}{'pred_rel':>9}{'prec':>7}{'rec':>7}{'F1':>7}{'acc':>7}   TP/FP/FN/TN")
    for name, thr in [("argmax .50", .5), ("thr .45", .45), ("thr .40", .40),
                      ("thr .35", .35), ("thr .55", .55), ("thr .60", .60)]:
        pred = (probs >= thr).astype(int); pr, re, f1, ac, tp, fp, fn, tn = m(pred)
        print(f"{name:<12}{int(pred.sum()):>9}{pr:>7.2f}{re:>7.2f}{f1:>7.2f}{ac:>7.2f}   {tp}/{fp}/{fn}/{tn}")


if __name__ == "__main__":
    main()
