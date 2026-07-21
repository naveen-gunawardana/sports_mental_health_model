"""Error analysis on the AND gate (held-out): categorize false positives / negatives by which
model erred, with examples. Babak: error analysis on the gate."""
import csv
import numpy as np, torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
csv.field_size_limit(2**31 - 1)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def predict(g, texts):
    p = f"models/filter_relevance_{g}"
    tok = RobertaTokenizerFast.from_pretrained(p)
    m = RobertaForSequenceClassification.from_pretrained(p).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            e = tok(texts[i:i + 32], truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEV)
            out.extend(m(**e).logits.softmax(1)[:, 1].tolist())
    return np.array(out)


def safe(s):
    return (s or "").replace("\n", " ").encode("ascii", "replace").decode()[:108]


rows = list(csv.DictReader(open("data/data_relevance_ratings/comments/holdout_labeled.csv", encoding="utf-8-sig")))
texts = [r["text"] for r in rows]
mh = np.array([int(r["mh"]) for r in rows]); sp = np.array([int(r["sport"]) for r in rows]); rel = np.array([int(r["relevant"]) for r in rows])
pm = predict("mh", texts); ps = predict("sport", texts)
gate = ((pm >= 0.5) & (ps >= 0.5)).astype(int)
fp = [i for i in range(len(rows)) if gate[i] == 1 and rel[i] == 0]
fn = [i for i in range(len(rows)) if gate[i] == 0 and rel[i] == 1]
print(f"AND gate @ mh.5/sp.5:  FP={len(fp)}  FN={len(fn)}")
mh_err = sum(1 for i in fp if mh[i] == 0); sp_err = sum(1 for i in fp if sp[i] == 0); both = sum(1 for i in fp if mh[i] == 0 and sp[i] == 0)
print(f"  FP causes: mh model wrong (mh_true=0): {mh_err} | sport model wrong (sport_true=0): {sp_err} | both wrong: {both}")
print("\n-- FALSE POSITIVES (gate says relevant, truth = not) --")
for i in fp[:9]:
    print(f"  mh_t={mh[i]} sp_t={sp[i]} p_mh={pm[i]:.2f} p_sp={ps[i]:.2f} | {safe(rows[i]['text'])}")
print("\n-- FALSE NEGATIVES (gate says not, truth = relevant) --")
for i in fn[:6]:
    print(f"  mh_t={mh[i]} sp_t={sp[i]} p_mh={pm[i]:.2f} p_sp={ps[i]:.2f} | {safe(rows[i]['text'])}")
