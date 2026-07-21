"""Score the fixed 1000-comment audit sample with the current mh + sport models.
Writes _audit1000_scored.csv (adds p_mh, p_sport) so we can apply thresholds and sample.
"""
import csv, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
csv.field_size_limit(2**31-1)
DEV="cuda" if torch.cuda.is_available() else "cpu"
IN="data/classified/_audit1000.csv"; OUT="data/classified/_audit1000_scored.csv"

def load(g):
    p=f"models/filter_relevance_{g}"
    m=AutoModelForSequenceClassification.from_pretrained(p).to(DEV).eval()
    if DEV=="cuda": m=m.half()
    return AutoTokenizer.from_pretrained(p), m

def score(tok,m,texts):
    out=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            e=tok(texts[i:i+32],truncation=True,padding=True,max_length=512,return_tensors="pt").to(DEV)
            out.extend(m(**e).logits.float().softmax(1)[:,1].tolist())
    return out

rows=list(csv.DictReader(open(IN,encoding="utf-8-sig",newline="")))
texts=[r.get("text","") or "" for r in rows]
tk,mm=load("mh"); pmh=score(tk,mm,texts)
tk,ms=load("sport"); psp=score(tk,ms,texts)
fields=list(rows[0].keys())+["p_mh","p_sport"]
with open(OUT,"w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r,a,b in zip(rows,pmh,psp):
        r["p_mh"]=f"{a:.4f}"; r["p_sport"]=f"{b:.4f}"; w.writerow(r)
# report gate counts at a few thresholds
for mt in (0.60,0.70,0.75,0.80,0.85):
    rel=sum(1 for a,b in zip(pmh,psp) if a>=mt and b>=0.40)
    print(f"thr mh>={mt:.2f} sp>=0.40 : {rel} / 1000 relevant ({rel/10:.1f}%)")
print(f"scored -> {OUT}")
