"""Layer-1 gate demo — type a comment in your browser, see if it's classified as
athlete mental health.

Loads the two trained twitter-roberta models once, then serves a tiny local web page.
No Flask/extra deps (stdlib http.server only). Runs the REAL gate:
    relevant = (P(mh) >= 0.45) AND (P(sport) >= 0.40)

Run:   .venv\\Scripts\\python.exe code\\demo_gate.py
Then open the URL it prints (default http://127.0.0.1:8000).
(or double-click demo_gate.bat)
"""
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MH_THR, SP_THR = 0.45, 0.40           # best-F1 operating point on the held-out grid
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 8000

print(f"[demo] loading models on {DEVICE} ...")
_MODELS = {}
for g in ("mh", "sport"):
    p = f"models/filter_relevance_{g}"
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE).eval()
    _MODELS[g] = (tok, model)
print("[demo] ready.")


def prob(group, text):
    tok, model = _MODELS[group]
    with torch.no_grad():
        enc = tok([text], truncation=True, padding=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        return float(model(**enc).logits.softmax(1)[0, 1])


def classify(text):
    p_mh, p_sp = prob("mh", text), prob("sport", text)
    relevant = (p_mh >= MH_THR) and (p_sp >= SP_THR)
    return {"p_mh": p_mh, "p_sport": p_sp,
            "mh_yes": p_mh >= MH_THR, "sport_yes": p_sp >= SP_THR,
            "relevant": relevant}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Layer-1 gate demo</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 16px;color:#1a1a2e}}
 h1{{font-size:1.4rem}} .sub{{color:#666;margin-top:-8px}}
 textarea{{width:100%;height:120px;font-size:1rem;padding:10px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box}}
 button{{margin-top:10px;padding:10px 20px;font-size:1rem;border:0;border-radius:8px;background:#3a5;color:#fff;cursor:pointer}}
 .verdict{{font-size:1.3rem;font-weight:700;margin:18px 0 6px}}
 .yes{{color:#1a7f37}} .no{{color:#b42318}}
 .bar{{height:22px;border-radius:6px;background:#eee;overflow:hidden;margin:4px 0}}
 .fill{{height:100%;background:#3a5}} .fill.lo{{background:#c33}}
 .row{{margin:12px 0}} code{{background:#f2f2f5;padding:2px 6px;border-radius:4px}}
 .thr{{color:#888;font-size:.85rem}}
</style></head><body>
<h1>Layer-1 relevance gate</h1>
<p class="sub">Is this comment about <b>athlete mental health</b>? &nbsp;(mh AND sport)</p>
<form method="post">
<textarea name="comment" placeholder="Paste a Reddit comment...">{comment}</textarea>
<button type="submit">Classify</button>
</form>
{result}
<p class="thr">Gate: relevant if P(mh) &ge; {mh_thr} AND P(sport) &ge; {sp_thr}. Model: cardiffnlp/twitter-roberta-base, held-out F1 0.92.</p>
</body></html>"""


def bar(p, thr):
    pct = int(round(p * 100))
    cls = "fill" if p >= thr else "fill lo"
    return f'<div class="bar"><div class="{cls}" style="width:{pct}%"></div></div>'


def result_html(text):
    if not text.strip():
        return ""
    r = classify(text)
    v = ('<div class="verdict yes">&#10003; RELEVANT &mdash; athlete mental health</div>'
         if r["relevant"] else
         '<div class="verdict no">&#10007; not relevant</div>')
    def line(label, p, yes, thr):
        tag = "yes" if yes else "no"
        mark = "&#10003;" if yes else "&#10007;"
        return (f'<div class="row"><b>{label}</b>: {p:.2f} '
                f'<span class="{tag}">{mark} {"pass" if yes else "below"} (thr {thr})</span>'
                f'{bar(p, thr)}</div>')
    return (v
            + line("P(mental health)", r["p_mh"], r["mh_yes"], MH_THR)
            + line("P(sport)", r["p_sport"], r["sport_yes"], SP_THR))


class H(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/favicon"):
            self.send_response(204); self.end_headers(); return
        self._send(PAGE.format(comment="", result="", mh_thr=MH_THR, sp_thr=SP_THR))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = parse_qs(self.rfile.read(n).decode("utf-8"))
        comment = (data.get("comment", [""])[0])
        self._send(PAGE.format(comment=html.escape(comment),
                               result=result_html(comment),
                               mh_thr=MH_THR, sp_thr=SP_THR))

    def log_message(self, format, *args):  # quiet
        pass


if __name__ == "__main__":
    print(f"[demo] open  http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
