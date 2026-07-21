"""Live Layer-1 monitor — shows all three at once: the MH model, the SPORT model, and the
gate (the deliverable metric on the held-out set).

- LIVE panel: whichever run is training right now (per-epoch loss/F1/prec/recall + step bar).
- BEST panels: best validation-split epoch found across ALL mh_* and sport_* run logs.
- GATE panel: latest held-out gate result, read from run_logs/gate_status.txt
  (eval_holdout_grid.py / eval_holdout_stack.py write this file when they run).

Run in its own window:  .venv\\Scripts\\python.exe code\\watch_training.py
(or double-click watch_monitor.bat).  Auto-refresh every 2s.
"""
import re, time, os, sys, glob

LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_logs")
EVAL = re.compile(
    r"'eval_loss':\s*'([0-9.]+)'.*?'eval_f1':\s*'([0-9.]+)'.*?"
    r"'eval_precision':\s*'([0-9.]+)'.*?'eval_recall':\s*'([0-9.]+)'.*?'epoch':\s*'([0-9.]+)'"
)
STEP = re.compile(r"(\d+)/(\d+)\s*\[")


def evals_in(path):
    """Return list of (epoch, loss, f1, prec, recall) tuples for one log."""
    try:
        t = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    return [(float(ep), float(loss), float(f1), float(pr), float(rc))
            for loss, f1, pr, rc, ep in EVAL.findall(t)]


def group_of(path):
    b = os.path.basename(path)
    if b.startswith("train_mh") or "_mh" in b:
        return "mh"
    if b.startswith("train_sport") or "_sport" in b:
        return "sport"
    return "other"


def best_across(group):
    """Best-F1 epoch across all logs of a group -> (f1, prec, recall, label) or None."""
    best = None
    for path in glob.glob(os.path.join(LOGDIR, "train_*.log")):
        if group_of(path) != group:
            continue
        for ep, loss, f1, pr, rc in evals_in(path):
            if best is None or f1 > best[0]:
                tag = os.path.basename(path)[6:-4]  # strip 'train_' and '.log'
                best = (f1, pr, rc, f"{tag} ep{int(ep)}")
    return best


def latest_log():
    logs = glob.glob(os.path.join(LOGDIR, "train_*.log"))
    return max(logs, key=os.path.getmtime) if logs else None


def read_gate():
    p = os.path.join(LOGDIR, "gate_status.txt")
    if not os.path.exists(p):
        return {}
    d = {}
    for tok in open(p, encoding="utf-8", errors="ignore").read().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def render():
    W = 60
    L = ["=" * W, "  AM LAYER-1 MONITOR" + time.strftime("%H:%M:%S").rjust(W - 21), "=" * W]

    # ---- LIVE panel ----
    live = latest_log()
    if live:
        t = open(live, encoding="utf-8", errors="ignore").read()
        done = "train_runtime" in t
        state = "[DONE]" if done else "[training...]"
        L.append(f"  LIVE: {os.path.basename(live)}  {state}")
        rows = evals_in(live)
        bestf1 = max((r[2] for r in rows), default=0.0)
        L.append(f"    {'epoch':>5}{'loss':>8}{'F1':>8}{'prec':>8}{'recall':>8}")
        for ep, loss, f1, pr, rc in rows[-6:]:
            star = "  <= best" if f1 == bestf1 and bestf1 > 0 else ""
            L.append(f"    {int(ep):>5}{loss:>8.3f}{f1:>8.3f}{pr:>8.3f}{rc:>8.3f}{star}")
        steps = STEP.findall(t)
        if steps and not done:
            cur, tot = int(steps[-1][0]), int(steps[-1][1])
            pct = int(100 * cur / tot) if tot else 0
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            L.append(f"    step {cur}/{tot}  [{bar}] {pct}%")
    else:
        L.append("  LIVE: (no training log yet)")

    # ---- BEST panels ----
    L.append("-" * W)
    L.append("  BEST SO FAR  (validation split, best epoch across all runs)")
    for g in ("mh", "sport"):
        b = best_across(g)
        if b:
            L.append(f"    {g.upper():<6} F1 {b[0]:.3f}  P {b[1]:.3f}  R {b[2]:.3f}   ({b[3]})")
        else:
            L.append(f"    {g.upper():<6} (no evals found)")

    # ---- GATE panel ----
    L.append("-" * W)
    L.append("  GATE  (held-out 292 - the deliverable metric)")
    g = read_gate()
    if g:
        L.append(f"    F1 {g.get('f1','?')}  P {g.get('prec','?')}  R {g.get('rec','?')}"
                 f"   thr {g.get('thr','?')}")
        extra = []
        if g.get("pmhmax"):
            extra.append(f"P(mh)max {g['pmhmax']}")
        if g.get("note"):
            extra.append(g["note"])
        if extra:
            L.append("    " + "  |  ".join(extra))
        L.append("    baseline (always-yes) F1 0.61")
    else:
        L.append("    (run eval_holdout_grid.py to populate)")
    L.append("=" * W)
    return "\n".join(L)


def main():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(render())
        print("\n  (auto-refresh 2s - close this window to stop)")
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
