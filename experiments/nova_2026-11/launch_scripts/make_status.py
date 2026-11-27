#!/usr/bin/env python3
"""Rewrites Nov25_STATUS.md from the logs and squeue. Run: python3 make_status.py (any python >= 3.8)."""
import glob, json, os, re, subprocess, time, datetime as dt
EXP = "/work/flemingc/akhan1/gnn_marl_experiments"; TAG = os.environ.get("TAG", "Nov25")
BUDGET = 2_000_000
q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%T|%S|%N"], capture_output=True, text=True).stdout
jobs = {}
for line in q.splitlines():
    jid, state, start, node = line.split("|")
    m = re.match(r"(\d+)_(\d+)$", jid)
    if m: jobs[int(m.group(2))] = (jid, state, start, node)
rows, notes = [], []
for log in sorted(glob.glob(f"{EXP}/{TAG}_*/logs/task_*.log"), key=lambda p: int(re.search(r"task_(\d+)", p).group(1))):
    tid = int(re.search(r"task_(\d+)", log).group(1)); method = log.split("/")[-3][len(TAG) + 1:]
    txt = open(log, errors="replace").read()
    rd = re.search(r"^run_dir: (.+)$", txt, re.M); rd = rd.group(1) if rd else ""
    seed = re.search(r"_s(\d+)_", rd); seed = seed.group(1) if seed else "?"
    steps = [int(s) for s in re.findall(r"^step=(\d+) ", txt, re.M)]; step = steps[-1] if steps else 0
    sps = re.findall(r"steps/s=(\d+)", txt); sps = sps[-1] if sps else "?"
    jid, state, start, node = jobs.get(tid, ("(not in queue)", "ENDED", "", ""))
    done = "wrote " in txt and "eval_zero_shot.csv" in txt
    rate, finish = "", ""
    hdr = re.search(r"started (.+)$", txt, re.M)
    if done: finish = "finished"
    elif step and start and state == "RUNNING":
        t0 = dt.datetime.fromisoformat(start); el = (dt.datetime.fromtimestamp(os.path.getmtime(log)) - t0).total_seconds()
        r = step / max(el, 1); rate = f"{r:.1f}"
        finish = (t0 + dt.timedelta(seconds=BUDGET / r)).strftime("%a %b %d %H:%M")
    ck = "yes" if os.path.exists(f"{rd}/checkpoints/latest.pt") else "no"
    fin = "yes" if os.path.exists(f"{rd}/checkpoints/final.pt") else "no"
    mrows = sum(1 for _ in open(f"{rd}/metrics.csv")) if os.path.exists(f"{rd}/metrics.csv") else 0
    cfg = "yes" if os.path.exists(f"{rd}/config.json") else "no"
    tb = len(re.findall(r"^Traceback", txt, re.M)); fatal = "no" if (state == "RUNNING" or done) else "CHECK"
    rows.append(f"| {tid} | {method} | {seed} | {jid} | {state} {node} | {step:,} | {sps} | {rate} | {finish} | "
                f"{len(re.findall(r'^.eval. step=0 ', txt, re.M))} | {cfg} | {mrows} | {ck} | {fin} | {tb} | `{os.path.basename(rd)}` |")
out = [f"# {TAG} experiments: status", "",
       f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `{EXP}/make_status.py` (rerun it to refresh).", "",
       f"Run directories: `{EXP}/{TAG}_<METHOD>/<run dir>`. Logs: `{EXP}/{TAG}_<METHOD>/logs/task_<id>.log`.", "",
       "Columns: `step` is per fleet size (budget 2,000,000). `log steps/s` is the last value printed by the training",
       "script (all fleet sizes together, between evaluations). `per-size steps/s` is step / wall time since job start,",
       "evaluations included; the projected end of training uses it. Add up to about 1.5 h for the final zero-shot",
       "evaluation (estimate from the smoke run, not measured at full size). `tracebacks` counts tracebacks in the log;",
       "see the notes below the table.", "",
       "| task | method | seed | job id | state, node | step | log steps/s | per-size steps/s | training ends (projected) | eval step=0 lines | config.json | metrics.csv rows | latest.pt | final.pt | tracebacks | run directory |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows
notes_path = f"{EXP}/{TAG}_STATUS_notes.md"
if os.path.exists(notes_path): out += ["", open(notes_path).read().rstrip()]
open(f"{EXP}/{TAG}_STATUS.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
