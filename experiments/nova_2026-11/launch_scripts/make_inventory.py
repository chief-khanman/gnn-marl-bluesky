#!/usr/bin/env python3
"""Writes runs_index.csv (one row per run directory) and prints per-run health and aggregate results."""
import csv, glob, json, os, statistics as st, collections, sys
W="/work/flemingc/akhan1"
SETS=[("Nov25_GNN-QMIX","paper"),("Nov25_VDN","paper"),("Nov25_IQL","paper"),("Nov25_ABL-VDN-GCN","paper (optional ablation)"),
      ("Nov26_QMIX","paper"),("Nov25_QMIX","superseded: diverged, do not report")]
rows=[]
def scan(d,setname,role):
    cfg=json.load(open(d+"/config.json")); a=cfg["args"]
    r=dict(set=setname,role=role,run_id=cfg["run_id"],path=d,method=cfg["method"],algo=a["algo"],train_n="-".join(map(str,a["train_n"])),seed=a["seed"],
           polyak=a["polyak"],qmix_airspeed_div=a.get("qmix_airspeed_div",1.0),steps_per_size=a["steps_per_size"],config_sha=cfg["config_sha"][:12],
           repo_commit=cfg["repo_commit"][:7],repo_dirty=cfg["repo_dirty"],created=cfg["created"])
    loss={}; last=0; ev=collections.defaultdict(dict); n=0
    for x in csv.DictReader(open(d+"/metrics.csv")):
        n+=1; s=int(x["step"]); last=max(last,s)
        if x["metric"]=="loss": loss[s]=float(x["value"])
        if x["split"]=="eval" and x["metric"] in ("ep_return_per_agent","conflict_free"): ev[(int(x["n_agents"]),x["metric"])][s]=float(x["value"])
    ks=sorted(loss)
    r.update(metrics_rows=n,last_step=last,loss_first=f"{loss[ks[0]]:.3g}" if ks else "",loss_max=f"{max(loss.values()):.3g}" if ks else "",loss_last=f"{loss[ks[-1]]:.3g}" if ks else "")
    z=d+"/eval_zero_shot.csv"; r["zero_shot_rows"]=sum(1 for _ in open(z))-1 if os.path.exists(z) else 0
    r["final_pt"]=os.path.exists(d+"/checkpoints/final.pt"); r["latest_pt"]=os.path.exists(d+"/checkpoints/latest.pt")
    r["bytes"]=sum(os.path.getsize(os.path.join(p,f)) for p,_,fs in os.walk(d) for f in fs)
    r["_ev"]=ev; return r
for s,role in SETS:
    for d in sorted(glob.glob(f"{W}/gnn_marl_experiments/{s}/*_nova")): rows.append(scan(d,s,role))
for d in sorted(glob.glob(f"{W}/gnn_marl_diag/G*"))+sorted(glob.glob(f"{W}/gnn_marl_diag/round2/G*")): rows.append(scan(d,"diag" if "round2" not in d else "diag/round2","diagnostic, not a result"))
for d in sorted(glob.glob(f"{W}/gnn_marl_smoke/G*"))+sorted(glob.glob(f"{W}/gnn_marl_smoke/*/G*")): rows.append(scan(d,"smoke","smoke test, not a result"))
cols=[c for c in rows[0] if not c.startswith("_")]
with open(f"{W}/gnn_marl_experiments/runs_index.csv","w",newline="") as f:
    w=csv.DictWriter(f,cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("runs:",collections.Counter(r["set"] for r in rows))
print("\nHEALTH (paper + superseded): set method n seed | last_step | loss first/max/last | per training n: return first3 -> last10 (best@step), conflict_free last10")
for r in rows:
    if r["set"] in("diag","diag/round2","smoke"): continue
    parts=[]
    for n in sorted({k[0] for k in r["_ev"]}):
        c=r["_ev"][(n,"ep_return_per_agent")]; ks=sorted(c); b=max(ks,key=lambda k:c[k]); cf=r["_ev"][(n,"conflict_free")]
        parts.append(f"n{n}: {st.mean(c[k] for k in ks[:3]):.1f}->{st.mean(c[k] for k in ks[-10:]):.1f} (best {c[b]:.1f}@{b//1000}k) cf {st.mean(cf[k] for k in ks[-10:]):.2f}")
    print(f'{r["set"]:18s} {r["method"]:9s} n{r["train_n"]:6s} s{r["seed"]} | {r["last_step"]} | {r["loss_first"]}/{r["loss_max"]}/{r["loss_last"]} | '+" ; ".join(parts))
print("\nZERO-SHOT (final.pt, 100 episodes/seed; mean±sd over seeds of per-seed means): method train_n airspace eval_n seeds | return/agent | conflict_free | los_rate | min_sep_nm")
per=collections.defaultdict(dict); st_counts=collections.Counter()
for r in rows:
    if not r["role"].startswith("paper"): continue
    g=collections.defaultdict(list)
    for x in csv.DictReader(open(r["path"]+"/eval_zero_shot.csv")):
        st_counts[(r["method"],x["status"])]+=1
        if x["status"]=="ok": g[(r["method"],x["train_n"],x["airspace"],int(x["eval_n"]))].append(x)
    for k,v in g.items(): per[k][r["seed"]]={m:st.mean(float(x[m]) for x in v) for m in("ep_return_per_agent","conflict_free","los_rate","min_sep_nm")}
def ms(vals): return f"{st.mean(vals):.3g}±{st.stdev(vals):.2g}" if len(vals)>1 else f"{vals[0]:.3g}"
for k in sorted(per):
    s=per[k]; print(*k,len(s),"|"," | ".join(ms([s[x][m] for x in s]) for m in("ep_return_per_agent","conflict_free","los_rate","min_sep_nm")))
print("status counts",dict(st_counts))
