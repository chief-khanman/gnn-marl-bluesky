"""Aggregate run directories into summary tables and paper figures (G06.3, G06.5, G06.9; S-10, S-14).

    python tools/plot_results.py --runs-dir runs --out results

Reads every <runs-dir>/*/config.json with its metrics.csv (training curves) and, if present,
eval_zero_shot.csv (written by algorithms.gnn_qmix.evaluate). Writes to --out:

    summary_training.csv    final in-training evaluation per method, fleet size and metric (mean, std, seeds)
    summary_zero_shot.csv   zero-shot evaluation per method, eval fleet size and metric (mean, std, seeds)
    curves_<metric>.png     learning curves, one panel per fleet size, mean +/- std over seeds
    zero_shot_<metric>_<airspace>.png  metric against evaluation fleet size, training sizes shaded;
                            airspace = fixed (density rises with n) or density (area grows with n)
    runs_used.csv           the run IDs behind every number (provenance)

A "method" is the algorithm plus its training sizes, e.g. "gnn_qmix (n=3-6-9)" or "qmix (n=6)".
Every table carries the number of seeds; S-10 asks for at least 3 before a number is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CURVE_METRICS = ("ep_return_per_agent", "los_rate", "conflict_free", "mean_abs_drift_deg")
ZERO_SHOT_METRICS = ("los_rate", "conflict_free", "los_events_per_agent", "ep_return_per_agent")
LABELS = {
    "ep_return_per_agent": "Episode return per agent",
    "los_rate": "LoS rate (fraction of pair-steps)",
    "conflict_free": "Conflict-free episode rate",
    "mean_abs_drift_deg": "Mean |track deviation| (deg)",
    "los_events_per_agent": "LoS events per agent",
}
# Colour follows the algorithm, in a fixed order; marker and line style repeat the identity.
STYLE = {
    "gnn_qmix": ("#2a78d6", "o", "-"),
    "qmix": ("#eb6834", "s", "--"),
    "vdn": ("#1baf7a", "^", "-."),
    "iql": ("#eda100", "D", ":"),
    "vdn-k3": ("#1baf7a", "^", "-."),
    "iql-k3": ("#eda100", "D", ":"),
    "qmix-k3": ("#eb6834", "s", "--"),
    "vdn-gcn": ("#e87ba4", "v", (0, (5, 1))),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AIRSPACE_LABEL = {"fixed": "fixed sector (density rises with n)", "density": "sector area scaled with n"}


def load_runs(runs_dir: Path):
    curves, zero_shot, used = [], [], []
    for cfg_path in sorted(runs_dir.glob("*/config.json")):
        meta = json.loads(cfg_path.read_text())
        a = meta["args"]
        algo = meta.get("method", a["algo"])
        train_n = "-".join(map(str, a["train_n"]))
        tags = {"run_id": meta["run_id"], "algo": algo, "train_n": train_n, "seed": a["seed"],
                "method": f"{algo} (n={train_n})"}
        metrics_path = cfg_path.parent / "metrics.csv"
        if metrics_path.exists():
            curves.append(pd.read_csv(metrics_path).assign(**tags))
        zs_path = cfg_path.parent / "eval_zero_shot.csv"
        if zs_path.exists():
            zero_shot.append(pd.read_csv(zs_path).drop(columns=["algo", "train_n", "seed"]).assign(**tags))
        used.append({**tags, "config_sha": meta["config_sha"], "repo_commit": meta["repo_commit"],
                     "has_zero_shot": zs_path.exists()})
    cat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # noqa: E731
    return cat(curves), cat(zero_shot), pd.DataFrame(used)


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def series_style(algo: str):
    return STYLE.get(algo, (MUTED, "x", "-"))


def plot_curves(curves: pd.DataFrame, out: Path) -> pd.DataFrame:
    ev = curves[curves["split"] == "eval"]
    per_seed = ev.groupby(["method", "algo", "train_n", "n_agents", "metric", "step", "seed"], as_index=False)["value"].mean()
    agg = per_seed.groupby(["method", "algo", "train_n", "n_agents", "metric", "step"])["value"].agg(
        mean="mean", std="std", seeds="count").reset_index().fillna({"std": 0.0})
    sizes = sorted(agg["n_agents"].unique())
    for metric in CURVE_METRICS:
        data = agg[agg["metric"] == metric]
        if data.empty:
            continue
        fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.4), squeeze=False, sharey=True)
        for ax, n in zip(axes[0], sizes):
            for (method, algo), d in data[data["n_agents"] == n].groupby(["method", "algo"]):
                color, _, ls = series_style(algo)
                d = d.sort_values("step")
                ax.plot(d["step"], d["mean"], color=color, linestyle=ls, linewidth=2, label=method)
                ax.fill_between(d["step"], d["mean"] - d["std"], d["mean"] + d["std"], color=color, alpha=0.15, linewidth=0)
            style_axes(ax)
            ax.set_title(f"n = {n}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("Environment steps per fleet size", fontsize=9, color=MUTED)
            ax.legend(fontsize=7, frameon=False)
        axes[0][0].set_ylabel(LABELS[metric], fontsize=9, color=MUTED)
        fig.suptitle(f"{LABELS[metric]} during training (greedy evaluation, mean ± std over seeds)",
                     fontsize=10, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"curves_{metric}.png", dpi=200)
        plt.close(fig)
    last = agg[agg["step"] == agg.groupby(["method", "n_agents"])["step"].transform("max")]
    return last.rename(columns={"n_agents": "eval_n"})[
        ["method", "algo", "train_n", "eval_n", "metric", "step", "mean", "std", "seeds"]]


def plot_zero_shot(zs: pd.DataFrame, out: Path) -> pd.DataFrame:
    ok = zs[zs["status"] == "ok"]
    metrics = [m for m in LABELS if m in ok.columns] + [m for m in ("min_sep_nm", "los_events", "los_pair_steps")
                                                        if m in ok.columns]
    keys = ["method", "algo", "train_n", "airspace", "eval_n"]
    per_seed = ok.groupby([*keys, "seed"], as_index=False)[metrics].mean()
    long = per_seed.melt(id_vars=[*keys, "seed"], var_name="metric")
    agg = long.groupby([*keys, "metric"])["value"].agg(
        mean="mean", std="std", seeds="count").reset_index().fillna({"std": 0.0})
    agg["status"] = "ok"
    blocked = zs[zs["status"] == "not_evaluable"].groupby(keys).agg(
        seeds=("seed", "nunique")).reset_index().assign(metric="", status="not_evaluable")
    for metric, airspace in [(m, a) for m in ZERO_SHOT_METRICS for a in sorted(agg["airspace"].unique())]:
        data = agg[(agg["metric"] == metric) & (agg["airspace"] == airspace)]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for (method, algo, train_n), d in data.groupby(["method", "algo", "train_n"]):
            color, marker, ls = series_style(algo)
            d = d.sort_values("eval_n")
            ax.errorbar(d["eval_n"], d["mean"], yerr=d["std"], color=color, marker=marker, markersize=6,
                        linestyle=ls, linewidth=2, capsize=3, label=method)
            if "-" in train_n:  # shade the sizes a jointly trained method has seen
                seen = [int(v) for v in train_n.split("-")]
                ax.axvspan(min(seen), max(seen), color=color, alpha=0.07, linewidth=0)
        style_axes(ax)
        ax.set_xscale("log")
        ticks = sorted(data["eval_n"].unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Evaluation fleet size n (shaded: jointly trained sizes)", fontsize=9, color=MUTED)
        ax.set_ylabel(LABELS[metric], fontsize=9, color=MUTED)
        ax.set_title(f"{LABELS[metric]} against fleet size, {AIRSPACE_LABEL[airspace]}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"zero_shot_{metric}_{airspace}.png", dpi=200)
        plt.close(fig)
    cols = ["method", "algo", "train_n", "airspace", "eval_n", "status", "metric", "mean", "std", "seeds"]
    return pd.concat([agg, blocked], ignore_index=True)[cols].sort_values(["method", "airspace", "eval_n", "metric"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curves, zero_shot, used = load_runs(Path(a.runs_dir))
    if used.empty:
        raise SystemExit(f"no runs found under {a.runs_dir}")
    used.to_csv(out / "runs_used.csv", index=False)
    if not curves.empty:
        plot_curves(curves, out).to_csv(out / "summary_training.csv", index=False, float_format="%.6g")
    if not zero_shot.empty:
        plot_zero_shot(zero_shot, out).to_csv(out / "summary_zero_shot.csv", index=False, float_format="%.6g")
    few = used.groupby("method")["seed"].nunique()
    for method, seeds in few[few < 3].items():
        print(f"WARNING: {method} has {seeds} seed(s); S-10 needs 3 before a number is reported")
    print(f"{len(used)} runs -> {out}/ ({', '.join(sorted(f.name for f in out.iterdir()))})")


if __name__ == "__main__":
    main()
