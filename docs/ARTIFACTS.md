# Artifacts: sharing trained weights and results

> Standard **S-23**. Goal: anyone, on any machine, can pull a trained policy by its ID and get
> exactly the bytes that produced a reported result, without re-training it.

## 1. What is an artifact

An **artifact** is one immutable bundle produced by one run:

```
<artifact_id>/                 artifact_id == run ID (S-10), e.g.
                               G05.4_qmixgnn_sector_cr_n3-5-8_s1_20261120-1430_hpc
├── manifest.json              provenance + checksums (schema §4)
├── config.json                fully resolved run config
├── checkpoint.pt              weights only (state_dicts), see §6
├── eval/                      optional: evaluation tables (csv) produced from this checkpoint
└── tb/                        optional: tensorboard event files, if they back a reported curve
```

Publish an artifact when (a) a result in the paper or record depends on it, (b) someone else
will evaluate or fine-tune it, or (c) the run was expensive (e.g. an HPC run). Exploratory
local runs stay in `runs/` and aren't shared.

## 2. Where artifacts live

- **The bytes:** the artifact store, one folder per `artifact_id`.
  **Proposed backend (D-013, pending Q-07): one private Hugging Face Hub model repository**,
  e.g. `<org>/gnn-marl-bluesky-artifacts`, under an organization all authors belong to.
- **The index:** `artifacts/registry.toml`, committed to git. An artifact that isn't in the
  registry doesn't officially exist.
- **Local copy:** `$GNN_MARL_ARTIFACTS` (gitignored; default `artifacts/cache/`).

Why the HF Hub was proposed:
- free private repos and versioned storage
- a CLI and a Python API that work the same on laptops and HPC login nodes (plain HTTPS)
- per-user tokens with read or write scope
- at paper release the repo can be made public as the reproducibility artifact

Alternatives considered:

| option | why not (for now) |
|---|---|
| Weights & Biases Artifacts | good, but we log with tensorboard, and it ties sharing to a second account system |
| DVC + remote (S3/SSH/Drive) | ties data to commits nicely, but needs more setup per machine; Drive auth is painful on HPC |
| Git LFS on GitHub | small free quotas for storage and bandwidth |
| University Box/Drive via rclone | workable fallback; weaker versioning and no checksum-based download |

Switching backends changes only §3 and §5. The bundle format, manifest and registry stay.

## 3. One-time setup

**Per user:** create a Hugging Face account and get added to the project org (by whoever
creates it, Q-07). Create an access token: `read` scope for everyone, `write` for anyone who
publishes.

**Per machine:**
```bash
pip install -U huggingface_hub          # will be added to pyproject.toml with the helper (INF-8)
hf auth login                           # older versions: huggingface-cli login
# headless/HPC jobs: export HF_TOKEN=... in the job environment instead of storing it in the repo
```
Then set `GNN_MARL_ARTIFACTS` and `GNN_MARL_ARTIFACT_REPO` in `.env` (see [HANDOFF.md](HANDOFF.md)).

## 4. `manifest.json` schema

```jsonc
{
  "artifact_id": "G05.4_qmixgnn_sector_cr_n3-5-8_s1_20261120-1430_hpc",
  "goal_ids": ["G05.4"],
  "created_utc": "2026-11-20T14:30:00Z",
  "created_by": "<person>", "agent": "<AI agent/tool or null>",   // S-16
  "machine": "hpc",
  "repo_commit": "<git sha>", "repo_dirty": false,
  "submodules": {"bluesky-gym": "3a536c0…", "cleanmarl": "5cf837e…"},
  "packages": {"python": "3.12.x", "torch": "2.9.0", "numpy": "2.3.4"},
  "config_sha": "<sha256 of canonical config.json>",                // duplicate-run detection
  "algo": "qmixgnn", "env": "sector_cr", "train_n": [3,5,8], "seed": 1,
  "env_steps": 2000000,
  "files": {"checkpoint.pt": "<sha256>", "config.json": "<sha256>"},
  "summary_metrics": {"eval/success_rate": 0.0},
  "notes": ""
}
```
`repo_dirty: true` (uncommitted changes at train time) makes the artifact ineligible for paper results.

## 5. Procedures

**Publish** (after a run you want to share):
1. Make sure the run started from a committed, pushed tree (`repo_dirty: false`).
2. Write `manifest.json` (the helper does this, INF-8; until then by hand) with checksums
   (`shasum -a 256 <file>` on macOS, `sha256sum <file>` on Linux).
3. Upload the folder:
   `hf upload $GNN_MARL_ARTIFACT_REPO <local_dir>/<artifact_id> <artifact_id> --repo-type model`
4. Append an `[[artifact]]` block to `artifacts/registry.toml` and commit it on your branch.
   It goes in the same PR as the result it backs (S-14).

**Pull** (any machine):
```bash
hf download $GNN_MARL_ARTIFACT_REPO --include "<artifact_id>/*" --local-dir $GNN_MARL_ARTIFACTS
```
Then verify checksums against the manifest before use. (On HPC, pull on a login node.)

**Find** an existing artifact before training: search `artifacts/registry.toml` for the
`config_sha` (or for goal, algo, env and train_n).

## 6. Rules

- **Immutable.** Never overwrite or re-upload under an existing `artifact_id`. A new run gets a new ID.
- **Never deleted, only deprecated.** Set `status = "deprecated"` with a reason. Paper-cited
  artifacts are never deprecated.
- **Weights only.** Save `{"agent": state_dict, "mixer": state_dict, …}`. Load with
  `torch.load(path, weights_only=True)`. No pickled modules or objects (portability, safety).
- **Status lifecycle** in the registry: `candidate` → `reference` (a baseline or model of record)
  → `paper` (cited in the manuscript) | `deprecated`.
- No secrets, no raw datasets, no `.env` inside an artifact.

## 7. Planned tooling (INF-8)

`tools/artifacts.py` with `publish`, `pull`, `verify`, `find --config-sha`, writing manifests
and registry entries automatically. Until it exists, use §5 by hand.
