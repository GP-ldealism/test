# AGENTS.md

## Scope & current state
- Windows/PowerShell repo for a LanPaint-based 256x256 image inpainting prototype (Python 3.12, PyTorch/CUDA). Main app code: `app/` (`data`, `masks`, `features`, `budget`, `restore`, `cache`, `config`, `metrics`, `experiment`, `ui`); entry scripts: `scripts/`; configs: `configs/*.yaml`.
- S1-S4 and basic S6 UI exist; S5 batch/experiment formal completion is paused. Do not claim frozen rules, full dev/test experiments, or complete UI acceptance unless you run/verify them (see `docs/项目进度.md`, `docs/下一步计划.md`).

## Environment & prerequisites
- UI/real inference expects the existing LanPaint env: `D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe` (`start_ui.ps1` hardcodes this and sets `GRADIO_ANALYTICS_ENABLED=False`).
- Real `app.restore.restore`/UI repair needs NVIDIA CUDA GPU; `scripts/smoke_inference.py` and experiments default to `--device cuda`. Many unit tests are offline/CPU-safe (no model required).
- Upstream is **not vendored**: `third_party/LanPaintBench/` and `checkpoints/256x256_diffusion_uncond.pt` are gitignored. Restore/fix commit and weight hash from `docs/上游溯源.md` before touching inference. Do **not** modify upstream; do **not** import upstream `run.py`.

## Common commands (run from repo root)
- UI: `./start_ui.ps1` or `python -m app.ui` (serves local Gradio, typically `127.0.0.1:7860`; Ctrl+C to stop). Quick real UI/API check: `python scripts/check_ui.py` (launches temp service, runs repair/download assertions).
- Tests/lint: `python -m pytest tests -q --tb=short` and `python -m ruff check app scripts tests` (last recorded: 106 passed, ruff clean).
- Smoke inference: `python scripts/smoke_inference.py --runs 1` (or `--runs 10` for gate; writes `results/raw/s1_smoke/`).
- Offline metrics: `python scripts/evaluate.py --exp <id>` (reads `results/raw/<id>/images/*.png`, resolves reference/mask from `report.json` config or `--reference/--mask`, writes `results/metrics/<id>/`).
- Data/manifest: `python scripts/prepare_samples.py` (50 orig images x 3 mask types x 2 areas = 300 rows in `data/manifest.csv`; prefers local parquet in `data/_cache/`).
- Experiments: `python scripts/run_experiment.py --config configs/dev.yaml --split dev` (see script header for `main/area/repeat` examples; supports `--seed`, `--policies`, `--image-offset/--image-limit`, `--target-area`, `--mask-type`, cache reuse on rerun).

## Architecture & data conventions
- Tensor/domain conventions: images `(1,3,256,256)` float32 in `[-1,1]`; masks `(1,1,256,256)` `{0,1}` where **1 = missing** internally. `app.data.mask_to_upstream` inverts for LanPaintBench (upstream 1 = known/keep).
- `app.restore`: builds/caches model per `(checkpoint, device)` and pipeline per `(K, sampling params, device, checkpoint)`; resets seeds per inference; measures NFE by wrapping UNet forward (do not infer by formula). Official sampling defaults come from upstream makefile (`K/n_steps`, `outer_steps=T=20`, etc.); budgets 2/5/8.
- Masks/features: `app.masks` generates deterministic `center/scratch/scatter` at 15%/30%; features (`area_ratio`, `boundary_norm`) come from mask only (no missing-pixel leakage). `app.budget` defines tiers/policies/allocation; `app.cache` keys include code commit, checkpoint SHA256, image/mask hashes, seed, precision, K, outer steps, all sampling params.
- `app.experiment` runs low/mid/high tiers per sample, reuses cache, then reuses tier outputs across policies (B1/B2/B3/M1/A1/C1) without extra inference; outputs `results/raw/<exp>/` and `results/metrics/<exp>/`. `configs/*` are experiment specs (relative paths resolved vs project root by `app.config`).
- Data/results layout (mostly gitignored): `data/manifest.csv`, `data/masks/` tracked samples/masks as present; `data/samples/original/`, `data/_cache/`, `checkpoints/`, `results/raw/*/images/`, `results/ui/`, `.cache/` are ignored/generated.

## Gotchas
- UI input: keep original from editor background; derive brush mask from layer alpha to avoid paint color contaminating source; empty mask -> original, all-missing mask rejected. GPU requests serialized; UI writes isolated run dirs under `results/ui/`.
- `dev_budget`/S5 trial artifacts are incomplete/minimal evidence; do not treat as final research conclusions. Known S5 issues to respect before resuming formal runs: seed-specific output isolation, repeat baseline coverage, divisible-by-3 allocation edge cases, and cache traceability when code is uncommitted.
- `README.md` / some docs may have encoding artifacts in certain viewers; prefer config/scripts/code as executable truth.
- Preserve existing verified guidance in `docs/` and `AI项目实施计划_LanPaint主方案与工业检测备选.md`; commit/push only when explicitly requested.