"""Read-only diagnosis of frozen experiments; run from the repository root.

python analysis/diagnose_budget.py
python analysis/diagnose_budget.py --cases-per-side 5   # 最终报告用：5 改善 + 5 退化
No inference or policy fitting. Outputs are separate from original run artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/budget_diagnosis_20261117"
MASKS = ["center", "scratch", "scatter"]
METRICS = ["full_lpips", "masked_psnr", "boundary_mae_3px"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_box(mask):
    # Fixed geometric rule, independent of any output quality: nearest inner
    # boundary pixel to image centre, fixed 64x64 crop. No best-error cropping.
    padded = np.pad(mask, 1, constant_values=True)
    inside = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    ys, xs = np.where(mask & ~inside)
    assert len(xs)
    index = np.argmin((ys - 127.5) ** 2 + (xs - 127.5) ** 2)
    x, y = int(np.clip(xs[index] - 32, 0, 192)), int(np.clip(ys[index] - 32, 0, 192))
    return x, y, 64, 64


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _shrink_png(path, colors):
    """把已保存的 PNG 量化到 ``colors`` 色调色板，减小体积（仅供交付归档）。"""
    frame = Image.open(path).convert("RGB")
    frame.quantize(colors=colors, method=Image.MEDIANCUT).save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="离线预算诊断与案例导出")
    parser.add_argument(
        "--cases-per-side",
        type=int,
        default=3,
        help="改善/退化各取多少个案例（默认 3；最终报告用 5，共 10 组）",
    )
    parser.add_argument("--out", default=None, help="输出目录（默认使用本脚本约定的诊断目录）")
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="导出 PNG 的 DPI（默认 140；仓库瘦身时用 100，可把 20 张案例图从约 13 MB 降到约 7 MB）",
    )
    parser.add_argument(
        "--palette",
        type=int,
        default=0,
        help="把导出的 PNG 量化到 N 色调色板（0 = 关闭；归档时用 256，可再降到约 3 MB）",
    )
    args = parser.parse_args()
    if args.cases_per_side < 1:
        raise SystemExit("--cases-per-side 必须 >= 1")
    if args.dpi < 40:
        raise SystemExit("--dpi 必须 >= 40")
    if args.palette < 0 or 1 < args.palette < 16 or args.palette > 256:
        raise SystemExit("--palette 需为 0（关闭）或 16–256 之间的色数")
    per_side = args.cases_per_side
    dpi = args.dpi
    palette = args.palette
    out = Path(args.out).resolve() if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    campaign = read_json(ROOT / "results/formal_followup_20261116_status.json")
    assert campaign["status"] == "complete"
    reports = {
        "development": read_json(ROOT / "results/dev_freeze_20261116_report.json"),
        **campaign["reports"],
    }
    frames, tiers, manifests = {}, {}, {}
    inputs = {}
    grouped, allocations, gains, paired_rows = [], [], [], []
    for name, report in reports.items():
        assert (
            report["status"] == "ok"
            and report["comparison_complete"]
            and report["equal_nfe"]
        )
        run = Path(report["output_dir"])
        for filename in [
            "samples.csv",
            "tier_results.csv",
            "manifest.csv",
            "frozen_policy.json",
        ]:
            path = run / filename
            inputs[str(path.relative_to(ROOT))] = sha(path)
        frame = pd.read_csv(run / "samples.csv")
        tier = pd.read_csv(run / "tier_results.csv")
        manifest = pd.read_csv(run / "manifest.csv").set_index("sample_id")
        assert not frame.duplicated(["sample_id", "policy"]).any()
        assert not tier.duplicated(["sample_id", "k"]).any()
        frames[name], tiers[name], manifests[name] = frame, tier, manifest
        for (area, mask, policy), group in frame.groupby(
            ["target_area", "mask_type", "policy"]
        ):
            grouped.append(
                dict(
                    dataset=name,
                    area=area,
                    mask_type=mask,
                    policy=policy,
                    n_images=group.original_image_id.nunique(),
                    nfe=int(group.nfe.sum()),
                    **{
                        metric: float(
                            group.groupby("original_image_id")[metric].mean().mean()
                        )
                        for metric in METRICS
                    },
                )
            )
            for k, count in group.k.value_counts().items():
                allocations.append(
                    {
                        "dataset": name,
                        "area": area,
                        "mask_type": mask,
                        "policy": policy,
                        "k": int(k),
                        "count": int(count),
                    }
                )
        lookup = tier.pivot(index="sample_id", columns="k", values="full_lpips")
        assert set(lookup.columns) == {2, 5, 8} and not lookup.isna().any().any()
        base = frame[frame.policy == "B2"].set_index("sample_id")
        adaptive = frame[frame.policy == "M1"].set_index("sample_id")
        for sample_id, row in adaptive.iterrows():
            observed = float(row.full_lpips - base.loc[sample_id, "full_lpips"])
            counterfactual = float(
                lookup.loc[sample_id, int(row.k)] - lookup.loc[sample_id, 5]
            )
            assert np.isclose(observed, counterfactual, atol=1e-12)
            paired_rows.append(
                {
                    "dataset": name,
                    "sample_id": sample_id,
                    "original_image_id": row.original_image_id,
                    "mask_type": row.mask_type,
                    "area": float(row.target_area),
                    "m1_k": int(row.k),
                    "lpips_M1_minus_B2": observed,
                    "psnr_M1_minus_B2": float(
                        row.masked_psnr - base.loc[sample_id, "masked_psnr"]
                    ),
                    "boundary_M1_minus_B2": float(
                        row.boundary_mae_3px - base.loc[sample_id, "boundary_mae_3px"]
                    ),
                }
            )
            gains.append(
                {
                    "dataset": name,
                    "sample_id": sample_id,
                    "mask_type": row.mask_type,
                    "area": float(row.target_area),
                    "lpips_k2": float(lookup.loc[sample_id, 2]),
                    "lpips_k5": float(lookup.loc[sample_id, 5]),
                    "lpips_k8": float(lookup.loc[sample_id, 8]),
                    "improvement_2_to_5": float(
                        lookup.loc[sample_id, 2] - lookup.loc[sample_id, 5]
                    ),
                    "improvement_5_to_8": float(
                        lookup.loc[sample_id, 5] - lookup.loc[sample_id, 8]
                    ),
                }
            )
    pd.DataFrame(grouped).to_csv(out / "mask_quality_and_cost.csv", index=False)
    allocation = pd.DataFrame(allocations)
    allocation.to_csv(out / "budget_allocation_counts.csv", index=False)
    gain = pd.DataFrame(gains)
    gain.to_csv(out / "per_sample_tier_gains.csv", index=False)
    pairs = pd.DataFrame(paired_rows)
    pairs.to_csv(out / "per_sample_policy_differences.csv", index=False)
    summary = (
        pairs.groupby(["dataset", "area", "mask_type"])
        .agg(
            n=("sample_id", "count"),
            lpips_delta=("lpips_M1_minus_B2", "mean"),
            psnr_delta=("psnr_M1_minus_B2", "mean"),
            boundary_delta=("boundary_M1_minus_B2", "mean"),
            improved=("lpips_M1_minus_B2", lambda x: int((x < 0).sum())),
            unchanged=("lpips_M1_minus_B2", lambda x: int((x == 0).sum())),
            worsened=("lpips_M1_minus_B2", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    summary.to_csv(out / "mask_policy_differences.csv", index=False)
    gain_summary = (
        gain.groupby(["dataset", "area", "mask_type"])
        .agg(
            n=("sample_id", "count"),
            gain_2_to_5=("improvement_2_to_5", "mean"),
            gain_5_to_8=("improvement_5_to_8", "mean"),
            positive_2_to_5=("improvement_2_to_5", lambda x: int((x > 0).sum())),
            positive_5_to_8=("improvement_5_to_8", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    gain_summary.to_csv(out / "mask_tier_gains.csv", index=False)

    main_pairs = pairs[pairs.dataset == "main"]
    image_deltas = (
        main_pairs.groupby("original_image_id").lpips_M1_minus_B2.mean().reset_index()
    )
    recorded = pd.read_csv(
        ROOT / "results/metrics/formal_campaign_20261116/main_30_images_paired.csv"
    ).set_index("original_image_id")
    assert np.allclose(
        image_deltas.set_index("original_image_id").lpips_M1_minus_B2.sort_index(),
        recorded.full_lpips.sort_index(),
    )
    # Predeclared selection: best N and worst N by per-original mean, ID breaks ties.
    best = (
        image_deltas[image_deltas.lpips_M1_minus_B2 < 0]
        .sort_values(["lpips_M1_minus_B2", "original_image_id"])
        .head(per_side)
    )
    worst = (
        image_deltas[image_deltas.lpips_M1_minus_B2 > 0]
        .sort_values(
            ["lpips_M1_minus_B2", "original_image_id"], ascending=[False, True]
        )
        .head(per_side)
    )
    selected = []
    base = frames["main"][frames["main"].policy == "B2"].set_index("sample_id")
    adaptive = frames["main"][frames["main"].policy == "M1"].set_index("sample_id")
    for label, chosen in [("improved", best), ("degraded", worst)]:
        for rank, item in enumerate(chosen.itertuples(), 1):
            image_id = item.original_image_id
            fig, axes = plt.subplots(3, 4, figsize=(12, 10.5))
            zoom, zaxes = plt.subplots(3, 3, figsize=(9, 9))
            crop_records = []
            for i, mask_type in enumerate(MASKS):
                sid = f"{image_id}_{mask_type}30"
                row = manifests["main"].loc[sid]
                reference = load_rgb(ROOT / row.image_path)
                mask = np.asarray(Image.open(ROOT / row.mask_path).convert("L")) >= 128
                x, y, w, h = crop_box(mask)
                masked = reference.copy()
                masked[mask] = 127
                baseline = load_rgb(ROOT / base.loc[sid, "image"])
                prediction = load_rgb(ROOT / adaptive.loc[sid, "image"])
                for path in [
                    ROOT / row.image_path,
                    ROOT / row.mask_path,
                    ROOT / base.loc[sid, "image"],
                    ROOT / adaptive.loc[sid, "image"],
                ]:
                    inputs[str(path.relative_to(ROOT))] = sha(path)
                k = int(adaptive.loc[sid, "k"])
                delta = float(
                    adaptive.loc[sid, "full_lpips"] - base.loc[sid, "full_lpips"]
                )
                for j, (data, title) in enumerate(
                    zip(
                        [reference, masked, baseline, prediction],
                        ["Reference", "Masked input", "B2: K=5", f"M1: K={k}"],
                    )
                ):
                    axes[i, j].imshow(data)
                    axes[i, j].add_patch(
                        Rectangle(
                            (x, y), w, h, fill=False, edgecolor="yellow", linewidth=1
                        )
                    )
                    axes[i, j].set_title(f"{mask_type} | {title}", fontsize=10)
                    axes[i, j].axis("off")
                axes[i, 2].text(
                    0.5,
                    -0.04,
                    f"LPIPS {base.loc[sid, 'full_lpips']:.4f}",
                    transform=axes[i, 2].transAxes,
                    ha="center",
                    fontsize=9,
                )
                axes[i, 3].text(
                    0.5,
                    -0.04,
                    f"LPIPS {adaptive.loc[sid, 'full_lpips']:.4f} | delta {delta:+.4f}",
                    transform=axes[i, 3].transAxes,
                    ha="center",
                    fontsize=9,
                )
                for j, (data, title) in enumerate(
                    zip(
                        [reference, baseline, prediction],
                        ["Reference", "B2: K=5", f"M1: K={k}"],
                    )
                ):
                    zaxes[i, j].imshow(
                        data[y : y + h, x : x + w], interpolation="nearest"
                    )
                    zaxes[i, j].set_title(f"{mask_type} | {title}", fontsize=10)
                    zaxes[i, j].axis("off")
                crop_records.append(
                    {
                        "sample_id": sid,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "m1_k": k,
                        "lpips_delta": delta,
                    }
                )
            fig.suptitle(
                f"{image_id} | {label} rank {rank} | mean M1-B2 LPIPS {item.lpips_M1_minus_B2:+.5f}",
                fontsize=13,
            )
            zoom.suptitle(
                f"{image_id} | 64x64 boundary crops (fixed geometric selection)",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
            zoom.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(out / f"{image_id}_comparison.png", dpi=dpi)
            zoom.savefig(out / f"{image_id}_boundary.png", dpi=dpi)
            if palette:
                _shrink_png(out / f"{image_id}_comparison.png", palette)
                _shrink_png(out / f"{image_id}_boundary.png", palette)
            plt.close(fig)
            plt.close(zoom)
            selected.append(
                {
                    "original_image_id": image_id,
                    "category": label,
                    "rank": rank,
                    "mean_lpips_delta": float(item.lpips_M1_minus_B2),
                    "crops": crop_records,
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = summary[summary.dataset == "main"].set_index("mask_type").loc[MASKS]
    axes[0].bar(MASKS, data.lpips_delta, color=["#4477AA", "#CC6677", "#999999"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].margins(y=0.2)
    axes[0].set(
        ylabel="Mean LPIPS: M1 minus B2", title="Main test: changes by mask (30 images)"
    )
    for i, value in enumerate(data.lpips_delta):
        axes[0].annotate(
            f"{value:+.5f}",
            (i, value),
            xytext=(0, 5 if value >= 0 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for mask in MASKS:
        subset = (
            tiers["main"][tiers["main"].mask_type == mask]
            .groupby("k")
            .full_lpips.mean()
        )
        axes[1].plot(subset.index, subset.values, "o-", label=mask)
    axes[1].set(
        xlabel="K (NFE per sample: 60 / 120 / 180)",
        ylabel="Mean full-image LPIPS",
        title="Observed fixed-budget outputs",
        xticks=[2, 5, 8],
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "mask_budget_diagnosis.png", dpi=dpi)
    if palette:
        _shrink_png(out / "mask_budget_diagnosis.png", palette)
    plt.close(fig)
    (out / "case_selection.json").write_text(
        json.dumps(
            {
                "selection": f"per-original LPIPS delta; best {per_side}/worst {per_side}; ID tie-break",
                "crop_rule": "64x64 around inner four-neighbour boundary nearest image centre; no quality-based crop selection",
                "cases": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "analysis_script_sha256": sha(Path(__file__)),
                "inference_performed": False,
                "policy_refit": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(
        "\nSelected cases:", [(c["original_image_id"], c["category"]) for c in selected]
    )
    print(
        "\nVerification: matched all M1 outputs to recorded K; matched 30 paired image differences; outputs:",
        out,
    )


if __name__ == "__main__":
    main()
