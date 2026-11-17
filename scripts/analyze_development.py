"""从已完成的开发集生成质量—成本图和分析；不修改冻结规则。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(run_dir: Path) -> Path:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if not report["comparison_complete"] or not report["frozen_policy"]:
        raise ValueError("需要完整且已经冻结的开发集运行")
    frame = pd.read_csv(run_dir / "tier_results.csv")
    if set(frame["split"]) != {"dev"}:
        raise ValueError("本脚本只分析开发集")
    frozen = json.loads((run_dir / "frozen_policy.json").read_text(encoding="utf-8"))
    metrics_dir = Path(report["metrics_dir"])
    summary = pd.read_csv(metrics_dir / "summary.csv").set_index("policy")
    quality = pd.read_csv(metrics_dir / "budget_quality.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, constrained_layout=True)
    for row_index, area in enumerate((0.15, 0.30)):
        for col_index, mask in enumerate(("center", "scratch", "scatter")):
            axis = axes[row_index, col_index]
            part = quality[(quality["mask_type"] == mask) & (quality["target_area"] == area)].sort_values("k")
            axis.plot(part["mean_nfe"], part["mean_full_lpips"], "o-", color="#2563eb")
            for item in part.itertuples():
                axis.annotate(f"K={item.k}", (item.mean_nfe, item.mean_full_lpips),
                              xytext=(4, 6), textcoords="offset points", fontsize=8)
            axis.set_title(f"{mask} / {area:.0%} missing")
            axis.set_xlabel("Mean NFE")
            axis.set_ylabel("Full-image LPIPS (lower is better)")
            axis.grid(alpha=0.2)
    fig.suptitle("Development set: quality vs. computation (10 original images)")
    fig.savefig(metrics_dir / "budget_quality.png", dpi=160)
    plt.close(fig)
    tiers = frozen["tiers"]
    wide = frame.pivot(index="sample_id", columns="k", values="full_lpips")
    gains = wide[tiers["middle"]] - wide[tiers["high"]]
    low_gains = wide[tiers["low"]] - wide[tiers["middle"]]
    gain_table = frame.drop_duplicates("sample_id")[["sample_id", "original_image_id", "mask_type", "target_area"]].set_index("sample_id")
    gain_table["lpips_gain_low_to_middle"] = low_gains
    gain_table["lpips_gain_middle_to_high"] = gains
    gain_table.to_csv(metrics_dir / "per_sample_budget_gain.csv")
    baseline = float(summary.loc["B2", "mean_full_lpips"])
    main = float(summary.loc["M1", "mean_full_lpips"])
    relative_gain = (baseline - main) / baseline * 100 if baseline else float("nan")
    lines = [
        "# 完整开发集预算分析", "",
        "本结果用于选规则，属于开发集内结果；不能代替冻结后的独立测试集结论。", "",
        f"- 原图 {report['num_images']} 张，图片与掩码组合 {report['num_samples']} 个，三档结果 {report['num_tier_runs']} 条。",
        f"- 失败 {report['num_failures']}；缓存命中 {report['num_cache_hits']}；完整集合等 NFE：{report['equal_nfe']}。",
        f"- M1 权重：{frozen['rules']['M1']['weights']}；阈值：{frozen['rules']['M1']['thresholds']}。",
        f"- A1 权重：{frozen['rules']['A1']['weights']}；阈值：{frozen['rules']['A1']['thresholds']}。",
        f"- M1 相对 B2 的 LPIPS 降低比例：{relative_gain:.2f}%（正值改善，负值下降）。",
        f"- 从中预算增加到高预算，{int((gains > 0).sum())}/{len(gains)} 个组合的 LPIPS 改善。",
        "", "| 策略 | LPIPS ↓ | 缺失区 PSNR ↑ | 边界 MAE ↓ | 总 NFE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("B1", "B2", "B3", "M1", "A1", "C1"):
        item = summary.loc[name]
        lines.append(f"| {name} | {item.mean_full_lpips:.6f} | {item.mean_masked_psnr:.4f} | {item.mean_boundary_mae_3px:.6f} | {int(item.nfe_total)} |")
    lines += ["", "## 成本与下一步", "",
              f"三档采样累计耗时：{frame.elapsed_seconds.sum() / 3600:.3f} 小时；不含模型加载、评价和预热。",
              f"峰值显存最大值：{frame.peak_vram_mb.max():.1f} MB。",
              "按原图汇总质量指标。预算收益表中的掩码组合不是独立原图样本。",
              "下一步在查看测试效果前锁定规模与子集，加载本次冻结配置执行正式六策略对照。",
              "开发集选择存在小样本过拟合风险，测试结果可能不复现这里的改善。", ""]
    path = metrics_dir / "development_analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(analyze(parser.parse_args().run_dir))
