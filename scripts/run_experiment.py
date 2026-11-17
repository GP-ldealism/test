r"""批量等预算实验入口（S5 / FR-09）。

按 PRD §7.6 的各子集运行：

```powershell
$py = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
# 开发集：10 图 × 3 掩码 × 2 面积 × 3 档 = 180 次
& $py scripts\run_experiment.py --config configs\dev.yaml --split dev --freeze-development --warmup
# $frozen 指向开发集报告中的 frozen_policy.json；正式实验只加载冻结配置。
# 主测试：测试集前 20 图 × 3 掩码 × 30% 面积 × 3 档 = 180 次
& $py scripts\run_experiment.py --config configs\main.yaml --split test --image-limit 20 --target-area 0.30 --frozen-policy $frozen
# $baseline 指向已完成主测试报告中的 output_dir（含 config.yaml、manifest.csv、report.json）
# 面积扩展：主测试内前 5 图 × 3 掩码 × 15% 面积 × 3 档 = 45 次
& $py scripts\run_experiment.py --config configs\area.yaml --split test --image-offset 0 --image-limit 5 --target-area 0.15 --baseline-run $baseline --frozen-policy $frozen
# 重复实验：同一主测试内前 5 图 × 3 掩码 × 30% 面积 × 3 档 × 2 个新种子 = 90 次
& $py scripts\run_experiment.py --config configs\repeat.yaml --split test --image-offset 0 --image-limit 5 --target-area 0.30 --seed 2026 --baseline-run $baseline --frozen-policy $frozen
& $py scripts\run_experiment.py --config configs\repeat.yaml --split test --image-offset 0 --image-limit 5 --target-area 0.30 --seed 3407 --baseline-run $baseline --frozen-policy $frozen
```

中断后重跑同一命令即可：已完成的「图片 + 掩码 + 种子 + 档位」从结果缓存直接复用（FR-09）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.budget import make_policies
from app.config import load_config
from app.experiment import run_experiment


def _parse_policies(text: str | None, tiers) -> dict | None:
    if not text:
        return None
    wanted = [name.strip() for name in text.split(",") if name.strip()]
    available = make_policies(tiers=tiers)
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise SystemExit(f"未知策略：{unknown}；可选 {sorted(available)}")
    return {name: available[name] for name in wanted}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量等预算实验（低/中/高三档 + 六条策略复用）。")
    parser.add_argument("--config", required=True, help="YAML 实验配置，例如 configs/dev.yaml。")
    parser.add_argument("--split", default=None, choices=["check", "dev", "test"], help="只跑该划分。")
    parser.add_argument(
        "--mask-type", action="append", default=None,
        help="掩码类型，可重复（center / scratch / scatter）；默认全部。",
    )
    parser.add_argument(
        "--target-area", action="append", type=float, default=None,
        help="目标面积，可重复（0.15 / 0.30）；默认全部。",
    )
    parser.add_argument("--image-offset", type=int, default=0, help="跳过前 N 张原图（默认 0）。")
    parser.add_argument("--image-limit", type=int, default=None, help="最多使用 N 张原图（默认全部）。")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子（重复实验用）。")
    parser.add_argument("--policies", default=None, help="逗号分隔的策略名，默认全部六条。")
    parser.add_argument("--manifest", default=None, help="覆盖配置中的数据清单路径。")
    parser.add_argument("--baseline-run", default=None, help="面积/重复实验必填：完整种子 42 主测试输出目录。")
    parser.add_argument("--frozen-policy", default=None, help="冻结策略 JSON；测试集必填。")
    parser.add_argument("--freeze-development", action="store_true", help="完整开发集运行后选择并冻结规则。")
    parser.add_argument("--warmup", action="store_true", help="计量前执行一次中预算预热，单独记录。")
    parser.add_argument("--report", default=None, help="额外写出报告 JSON 的路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    policies = _parse_policies(args.policies, config.budget.to_tiers())

    report = run_experiment(
        config,
        split=args.split,
        mask_types=args.mask_type,
        target_areas=args.target_area,
        image_offset=args.image_offset,
        image_limit=args.image_limit,
        manifest_path=args.manifest,
        baseline_run=args.baseline_run,
        frozen_policy=args.frozen_policy,
        freeze_development=args.freeze_development,
        warmup=args.warmup,
        device=args.device,
        policies=policies,
    )

    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
        print(f"[report] {report_path}", flush=True)
    print(payload, flush=True)
    print(f"[verdict] {report.status.upper()}", flush=True)
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
