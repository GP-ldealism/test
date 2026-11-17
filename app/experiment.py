"""批量等预算实验编排（FR-05 / FR-09 / FR-10，PRD §7.5 / §7.6 / §8.3）。

执行流程：

1. 读取实验配置与数据清单，按 ``split`` / 掩码类型 / 目标面积 / 原图区间选出本次样本；
2. 对每个「图片 + 掩码 + 种子」分别跑低 / 中 / 高三档预算 ``K``（PRD §7.5），
   每档先查 :class:`app.cache.ResultCache`，命中则直接复用、不重复推理（FR-09）；
   单个样本失败写入 ``failures.jsonl`` 且不阻断后续样本（PRD §9.2）；
3. 复用同一批三档输出，按 B1/B2/B3/M1/A1/C1 分配预算并取对应档位结果；
   同一输出被多个策略复用，不计为独立重复实验（PRD §7.6）；
4. 每个档位输出只评价一次（缺失区 PSNR / 全图 LPIPS / 边界带 MAE），
   再按策略与按原图汇总（PRD §7.4 / §7.7）。

输出结构（PRD §8.3）::

    results/raw/<experiment_id>/seed_<seed>/<run_id>/
    ├─ config.yaml
    ├─ manifest.csv
    ├─ samples.csv
    ├─ run_log.jsonl
    ├─ images/
    └─ failures.jsonl
    results/metrics/<experiment_id>/seed_<seed>/<run_id>/
    ├─ per_sample.csv
    ├─ per_image.csv
    └─ summary.csv

``samples.csv`` 覆盖每一个「样本 × 策略」组合（策略只决定取哪一档输出），
因此六条策略共用同一次真实推理结果，真实推理次数等于「样本数 × 3」。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import traceback
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
import yaml

from app.budget import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MIDDLE,
    BudgetPolicy,
    allocate_batch,
    make_policies,
    mean_k,
)
from app.cache import ResultCache, make_cache_entry, source_provenance
from app.config import PRECISION_FP32, PROJECT_ROOT, ExperimentConfig
from app.data import load_image, load_mask, save_result, sha256_file
from app.features import MaskFeatures
from app.metrics import EvalMetrics, evaluate
from app.policy_freeze import (
    fit_development_rules,
    load_frozen_rules,
    policies_from_frozen,
    save_frozen_rules,
    validate_application,
)
from app.restore import RestoreConfig, RestoreResult, restore

#: 数据清单字段（PRD §7.2）
MANIFEST_FIELDS = (
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "actual_area",
    "boundary_norm",
    "seed",
    "image_path",
    "mask_path",
    "sha256",
)

#: 逐样本运行记录字段（results/raw/<exp>/samples.csv）
SAMPLE_FIELDS = (
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "policy",
    "level",
    "k",
    "nfe",
    "elapsed_seconds",
    "peak_vram_mb",
    "cache_hit",
    "image",
    "masked_psnr",
    "full_lpips",
    "boundary_mae_3px",
)

#: 逐样本指标字段（results/metrics/<exp>/per_sample.csv）
METRIC_FIELDS = (
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "policy",
    "k",
    "masked_psnr",
    "full_lpips",
    "boundary_mae_3px",
)

METRIC_COLUMNS = ("masked_psnr", "full_lpips", "boundary_mae_3px")

#: 三档按样本索引轮换先后，减少温度/功耗顺序偏差。
TIER_LEVELS = (LEVEL_LOW, LEVEL_MIDDLE, LEVEL_HIGH)

RestoreRunner = Callable[[torch.Tensor, torch.Tensor, RestoreConfig], RestoreResult]
Evaluator = Callable[[Any, Any, Any], EvalMetrics]


@dataclass(frozen=True)
class SampleRow:
    """清单中的一条样本（PRD §7.2）。"""

    sample_id: str
    original_image_id: str
    split: str
    mask_type: str
    target_area: float
    actual_area: float
    boundary_norm: float
    seed: int
    image_path: Path
    mask_path: Path
    sha256: str

    @property
    def features(self) -> MaskFeatures:
        """掩码特征直接取自清单记录，无需重新解码图片（FR-04 特征只依赖掩码）。"""
        return MaskFeatures(area_ratio=self.actual_area, boundary_norm=self.boundary_norm)


@dataclass(frozen=True)
class TierRun:
    """某一档预算的一次运行记录（真实推理或缓存命中）。"""

    sample_id: str
    level: str
    k: int
    image_path: Path
    nfe: int
    elapsed_seconds: float
    peak_vram_mb: float
    cache_hit: bool
    blank: bool


@dataclass
class ExperimentReport:
    """一次批量实验的汇总。"""

    experiment_id: str
    output_dir: Path
    metrics_dir: Path
    num_images: int
    num_samples: int
    num_tier_runs: int
    num_cache_hits: int
    num_failures: int
    num_records: int
    policy_mean_k: dict[str, float] = field(default_factory=dict)
    comparison_complete: bool = False
    equal_nfe: bool = False
    frozen_policy: str | None = None

    @property
    def status(self) -> str:
        """``ok``（无失败）/ ``partial``（有失败但仍产出结果图）/ ``failed``（未产出任何结果）。

        有失败的运行仅为部分结果，不可作为完整对照。
        """
        if self.num_tier_runs == 0:
            return "failed"
        return "ok" if self.num_failures == 0 else "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "num_images": self.num_images,
            "num_samples": self.num_samples,
            "num_tier_runs": self.num_tier_runs,
            "num_cache_hits": self.num_cache_hits,
            "num_failures": self.num_failures,
            "num_records": self.num_records,
            "comparison_complete": self.comparison_complete,
            "equal_nfe": self.equal_nfe,
            "frozen_policy": self.frozen_policy,
            "policy_mean_k": {name: round(value, 4) for name, value in sorted(self.policy_mean_k.items())},
            "output_dir": str(self.output_dir),
            "metrics_dir": str(self.metrics_dir),
        }


# --------------------------------------------------------------------------- #
# 清单读取与样本筛选
# --------------------------------------------------------------------------- #


def _resolve_path(text: str | Path, project_root: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _row_from_mapping(raw: Mapping[str, str], project_root: Path) -> SampleRow:
    return SampleRow(
        sample_id=raw["sample_id"],
        original_image_id=raw["original_image_id"],
        split=raw["split"],
        mask_type=raw["mask_type"],
        target_area=float(raw["target_area"]),
        actual_area=float(raw["actual_area"]),
        boundary_norm=float(raw["boundary_norm"]),
        seed=int(raw["seed"]),
        image_path=_resolve_path(raw["image_path"], project_root),
        mask_path=_resolve_path(raw["mask_path"], project_root),
        sha256=raw["sha256"],
    )


def load_manifest(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> list[SampleRow]:
    """读取数据清单；相对路径相对 ``project_root`` 解析（PRD §7.2）。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到数据清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [name for name in MANIFEST_FIELDS if name not in header]
        if missing:
            raise ValueError(f"清单 {manifest_path} 缺少字段：{missing}")
        rows = [_row_from_mapping(raw, project_root) for raw in reader]
    if not rows:
        raise ValueError(f"清单为空：{manifest_path}")
    for key in ("original_image_id", "sha256"):
        seen: dict[str, str] = {}
        for row in rows:
            value = getattr(row, key)
            if value in seen and seen[value] != row.split:
                raise ValueError(f"清单跨划分图片泄漏：{key}={value}")
            seen[value] = row.split
    return rows


def select_samples(
    rows: Sequence[SampleRow],
    *,
    split: str | None = None,
    mask_types: Sequence[str] | None = None,
    target_areas: Sequence[float] | None = None,
    image_offset: int = 0,
    image_limit: int | None = None,
) -> list[SampleRow]:
    """按划分 / 掩码类型 / 目标面积 / 原图区间筛选样本，保持清单原顺序。

    ``image_offset`` 与 ``image_limit`` 以**原图**为单位（PRD §7.6 的各子集都按原图计），
    用于从测试集中切出「主测试前 20 张 / 面积扩展 5 张 / 重复实验 5 张」。
    """
    if image_offset < 0 or (image_limit is not None and image_limit < 1):
        raise ValueError("原图偏移须非负，数量须为正")
    selected = [row for row in rows if split is None or row.split == split]
    if mask_types:
        wanted = set(mask_types)
        selected = [row for row in selected if row.mask_type in wanted]
    if target_areas:
        areas = [float(area) for area in target_areas]
        selected = [
            row for row in selected if any(abs(row.target_area - area) < 1e-6 for area in areas)
        ]

    image_ids = list(dict.fromkeys(row.original_image_id for row in selected))
    start = max(image_offset, 0)
    kept = image_ids[start:] if image_limit is None else image_ids[start : start + image_limit]
    allowed = set(kept)
    return [row for row in selected if row.original_image_id in allowed]


def validate_baseline(
    selected: Sequence[SampleRow], baseline_run: Path, config: ExperimentConfig,
    *, project_root: Path = PROJECT_ROOT,
) -> None:
    """扩展实验必须有完整的同图、同设置种子 42 主测试基准。"""
    baseline_config = yaml.safe_load((baseline_run / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((baseline_run / "report.json").read_text(encoding="utf-8"))
    if baseline_config["seed"] != 42 or not report.get("comparison_complete"):
        raise ValueError("需要完整的种子 42 主测试基准")
    for key in ("budget", "outer_steps", "image_size", "precision", "method"):
        if baseline_config[key] != config.to_dict()[key]:
            raise ValueError(f"基准采样配置不一致：{key}")
    if baseline_config.get("checkpoint_sha256", "").lower() != config.checkpoint_sha256.lower():
        raise ValueError("基准权重不一致")
    baseline = load_manifest(baseline_run / "manifest.csv", project_root=project_root)
    if any(row.split != "test" or abs(row.target_area - 0.30) > 1e-6 for row in baseline):
        raise ValueError("基准必须是测试集 30% 面积")
    selection = json.loads((baseline_run / "selection.json").read_text(encoding="utf-8"))
    expected_ids = set(selection["extension_image_ids"])
    if {row.original_image_id for row in selected} != expected_ids:
        raise ValueError("扩展与重复实验必须使用主测试预先固定的同图扩展子集")
    input_hashes = json.loads((baseline_run / "input_hashes.json").read_text(encoding="utf-8"))
    by_key = {(row.original_image_id, row.mask_type): row for row in baseline}
    for row in selected:
        reference = by_key.get((row.original_image_id, row.mask_type))
        if row.split != "test" or reference is None or reference.sha256 != row.sha256:
            raise ValueError(f"样本缺少同图种子 42 基准：{row.sample_id}")
        if abs(row.target_area - 0.30) < 1e-6 and (
            row.sample_id != reference.sample_id
            or sha256_file(row.mask_path) != input_hashes[reference.sample_id]["mask_sha256"]
        ):
            raise ValueError(f"重复实验掩码样本不一致：{row.sample_id}")


# --------------------------------------------------------------------------- #
# 缓存键与权重校验
# --------------------------------------------------------------------------- #


def sampling_parameters(config: RestoreConfig) -> dict[str, float]:
    """全部会影响输出的采样参数（缓存键的第 9 个字段，FR-09）。"""
    return {
        "step_size": float(config.step_size),
        "alpha": float(config.alpha),
        "friction": float(config.friction),
        "m": float(config.m),
        "chara_lamb": float(config.chara_lamb),
        "chara_beta": float(config.chara_beta),
    }


def sampling_contract(config: ExperimentConfig, checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "budget": asdict(config.budget), "outer_steps": config.outer_steps,
        "image_size": config.image_size, "precision": config.precision,
        "method": config.method, "checkpoint_sha256": checkpoint_sha256.lower(),
        "sampling": sampling_parameters(RestoreConfig(outer_steps=config.outer_steps)),
        "enable_dynamic_stopping": config.enable_dynamic_stopping,
    }


def validate_development_selection(selected: Sequence[SampleRow]) -> None:
    """正式冻结需要 10 原图 × 三类掩码 × 两种面积，禁止最小试跑误冻结。"""
    groups: dict[str, list[SampleRow]] = {}
    for row in selected:
        if row.split != "dev":
            raise ValueError("冻结规则只能使用开发集")
        groups.setdefault(row.original_image_id, []).append(row)
    expected = {(kind, area) for kind in ("center", "scratch", "scatter") for area in (0.15, 0.30)}
    if len(groups) != 10 or any(
        len(group) != 6 or {(row.mask_type, round(row.target_area, 6)) for row in group} != expected
        for group in groups.values()
    ):
        raise ValueError("冻结需要完整开发集：10 原图 × 3 类掩码 × 2 种面积")


def resolve_checkpoint_sha256(config: ExperimentConfig) -> str:
    """返回实际权重 SHA-256：配置已给出则校验一致，未给出则就地计算。

    校验失败直接抛错——缓存键以权重哈希为准，静默复用旧结果会污染实验记录（FR-09）。
    哈希比较忽略大小写（Windows 的 ``Get-FileHash`` 输出大写，``hashlib`` 输出小写）。
    """
    expected = config.checkpoint_sha256.strip()
    if not config.checkpoint.is_file():
        if expected:
            raise FileNotFoundError(f"未找到模型权重：{config.checkpoint}")
        return ""
    actual = sha256_file(config.checkpoint)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"权重 SHA-256 与配置不符：配置 {expected}，实际 {actual}（{config.checkpoint}）"
        )
    return actual


# --------------------------------------------------------------------------- #
# 单档执行
# --------------------------------------------------------------------------- #


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _run_tier(
    *,
    row: SampleRow,
    level: str,
    k: int,
    image: torch.Tensor,
    mask: torch.Tensor,
    mask_sha256: str,
    images_dir: Path,
    config: ExperimentConfig,
    cache: ResultCache,
    code_commit: str,
    checkpoint_sha256: str,
    device: str,
    sampling: Mapping[str, float],
    runner: RestoreRunner,
) -> TierRun:
    """跑一档预算：命中缓存则复制结果图，否则真实推理并回写缓存。"""
    target = images_dir / f"{row.sample_id}__k{k}.png"
    restore_config = RestoreConfig(
        n_steps=k,
        outer_steps=config.outer_steps,
        seed=config.seed,
        device=device,
        checkpoint=config.checkpoint,
    )
    entry = make_cache_entry(
        code_commit=code_commit,
        checkpoint_sha256=checkpoint_sha256,
        image_sha256=row.sha256,
        mask_sha256=mask_sha256,
        seed=config.seed,
        precision=config.precision,
        budget_k=k,
        outer_steps=config.outer_steps,
        sampling=dict(sampling),
    )

    record = cache.lookup(entry)
    if record is not None:
        shutil.copyfile(record.image_path, target)
        extra = record.meta.get("extra", {})
        return TierRun(
            sample_id=row.sample_id,
            level=level,
            k=k,
            image_path=target,
            nfe=int(extra.get("nfe", 0)),
            elapsed_seconds=float(extra.get("elapsed_seconds", 0.0)),
            peak_vram_mb=float(extra.get("peak_vram_mb", 0.0)),
            cache_hit=True,
            blank=bool(extra.get("blank", False)),
        )

    result = runner(image, mask, restore_config)
    if not bool(torch.isfinite(result.image).all().item()):
        raise ValueError(f"推理结果包含 NaN/Inf（sample={row.sample_id} K={k}）")
    blank = float(result.image.std().item()) < 1e-3

    save_result(target, result.image)
    cache.store(
        entry,
        result.image,
        extra={
            "nfe": int(result.nfe),
            "elapsed_seconds": float(result.elapsed_seconds),
            "peak_vram_mb": float(result.peak_vram_mb),
            "blank": blank,
        },
    )
    return TierRun(
        sample_id=row.sample_id,
        level=level,
        k=k,
        image_path=target,
        nfe=int(result.nfe),
        elapsed_seconds=float(result.elapsed_seconds),
        peak_vram_mb=float(result.peak_vram_mb),
        cache_hit=False,
        blank=blank,
    )


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def _dump_config(
    path: Path, config: ExperimentConfig, checkpoint_sha256: str, project_root: Path
) -> None:
    payload = config.to_dict()
    for key in ("data_manifest", "checkpoint", "output_dir", "cache_dir"):
        payload[key] = _relative(Path(payload[key]), project_root)
    payload["checkpoint_sha256"] = checkpoint_sha256
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_manifest(path: Path, rows: Sequence[SampleRow], project_root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "original_image_id": row.original_image_id,
                    "split": row.split,
                    "mask_type": row.mask_type,
                    "target_area": f"{row.target_area:.4f}",
                    "actual_area": f"{row.actual_area:.6f}",
                    "boundary_norm": f"{row.boundary_norm:.6f}",
                    "seed": row.seed,
                    "image_path": _relative(row.image_path, project_root),
                    "mask_path": _relative(row.mask_path, project_root),
                    "sha256": row.sha256,
                }
            )


def _summarize(records: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """按策略汇总；无记录时返回仅含表头的空表。"""
    frame = pd.DataFrame(records, columns=list(columns))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "experiment_id", "policy", "num_samples", "mean_k", "nfe_total",
                *[f"mean_{name}" for name in METRIC_COLUMNS],
            ]
        )
    grouped = frame.groupby("policy", as_index=False)
    summary = grouped.agg(
        num_samples=("sample_id", "count"),
        mean_k=("k", "mean"),
        nfe_total=("nfe", "sum"),
        **{f"mean_{name}": (name, "mean") for name in METRIC_COLUMNS},
    )
    return summary


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_experiment(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    mask_types: Sequence[str] | None = None,
    target_areas: Sequence[float] | None = None,
    image_offset: int = 0,
    image_limit: int | None = None,
    manifest_path: str | Path | None = None,
    baseline_run: str | Path | None = None,
    frozen_policy: str | Path | None = None,
    freeze_development: bool = False,
    warmup: bool = False,
    device: str = "cuda",
    policies: Mapping[str, BudgetPolicy] | None = None,
    runner: RestoreRunner = restore,
    evaluator: Evaluator = evaluate,
    project_root: Path = PROJECT_ROOT,
    code_commit: str | None = None,
    log: Callable[[str], None] = print,
) -> ExperimentReport:
    """执行一次批量等预算实验，返回汇总报告（PRD §7.5 / §8.3）。"""
    if config.precision != PRECISION_FP32:
        raise ValueError(
            f"当前 restore 实现仅支持 {PRECISION_FP32}，收到 precision={config.precision!r}"
        )

    rows = load_manifest(manifest_path or config.data_manifest, project_root=project_root)
    selected = select_samples(
        rows,
        split=split,
        mask_types=mask_types,
        target_areas=target_areas,
        image_offset=image_offset,
        image_limit=image_limit,
    )
    if not selected:
        available = sorted({row.split for row in rows})
        raise ValueError(
            f"筛选后没有样本（split={split!r}）；清单可用划分：{available}"
        )

    if len({row.sample_id for row in selected}) != len(selected):
        raise ValueError("样本 ID 重复")
    if config.experiment_id in {"area_extension", "repeat_seed"} and baseline_run is None:
        raise ValueError("面积/重复实验必须指定种子 42 主测试的 baseline_run")
    if baseline_run is not None:
        validate_baseline(selected, Path(baseline_run), config, project_root=project_root)

    if freeze_development:
        if frozen_policy is not None or policies is not None:
            raise ValueError("开发集冻结不能同时指定外部策略")
        validate_development_selection(selected)
    if any(row.split == "test" for row in selected) and frozen_policy is None:
        raise ValueError("测试集必须指定 frozen_policy 冻结配置")
    if len({row.split for row in selected}) != 1:
        raise ValueError("一次实验只能使用一个划分")
    checkpoint_sha256 = resolve_checkpoint_sha256(config)
    provenance = source_provenance(project_root)
    contract = sampling_contract(config, checkpoint_sha256)
    frozen = load_frozen_rules(frozen_policy) if frozen_policy else None
    if frozen is not None:
        validate_application(
            frozen, contract=contract, source_sha256=provenance["source_sha256"],
            image_ids={row.original_image_id for row in selected},
            image_hashes={row.sha256 for row in selected},
        )
        if baseline_run is not None:
            baseline_frozen = load_frozen_rules(Path(baseline_run) / "frozen_policy.json")
            if baseline_frozen["sha256"] != frozen["sha256"]:
                raise ValueError("基准与扩展实验必须使用同一冻结策略")
    head = code_commit if code_commit is not None else provenance["git_commit"]
    resolved_commit = f"{head}:source:{provenance['source_sha256']}"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    relative_run = Path(config.experiment_id) / f"seed_{config.seed}" / run_id
    output_dir = config.output_dir / relative_run
    metrics_dir = project_root / "results" / "metrics" / relative_run
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 保存实际源码快照，未提交的运行也能还原；写入时再核对文件哈希。
    with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, expected_hash in provenance["files"].items():
            content = (project_root / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("源码在启动时发生变化，请重试")
            archive.writestr(relative, content)
    if frozen is not None:
        save_frozen_rules(output_dir / "frozen_policy.json", frozen)
    cache = ResultCache(config.cache_dir)
    tiers = config.budget.to_tiers()
    if frozen is not None:
        available_policies = policies_from_frozen(frozen)
        resolved_policies = {name: available_policies[name] for name in (policies or available_policies)}
    else:
        resolved_policies = dict(policies) if policies is not None else make_policies(tiers=tiers)

    _dump_config(output_dir / "config.yaml", config, checkpoint_sha256, project_root)
    _write_manifest(output_dir / "manifest.csv", selected, project_root)
    original_ids = list(dict.fromkeys(row.original_image_id for row in selected))
    extension_count = 10 if len(original_ids) > 20 else 5
    (output_dir / "selection.json").write_text(json.dumps({
        "sample_ids": [row.sample_id for row in selected],
        "original_image_ids": original_ids,
        "extension_image_ids": original_ids[:extension_count],
        "baseline_run": str(Path(baseline_run).resolve()) if baseline_run else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    run_log_path = output_dir / "run_log.jsonl"
    failures_path = output_dir / "failures.jsonl"
    run_log_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    sampling = sampling_parameters(RestoreConfig(outer_steps=config.outer_steps))
    _append_jsonl(
        run_log_path,
        {
            "event": "start",
            "experiment_id": config.experiment_id,
            "num_samples": len(selected),
            "num_images": len({row.original_image_id for row in selected}),
            "split": split,
            "device": device,
            "seed": config.seed,
            "precision": config.precision,
            "tiers": {"low": tiers.low, "middle": tiers.middle, "high": tiers.high},
            "code_commit": resolved_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "policy_state": "frozen" if frozen else "unfrozen",
            "warmup": warmup,
        },
    )

    tier_runs: dict[tuple[str, int], TierRun] = {}
    tier_metrics: dict[tuple[str, int], EvalMetrics] = {}
    failed_samples: set[str] = set()
    num_failures = 0
    num_cache_hits = 0
    input_hashes: dict[str, dict[str, str]] = {}

    if warmup:
        row = selected[0]
        warm_result = runner(load_image(row.image_path, config.image_size),
                             load_mask(row.mask_path, config.image_size),
                             RestoreConfig(n_steps=tiers.middle, outer_steps=config.outer_steps,
                                           seed=config.seed, device=device, checkpoint=config.checkpoint))
        _append_jsonl(run_log_path, {"event": "warmup", "nfe": warm_result.nfe,
                                   "elapsed_seconds": warm_result.elapsed_seconds})
        log(f"[warmup] K={tiers.middle}, NFE={warm_result.nfe}")
    for row_index, row in enumerate(selected):
        try:
            if sha256_file(row.image_path).lower() != row.sha256.lower():
                raise ValueError("图片内容与清单 SHA-256 不符，请重新准备清单")
            image = load_image(row.image_path, config.image_size)
            mask = load_mask(row.mask_path, config.image_size)
            mask_sha256 = sha256_file(row.mask_path)
            input_hashes[row.sample_id] = {"image_sha256": row.sha256, "mask_sha256": mask_sha256}
        except (OSError, ValueError) as exc:
            num_failures += 1
            failed_samples.add(row.sample_id)
            _append_jsonl(
                run_log_path,
                {"event": "fail", "sample_id": row.sample_id, "stage": "load"},
            )
            _append_jsonl(
                failures_path,
                {
                    "sample_id": row.sample_id,
                    "stage": "load",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            log(f"[fail] {row.sample_id} 读取失败：{exc}")
            continue

        offset = row_index % len(TIER_LEVELS)
        for level in TIER_LEVELS[offset:] + TIER_LEVELS[:offset]:
            k = tiers.k(level)
            try:
                run = _run_tier(
                    row=row,
                    level=level,
                    k=k,
                    image=image,
                    mask=mask,
                    mask_sha256=mask_sha256,
                    images_dir=images_dir,
                    config=config,
                    cache=cache,
                    code_commit=resolved_commit,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                    sampling=sampling,
                    runner=runner,
                )
                metrics = evaluator(load_image(run.image_path, config.image_size), image, mask)
                if metrics.status == "ok" and not all(math.isfinite(value) for value in (metrics.full_lpips, metrics.boundary_mae_3px)):
                    raise ValueError("评价返回非有限 LPIPS 或边界误差")
            except Exception as exc:  # noqa: BLE001 - 单样本失败必须可记录并继续
                num_failures += 1
                failed_samples.add(row.sample_id)
                _append_jsonl(
                    run_log_path,
                    {"event": "fail", "sample_id": row.sample_id, "level": level, "k": k},
                )
                _append_jsonl(
                    failures_path,
                    {
                        "sample_id": row.sample_id,
                        "level": level,
                        "k": k,
                        "stage": "tier",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                log(f"[fail] {row.sample_id} K={k}：{exc}")
                continue

            tier_runs[(row.sample_id, k)] = run
            tier_metrics[(row.sample_id, k)] = metrics
            num_cache_hits += int(run.cache_hit)
            _append_jsonl(
                run_log_path,
                {
                    "event": "tier",
                    "sample_id": row.sample_id,
                    "level": level,
                    "k": k,
                    "nfe": run.nfe,
                    "elapsed_seconds": round(run.elapsed_seconds, 4),
                    "peak_vram_mb": round(run.peak_vram_mb, 1),
                    "cache_hit": run.cache_hit,
                    "blank": run.blank,
                    "status": metrics.status,
                    "full_lpips": metrics.full_lpips,
                    "masked_psnr": metrics.masked_psnr,
                    "boundary_mae_3px": metrics.boundary_mae_3px,
                },
            )
            if metrics.status != "ok":
                num_failures += 1
                failed_samples.add(row.sample_id)
                _append_jsonl(
                    failures_path,
                    {
                        "sample_id": row.sample_id,
                        "level": level,
                        "k": k,
                        "stage": "evaluate",
                        "error": metrics.error or "评价失败",
                    },
                )
        state = "partial" if row.sample_id in failed_samples else "ok"
        log(f"[{state}] {row.sample_id} 三档处理结束")

    tier_records = []
    for row in selected:
        for level in TIER_LEVELS:
            k = tiers.k(level)
            key = (row.sample_id, k)
            if key not in tier_runs:
                continue
            run, metric = tier_runs[key], tier_metrics[key]
            tier_records.append({
                "sample_id": row.sample_id, "original_image_id": row.original_image_id,
                "split": row.split, "mask_type": row.mask_type, "target_area": row.target_area,
                "actual_area": row.actual_area, "boundary_norm": row.boundary_norm,
                "image_sha256": row.sha256, "k": k, "status": metric.status,
                "full_lpips": metric.full_lpips, "masked_psnr": metric.masked_psnr,
                "boundary_mae_3px": metric.boundary_mae_3px, "nfe": run.nfe,
                "elapsed_seconds": run.elapsed_seconds, "peak_vram_mb": run.peak_vram_mb,
                "cache_hit": run.cache_hit,
            })
    tier_frame = pd.DataFrame(tier_records)
    tier_frame.to_csv(output_dir / "tier_results.csv", index=False, encoding="utf-8")
    if freeze_development and not failed_samples:
        frozen = fit_development_rules(
            tier_frame, tiers=tiers, contract=contract, provenance=provenance,
            manifest_sha256=sha256_file(output_dir / "manifest.csv"), seed=config.seed,
        )
        save_frozen_rules(output_dir / "frozen_policy.json", frozen)
        pd.DataFrame(frozen["candidates"]).to_csv(metrics_dir / "candidate_scores.csv", index=False)
        resolved_policies = policies_from_frozen(frozen)
        log(f"[freeze] {output_dir / 'frozen_policy.json'}")
    if not tier_frame.empty:
        good = tier_frame.loc[tier_frame["status"] == "ok"]
        by_image = good.groupby(["original_image_id", "mask_type", "target_area", "k"], as_index=False).agg(
            **{name: (name, "mean") for name in METRIC_COLUMNS},
            nfe=("nfe", "mean"), elapsed_seconds=("elapsed_seconds", "mean"),
        )
        by_image.groupby(["mask_type", "target_area", "k"], as_index=False).agg(
            num_images=("original_image_id", "count"),
            **{f"mean_{name}": (name, "mean") for name in (*METRIC_COLUMNS, "nfe", "elapsed_seconds")},
        ).to_csv(metrics_dir / "budget_quality.csv", index=False)

    usable = [row for row in selected if row.sample_id not in failed_samples]
    records: list[dict[str, Any]] = []
    policy_mean_k: dict[str, float] = {}

    if usable:
        sample_ids = [row.sample_id for row in usable]
        features = [row.features for row in usable]
        for name, policy in resolved_policies.items():
            assignments = allocate_batch(sample_ids, features, policy)
            policy_mean_k[name] = mean_k(assignments)
            for row, assignment in zip(usable, assignments, strict=True):
                run = tier_runs[(row.sample_id, assignment.k)]
                metrics = tier_metrics[(row.sample_id, assignment.k)]
                records.append(
                    {
                        "sample_id": row.sample_id,
                        "original_image_id": row.original_image_id,
                        "split": row.split,
                        "mask_type": row.mask_type,
                        "target_area": row.target_area,
                        "policy": name,
                        "level": assignment.level,
                        "k": assignment.k,
                        "nfe": run.nfe,
                        "elapsed_seconds": run.elapsed_seconds,
                        "peak_vram_mb": run.peak_vram_mb,
                        "cache_hit": run.cache_hit,
                        "image": _relative(run.image_path, project_root),
                        "masked_psnr": metrics.masked_psnr,
                        "full_lpips": metrics.full_lpips,
                        "boundary_mae_3px": metrics.boundary_mae_3px,
                    }
                )
            _append_jsonl(
                run_log_path,
                {
                    "event": "policy",
                    "policy": name,
                    "num_samples": len(sample_ids),
                    "mean_k": round(policy_mean_k[name], 4),
                    "nfe_total": int(
                        sum(tier_runs[(row.sample_id, item.k)].nfe for row, item in zip(usable, assignments, strict=True))
                    ),
                },
            )
            log(f"[policy] {name} 平均 K={policy_mean_k[name]:.3f}（{len(sample_ids)} 个样本）")
    else:
        _append_jsonl(
            run_log_path,
            {
                "event": "policy_skip",
                "num_usable": len(usable),
                "reason": "没有三档均推理和评价成功的样本",
            },
        )
        log("[warn] 没有可用样本，跳过策略分配")

    frame = pd.DataFrame(records, columns=list(SAMPLE_FIELDS))
    frame.to_csv(output_dir / "samples.csv", index=False, encoding="utf-8")

    metric_frame = frame.loc[:, list(METRIC_FIELDS)] if not frame.empty else pd.DataFrame(
        columns=list(METRIC_FIELDS)
    )
    metric_frame.to_csv(metrics_dir / "per_sample.csv", index=False, encoding="utf-8")

    if frame.empty:
        per_image = pd.DataFrame(columns=["original_image_id", "policy", "num_samples", *METRIC_COLUMNS])
        summary = pd.DataFrame(
            columns=[
                "experiment_id", "policy", "num_samples", "mean_k", "nfe_total",
                *[f"mean_{name}" for name in METRIC_COLUMNS],
            ]
        )
    else:
        per_image = (
            frame.groupby(["original_image_id", "policy"], as_index=False)
            .agg(
                num_samples=("sample_id", "count"),
                **{name: (name, "mean") for name in METRIC_COLUMNS},
            )
            .sort_values(["policy", "original_image_id"])
        )
        summary = _summarize(records, SAMPLE_FIELDS)
        # 质量先按原图汇总，再跨图平均，避免掩码数不同导致图片权重不同。
        image_means = per_image.groupby("policy")[list(METRIC_COLUMNS)].mean()
        for name in METRIC_COLUMNS:
            summary[f"mean_{name}"] = summary["policy"].map(image_means[name])
        summary["num_images"] = summary["policy"].map(per_image.groupby("policy").size())
        summary.insert(0, "experiment_id", config.experiment_id)
    comparison_complete = not failed_samples and len(usable) == len(selected)
    totals = {name: sum(r["nfe"] for r in records if r["policy"] == name)
              for name in resolved_policies}
    equal_nfe = comparison_complete and all(name in totals for name in ("B2", "M1", "A1", "C1"))
    equal_nfe = equal_nfe and all(totals[name] == totals["B2"] for name in ("M1", "A1", "C1"))
    summary["policy_state"] = "frozen" if frozen else "unfrozen"
    summary["comparison_complete"] = comparison_complete
    summary["equal_nfe"] = equal_nfe
    per_image.to_csv(metrics_dir / "per_image.csv", index=False, encoding="utf-8")
    summary.to_csv(metrics_dir / "summary.csv", index=False, encoding="utf-8")

    report = ExperimentReport(
        experiment_id=config.experiment_id,
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        num_images=len({row.original_image_id for row in selected}),
        num_samples=len(selected),
        num_tier_runs=len(tier_runs),
        num_cache_hits=num_cache_hits,
        num_failures=num_failures,
        num_records=len(records),
        policy_mean_k=policy_mean_k,
        comparison_complete=comparison_complete,
        equal_nfe=equal_nfe,
        frozen_policy=str(output_dir / "frozen_policy.json") if frozen else None,
    )
    (output_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _append_jsonl(run_log_path, {"event": "end", **report.to_dict()})
    log(f"[report] {json.dumps(report.to_dict(), ensure_ascii=False)}")
    return report
