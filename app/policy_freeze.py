"""开发集规则选择与冻结文件；测试集只能加载，不参与拟合。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from app.budget import (
    CANDIDATE_WEIGHTS_AREA,
    CANDIDATE_WEIGHTS_AREA_BOUNDARY,
    BudgetPolicy,
    BudgetTiers,
    FeatureNormalizer,
    allocate_batch,
    fit_thresholds,
    make_policies,
    policy_score,
    select_weights,
)
from app.features import MaskFeatures

SCHEMA_VERSION = 1
SELECTION_METRIC = "mean_per_image_full_lpips"


def canonical_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def fit_development_rules(
    frame: pd.DataFrame, *, tiers: BudgetTiers, contract: dict[str, Any],
    provenance: dict[str, Any], manifest_sha256: str, seed: int,
) -> dict[str, Any]:
    """从完整三档评价表选择受限候选；按原图平均 LPIPS，同分按候选顺序。"""
    if frame.empty or set(frame["split"]) != {"dev"}:
        raise ValueError("只能用非空开发集拟合冻结规则")
    if set(frame["status"]) != {"ok"}:
        raise ValueError("开发集评价存在失败，不能冻结")
    if frame.duplicated(["sample_id", "k"]).any():
        raise ValueError("开发集样本预算记录重复")
    ks = {tiers.low, tiers.middle, tiers.high}
    if any(set(group["k"]) != ks for _, group in frame.groupby("sample_id")):
        raise ValueError("每个开发集样本都必须有完整三档结果")
    if not all(math.isfinite(float(value)) for value in frame["full_lpips"]):
        raise ValueError("开发集 LPIPS 必须全部有效且有限")
    identity = ["original_image_id", "actual_area", "boundary_norm", "image_sha256"]
    if any((group[identity].nunique(dropna=False) != 1).any()
           for _, group in frame.groupby("sample_id")):
        raise ValueError("同一样本的三档特征或原图不一致")
    samples = frame.drop_duplicates("sample_id").sort_values("sample_id")
    ids = samples["sample_id"].tolist()
    features = [MaskFeatures(float(r.actual_area), float(r.boundary_norm))
                for r in samples.itertuples()]
    normalizer = FeatureNormalizer.fit(features)
    lookup = frame.set_index(["sample_id", "k"])["full_lpips"].to_dict()
    image_ids = dict(zip(samples["sample_id"], samples["original_image_id"], strict=True))
    candidate_results: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for name, candidates in (("M1", CANDIDATE_WEIGHTS_AREA_BOUNDARY), ("A1", CANDIDATE_WEIGHTS_AREA)):
        losses = {}
        for weights in candidates:
            policy = BudgetPolicy(name, "score", tiers=tiers, normalizer=normalizer, weights=weights)
            assignments = allocate_batch(ids, features, policy)
            values = pd.DataFrame([
                {"image_id": image_ids[a.sample_id], "lpips": lookup[(a.sample_id, a.k)]}
                for a in assignments
            ])
            loss = float(values.groupby("image_id")["lpips"].mean().mean())
            losses[weights] = loss
            candidate_results.append({"policy": name, "weights": list(weights), SELECTION_METRIC: loss})
        weights = select_weights(losses, candidates)
        policy = BudgetPolicy(name, "score", tiers=tiers, normalizer=normalizer, weights=weights)
        thresholds = fit_thresholds([policy_score(f, policy) for f in features])
        selected[name] = {"weights": list(weights), "thresholds": list(thresholds)}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "normalization": "min_max_clipped_0_1",
        "selection_metric": SELECTION_METRIC,
        "normalizer": asdict(normalizer),
        "tiers": asdict(tiers),
        "rules": selected,
        "shuffle_seed": seed,
        "development": {
            "split": "dev", "seed": seed, "num_samples": len(samples),
            "original_image_ids": sorted(set(samples["original_image_id"])),
            "image_sha256": sorted(set(samples["image_sha256"])),
            "sample_ids": ids, "manifest_sha256": manifest_sha256,
        },
        "contract": contract,
        "source_provenance": provenance,
        "candidates": candidate_results,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def policies_from_frozen(payload: dict[str, Any]) -> dict[str, BudgetPolicy]:
    """校验冻结文件并恢复各策略独立的单图阈值。"""
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_hash(content):
        raise ValueError("冻结配置哈希不一致")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的冻结配置版本")
    if payload.get("normalization") != "min_max_clipped_0_1":
        raise ValueError("冻结配置归一化口径不匹配")
    if payload.get("selection_metric") != SELECTION_METRIC or payload["development"]["split"] != "dev":
        raise ValueError("冻结规则必须来自开发集 LPIPS 选择")
    normalizer = FeatureNormalizer(**payload["normalizer"])
    tiers = BudgetTiers(**payload["tiers"])
    rules = payload["rules"]
    for name, candidates in (("M1", CANDIDATE_WEIGHTS_AREA_BOUNDARY), ("A1", CANDIDATE_WEIGHTS_AREA)):
        if tuple(rules[name]["weights"]) not in candidates:
            raise ValueError(f"非法候选权重：{name}")
        thresholds = rules[name]["thresholds"]
        if len(thresholds) != 2 or not all(math.isfinite(value) for value in thresholds):
            raise ValueError("冻结阈值必须是两个有限值")
        if thresholds[0] > thresholds[1]:
            raise ValueError("冻结阈值顺序错误")
    policies = make_policies(
        tiers=tiers, normalizer=normalizer,
        thresholds=tuple(rules["M1"]["thresholds"]),
        m1_weights=tuple(rules["M1"]["weights"]),
        a1_weights=tuple(rules["A1"]["weights"]),
        shuffle_seed=payload["shuffle_seed"],
    )
    policies["A1"] = replace(policies["A1"], thresholds=tuple(rules["A1"]["thresholds"]))
    return policies


def save_frozen_rules(path: Path, payload: dict[str, Any]) -> None:
    policies_from_frozen(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 不静默覆盖已经发布的冻结规则。
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_frozen_rules(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policies_from_frozen(payload)
    return payload


def validate_application(
    payload: dict[str, Any], *, contract: dict[str, Any], source_sha256: str,
    image_ids: set[str], image_hashes: set[str],
) -> None:
    if payload["contract"] != contract:
        raise ValueError("采样配置或模型权重与冻结配置不一致")
    if payload["source_provenance"]["source_sha256"] != source_sha256:
        raise ValueError("实际源码与开发集冻结版本不一致")
    development = payload["development"]
    if image_ids.intersection(development["original_image_ids"]) or image_hashes.intersection(development["image_sha256"]):
        raise ValueError("测试图片与开发集重叠，禁止数据泄漏")
