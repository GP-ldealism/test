"""离线验证规则选择、独立阈值、冻结校验与数据隔离。"""

import copy

import pandas as pd
import pytest

from app.budget import BudgetTiers, allocate_batch
from app.features import MaskFeatures
from app.policy_freeze import (
    canonical_hash,
    fit_development_rules,
    load_frozen_rules,
    policies_from_frozen,
    save_frozen_rules,
    validate_application,
)


def observations():
    # 面积越大，越受益于低预算；确定性地让 (-1, 0) 胜出。
    return pd.DataFrame([
        {"sample_id": f"s{i}", "original_image_id": f"img{i}", "split": "dev",
         "actual_area": 0.1 + 0.1 * i, "boundary_norm": 4.0,
         "image_sha256": f"hash{i}", "k": k,
         "full_lpips": float(abs(k - preferred)), "status": "ok"}
        for i, preferred in enumerate((8, 5, 2)) for k in (2, 5, 8)
    ])


def fit(frame=None):
    return fit_development_rules(
        observations() if frame is None else frame, tiers=BudgetTiers(),
        contract={"outer_steps": 20}, provenance={"source_sha256": "source-v1"},
        manifest_sha256="manifest", seed=42,
    )


def test_selects_best_candidate_with_deterministic_tie_break():
    frozen = fit()
    assert frozen["rules"]["A1"]["weights"] == [-1.0, 0.0]
    # 边界恒为零，(-1, 1) 与 (-1, -1) 同分，取前者。
    assert frozen["rules"]["M1"]["weights"] == [-1.0, 1.0]
    policies = policies_from_frozen(frozen)
    assigned = allocate_batch(["s0", "s1", "s2"],
                              [MaskFeatures(0.1 + i * 0.1, 4.0) for i in range(3)], policies["M1"])
    assert [item.k for item in assigned] == [8, 5, 2]


def test_roundtrip_retains_independent_thresholds_and_no_overwrite(tmp_path):
    frame = observations()
    frame["boundary_norm"] = frame["sample_id"].map({"s0": 2.0, "s1": 9.0, "s2": 4.0})
    frozen = fit(frame)
    path = tmp_path / "frozen.json"
    save_frozen_rules(path, frozen)
    loaded = load_frozen_rules(path)
    assert loaded == frozen
    for name in ("M1", "A1"):
        assert policies_from_frozen(loaded)[name].thresholds == tuple(frozen["rules"][name]["thresholds"])
    with pytest.raises(FileExistsError):
        save_frozen_rules(path, frozen)


@pytest.mark.parametrize("problem", ["test", "missing", "duplicate", "nan", "failed"])
def test_invalid_development_cannot_freeze(problem):
    frame = observations()
    if problem == "test":
        frame.loc[0, "split"] = "test"
    elif problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "nan":
        frame.loc[0, "full_lpips"] = float("nan")
    else:
        frame.loc[0, "status"] = "error"
    with pytest.raises(ValueError):
        fit(frame)


def test_tampered_or_incompatible_freeze_rejected():
    frozen = fit()
    altered = copy.deepcopy(frozen)
    altered["normalizer"]["area_min"] = 0.0
    with pytest.raises(ValueError, match="哈希"):
        policies_from_frozen(altered)
    altered["schema_version"] = 99
    altered["sha256"] = canonical_hash({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="版本"):
        policies_from_frozen(altered)


def test_apply_checks_contract_source_and_original_image_leakage():
    frozen = fit()
    args = {"contract": {"outer_steps": 20}, "source_sha256": "source-v1",
            "image_ids": {"test0"}, "image_hashes": {"test-hash"}}
    validate_application(frozen, **args)
    for change in ({"contract": {"outer_steps": 10}}, {"source_sha256": "changed"},
                   {"image_ids": {"img0"}}, {"image_hashes": {"hash0"}}):
        with pytest.raises(ValueError):
            validate_application(frozen, **(args | change))
