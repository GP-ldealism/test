import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
from app.cache import source_provenance
from app.config import load_config
from app.data import sha256_file
from app.experiment import run_experiment
from app.policy_freeze import load_frozen_rules

validation = root/'results/validation/formal_preflight_20261116'
state_path = root/'results/main_test_20261116_status.json'
state = {'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

dump(state_path, state)
try:
    gate = json.loads((validation/'report.json').read_text(encoding='utf-8'))
    lock_path = validation/'selection_lock.json'
    assert gate['status'] == 'passed' and sha256_file(lock_path) == gate['selection_lock_sha256']
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert sha256_file(root/'data/manifest.csv') == lock['manifest_sha256']
    assert source_provenance(root)['source_sha256'] == lock['source_sha256']
    for row in lock['samples']:
        assert sha256_file(row['image_path']) == row['sha256']
        assert sha256_file(row['mask_path']) == row['mask_sha256']
    frozen_path = root/'results/raw/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/frozen_policy.json'
    assert load_frozen_rules(frozen_path)['sha256'] == lock['frozen_policy_sha256']
    report = run_experiment(load_config(root/'configs/main.yaml'), split='test',
        image_limit=30, target_areas=[.30], frozen_policy=frozen_path, warmup=True)
    payload = report.to_dict()
    dump(root/'results/main_test_20261116_report.json', payload)
    state.update(output_dir=payload['output_dir'], metrics_dir=payload['metrics_dir'])
    assert payload['status'] == 'ok' and payload['comparison_complete'] and payload['equal_nfe']
    assert (payload['num_images'],payload['num_samples'],payload['num_tier_runs'],payload['num_records'],payload['num_failures']) == (30,90,270,540,0)
    run, metrics = Path(payload['output_dir']), Path(payload['metrics_dir'])
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    assert selection['original_image_ids'] == lock['original_image_ids']
    assert selection['extension_image_ids'] == lock['extension_image_ids']
    hashes = json.loads((run/'input_hashes.json').read_text(encoding='utf-8'))
    for row in lock['samples']:
        if row['target_area'] == .30:
            assert hashes[row['sample_id']] == {'image_sha256':row['sha256'],'mask_sha256':row['mask_sha256']}
    tiers = pd.read_csv(run/'tier_results.csv')
    records = pd.read_csv(run/'samples.csv')
    summary = pd.read_csv(metrics/'summary.csv').set_index('policy')
    assert len(tiers) == 270 and not tiers.duplicated(['sample_id','k']).any()
    assert (tiers.status == 'ok').all() and (tiers.nfe == tiers.k.map({2:60,5:120,8:180})).all()
    assert len(list((run/'images').glob('*.png'))) == 270
    assert len(records) == 540 and not records.duplicated(['sample_id','policy']).any()
    assert set(records.policy) == {'B1','B2','B3','M1','A1','C1'}
    assert all(summary.loc[p,'nfe_total'] == 10800 for p in ('B2','M1','A1','C1'))
    for row in tiers.itertuples():
        assert (run/'images'/f'{row.sample_id}__k{row.k}.png').is_file()
    image_metrics = pd.read_csv(metrics/'per_image.csv')
    columns = ['full_lpips','masked_psnr','boundary_mae_3px']
    left = image_metrics[image_metrics.policy == 'M1'].set_index('original_image_id')
    right = image_metrics[image_metrics.policy == 'B2'].set_index('original_image_id')
    differences = (left[columns]-right[columns]).sort_index()
    assert len(differences) == 30 and not differences.isna().any().any()
    differences.to_csv(metrics/'paired_M1_minus_B2.csv')
    rng = np.random.default_rng(42)
    indices = rng.integers(0,30,size=(10000,30))
    paired = {}
    for col in columns:
        values = differences[col].to_numpy()
        interval = np.percentile(values[indices].mean(axis=1), [2.5,97.5])
        paired[col] = {'mean_difference':float(values.mean()),'ci95_percentile':interval.tolist(),
                       'negative':int((values<0).sum()),'zero':int((values==0).sum()),'positive':int((values>0).sum())}
    dump(metrics/'paired_statistics.json', {'comparison':'M1 minus B2','n_original_images':30,
         'bootstrap_seed':42,'bootstrap_resamples':10000,'metrics':paired})
    records.groupby(['original_image_id','mask_type','policy'])[columns].mean().reset_index().groupby(
        ['mask_type','policy'])[columns].agg(['mean','std','count']).to_csv(metrics/'by_mask_type.csv')
    verification = {'status':'passed','num_tiers':270,'num_policy_records':540,
                    'nfe_per_equal_budget_policy':10800,'selection_matches_preflight':True,
                    'source_sha256':lock['source_sha256'],'frozen_policy_sha256':lock['frozen_policy_sha256']}
    dump(metrics/'verification.json',verification)
    table = '| 策略 | 全图 LPIPS ↓ | 缺失区 PSNR ↑ | 边界 MAE ↓ | 总 NFE |\n|---|---:|---:|---:|---:|\n'
    for name in ('B1','B2','B3','M1','A1','C1'):
        r = summary.loc[name]
        table += f'| {name} | {r.mean_full_lpips:.6f} | {r.mean_masked_psnr:.4f} | {r.mean_boundary_mae_3px:.6f} | {int(r.nfe_total)} |\n'
    effect = paired['full_lpips']
    analysis = ('# 独立主测试报告\n\n30 张原图、3 类掩码、30% 目标面积、种子 42；270 条档位结果、540 条策略记录。'
                '使用开发集冻结规则，未根据测试效果调整参数。\n\n'+table+
                f'\nM1−B2 的逐原图 LPIPS 平均差为 {effect["mean_difference"]:.6f}，'
                f'95% 配对 bootstrap 百分位区间为 [{effect["ci95_percentile"][0]:.6f}, {effect["ci95_percentile"][1]:.6f}]。'
                f'改善 {effect["negative"]} 图、相同 {effect["zero"]} 图、退化 {effect["positive"]} 图。'
                '\n\n统计先在每张原图内平均三种掩码，再跨原图配对；重采样单位为原图，10000 次，分析种子 42。'
                '辅助指标和分掩码统计另见 paired_statistics.json 与 by_mask_type.csv。'
                '\n\n本报告仅覆盖主测试；面积扩展、另外两个种子及案例分析尚未完成，不据此声称跨面积或跨种子稳定。\n')
    (metrics/'main_test_analysis.md').write_text(analysis,encoding='utf-8')
    state['status'] = 'complete'
    print('[verification] formal main test PASSED',flush=True)
except BaseException as exc:
    state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    traceback.print_exc()
    raise
finally:
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    dump(state_path,state)
