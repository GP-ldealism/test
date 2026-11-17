"""Resume the preregistered formal campaign without changing frozen inference source."""
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.cache import source_provenance
from app.config import load_config
from app.data import sha256_file
from app.experiment import run_experiment, sampling_contract
from app.policy_freeze import load_frozen_rules, validate_application

VALIDATION = ROOT/'results/validation/formal_preflight_20261116'
STATE = ROOT/'results/formal_followup_20261116_status.json'
FROZEN = ROOT/'results/raw/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/frozen_policy.json'
MAIN_STATE = ROOT/'results/main_test_20261116_status.json'
MAIN_REPORT = ROOT/'results/main_test_20261116_report.json'
METRICS = ['full_lpips','masked_psnr','boundary_mae_3px']
POLICIES = ['B1','B2','B3','M1','A1','C1']

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def check_inputs():
    gate = read(VALIDATION/'report.json')
    lock = read(VALIDATION/'selection_lock.json')
    assert gate['status'] == 'passed'
    assert sha256_file(VALIDATION/'selection_lock.json') == gate['selection_lock_sha256']
    assert sha256_file(ROOT/'data/manifest.csv') == lock['manifest_sha256']
    assert source_provenance(ROOT)['source_sha256'] == lock['source_sha256']
    frozen = load_frozen_rules(FROZEN)
    assert frozen['sha256'] == lock['frozen_policy_sha256']
    for config_name in ('main','area','repeat'):
        config = load_config(ROOT/f'configs/{config_name}.yaml')
        validate_application(frozen,contract=sampling_contract(config,lock['checkpoint_sha256']),
            source_sha256=lock['source_sha256'],image_ids=set(lock['original_image_ids']),
            image_hashes={r['sha256'] for r in lock['samples']})
    for row in lock['samples']:
        assert sha256_file(row['image_path']) == row['sha256']
        assert sha256_file(row['mask_path']) == row['mask_sha256']
    return lock

def verify(payload, lock, main=False):
    n = 30 if main else 10
    assert payload['status'] == 'ok' and payload['comparison_complete'] and payload['equal_nfe']
    assert [payload[k] for k in ('num_images','num_samples','num_tier_runs','num_records','num_failures')] == [n,n*3,n*9,n*18,0]
    run, metrics = Path(payload['output_dir']), Path(payload['metrics_dir'])
    selected = read(run/'selection.json')
    ids = lock['original_image_ids'] if main else lock['extension_image_ids']
    assert selected['original_image_ids'] == ids
    assert read(run/'source_provenance.json')['source_sha256'] == lock['source_sha256']
    assert load_frozen_rules(run/'frozen_policy.json')['sha256'] == lock['frozen_policy_sha256']
    inputs = read(run/'input_hashes.json')
    frozen_inputs = {r['sample_id']:r for r in lock['samples']}
    assert set(inputs) == set(selected['sample_ids']) and len(inputs) == n*3
    for sample_id, hashes in inputs.items():
        expected = frozen_inputs[sample_id]
        assert hashes == {'image_sha256':expected['sha256'],'mask_sha256':expected['mask_sha256']}
    tiers = pd.read_csv(run/'tier_results.csv')
    records = pd.read_csv(run/'samples.csv')
    assert len(tiers) == n*9 and not tiers.duplicated(['sample_id','k']).any()
    assert set(tiers.sample_id) == set(inputs) and (tiers.status == 'ok').all()
    assert (tiers.nfe == tiers.k.map({2:60,5:120,8:180})).all()
    assert len(records) == n*18 and not records.duplicated(['sample_id','policy']).any()
    assert set(records.policy) == set(POLICIES) and set(records.sample_id) == set(inputs)
    assert np.isfinite(records[METRICS].to_numpy()).all()
    totals = records.groupby('policy').nfe.sum()
    assert all(totals[p] == n*360 for p in ('B2','M1','A1','C1'))
    for row in tiers.itertuples():
        assert (run/'images'/f'{row.sample_id}__k{row.k}.png').is_file()
    write(metrics/'campaign_verification.json',{'status':'passed','verified_at':now(),
          'num_images':n,'num_tiers':n*9,'num_policy_records':n*18,
          'nfe_per_equal_budget_policy':n*360,'inputs_match_preregistered_hashes':True})
    return records

def main_alive():
    process = read(ROOT/'.cache/formal-main-20261116-process.json')
    pid = int(process['pid'])
    result = subprocess.run(['pwsh','-NoProfile','-Command',
        f'Get-Process -Id {pid} -ErrorAction Stop | Select-Object -ExpandProperty ProcessName'],
        capture_output=True,text=True)
    return result.returncode == 0 and result.stdout.strip() == 'pwsh'

def paired(frame):
    per_image = frame.groupby(['original_image_id','policy'])[METRICS].mean()
    difference = per_image.xs('M1',level='policy')-per_image.xs('B2',level='policy')
    difference = difference.sort_index()
    n = len(difference)
    indices = np.random.default_rng(42).integers(0,n,size=(10000,n))
    result = {'num_original_images':n,'bootstrap_seed':42,'resamples':10000,'metrics':{}}
    for name in METRICS:
        values = difference[name].to_numpy()
        result['metrics'][name] = {'mean_M1_minus_B2':float(values.mean()),
            'ci95_percentile':np.percentile(values[indices].mean(axis=1),[2.5,97.5]).tolist(),
            'negative':int((values<0).sum()),'zero':int((values==0).sum()),'positive':int((values>0).sum())}
    return difference,result

def analyze(reports, lock, output=None):
    output = Path(output) if output is not None else ROOT/'results/metrics/formal_campaign_20261116'
    output.mkdir(parents=True,exist_ok=True)
    frames = {name:pd.read_csv(Path(r['output_dir'])/'samples.csv') for name,r in reports.items()}
    subset = frames['main'][frames['main'].original_image_id.isin(lock['extension_image_ids'])].copy()
    groups = {'main_30_images':frames['main'], 'subset_area30_seed42':subset,
              'subset_area15_seed42':frames['area'],
              'subset_area30_seed2026':frames['seed2026'],
              'subset_area30_seed3407':frames['seed3407']}
    # Each area/seed contributes equally within each original image. Keep existing assignments.
    groups['subset_both_areas_seed42'] = pd.concat([subset,frames['area']],ignore_index=True)
    groups['subset_area30_three_seeds'] = pd.concat([subset,frames['seed2026'],frames['seed3407']],ignore_index=True)
    results, summaries = {}, []
    for name,frame in groups.items():
        differences,results[name] = paired(frame)
        differences.to_csv(output/f'{name}_paired.csv')
        quality = frame.groupby(['original_image_id','policy'])[METRICS].mean().groupby('policy').mean()
        costs = frame.groupby('policy').agg(nfe_total=('nfe','sum'),
                 recorded_sampling_seconds=('elapsed_seconds','sum'),peak_vram_mb=('peak_vram_mb','max'))
        table = quality.join(costs).reset_index()
        table.insert(0,'group',name)
        table['num_original_images'] = frame.original_image_id.nunique()
        summaries.append(table)
    summary = pd.concat(summaries,ignore_index=True)
    summary.to_csv(output/'summary.csv',index=False)
    write(output/'paired_statistics.json',results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    main = summary[summary.group == 'main_30_images'].set_index('policy')
    for ax,metric in zip(axes,METRICS):
        ax.plot([main.loc[p,'nfe_total'] for p in ['B1','B2','B3']],
                [main.loc[p,metric] for p in ['B1','B2','B3']], 'o-',label='Fixed K')
        for p in ['M1','A1','C1']:
            ax.scatter(main.loc[p,'nfe_total'],main.loc[p,metric],label=p)
        ax.set(xlabel='Total NFE (30 images)',ylabel=metric)
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output/'main_quality_cost.png',dpi=160)
    plt.close(fig)
    lines = ['# 独立测试与复核报告','',
        '主测试 30 张独立原图，面积与三种子复核共用预选 10 张子集；四次运行共 540 条档位结果、1080 条策略记录。',
        '各次运行通过完整性、输入哈希和实际 NFE 核验。使用开发集冻结参数，未依据测试效果重新选规则。','',
        '| 比较范围 | 原图数 | M1−B2 LPIPS | 95% 配对 bootstrap 区间 | 改善/持平/退化图数 |',
        '|---|---:|---:|---|---|']
    for name,entry in results.items():
        value = entry['metrics']['full_lpips']
        lo,hi = value['ci95_percentile']
        lines.append(f'| {name} | {entry["num_original_images"]} | {value["mean_M1_minus_B2"]:.6f} | [{lo:.6f}, {hi:.6f}] | {value["negative"]}/{value["zero"]}/{value["positive"]} |')
    lines += ['', '差值小于零表示 M1 的 LPIPS 更好；主比较为 30 图主测试，其余为子集复核。区间跨零时，收益证据有限。',
        '先按原图平均掩码、面积或种子，再按原图配对重采样（10000 次，分析种子 42）。同图重复项不作为独立原图。',
        '混合面积沿用各自运行的预算分配，不重新排序。跨三种子的合并区间反映原图不确定性，不能替代更多种子的验证。',
        '完整六策略与辅助指标见 summary.csv，逐组配对指标见 paired_statistics.json。成本记录包含缓存的历史推理时间，不等于当前运行墙钟时间。',
        '', '![主测试质量与计算量](main_quality_cost.png)', '',
        '案例图与结构、纹理、语义失败分析仍需单独完成；不据单项指标概括为全面提升。','']
    (output/'formal_campaign_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return str(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    lock = check_inputs()
    if args.check_only:
        print('[check] frozen source, configs, selection and all input hashes PASSED',flush=True)
        return
    with (ROOT/'.cache/formal-followup-20261116.lock').open('a+b') as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        previous = read(STATE) if STATE.exists() else {}
        state = {'status':'waiting_main','started_at':now(),'pid':os.getpid(),
                 'reports':previous.get('reports',{})}
        write(STATE,state)
        try:
            deadline = time.monotonic()+8*3600
            print('[queue] waiting for successful main test verification; GPU idle in this worker',flush=True)
            while True:
                status = read(MAIN_STATE)
                if status['status'] == 'complete':
                    break
                if status['status'] == 'failed':
                    raise RuntimeError(f'Main test failed: {status.get("error")}')
                if not main_alive():
                    # Allow a process exit and final status write to settle once.
                    time.sleep(2)
                    if read(MAIN_STATE)['status'] == 'complete':
                        break
                    raise RuntimeError('Main process stopped before verification; resume main before follow-up')
                if time.monotonic()>deadline:
                    raise TimeoutError('Main test did not complete within 8 hours')
                time.sleep(20)
            check_inputs()
            baseline = read(MAIN_REPORT)
            verify(baseline,lock,main=True)
            state['reports']['main'] = baseline
            for name,config_name,area,seed in [('area','area',.15,42),('seed2026','repeat',.30,2026),('seed3407','repeat',.30,3407)]:
                if name in state['reports']:
                    verify(state['reports'][name],lock)
                    continue
                check_inputs()
                state.update(status='running',stage=name)
                write(STATE,state)
                print(f'[queue] starting {name}',flush=True)
                config = replace(load_config(ROOT/f'configs/{config_name}.yaml'),seed=seed)
                report = run_experiment(config,split='test',image_offset=0,image_limit=10,
                    target_areas=[area],baseline_run=baseline['output_dir'],frozen_policy=FROZEN,warmup=True)
                payload = report.to_dict()
                write(ROOT/f'results/formal_{name}_20261116_report.json',payload)
                verify(payload,lock)
                state['reports'][name] = payload
                write(STATE,state)
                print(f'[queue] {name} verified: 90 tiers, 180 records, 3600 NFE per equal-budget policy',flush=True)
            state.update(status='analyzing',stage='combined_analysis')
            write(STATE,state)
            state['metrics_dir'] = analyze(state['reports'],lock)
            state.update(status='complete',finished_at=now())
            print('[queue] all formal runs and combined statistics complete',flush=True)
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now())
            traceback.print_exc()
            raise
        finally:
            write(STATE,state)

if __name__ == '__main__':
    main()
