import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
from PIL import Image
from app.cache import source_provenance
from app.config import load_config
from app.data import load_image, load_mask, save_result, sha256_file
from app.experiment import load_manifest, select_samples, sampling_contract, resolve_checkpoint_sha256
from app.metrics import evaluate
from app.policy_freeze import load_frozen_rules, validate_application
from app.restore import RestoreConfig, restore

out = root / 'results/validation/formal_preflight_20261116'
out.mkdir(parents=True, exist_ok=True)
def write(name, payload):
    (out/name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

dev = root/'results/raw/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb'
frozen = load_frozen_rules(dev/'frozen_policy.json')
config = load_config(root/'configs/main.yaml')
rows = load_manifest(config.data_manifest)
main = select_samples(rows, split='test', target_areas=[0.30], image_limit=30)
extension = select_samples(rows, split='test', target_areas=[0.15], image_limit=10)
ids = list(dict.fromkeys(row.original_image_id for row in main))
assert len(ids) == 30 and len(main) == 90 and len(extension) == 30
assert list(dict.fromkeys(row.original_image_id for row in extension)) == ids[:10]
provenance = source_provenance(root)
checkpoint_hash = resolve_checkpoint_sha256(config)
validate_application(frozen, contract=sampling_contract(config, checkpoint_hash),
    source_sha256=provenance['source_sha256'], image_ids=set(ids),
    image_hashes={r.sha256 for r in main+extension})
checked = []
for row in main+extension:
    actual_image_hash = sha256_file(row.image_path)
    assert actual_image_hash.lower() == row.sha256.lower()
    actual_area = float(load_mask(row.mask_path).mean())
    assert abs(actual_area-row.target_area) <= .02
    assert abs(actual_area-row.actual_area) <= 1e-6
    checked.append({**asdict(row), 'image_path': str(row.image_path), 'mask_path': str(row.mask_path),
                    'actual_area_verified': actual_area, 'mask_sha256': sha256_file(row.mask_path)})
cases = [('img0010_center15', 2), ('img0011_scratch30', 5), ('img0012_scatter30', 8)]
row_by_id = {r.sample_id:r for r in rows}
assert all(s in row_by_id and row_by_id[s].split == 'dev' for s,k in cases)
lock = {'created_at':datetime.now(timezone.utc).isoformat(), 'status':'locked_before_test',
        'reason':'Original manifest order; full 30/10 protocol chosen before test outcomes',
        'manifest_sha256':sha256_file(config.data_manifest), 'source_sha256':provenance['source_sha256'],
        'frozen_policy_sha256':frozen['sha256'], 'checkpoint_sha256':checkpoint_hash,
        'original_image_ids':ids, 'extension_image_ids':ids[:10], 'samples':checked,
        'seeds':[42,2026,3407], 'main_expected_tiers':270, 'main_expected_records':540,
        'primary_comparison':'M1 minus B2 full_lpips; per-original paired mean',
        'bootstrap':{'unit':'original_image','seed':42,'resamples':10000,'interval':'percentile 95%'},
        'direct_rerun_cases':[{'sample_id':s,'k':k} for s,k in cases]}
lock_path = out/'selection_lock.json'
if lock_path.exists():
    prior = json.loads(lock_path.read_text(encoding='utf-8'))
    for key in lock:
        if key != 'created_at':
            assert prior[key] == lock[key], f'Locked protocol changed: {key}'
else:
    write('selection_lock.json', lock)
print('[preflight] source, checkpoint, data isolation, 30/10 selection and masks passed', flush=True)
expected = pd.read_csv(dev/'tier_results.csv').set_index(['sample_id','k'])
results = []
for sample_id,k in cases:
    row = row_by_id[sample_id]
    reference, mask = load_image(row.image_path), load_mask(row.mask_path)
    result = restore(reference, mask, RestoreConfig(n_steps=k, outer_steps=20, seed=42,
                     device='cuda', checkpoint=config.checkpoint))
    target = out/f'{sample_id}__k{k}__direct.png'
    save_result(target, result.image)
    cached = dev/'images'/f'{sample_id}__k{k}.png'
    pixels_equal = np.array_equal(np.asarray(Image.open(target)), np.asarray(Image.open(cached)))
    metrics = evaluate(load_image(target), reference, mask)
    prior = expected.loc[(sample_id,k)]
    errors = {name:abs(float(getattr(metrics,name))-float(prior[name]))
              for name in ('full_lpips','masked_psnr','boundary_mae_3px')}
    passed = pixels_equal and metrics.status == 'ok' and result.nfe == int(prior.nfe) and max(errors.values()) <= 1e-6
    results.append({'sample_id':sample_id,'k':k,'passed':passed,'pixels_equal':pixels_equal,
                    'direct_sha256':sha256_file(target),'cached_sha256':sha256_file(cached),
                    'nfe':result.nfe,'elapsed_seconds':result.elapsed_seconds,
                    'metrics':asdict(metrics),'metric_absolute_errors':errors})
    write('direct_rerun.json', {'status':'running','cases':results})
    print(f'[direct] {sample_id} K={k}: pixels_equal={pixels_equal}, NFE={result.nfe}, passed={passed}', flush=True)
    if not passed:
        write('direct_rerun.json', {'status':'failed','cases':results})
        raise RuntimeError('Direct rerun differs from frozen development result; formal test stopped')
write('direct_rerun.json', {'status':'passed','cases':results})
write('report.json', {'status':'passed','checked_samples':len(checked),'num_main_images':30,
      'num_extension_images':10,'direct_reruns':3,'selection_lock_sha256':sha256_file(lock_path),
      'finished_at':datetime.now(timezone.utc).isoformat()})
print('[preflight] PASSED', flush=True)
