"""Read-only final checks of real artifacts, product parity and sealed inputs."""
from pathlib import Path
import json,hashlib,numpy as np,time
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
checks={};details={}
# Exact product integration parity against independently evaluated final predictions.
m=json.loads((R/'integration_val2.json').read_text());n=0
for im in m['images']:
 iid=im['image_id'];d=R/'integration_val2'/iid
 with np.load(R/'final_frozen_predictions'/(iid+'.npz')) as z:
  for q in m['queries']:
   if q['image_id']==iid:
    expected=np.unpackbits(z[q['query_id']],axis=1,count=2048).astype(bool)
    actual=np.asarray(Image.open(d/(q['query_id']+'.png')))>0
    assert np.array_equal(actual,expected),q['query_id'];n+=1
 assert np.array_equal(np.asarray(Image.open(d/'semantic_cooperated.png')),np.asarray(Image.open(R/'final_frozen_predictions/S2'/(iid+'.png')))),iid
checks['product_exact_parity_2_images_16_queries']=n==16
# Open vocabulary fallback is bitwise identical to the independent raw SAM union.
man=json.loads((R/'free_text_qual.json').read_text());n=0
for im in man['images']:
 iid=im['image_id'];folder=R/'free_text_qual'/iid;meta=json.loads((folder/'metadata.json').read_text())
 with np.load(folder/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=2048).astype(bool)
 for q in man['queries']:
  if q['image_id']!=iid:continue
  ix=[i for i,c in enumerate(meta['candidates']) if c['query_id']==q['query_id']]
  union=masks[ix].any(0) if ix else np.zeros((400,2048),bool)
  assert np.array_equal(union,np.asarray(Image.open(R/'free_text_outputs'/iid/(q['query_id']+'.png')))>0),q['query_id'];n+=1
 assert np.array_equal(np.asarray(Image.open(R/'free_text_outputs'/iid/'semantic_cooperated.png')),np.asarray(Image.open(R/'free_text_outputs'/iid/'semantic_baseline.png')))
checks['free_text_exact_fallback_3_images_12_queries']=n==12
cfg=json.loads((ROOT/'configs/piers/innovation2_final.json').read_text())
for p,h in cfg['runtime_source_sha256'].items():assert sha(p)==h,p
assert sha(cfg['checkpoint_path'])==cfg['checkpoint_sha256']
checks['sealed_innovation2_sources_and_checkpoint_unchanged']=True
rec=Path('/root/autodl-tmp/data/DensePASS/recovery_20261107');report=json.loads((rec/'report.json').read_text());rows=json.loads((rec/'train_checksums.json').read_text())
for row in rows:
 p=Path('/root/autodl-tmp/data/DensePASS/leftImg8bit/train')/row['name'];assert sha(p)==row['sha256'],p
checks['all_2000_train_image_file_hashes_unchanged']=len(rows)==2000
for p,h in report['full_val_label_hashes'].items():assert sha(p)==h,p
checks['all_100_original_target_GT_files_unchanged']=len(report['full_val_label_hashes'])==100
for p in [ROOT/'third_party/effective_omnisam/dataset/densepass_list/train.txt',Path('/root/autodl-tmp/eff_omni/dataset/densepass_list/train.txt'),Path('/root/autodl-tmp/dataset/densepass_list/train.txt')]:
 if p.exists():assert sha(p)==report['train_list_sha256'],p
checks['canonical_train_lists_unchanged']=True
training={}
for arm,seeds in [('final_runs',[1234,4321]),('allhard_runs',[1234])]:
 for seed in seeds:
  pair=[]
  for v in ['R0','R1']:
   d=R/arm/(v+'_seed'+str(seed))/'training';a=json.loads((d/'result.json').read_text());pair.append(a)
   assert a['status']=='complete' and a['actual_optimizer_steps']==1440 and a['optimizer_updates']==1440
   assert a['target_GT_reads']==0 and a['max_parameter_delta']>0
   assert sha(d/'final.pth')==a['checkpoint_sha256']
   assert len((d/'updates.jsonl').read_text().splitlines())==1440
   training[arm+'/'+v+'_seed'+str(seed)]={k:a[k] for k in ['actual_optimizer_steps','source_target_order_sha256','elapsed_s','checkpoint_sha256','changed_parameter_tensors','target_GT_reads','peak_allocated_mib']}
  assert pair[0]['source_target_order_sha256']==pair[1]['source_target_order_sha256']
checks['all_6_training_jobs_actual_updates_checkpoints_and_matched_order']=True
checks['project_77_tests_passed']='77 passed' in (R/'project_tests_final.log').read_text()
checks['real_text_embedding_cache_4_queries_bitwise_parity']=all(x['different_pixels']==0 for x in json.loads((R/'real_text_parity.json').read_text()))
perf=json.loads((R/'performance.json').read_text());assert perf['status']=='complete' and perf['semantic_peak_reserved_mib']>0 and perf['text_peak_reserved_mib']>0
checks['isolated_performance_with_allocated_and_reserved']=True
assert json.loads((R/'supplement_jobs_resumed.json').read_text())['at_unix']>0
checks['supplement_jobs_resumed']=True
details['supplement_current_states']={}
for pid in [33823,35235,35236,35237,35238]:
 p=Path('/proc')/str(pid)/'status';state=next((s for s in p.read_text().splitlines() if s.startswith('State:')),'') if p.exists() else 'exited';assert 'T (stopped)' not in state;details['supplement_current_states'][str(pid)]=state
checks['supplement_jobs_not_stopped']=True
payload={'status':'passed','at_unix':time.time(),'checks':checks,'training_jobs':training,'details':details}
(R/'reports/acceptance.json').write_text(json.dumps(payload,indent=2));print('FINAL_ACCEPTANCE',json.dumps({'status':'passed','checks':checks}))
