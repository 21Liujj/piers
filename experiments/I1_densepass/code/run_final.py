"""Fixed two-seed matched final continuation and full development evaluation."""
from pathlib import Path
import subprocess,concurrent.futures,json,time,os
RUN=Path(__file__).resolve().parent;PY='/root/autodl-tmp/envs/piers/bin/python'
def one(job):
 variant,seed=job;out=RUN/'final_runs'/(variant+'_seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
 deadline=time.monotonic()+300
 while not (RUN/'pseudo_final/report.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Final pseudo generation stalled')
  time.sleep(2)
 if not (out/'training/result.json').exists():
  with (out/'train.log').open('w') as log:
   subprocess.run([PY,'-u',str(RUN/'train_driver.py'),'--pseudo',str(RUN/'pseudo_final'/variant/'epoch1'),'--out',str(out/'training'),'--iters','160','--seed',str(seed),'--variant',variant,'--shuffle-target'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('FINAL_TRAINED',variant,seed,flush=True)
 with (out/'eval.log').open('w') as log:
  subprocess.run([PY,'-u',str(RUN/'semantic_worker.py'),'--manifest',str(RUN/'val100.json'),'--out',str(out/'cache'),'--ckpt',str(out/'training/final.pth')],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out/'score.log').open('w') as log:
  subprocess.run([PY,str(RUN/'score_trained.py'),'--cache',str(out/'cache'),'--out',str(out/'evaluation')],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((out/'evaluation/metrics.json').read_text());print('FINAL_SCORED',variant,seed,result['mIoU'],flush=True);return variant,seed,result
if __name__=='__main__':
 os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');start=time.perf_counter();results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for v,seed,res in pool.map(one,[('R0',1234),('R1',1234),('R0',4321),('R1',4321)]):results[v+'_seed'+str(seed)]=res
 orders={}
 for seed in [1234,4321]:
  rows=[json.loads((RUN/'final_runs'/(v+'_seed'+str(seed))/'training/result.json').read_text()) for v in ['R0','R1']]
  assert rows[0]['source_target_order_sha256']==rows[1]['source_target_order_sha256']
  assert all(x['actual_optimizer_steps']==1440 for x in rows)
  orders[str(seed)]=rows[0]['source_target_order_sha256']
 (RUN/'final_runs/report.json').write_text(json.dumps({'status':'complete','metrics':results,'matched_data_order':orders,'elapsed_s':time.perf_counter()-start},indent=2));print('FINAL_MATCHED_ALL_COMPLETE',flush=True)
