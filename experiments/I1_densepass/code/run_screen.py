"""Execute a fixed four-arm matched screen and full development scoring."""
from pathlib import Path
import subprocess,concurrent.futures,json,time,os
RUN=Path(__file__).resolve().parent;PY='/root/autodl-tmp/envs/piers/bin/python'
def one(variant):
 out=RUN/'screens'/variant;out.mkdir(parents=True,exist_ok=True)
 if not (out/'training/result.json').exists():
  with (out/'train.log').open('w') as log:
   subprocess.run([PY,'-u',str(RUN/'train_driver.py'),'--pseudo',str(RUN/'pseudo'/variant/'epoch1'),'--out',str(out/'training'),'--iters','20','--variant',variant],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out/'eval.log').open('w') as log:
  subprocess.run([PY,'-u',str(RUN/'semantic_worker.py'),'--manifest',str(RUN/'val100.json'),'--out',str(out/'cache'),'--ckpt',str(out/'training/final.pth')],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out/'score.log').open('w') as log:
  subprocess.run([PY,str(RUN/'score_trained.py'),'--cache',str(out/'cache'),'--out',str(out/'evaluation')],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((out/'evaluation/metrics.json').read_text());print('SCREEN_SCORED',variant,result['mIoU'],flush=True);return variant,result
if __name__=='__main__':
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
 os.environ['OPENBLAS_NUM_THREADS']='1';start=time.perf_counter();res={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for variant,result in pool.map(one,['R0','R1','Rnaive','Rsingle']):res[variant]=result
 order={v:json.loads((RUN/'screens'/v/'training/result.json').read_text())['source_target_order_sha256'] for v in res};assert len(set(order.values()))==1,order
 (RUN/'screens/report.json').write_text(json.dumps({'status':'complete','metrics':res,'data_order':order,'elapsed_s':time.perf_counter()-start},indent=2))
 print('MATCHED_SCREEN_COMPLETE',flush=True)
