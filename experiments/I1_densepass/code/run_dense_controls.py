from pathlib import Path
import os,sys,time,json,subprocess,concurrent.futures,signal,traceback
R=Path(__file__).resolve().parent;D=R/'dense_controls';PY='/root/autodl-tmp/envs/piers/bin/python';env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');start=time.time()
def state(stage,**kw):
 (D/'progress.json').write_text(json.dumps({'stage':stage,'at_unix':time.time(),'elapsed_s':time.time()-start,**kw},indent=2));print('DENSE_CONTROL_STAGE',stage,kw,flush=True)
def command(cmd,log):
 with Path(log).open('w') as f:subprocess.run([PY,'-u',*map(str,cmd)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def cache(split,shard):command([R/'semantic_worker_dense.py','--manifest',R/('val100.json' if split=='val' else 'train160.json'),'--out',D/('cache_'+split),'--shard',shard,'--shards',2],D/(split+'_shard'+str(shard)+'.log'))
def train(job):
 v,seed=job;out=D/'trained'/(v+'_seed'+str(seed));out.mkdir(parents=True,exist_ok=True)
 if not (out/'training/result.json').exists():command([R/'train_driver_dense.py','--pseudo',D/'pseudo'/v/'epoch1','--out',out/'training','--iters',160,'--seed',seed,'--variant',v,'--shuffle-target'],out/'train.log')
 command([R/'semantic_worker_dense.py','--manifest',R/'val100.json','--out',out/'cache','--ckpt',out/'training/final.pth'],out/'eval.log')
 command([R/'score_trained.py','--cache',out/'cache','--out',out/'evaluation'],out/'score.log')
 return v+'_seed'+str(seed),json.loads((out/'evaluation/metrics.json').read_text())
def resume():
 for j in json.loads((D/'paused_jobs.json').read_text()):
  p=Path('/proc')/str(j['pid'])/'cmdline'
  if p.exists() and p.read_bytes().hex()==j['cmdline_hex']:os.kill(j['pid'],signal.SIGCONT)
 (D/'supplements_resumed.json').write_text(json.dumps({'at_unix':time.time()}))
if '--watchdog' in sys.argv:
 deadline=time.time()+3600
 while time.time()<deadline and not (D/'resume.flag').exists():time.sleep(15)
 resume();sys.exit(0)
try:
 jobs=[]
 for j in json.loads((R/'paused_supplement_jobs.json').read_text()):
  p=Path('/proc')/str(j['pid'])/'cmdline'
  if p.exists() and p.read_bytes().hex()==j['cmdline_hex']:jobs.append(j)
 (D/'paused_jobs.json').write_text(json.dumps(jobs,indent=2))
 for j in jobs:os.kill(j['pid'],signal.SIGSTOP)
 with (D/'watchdog.log').open('w') as f:subprocess.Popen([PY,str(Path(__file__).resolve()),'--watchdog'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 state('uncompressed_teacher_val100')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:cache('val',i),[0,1]))
 state('uncompressed_teacher_train160')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:cache('train',i),[0,1]))
 state('frozen_comparison_and_no_GT_pseudos');command([R/'dense_control_build.py'],D/'build.log')
 state('matched_dense_R0_R1_two_seeds')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:metrics=dict(pool.map(train,[('R0',1234),('R1',1234),('R0',4321),('R1',4321)]))
 orders={}
 for seed in [1234,4321]:
  pair=[json.loads((D/'trained'/(v+'_seed'+str(seed))/'training/result.json').read_text()) for v in ['R0','R1']];assert all(x['actual_optimizer_steps']==1440 and x['target_GT_reads']==0 for x in pair);assert pair[0]['source_target_order_sha256']==pair[1]['source_target_order_sha256'];orders[str(seed)]=pair[0]['source_target_order_sha256']
 (D/'trained/report.json').write_text(json.dumps({'status':'complete','metrics':metrics,'matched_data_order':orders,'elapsed_s':time.time()-start},indent=2));state('all_complete')
except BaseException:
 (D/'failure.txt').write_text(traceback.format_exc());state('failed');raise
finally:
 (D/'resume.flag').touch()
 if (D/'paused_jobs.json').exists():resume()
