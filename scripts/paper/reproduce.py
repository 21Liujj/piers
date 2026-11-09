"""Paper entry point: prepare, real inference, scoring and archived-ledger replay."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/piers'),str(Path(__file__).parent)]
from prepare import write,sha

def score(run,prepared,variants):
    import numpy as np
    from PIL import Image
    from score_zero_shot_native import confusion,metrics
    run=Path(run);prepared=Path(prepared)
    manifest=json.loads((prepared/'manifest.json').read_text())['images']
    index=json.loads((prepared/'SCORER_ONLY_INDEX.json').read_text());reports={};allh={}
    if len(manifest)!=len(index['images']):raise ValueError('Prediction and scorer image counts differ')
    if [r['image_id'] for r in manifest]!=[r['image_id'] for r in index['images']]:raise ValueError('Prediction/scorer order differs')
    lut=None
    if 'map' in index:lut=np.full(256,255,np.uint8);lut[:len(index['map'])]=index['map']
    for variant in variants:
        rows=[]
        for image,gtrow in zip(manifest,index['images']):
            folder=run/variant/image['image_id'];audit=json.loads((folder/'audit.json').read_text())
            if audit['input_sha256']!=image['input_sha256'] or audit['classes']!=index['classes']:raise ValueError('Mismatched prediction protocol')
            if audit['GT_reads']!=0 or audit['new_training']:raise ValueError('Invalid inference label access')
            if sha(folder/'prediction.png')!=audit['prediction_sha256'] or sha(gtrow['GT_path'])!=gtrow['GT_sha256']:raise ValueError('Output/GT changed')
            with Image.open(folder/'prediction.png') as im:prediction=np.array(im)
            with Image.open(gtrow['GT_path']) as im:gt=np.array(im)
            if lut is not None:gt=lut[gt]
            rows.append(confusion(prediction,gt,index['raw_ids'],index['ignore_raw_ids']))
        h=np.stack(rows);allh[variant]=h;report=metrics(h.sum(0));report.update(images=len(rows),classes=index['classes'])
        reports[variant]=report;(run/'results').mkdir(exist_ok=True);np.save(run/'results'/(variant+'_confusion.npy'),h.sum(0))
    np.savez_compressed(run/'results/per_image_histograms.npz',image_ids=np.array([r['image_id'] for r in manifest]),**allh)
    write(run/'results/summary.json',{'status':'complete','task':index['task'],'scope':index['scope'],'variants':reports})
    return reports

def run(a):
    from backend import configuration
    from benchmark_complete_model_v3 import gpu_pids
    prepared=Path(a.prepared).resolve();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh run folder')
    if gpu_pids():raise RuntimeError('GPU busy; do not stop unrelated jobs')
    out.mkdir(parents=True,exist_ok=True);variants=a.variants.split(',')
    if len(variants)!=len(set(variants)):raise ValueError('Duplicate variants')
    if min(a.workers_per_variant,a.max_gpu_workers,a.text_workers)<1:raise ValueError('Worker counts must be positive')
    for v in variants:configuration(v)
    task=json.loads((prepared/'SCORER_ONLY_INDEX.json').read_text())['task']
    if task not in ['densepass13'] and set(variants)-{'original','full'}:
        raise ValueError('Component ablation protocols are registered only for DensePASS13')
    cfgs={v:configuration(v) for v in variants}
    write(out/'PROTOCOL.json',{'task':task,'variants':cfgs,'manifest_sha256':sha(prepared/'manifest.json'),
        'new_training':False,'target_parameter_selection':False,'accuracy_text_cache_allowed':True,
        'latency_claim_allowed':False,'inference_inputs':'image,geometry,category names only'})
    text=Path(a.text_cache).resolve() if a.text_cache else out/'text'
    jobs=[];pending=[];logs=out/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    backend=str(Path(__file__).with_name('backend.py'))
    if any(c['i1'] for c in cfgs.values()) and not a.text_cache:
        for shard in range(a.text_workers):
            pending.append(('text',shard,[sys.executable,'-u',backend,'--out',str(out),'--manifest',str(prepared/'manifest.json'),
                 '--variant','full_cached' if set(variants)=={'full_cached'} else 'full','--text-worker','--shard',str(shard),'--shards',str(a.text_workers)]))
    for variant in variants:
        for shard in range(a.workers_per_variant):
            pending.append((variant,shard,[sys.executable,'-u',backend,'--out',str(out),'--manifest',str(prepared/'manifest.json'),
                '--text',str(text),'--variant',variant,'--shard',str(shard),'--shards',str(a.workers_per_variant)]))
    maximum=a.max_gpu_workers;running=[]
    if maximum<a.text_workers+1 and any(c['i1'] for c in cfgs.values()) and not a.text_cache:
        raise ValueError('Leave at least one semantic slot beyond text workers')
    while pending or running:
        while pending and len(running)<maximum:
            role,shard,command=pending.pop(0)
            with (logs/(role+'_s'+str(shard)+'.log')).open('w') as log:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running.append((process,role,shard));jobs.append({'pid':process.pid,'variant':role,'shard':shard,'command':command})
            write(out/'jobs.json',{'jobs':jobs});time.sleep(.2)
        remaining=[]
        for process,role,shard in running:
            code=process.poll()
            if code not in [None,0]:
                for control in (out/'services').glob('*/control/*'):
                    write(control/'request_999999_stop.json',{'operation':'stop'})
                raise RuntimeError('Worker failed: '+role+' shard'+str(shard)+'; see '+str(logs))
            if code is None:remaining.append((process,role,shard))
        running=remaining
        write(out/'STATUS.json',{'status':'running','active_workers':len(running),'queued_workers':len(pending),'time':time.time()})
        if pending or running:time.sleep(3)
    reports=score(out,prepared,variants)
    write(out/'COMPLETE.json',{'status':'complete','time':time.time()});write(out/'STATUS.json',{'status':'complete'})
    print(json.dumps({k:v['mIoU_pct'] for k,v in reports.items()},indent=2))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--task',required=True,choices=['densepass13','wildpass8','synpass22','synpass13'])
    prep.add_argument('--out',required=True);prep.add_argument('--data-root');prep.add_argument('--extra-root');prep.add_argument('--limit',type=int,default=0)
    r=sub.add_parser('run');r.add_argument('--prepared',required=True);r.add_argument('--out',required=True);r.add_argument('--variants',default='original,full')
    r.add_argument('--workers-per-variant',type=int,default=2);r.add_argument('--max-gpu-workers',type=int,default=8)
    r.add_argument('--text-workers',type=int,default=2);r.add_argument('--text-cache')
    s=sub.add_parser('score');s.add_argument('--prepared',required=True);s.add_argument('--run',required=True);s.add_argument('--variants',default='original,full')
    sub.add_parser('verify-ledgers')
    f=sub.add_parser('fetch-assets');f.add_argument('--weights-root');f.add_argument('--only',nargs='*')
    a=p.parse_args()
    if a.command=='prepare':
        from prepare import prepare
        print(prepare(a.task,a.out,a.data_root,a.extra_root,a.limit))
    elif a.command=='run':run(a)
    elif a.command=='score':print(json.dumps(score(a.run,a.prepared,a.variants.split(',')),indent=2))
    elif a.command=='verify-ledgers':
        from ledgers import verify
        verify()
    elif a.command=='fetch-assets':
        from assets import fetch
        fetch(a.weights_root,a.only)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
