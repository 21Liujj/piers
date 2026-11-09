"""Prediction-only worker: image paths and category names; no GT or dataset ID.

Text candidates are produced in independent Grounded-SAM2 processes. This
worker retains one semantic model and writes only final requested-class labels.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from full_model_fusion import make_queries,requested_probabilities,FrozenCooperation

FORBIDDEN=('/gtFine/','/evaluation_gt/','/labels/','/labels_raw/','/labels_trainid/','/SynPASS/semantic/')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def validate_manifest(manifest):
    for row in manifest['images']:
        if set(row)-{'image_id','path','width','height','original_size','classes','input_sha256','source_sha256'}:
            raise ValueError('Inference accepts only image geometry, hashes and category names')
        if any(x in str(Path(row['path']).resolve()) for x in FORBIDDEN):raise ValueError('GT path forbidden')
        if (row['width'],row['height'])!=(2048,400):raise ValueError('Canonical input must be 2048x400')
        make_queries(row['image_id'],row['classes'])
    ids=[x['image_id'] for x in manifest['images']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate inference image IDs')

def load_text(folder,row,timeout):
    start=time.monotonic();meta_path=folder/row['image_id']/'metadata.json';mask_path=meta_path.with_name('candidates.npz')
    while True:
        failures=list(folder.glob('failure_s*.txt'))
        if failures:raise RuntimeError('Text worker failed: '+failures[0].read_text()[-1000:])
        if meta_path.is_file() and mask_path.is_file():
            try:
                meta=json.loads(meta_path.read_text())
                with np.load(mask_path,allow_pickle=False) as z:
                    shape=tuple(z['shape']);packed=z['masks_packed'].copy()
                if shape!=(400,2048) or meta['shape']!=[400,2048]:raise ValueError('Text shape mismatch')
                assert meta['status']=='complete' and meta['no_GT_read']
                assert meta['image_id']==row['image_id'] and Path(meta['path'])==Path(row['path'])
                assert [f['x_start'] for f in meta['fovs']]==[0,512,1024,1536]
                assert len(meta['candidates'])==len(packed)
                assert [c['mask_index'] for c in meta['candidates']]==list(range(len(packed)))
                masks=np.unpackbits(packed,axis=2,count=2048).astype(bool)
                return meta,masks
            except (json.JSONDecodeError,EOFError,OSError):pass
        if time.monotonic()-start>timeout:raise TimeoutError('Text input not ready: '+row['image_id'])
        time.sleep(1)

def main(a):
    import cv2
    from full_model_semantic import SemanticEngine
    repo=Path(a.repo);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(a.manifest).read_text());validate_manifest(manifest)
    protocol=json.loads(Path(a.protocol).read_text());protocol_sha=sha(a.protocol)
    fingerprint={'manifest_sha256':sha(a.manifest),'protocol_sha256':protocol_sha,
                 'worker_sha256':sha(__file__),'fusion_sha256':sha(Path(__file__).with_name('full_model_fusion.py')),
                 'semantic_engine_sha256':sha(Path(__file__).with_name('full_model_semantic.py')),'method':a.method}
    rows=manifest['images'][a.shard::a.shards]
    if a.limit:rows=rows[:a.limit]
    model=SemanticEngine(repo,'dense' if a.method=='original' else 'i2',0 if a.method=='original' else 2)
    cooperation=FrozenCooperation(repo,a.calibrators) if a.method=='full' else None
    model_time=0.;begin=time.monotonic();done=0
    for row in rows:
        iid=row['image_id'];dest=out/iid;dest.mkdir(exist_ok=True);audit_path=dest/'audit.json';prediction=dest/'prediction.png'
        if prediction.exists() and audit_path.exists():
            prior=json.loads(audit_path.read_text())
            if prior['fingerprint']!=fingerprint or prior['input_sha256']!=row['input_sha256']:raise RuntimeError('Stale prediction cache')
            done+=1;continue
        if sha(row['path'])!=row['input_sha256']:raise RuntimeError('Input image changed')
        started=time.monotonic()
        if a.method=='full':meta,masks=load_text(Path(a.text),row,a.wait_timeout)
        probs,label,semantic_audit=model.predict(row['path']);model_time+=semantic_audit['elapsed_s']
        queries,bank,output_queries,channels=make_queries(iid,row['classes'])
        if a.method=='full':
            assert {q['query_id'] for q in queries}=={q['query_id'] for q in meta['queries']}
            pred,cooperation_audit,guided=cooperation.apply(meta,masks,probs,label,bank,output_queries,row['classes'])
            del masks,guided
        else:
            _,pred=requested_probabilities(probs,row['classes'])
            cooperation_audit={'unsupported_original_head_classes':[name for name,c in zip(row['classes'],channels) if c is None],
                               'original_model_has_no_open_vocabulary_output_for_these_classes':True}
        width,height=row['original_size'];native=cv2.resize(pred,(width,height),interpolation=cv2.INTER_NEAREST)
        tmp=prediction.with_name('prediction.partial.png');Image.fromarray(native).save(tmp);tmp.replace(prediction)
        report={'status':'complete','image_id':iid,'method':a.method,'classes':row['classes'],
                'input_sha256':row['input_sha256'],'source_sha256':row['source_sha256'],
                'fingerprint':fingerprint,'semantic':semantic_audit,'cooperation':cooperation_audit,
                'native_output_WH':[width,height],'canonical_input_WH':[2048,400],
                'prediction_sha256':sha(prediction),'unknown_pixels':int((native==255).sum()),
                'dataset_identity_input':False,'target_GT_reads':0,'target_training':False,
                'elapsed_s_including_text_wait_not_isolated_latency':time.monotonic()-started}
        atomic_json(audit_path,report);done+=1
        atomic_json(out/('progress_s'+str(a.shard)+'.json'),{'done':done,'total':len(rows),'elapsed_s':time.monotonic()-begin,'semantic_seconds':model_time,'last_image':iid})
        print('PREDICTION',a.method,a.shard,done,len(rows),iid,round(report['elapsed_s_including_text_wait_not_isolated_latency'],2),flush=True)
    atomic_json(out/('DONE_s'+str(a.shard)+'.json'),{'status':'complete','images':len(rows),'fingerprint':fingerprint,'elapsed_s':time.monotonic()-begin})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=['original','full'],required=True)
    p.add_argument('--text');p.add_argument('--calibrators');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--wait-timeout',type=int,default=21600);a=p.parse_args()
    try:main(a)
    except BaseException:
        Path(a.out).mkdir(parents=True,exist_ok=True);atomic_json(Path(a.out)/('FAIL_s'+str(a.shard)+'.json'),{'error':traceback.format_exc()});raise
