"""Reproducible same-checkpoint indoor original/full experiment commands."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'runs/INDOOR-FOLD1-20261109'
DEFAULT=REPO/'runs/INDOOR-MATCHED-20261112'
CLASSES=['ceiling','chair','door','floor','sofa','table','wall','window']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(a):
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    ckpt=OLD/'train/indoor_lora_last.pth'
    stats=json.loads((OLD/'SPAN8_CLASS_STATS.json').read_text())['fit']['class_pixel_pct']
    rare=[i for i,c in enumerate(CLASSES) if stats[c]<1.]
    manifests={}
    for split in ('calib','heldout'):
        rows=[json.loads(s) for s in (OLD/f'split_{split}.jsonl').read_text().splitlines() if s.strip()]
        manifests[split]=[{'image_id':r['id'],'path':str(Path('/root/autodl-tmp/data_extra/Stanford2d3d_Seg')/r['image']),
                          'input_sha256':sha(Path('/root/autodl-tmp/data_extra/Stanford2d3d_Seg')/r['image'])} for r in rows]
        write_json(root/f'inference_{split}.json',manifests[split])
    protocol={'version':'indoor-matched-full-v1','classes':CLASSES,'shared_checkpoint':str(ckpt),
        'checkpoint_sha256':sha(ckpt),'fit_images':832,'calibration_images':104,'heldout_images':373,
        'rare_class_ids':rare,'rare_class_source':'fit split only','shared_8class_head':True,
        'original':{'memory':'dense','I3_budget':0,'I1':False,'forwards':18},
        'full':{'memory':'i2','I3_budget':2,'I1':'GroundingDINO+SAM2 calibrated cooperation','forwards':20},
        'geometry':{'canvas':[1024,512],'window':512,'stride':64,'passes':2},
        'I3_candidate_starts':[640,704,768,832,896],
        'I1_gate':{'pixel_text_confidence_threshold':.9,'distinct_FoV_votes_required':2,
                   'baseline_ignore_confidence':.85,'teacher_high_conflict_protection':.8},
        'calibrator_fit':{'steps':1000,'lr':.05,'l2':.001,'seed':20261107,'dtype':'float64',
                          'optimizer':'Adam full batch','max_samples_per_class_view':256},
        'text':{'classes':CLASSES,'fovs':4,'window':512,'stride':256,'box_thr':.35,'text_thr':.25,
                'dtype':'bf16','feature_cache':False},
        'heldout_GT_for_parameter_selection':False,'stage2_CLIP_checkpoint_used':False,
        'limitations':['Single area5 heldout fold; fit/calib partition uses camera UUIDs, so room-disjointness is not established. Calibration remains in training areas only.'],
        'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('indoor_matched_*.py')}}
    write_json(root/'protocol.json',protocol)
    print(json.dumps(protocol,indent=2),flush=True)

def semantic(a):
    import numpy as np
    from indoor_matched_semantic import IndoorSemanticEngine
    root=Path(a.out);protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/f'inference_{a.split}.json').read_text())
    if a.limit: rows=rows[:a.limit]
    assert sha(protocol['shared_checkpoint'])==protocol['checkpoint_sha256']
    engine=IndoorSemanticEngine(protocol['shared_checkpoint'], 'dense' if a.arm=='original' else 'i2',
                               0 if a.arm=='original' else 2,protocol['rare_class_ids'])
    target=root/'semantic'/a.split/a.arm;target.mkdir(parents=True,exist_ok=True)
    fingerprint={'protocol_sha256':sha(root/'protocol.json'),'semantic_source_sha256':sha(Path(__file__).with_name('indoor_matched_semantic.py')),
                 'checkpoint_sha256':engine.checkpoint_sha256,'lora_sha256':engine.lora_sha256}
    started=time.monotonic()
    for index,row in enumerate(rows):
        audit=target/(row['image_id']+'.json');pred=target/(row['image_id']+'.npz')
        if audit.exists() and pred.exists():
            assert json.loads(audit.read_text())['fingerprint']==fingerprint
            continue
        assert sha(row['path'])==row['input_sha256']
        probs,label,report=engine.predict(row['path'])
        arrays={'prediction':label}
        if a.arm!='original': arrays['probabilities']=probs
        tmp=pred.with_name(pred.name+'.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
        tmp.replace(pred)
        write_json(audit,{'fingerprint':fingerprint,'input_sha256':row['input_sha256'],**report})
        write_json(target/'progress.json',{'done':index+1,'total':len(rows),'elapsed_s':time.monotonic()-started,'last_image':row['image_id']})
        print('SEMANTIC',a.split,a.arm,index+1,len(rows),round(report['elapsed_s'],3),flush=True)
    write_json(target/'DONE.json',{'n':len(rows),'fingerprint':fingerprint,'elapsed_s':time.monotonic()-started})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','semantic','text','calibrate','fuse','score'])
    parser.add_argument('--out',default=str(DEFAULT));parser.add_argument('--split',default='heldout',choices=['calib','heldout'])
    parser.add_argument('--arm',default='original',choices=['original','full_core']);parser.add_argument('--limit',type=int,default=0)
    a=parser.parse_args()
    if a.action=='prepare':prepare(a)
    elif a.action=='semantic':semantic(a)
    else:
        from indoor_matched_finish import run
        run(a)

if __name__=='__main__':main()
