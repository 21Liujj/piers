"""Score original native target labels after GT-free prediction is complete.

Only the scorer receives dataset identities, GT paths and raw label IDs. Unknown
predictions count as misses through an extra confusion-matrix column.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def confusion(pred,gt,raw_ids,ignore_ids):
    pred=np.asarray(pred);gt=np.asarray(gt);n=len(raw_ids)
    if pred.shape!=gt.shape:raise ValueError('Native GT/prediction dimensions differ')
    gt_values=set(np.unique(gt).tolist());allowed=set(raw_ids)|set(ignore_ids)
    if not gt_values<=allowed:raise ValueError('Unregistered GT IDs: '+str(gt_values-allowed))
    if not set(np.unique(pred).tolist())<=set(range(n))|{255}:raise ValueError('Prediction has an invalid requested-class ID')
    mapped=np.full(gt.shape,-1,np.int16)
    for i,raw in enumerate(raw_ids):mapped[gt==raw]=i
    valid=mapped>=0;p=pred[valid].astype(np.int64);p[p==255]=n
    return np.bincount(mapped[valid].astype(np.int64)*(n+1)+p,minlength=n*(n+1)).reshape(n,n+1)

def metrics(hist):
    h=np.asarray(hist,np.float64);n=h.shape[0];tp=np.diag(h[:,:n]);gt=h.sum(1);pr=h[:,:n].sum(0)
    union=gt+pr-tp;den=gt+pr
    iou=np.divide(tp,union,out=np.zeros(n),where=union>0)
    acc=np.divide(tp,gt,out=np.zeros(n),where=gt>0)
    dice=np.divide(2*tp,den,out=np.zeros(n),where=den>0);frequency=gt/max(gt.sum(),1)
    return {'mIoU_pct':float(iou.mean()*100),'mAcc_pct':float(acc.mean()*100),
            'pixel_accuracy_pct':float(tp.sum()/max(gt.sum(),1)*100),
            'FWIoU_pct':float(np.sum(frequency*iou)*100),'mDice_pct':float(dice.mean()*100),
            'per_class_IoU_pct':(iou*100).tolist(),'GT_pixels_per_class':gt.astype(np.int64).tolist(),
            'valid_GT_pixels':int(gt.sum()),'abstained_valid_pixels':int(h[:,n].sum()),
            'zero_union_class_indices':np.flatnonzero(union==0).tolist(),'macro_denominator':n}

def main(a):
    index=json.loads(Path(a.index).read_text());run=Path(a.run);reports={}
    for dataset,spec in index['datasets'].items():
        n=len(spec['raw_ids']);group={}
        for method in ['original','full']:
            hist=np.zeros((n,n+1),np.int64);ids=[];inputs=[]
            for row in spec['images']:
                folder=run/method/row['image_id'];audit=json.loads((folder/'audit.json').read_text())
                if audit['method']!=method or audit['classes']!=spec['classes']:raise ValueError('Method/class protocol mismatch')
                prediction=folder/'prediction.png'
                if hashlib.sha256(prediction.read_bytes()).hexdigest()!=audit['prediction_sha256']:raise ValueError('Prediction changed')
                if audit['target_GT_reads']!=0 or audit['target_training']:raise ValueError('Prediction leaked target labels')
                with Image.open(prediction) as im:pred=np.array(im)
                if hashlib.sha256(Path(row['GT_path']).read_bytes()).hexdigest()!=row['GT_sha256']:raise ValueError('Native GT file changed')
                with Image.open(row['GT_path']) as im:gt=np.array(im)
                hist+=confusion(pred,gt,spec['raw_ids'],spec['ignore_raw_ids']);ids.append(row['image_id']);inputs.append(audit['input_sha256'])
            result=metrics(hist);result.update(images=len(ids),classes=spec['classes'],official_class_names=spec['official_class_names'],raw_ids=spec['raw_ids'],input_hashes=inputs,image_ids=ids)
            target=run/'results'/dataset/method;target.mkdir(parents=True,exist_ok=True);np.save(target/'confusion.npy',hist)
            (target/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');group[method]=result
        if group['original']['image_ids']!=group['full']['image_ids'] or group['original']['input_hashes']!=group['full']['input_hashes']:
            raise ValueError('Paired methods saw different inputs')
        if group['original']['GT_pixels_per_class']!=group['full']['GT_pixels_per_class']:raise ValueError('Paired GT pixel counts differ')
        reports[dataset]={'images':len(spec['images']),'class_count':n,'original':{k:v for k,v in group['original'].items() if k not in ['input_hashes','image_ids']},'full':{k:v for k,v in group['full'].items() if k not in ['input_hashes','image_ids']},'delta_mIoU_pp':group['full']['mIoU_pct']-group['original']['mIoU_pct']}
    dest=run/'results/paired_summary.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n');print(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--run',required=True);main(p.parse_args())
