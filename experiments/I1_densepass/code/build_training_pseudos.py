"""Defined training-label variants on target train only; no evaluation-GT reads."""
from pathlib import Path
from collections import Counter
import os,json,time,concurrent.futures
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from PIL import Image
from cooperation import prepare,text_outputs,semantic_revision,morphology
RUN=Path(__file__).resolve().parent
MAN=json.loads((RUN/'train160.json').read_text());Q={q['query_id']:q for q in MAN['queries']}

def one(im):
 iid=im['image_id'];folder=RUN/'text_train'/iid
 meta=json.loads((folder/'metadata.json').read_text())
 with np.load(folder/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=2048).astype(bool)
 with np.load(RUN/'cache_train_v2'/(iid+'.npz')) as z:
  hard=z['label'];probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);conf=probs.max(0);base=hard.copy();base[conf<.85]=255
 records=prepare(meta,masks,probs,Q);qids=[q['query_id'] for q in meta['queries']]
 t2,tstats=text_outputs(records,probs,qids,alpha=.5,mode='gated_consensus')
 feedback=[{**c,'mask':t2[c['query_id']]&morphology(c['mask'],2,'dilate')&c['valid']} for c in records]
 outputs={'R0':base};stats={}
 for variant,mode,rec in [('R1','gated',feedback),('Rsingle','quality',records),('Rnaive','naive',records)]:
  outputs[variant],stats[variant]=semantic_revision(base,probs,rec,mode=mode)
 for variant,label in outputs.items():
  path=RUN/'pseudo'/variant/'epoch1/gtFine'/im['src_file'].replace('.jpg','_labelTrainIds.png');path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(label.astype(np.uint8)).save(path,compress_level=1)
 stats['R0']={'coverage':float((base!=255).mean()),'class_hist':np.bincount(np.where(base==255,13,base).flatten(),minlength=14).tolist()}
 stats['text']=tstats;stats['image_id']=iid
 changes=RUN/'pseudo/changes';changes.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(changes/(iid+'.npz'),R1=np.packbits(outputs['R1']!=base),Rnaive=np.packbits(outputs['Rnaive']!=base),shape=np.array(base.shape))
 return stats

if __name__=='__main__':
 t0=time.perf_counter();records=[]
 for v in ['R0','R1','Rsingle','Rnaive']:
  p=RUN/'pseudo'/v;p.mkdir(parents=True,exist_ok=True);(p/'used_path.txt').write_text('\n'.join(im['src_file'] for im in MAN['images'])+'\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,rec in enumerate(pool.map(one,MAN['images']),1):
   records.append(rec)
   if i%10==0:print('PSEUDO_LABELS',i,160,flush=True)
 totals={}
 for v in ['R1','Rsingle','Rnaive']:
  totals[v]={k:sum(x[v].get(k,0) for x in records) for k in ['candidates','accepted','agree','conflict','uncertain','changed_pixels','recovered_ignore','written_pixels','blocked_conflicts','cross_class_conflicts']}
 report={'status':'complete','images':160,'no_target_GT_read':True,'frozen_teacher_threshold':.85,'quality_threshold':.8,'gate_support':.5,'pixel_conflict_threshold':.8,'edge_width':2,'class_space':'densepass13@v1','totals':totals,'elapsed_s':time.perf_counter()-t0,'records':records}
 (RUN/'pseudo/report.json').write_text(json.dumps(report,indent=2));print('PSEUDOS_COMPLETE',json.dumps(totals),flush=True)
