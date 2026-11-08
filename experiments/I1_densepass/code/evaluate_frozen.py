"""Independent full-label evaluation; never called by training/pseudo generator."""
from pathlib import Path
import os,json,time,concurrent.futures,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from PIL import Image
from cooperation import prepare,text_outputs,semantic_revision,morphology,confusion,scores
RUN=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/piers-main')
MAN=json.loads((RUN/'val100.json').read_text());Q={q['query_id']:q for q in MAN['queries']};MAP=np.full(256,255,np.uint8);MAP[:19]=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
FINE={'car':13,'person':11,'road':0,'vegetation':8,'sky':10,'train':16}
TEXT_VARIANTS={'T1':('independent',.5),'T2_score_only':('score_only',.5),'T2':('gated_consensus',.5),'T2_alpha025':('gated_consensus',.25),'T2_boundary':('boundary',.5),'T2_no_semantic':('no_semantic_consensus',.5),'T2_no_gate':('ungated_semantic_consensus',.5)}
OLD=ROOT/'runs/I1-collab-20261106_213607/t1_candidate_shards'

def load_old(iid):
 folders=list(OLD.glob('*/'+iid));assert len(folders)==1,(iid,len(folders))
 f=folders[0];meta=json.loads((f/'metadata.json').read_text())
 with np.load(f/'candidates.npz') as z:
  shape=z['mask_shape'];masks=np.unpackbits(z['masks_packed'],axis=2,count=int(shape[1])).astype(bool)
  lookup={str(cid):i for i,cid in enumerate(z['candidate_ids'])};masks=np.stack([masks[lookup[c['id']]] for c in meta['candidates']]) if meta['candidates'] else masks
 return meta,masks

def query_metrics(mask,raw,cls):
 valid=raw!=255;g=(raw==cls)&valid;p=np.asarray(mask,bool)&valid
 inter=int((p&g).sum());gs=int(g.sum());ps=int(p.sum());union=gs+ps-inter
 return {'positive':gs>0,'intersection':inter,'gt_pixels':gs,'pred_pixels':ps,'iou':inter/max(union,1) if gs else None,'dice':2*inter/max(gs+ps,1) if gs else None,'recall':inter/max(gs,1) if gs else None,'false_positive':bool(ps>=16) if not gs else None}

def one(im):
 iid=im['image_id'];meta,masks=load_old(iid)
 with np.load(RUN/'cache_val_v2'/(iid+'.npz')) as z:hard=z['label'];probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);records=prepare(meta,masks,probs,Q)
 qs=[q for q in MAN['queries'] if q['image_id']==iid];qids=[q['query_id'] for q in qs]
 outputs={};text_stats={}
 for variant,(mode,alpha) in TEXT_VARIANTS.items():outputs[variant],text_stats[variant]=text_outputs(records,probs,qids,mode=mode,alpha=alpha)
 t2=outputs['T2'];feedback=[{**c,'mask':t2[c['query_id']]&morphology(c['mask'],2,'dilate')&c['valid']} for c in records]
 semantic={'S0':hard};semantic_stats={}
 for name,mode,rec in [('S1','quality',records),('S2','gated',feedback),('Naive','naive',records)]:semantic[name],semantic_stats[name]=semantic_revision(hard,probs,rec,mode=mode)
 # Full original semantic GT, with ignore preserved. No __q masks or instance proxies.
 label=Path('/root/autodl-tmp/data/DensePASS/gtFine/val')/Path(im['path']).name.replace('_.png','_labelTrainIds.png');raw=np.asarray(Image.open(label));gt=MAP[raw]
 per_sem={name:confusion(pred,gt).tolist() for name,pred in semantic.items()}
 for name in semantic_stats:
  changed=semantic[name]!=hard;valid=gt!=255
  semantic_stats[name].update(correct_changes=int((changed&valid&(semantic[name]==gt)).sum()),wrong_changes=int((changed&valid&(semantic[name]!=gt)).sum()))
 text=[]
 t0path=RUN/'cache_T0'/(iid+'.npz')
 if t0path.exists():
  with np.load(t0path) as z:outputs['T0']={qid:np.unpackbits(z[qid],axis=1,count=2048).astype(bool) for qid in z.files}
 for q in qs:
  key=q['raw_text'].strip().lower()
  if q['task_type']!='category_union' or key not in FINE:continue
  for name,preds in outputs.items():
   if q['query_id'] not in preds:continue
   text.append({'image_id':iid,'query_id':q['query_id'],'text':key,'method':name,**query_metrics(preds[q['query_id']],raw,FINE[key])})
 # Prediction artifacts enable fixed successes/failures without rerunning models.
 pred=RUN/'frozen_predictions';pred.mkdir(exist_ok=True)
 for name in semantic:
  d=pred/name;d.mkdir(exist_ok=True);Image.fromarray(semantic[name].astype(np.uint8)).save(d/(iid+'.png'))
 for name in ['T1','T2']:
  d=pred/name;d.mkdir(exist_ok=True);np.savez_compressed(d/(iid+'.npz'),**{qid:np.packbits(m,axis=1) for qid,m in outputs[name].items()})
 return {'image_id':iid,'semantic_hist':per_sem,'semantic_stats':semantic_stats,'text_stats':text_stats,'text_queries':text,'gate_states':{state:sum(c['verdict']['state']==state for c in records) for state in ['agree','conflict','uncertain']},'candidates':len(records)}

if __name__=='__main__':
 start=time.perf_counter();out=RUN/'frozen_eval';out.mkdir(exist_ok=True);recs=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,rec in enumerate(pool.map(one,MAN['images']),1):
   recs.append(rec)
   if i%10==0:print('FROZEN_EVAL',i,100,flush=True)
 semantic={name:scores(np.asarray([r['semantic_hist'][name] for r in recs]).sum(0)) for name in ['S0','S1','S2','Naive']}
 allq=[q for r in recs for q in r['text_queries']];text={}
 for name in sorted({x['method'] for x in allq}):
  rows=[x for x in allq if x['method']==name];positive=[x for x in rows if x['positive']];negative=[x for x in rows if not x['positive']]
  text[name]={'queries':len(rows),'positive_queries':len(positive),'negative_queries':len(negative),'positive_mask_IoU':float(np.mean([x['iou'] for x in positive])),'positive_mask_Dice':float(np.mean([x['dice'] for x in positive])),'positive_recall':float(np.mean([x['recall'] for x in positive])),'negative_FP_area16':sum(x['false_positive'] for x in negative),'negative_FPR_area16':float(np.mean([x['false_positive'] for x in negative])) if negative else None}
 report={'status':'complete','protocol':'DensePASS100 development evaluation (historically selected upon); true full semantic GT; original19 fine text labels; instance/referring qualitative only','images':100,'semantic':semantic,'text':text,'gate_states':{k:sum(r['gate_states'][k] for r in recs) for k in ['agree','conflict','uncertain']},'elapsed_s':time.perf_counter()-start}
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'per_image.json').write_text(json.dumps(recs));print('FROZEN_REPORT',json.dumps(report),flush=True)
