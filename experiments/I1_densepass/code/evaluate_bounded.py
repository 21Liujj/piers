from pathlib import Path
import json,time,concurrent.futures,numpy as np
from PIL import Image
from evaluate_frozen import RUN,MAN,Q,FINE,MAP,load_old,query_metrics
from cooperation import prepare,bounded_text_fusion,morphology,semantic_revision,confusion,scores

def one(im):
 iid=im['image_id'];meta,masks=load_old(iid)
 with np.load(RUN/'cache_val_v2'/(iid+'.npz')) as z:hard=z['label'];probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);records=prepare(meta,masks,probs,Q);qs=[q for q in MAN['queries'] if q['image_id']==iid];qids=[q['query_id'] for q in qs]
 outputs={};statistics={}
 for name,gated in [('T2_bounded',True),('T2_bounded_no_gate',False)]:outputs[name],statistics[name]=bounded_text_fusion(records,probs,qids,gated=gated)
 teacher_gt=Path('/root/autodl-tmp/data/DensePASS/gtFine/val')/Path(im['path']).name.replace('_.png','_labelTrainIds.png');raw=np.asarray(Image.open(teacher_gt));gt=MAP[raw]
 rows=[]
 for q in qs:
  key=q['raw_text'].strip().lower()
  if q['task_type']!='category_union' or key not in FINE:continue
  for name,masks_by_id in outputs.items():rows.append({'image_id':iid,'query_id':q['query_id'],'text':key,'method':name,**query_metrics(masks_by_id[q['query_id']],raw,FINE[key])})
 feedback=[{**c,'mask':outputs['T2_bounded'][c['query_id']]&c['valid']&morphology(c['mask'],16,'dilate')} for c in records]
 s2,stats=semantic_revision(hard,probs,feedback,mode='gated')
 folder=RUN/'bounded_predictions';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(iid+'.npz'),**{qid:np.packbits(mask,axis=1) for qid,mask in outputs['T2_bounded'].items()});d=folder/'S2';d.mkdir(exist_ok=True);Image.fromarray(s2.astype(np.uint8)).save(d/(iid+'.png'))
 return {'image_id':iid,'text_queries':rows,'text_stats':statistics,'semantic_hist':confusion(s2,gt).tolist(),'semantic_stats':stats}
if __name__=='__main__':
 start=time.perf_counter();records=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,rec in enumerate(pool.map(one,MAN['images']),1):
   records.append(rec)
   if i%10==0:print('BOUNDED_EVAL',i,flush=True)
 rows=[x for r in records for x in r['text_queries']];text={}
 for name in ['T2_bounded','T2_bounded_no_gate']:
  subset=[x for x in rows if x['method']==name];pos=[x for x in subset if x['positive']];neg=[x for x in subset if not x['positive']]
  text[name]={'positive_mask_IoU':float(np.mean([x['iou'] for x in pos])),'positive_mask_Dice':float(np.mean([x['dice'] for x in pos])),'positive_recall':float(np.mean([x['recall'] for x in pos])),'positive_queries':len(pos),'negative_queries':len(neg),'negative_FP_area16':sum(x['false_positive'] for x in neg),'negative_FPR_area16':float(np.mean([x['false_positive'] for x in neg]))}
 report={'status':'complete','text':text,'semantic_S2':scores(np.asarray([r['semantic_hist'] for r in records]).sum(0)),'elapsed_s':time.perf_counter()-start,'hypothesis':'Error analysis showed consensus trimming loses recall; bound exact-class probability fusion to original DINO boxes, preserve coarse/free queries and conflict fallback.'}
 (RUN/'bounded_report.json').write_text(json.dumps(report,indent=2));(RUN/'bounded_per_image.json').write_text(json.dumps(records));print('BOUNDED_REPORT',json.dumps(report),flush=True)
