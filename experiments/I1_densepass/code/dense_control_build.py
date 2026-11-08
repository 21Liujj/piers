from pathlib import Path
import json,numpy as np,concurrent.futures,time,sys
from PIL import Image
from cooperation import prepare,bounded_text_fusion,semantic_revision,confusion,scores
from evaluate_frozen import load_old,FINE,MAP,query_metrics
from analyze_results import paired_hist_boot,text_cluster_boot
R=Path(__file__).resolve().parent;D=R/'dense_controls'
def valone(im):
 iid=im['image_id'];meta,masks=load_old(iid)
 with np.load(D/'cache_val'/(iid+'.npz')) as z:hard=z['label'];probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);man=json.loads((R/'val100.json').read_text());qs=[q for q in man['queries'] if q['image_id']==iid];Q={q['query_id']:q for q in qs};cs=prepare(meta,masks,probs,Q)
 output,ts=bounded_text_fusion(cs,probs,list(Q),gated=True);sem,ss=semantic_revision(hard,probs,cs,mode='gated')
 raw=np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+Path(im['path']).name.replace('_.png','_labelTrainIds.png')));gt=MAP[raw]
 rows=[]
 for q in qs:
  key=q['raw_text'].strip().lower()
  if q['task_type']=='category_union' and key in FINE:rows.append({'image_id':iid,'query_id':q['query_id'],'text':key,**query_metrics(output[q['query_id']],raw,FINE[key])})
 dest=D/'predictions'/iid;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(hard).save(dest/'baseline.png');Image.fromarray(sem).save(dest/'I1.png')
 np.savez_compressed(dest/'text.npz',**{qid:np.packbits(mask,axis=1) for qid,mask in output.items()})
 return {'image_id':iid,'baseline_hist':confusion(hard,gt).tolist(),'I1_hist':confusion(sem,gt).tolist(),'text_queries':rows,'semantic_stats':ss,'text_stats':ts}
def trainone(im):
 iid=im['image_id'];folder=R/'text_train'/iid;meta=json.loads((folder/'metadata.json').read_text())
 with np.load(folder/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=2048).astype(bool)
 with np.load(D/'cache_train'/(iid+'.npz')) as z:base=z['label'].copy();probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);base[probs.max(0)<.85]=255
 man=json.loads((R/'train160.json').read_text());Q={q['query_id']:q for q in man['queries']};cs=prepare(meta,masks,probs,Q);revised,stats=semantic_revision(base,probs,cs,mode='gated')
 for v,label in [('R0',base),('R1',revised)]:
  path=D/'pseudo'/v/'epoch1/gtFine'/im['src_file'].replace('.jpg','_labelTrainIds.png');path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(label).save(path,compress_level=1)
 return {'image_id':iid,**stats}
def main():
 start=time.perf_counter();val=json.loads((R/'val100.json').read_text());rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for row in pool.map(valone,val['images']):rows.append(row)
 a=np.array([x['baseline_hist'] for x in rows]);b=np.array([x['I1_hist'] for x in rows]);qrows=[x for r in rows for x in r['text_queries']];pos=[x for x in qrows if x['positive']];neg=[x for x in qrows if not x['positive']]
 origq=[q for r in json.loads((R/'frozen_eval/per_image.json').read_text()) for q in r['text_queries'] if q['method']=='T1']
 report={'status':'complete','definition':'Original uncompressed OmniSAM-S vs same model plus Innovation1 only; no I2 compression; all thresholds identical to frozen joint method, no retuning.','baseline':scores(a.sum(0)),'I1_only':scores(b.sum(0)),'semantic_bootstrap':paired_hist_boot(a,b),'text':{'positive_mask_IoU':float(np.mean([x['iou'] for x in pos])),'positive_mask_Dice':float(np.mean([x['dice'] for x in pos])),'positive_recall':float(np.mean([x['recall'] for x in pos])),'positive_queries':len(pos),'negative_queries':len(neg),'negative_FP_area16':sum(x['false_positive'] for x in neg),'negative_FPR_area16':float(np.mean([x['false_positive'] for x in neg]))},'text_vs_T1_bootstrap':text_cluster_boot(origq,qrows),'elapsed_s':time.perf_counter()-start}
 (D/'frozen_report.json').write_text(json.dumps(report,indent=2));(D/'frozen_per_image.json').write_text(json.dumps(rows));np.savez_compressed(D/'frozen_hists.npz',baseline=a,I1_only=b)
 print('DENSE_FROZEN_COMPLETE',json.dumps(report),flush=True)
 if '--val-only' in sys.argv:return
 train=json.loads((R/'train160.json').read_text())
 for v in ['R0','R1']:
  p=D/'pseudo'/v;p.mkdir(parents=True,exist_ok=True);(p/'used_path.txt').write_text('\n'.join(x['src_file'] for x in train['images'])+'\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(trainone,train['images']))
 (D/'pseudo/report.json').write_text(json.dumps({'status':'complete','images':len(rows),'target_GT_read':False,'rows':rows},indent=2));print('DENSE_PSEUDO_COMPLETE',flush=True)
if __name__=='__main__':main()
