from pathlib import Path
import json,concurrent.futures,time,numpy as np
from PIL import Image
from cooperation import prepare,semantic_revision
RUN=Path(__file__).resolve().parent;MAN=json.loads((RUN/'train160.json').read_text());Q={q['query_id']:q for q in MAN['queries']}
def one(im):
 iid=im['image_id'];f=RUN/'text_train'/iid;meta=json.loads((f/'metadata.json').read_text())
 with np.load(f/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=2048).astype(bool)
 with np.load(RUN/'cache_train_v2'/(iid+'.npz')) as z:hard=z['label'];probs=z['probs'].astype(np.float32)
 probs/=np.maximum(probs.sum(0,keepdims=True),1e-8);base=hard.copy()
 records=prepare(meta,masks,probs,Q);revised,stats=semantic_revision(base,probs,records,mode='gated')
 path=RUN/'pseudo_allhard/R1/epoch1/gtFine'/im['src_file'].replace('.jpg','_labelTrainIds.png');path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(revised).save(path,compress_level=1)
 zero=RUN/'pseudo_allhard/R0/epoch1/gtFine'/im['src_file'].replace('.jpg','_labelTrainIds.png');zero.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(base).save(zero,compress_level=1)
 (RUN/'pseudo_allhard/changes').mkdir(parents=True,exist_ok=True);np.savez_compressed(RUN/'pseudo_allhard/changes'/(iid+'.npz'),changed=np.packbits(revised!=base),shape=np.array(base.shape))
 return {'image_id':iid,**stats}
if __name__=='__main__':
 root=RUN/'pseudo_allhard';root.mkdir(exist_ok=True)
 (root/'R0').mkdir(exist_ok=True)
 
 for variant in ['R0','R1']:
  d=root/variant;d.mkdir(exist_ok=True);(d/'used_path.txt').write_text('\n'.join(im['src_file'] for im in MAN['images'])+'\n')
 start=time.perf_counter();rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,row in enumerate(pool.map(one,MAN['images']),1):
   rows.append(row)
   if i%20==0:print('FINAL_PSEUDO',i,flush=True)
 report={'status':'complete','images':160,'no_GT_read':True,'method':'All-hard pseudo diagnostic; no confidence filtering. Shared frozen-teacher candidate gate. Semantic feedback takes independent raw SAM masks after acceptance; it never reuses semantic-fused text masks as independent evidence.','changed_pixels':sum(x['changed_pixels'] for x in rows),'recovered_ignore':sum(x['recovered_ignore'] for x in rows),'blocked_conflict_pixels':sum(x['blocked_conflicts'] for x in rows),'elapsed_s':time.perf_counter()-start,'rows':rows}
 (root/'report.json').write_text(json.dumps(report,indent=2));print('FINAL_PSEUDO_COMPLETE',report['changed_pixels'],flush=True)
