"""Real free-text FoV inference with last-crop SAM embedding reuse. No GT."""
import sys,os,json,time,argparse,hashlib,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/piers-main')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'third_party/grounded_sam2'))
import numpy as np,torch
from PIL import Image
from piers_ljj.workers.entry import _build_hf_backend
from piers_ljj.text.grounded_adapter import GroundedConfig,run_fov_text_branch
from piers_ljj.geometry.erp import plan_fovs,crop_erp_wrap,box_local_to_global

class LastCropCache:
 def __init__(self,base):self.base=base;self.key=None;self.enc=None;self.encodes=0;self.hits=0
 def set_image(self,pil):
  key=hashlib.sha256(np.asarray(pil.convert('RGB')).tobytes()).digest()
  if key!=self.key:self.enc=self.base.set_image(pil);self.key=key;self.encodes+=1
  else:self.hits+=1
  return self.enc
 def predict_boxes(self,enc,boxes):return self.base.predict_boxes(enc,boxes)

def main(a):
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 man=json.loads(Path(a.manifest).read_text());images=man['images'][a.shard::a.shards]
 if a.limit:images=images[:a.limit]
 qby={}
 for q in man['queries']:qby.setdefault(q['image_id'],[]).append(q)
 cfg=GroundedConfig(grounding_model_id='/root/autodl-tmp/weights/grounding-dino-tiny-hf',sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',sam2_ckpt='/root/autodl-tmp/weights/sam2.1_hiera_small.pt',device='cuda',dtype='bf16')
 dino,sam=_build_hf_backend(cfg,'cuda');cache=LastCropCache(sam);t0=time.perf_counter();done=0
 torch.cuda.reset_peak_memory_stats()
 for im in images:
  folder=out/im['image_id'];folder.mkdir(exist_ok=True)
  if (folder/'metadata.json').exists():done+=1;continue
  rgb=np.asarray(Image.open(im['path']).convert('RGB'));H,W=rgb.shape[:2]
  fovs=plan_fovs(W,H,win_w=1024,win_h=H,stride_x=512,y_start=0)
  masks=[];candidates=[];nqueries=0;start=time.perf_counter()
  for f in fovs:
   crop,_=crop_erp_wrap(rgb,f['x_start'],0,f['win_w'],H);pil=Image.fromarray(crop)
   for q in qby.get(im['image_id'],[]):
    result=run_fov_text_branch(pil,q['raw_text'],cfg,(dino,cache));nqueries+=1
    for k,c in enumerate(result['candidates']):
     lm=np.asarray(c.mask,bool);assert lm.shape==(H,1024),lm.shape
     gm=np.zeros((H,W),bool);cols=(f['x_start']+np.arange(1024))%W;gm[:,cols]=lm
     cid=q['query_id']+':'+f['view_id']+':'+str(k)
     candidates.append({'id':cid,'query_id':q['query_id'],'fov_id':f['view_id'],'detector_score':c.detector_score,'sam_quality':c.sam_quality,'phrase':c.phrase,'local_box':list(c.box_xyxy),'ontology':q.get('ontology'),'mask_index':len(masks)})
     masks.append(gm)
  packed=np.packbits(np.stack(masks),axis=2) if masks else np.zeros((0,H,(W+7)//8),np.uint8)
  np.savez_compressed(folder/'candidates.npz',masks_packed=packed,shape=np.array([H,W]))
  metadata={'status':'complete','image_id':im['image_id'],'path':im['path'],'shape':[H,W],'fovs':fovs,'queries':qby.get(im['image_id'],[]),'candidates':candidates,'elapsed_s':time.perf_counter()-start,'detector_calls':nqueries,'sam_real_encodes_total':cache.encodes,'sam_cache_hits_total':cache.hits,'weights':{'dino':cfg.grounding_model_id,'sam':cfg.sam2_ckpt},'no_GT_read':True}
  (folder/'metadata.json').write_text(json.dumps(metadata,indent=2));done+=1
  progress={'status':'running','done':done,'total':len(images),'elapsed_s':time.perf_counter()-t0,'last_image_s':metadata['elapsed_s'],'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
  (out/('progress_s'+str(a.shard)+'.json')).write_text(json.dumps(progress));print('TEXT',a.shard,done,len(images),len(candidates),round(metadata['elapsed_s'],2),flush=True)
 (out/('done_s'+str(a.shard)+'.json')).write_text(json.dumps({'status':'complete','images':len(images),'elapsed_s':time.perf_counter()-t0,'sam_encodes':cache.encodes,'sam_cache_hits':cache.hits}));print('TEXT_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 try:main(a)
 except BaseException:
  Path(a.out).mkdir(parents=True,exist_ok=True);(Path(a.out)/('failure_s'+str(a.shard)+'.txt')).write_text(traceback.format_exc());raise
