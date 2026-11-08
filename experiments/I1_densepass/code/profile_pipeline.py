"""Isolated real two-process end-to-end latency, including IPC and collaboration."""
from pathlib import Path
import os,sys,json,time,subprocess,multiprocessing as mp
ROOT=Path('/root/autodl-tmp/piers-main');RUN=Path(__file__).resolve().parent

def sem_child(pipe):
 sys.path.insert(0,str(ROOT/'third_party/effective_omnisam'))
 import utils.sam2_bootstrap
 import torch,numpy as np,cv2,argparse
 from PIL import Image
 import validation as v
 from utils.tools import merge_width_scatter_np
 from i1_runtime import install_fixed_memory,I2
 torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(0);v.setup_seed(1234)
 ns=argparse.Namespace(seed=1234,backbone='sam2_s',dataset='DensePASS',class_set=13,num_classes=13,use_mem_bank=True,num_maskmem=9,enable_scm=False,enable_bepl=False)
 model=v.build_sam_model(ns,torch.device('cuda'),ckpt_path=I2['checkpoint_path']).eval();install_fixed_memory(model)
 mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None];pipe.send({'ready':True})
 while True:
  path=pipe.recv()
  if path is None:break
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((4096,1024),Image.Resampling.BICUBIC))
  x=torch.from_numpy(rgb).permute(2,0,1).contiguous().float().div(255);x=(x-mean)/std
  logits=np.zeros((1,18,13,1024,1024),np.float32);xs=list(range(0,3073,384))
  with torch.no_grad():
   for pas,starts in enumerate([xs,list(reversed(xs))]):
    model.memory_bank._init_output_dict_source()
    for f,left in enumerate(starts):logits[0,pas*9+f]=model(f,x[:,:,left:left+1024].unsqueeze(0).contiguous().cuda())[0][0].cpu().numpy()
   model.memory_bank._init_output_dict_source();summed=merge_width_scatter_np(logits,(1024,4096),1024,384,2,'cuda',True)[0]
  label=cv2.resize(summed.argmax(0).astype(np.uint8),(2048,400),interpolation=cv2.INTER_NEAREST);summed/=summed.sum(0,keepdims=True)
  probs=np.stack([cv2.resize(p,(2048,400),interpolation=cv2.INTER_NEAREST) for p in summed]).astype(np.float16)
  torch.cuda.synchronize();pipe.send({'probs':probs,'label':label,'compute_s':time.perf_counter()-start,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'reserved_mib':torch.cuda.max_memory_reserved()/2**20})

def text_child(pipe):
 sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'third_party/grounded_sam2'))
 import torch,numpy as np
 from PIL import Image
 from text_worker import LastCropCache
 from piers_ljj.workers.entry import _build_hf_backend
 from piers_ljj.text.grounded_adapter import GroundedConfig,run_fov_text_branch
 from piers_ljj.geometry.erp import plan_fovs,crop_erp_wrap,box_local_to_global
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
 cfg=GroundedConfig(grounding_model_id='/root/autodl-tmp/weights/grounding-dino-tiny-hf',sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',sam2_ckpt='/root/autodl-tmp/weights/sam2.1_hiera_small.pt',device='cuda',dtype='bf16')
 dino,sam=_build_hf_backend(cfg,'cuda');cache=LastCropCache(sam);pipe.send({'ready':True})
 while True:
  request=pipe.recv()
  if request is None:break
  path,queries=request;torch.cuda.reset_peak_memory_stats();start=time.perf_counter();before=cache.encodes
  rgb=np.array(Image.open(path).convert('RGB'));H,W=rgb.shape[:2];fovs=plan_fovs(W,H,win_w=1024,win_h=H,stride_x=512,y_start=0);masks=[];cs=[]
  for f in fovs:
   crop,_=crop_erp_wrap(rgb,f['x_start'],0,1024,H);pil=Image.fromarray(crop)
   for q in queries:
    response=run_fov_text_branch(pil,q['raw_text'],cfg,(dino,cache))
    for k,c in enumerate(response['candidates']):
     gm=np.zeros((H,W),bool);cols=(f['x_start']+np.arange(1024))%W;gm[:,cols]=np.asarray(c.mask,bool)
     box=box_local_to_global(c.box_xyxy,f['x_start'],0,W)
     cs.append({'id':q['query_id']+':'+f['view_id']+':'+str(k),'query_id':q['query_id'],'fov_id':f['view_id'],'detector_score':c.detector_score,'sam_quality':c.sam_quality,'box_segs':box['segs']});masks.append(gm)
  packed=np.packbits(np.stack(masks),axis=2) if masks else np.zeros((0,H,W//8),np.uint8)
  torch.cuda.synchronize();pipe.send({'meta':{'fovs':fovs,'candidates':cs},'packed':packed,'compute_s':time.perf_counter()-start,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'reserved_mib':torch.cuda.max_memory_reserved()/2**20,'sam_image_encodes':cache.encodes-before,'DINO_calls':len(fovs)*len(queries)})

def active_gpu():
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True);active=[]
 for line in r.stdout.splitlines():
  if not line.strip().isdigit():continue
  pid=int(line);status=Path('/proc')/str(pid)/'status'
  if status.exists():
   state=next((s for s in status.read_text().splitlines() if s.startswith('State:')),'')
   if 'T (stopped)' not in state:active.append(pid)
 return active

def main():
 import numpy as np
 from cooperation import prepare,bounded_text_fusion,semantic_revision
 deadline=time.monotonic()+1200
 while active_gpu():
  (RUN/'profile_wait.json').write_text(json.dumps({'waiting_for_active_GPU_PIDs':active_gpu()}))
  if time.monotonic()>deadline:raise TimeoutError('GPU remained shared; no formal speed claim allowed')
  time.sleep(5)
 ctx=mp.get_context('spawn');pa,ca=ctx.Pipe();pb,cb=ctx.Pipe();startup=time.perf_counter();sa=ctx.Process(target=sem_child,args=(ca,));sb=ctx.Process(target=text_child,args=(cb,));sa.start();sb.start();ca.close();cb.close();assert pa.poll(180) and pb.poll(180),'GPU worker initialization timed out';assert pa.recv()['ready'] and pb.recv()['ready'];startup=time.perf_counter()-startup
 man=json.loads((RUN/'val100.json').read_text());images=[man['images'][i] for i in [0,10,20,30,40,50]]
 q={'query_id':'profile_road','raw_text':'road','ontology':{'map13':0,'relation':'exact'}};qm={q['query_id']:q};rows=[]
 try:
  for i,im in enumerate(images):
   # Only the two own resident workers may be active during a measurement.
   assert set(active_gpu()).issubset({sa.pid,sb.pid}),active_gpu()
   start=time.perf_counter();pa.send(im['path']);sem=pa.recv();pb.send((im['path'],[q]));text=pb.recv();t=time.perf_counter()
   probs=sem['probs'].astype(np.float32);probs/=probs.sum(0,keepdims=True);masks=np.unpackbits(text['packed'],axis=2,count=2048).astype(bool)
   records=prepare(text['meta'],masks,probs,qm);bounded_text_fusion(records,probs,list(qm));semantic_revision(sem['label'],probs,records,mode='gated')
   end=time.perf_counter();full=end-start;coop=end-t
   cached=time.perf_counter();pb.send((im['path'],[q]));again=pb.recv();recs=prepare(again['meta'],np.unpackbits(again['packed'],axis=2,count=2048).astype(bool),probs,qm);bounded_text_fusion(recs,probs,list(qm));increment=time.perf_counter()-cached
   t1start=time.perf_counter();pb.send((im['path'],[q]));independent=pb.recv();independent_union=np.any(np.unpackbits(independent['packed'],axis=2,count=2048),axis=0);t1wall=time.perf_counter()-t1start
   row={'T1_text_only_end_to_end_s':t1wall,'semantic_peak_reserved_mib':sem['reserved_mib'],'text_peak_reserved_mib':text['reserved_mib'],'image_id':im['image_id'],'warmup':i==0,'full_end_to_end_s':full,'cached_semantic_query_s':increment,'semantic_compute_s':sem['compute_s'],'text_compute_s':text['compute_s'],'collaboration_and_decode_IPC_s':coop,'semantic_peak_allocated_mib':sem['peak_mib'],'text_peak_allocated_mib':text['peak_mib'],'DINO_calls':text['DINO_calls'],'SAM_encodes':text['sam_image_encodes']};rows.append(row);print('PROFILE_IMAGE',json.dumps(row),flush=True)
  # Same-image query batching, with all masks still produced independently.
  qs=[{'query_id':'batch'+str(i),'raw_text':raw,'ontology':{'map13':cls,'relation':rel}} for i,(raw,cls,rel) in enumerate([('car',12,'coarse_compatible'),('person',11,'coarse_compatible'),('road',0,'exact')])]
  start=time.perf_counter();pb.send((images[-1]['path'],qs));bat=pb.recv();duration=time.perf_counter()-start
  fullbatch_start=time.perf_counter();pa.send(images[-1]['path']);bs=pa.recv();pb.send((images[-1]['path'],qs));bt=pb.recv();bp=bs['probs'].astype(np.float32);bp/=bp.sum(0,keepdims=True);bm=np.unpackbits(bt['packed'],axis=2,count=2048).astype(bool);bq={q['query_id']:q for q in qs};br=prepare(bt['meta'],bm,bp,bq);bounded_text_fusion(br,bp,list(bq));semantic_revision(bs['label'],bp,br,mode='gated');fullbatch_duration=time.perf_counter()-fullbatch_start
  used=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
  measured=rows[1:];report={'status':'complete','isolation':'All other GPU processes stopped; five supplementary contexts still reside in memory. No concurrent GPU compute allowed.','startup_both_resident_workers_s':startup,'cold_first_end_to_end_s':startup+rows[0]['full_end_to_end_s'],'warmup_images':1,'measured_images':5,'query':'road','precision':'semantic FP32, text DINO/SAM decode bf16, SAM image encoder original backend dtype','T1_text_only_end_to_end_mean_s':float(np.mean([x['T1_text_only_end_to_end_s'] for x in measured])),'semantic_peak_allocated_mib':max(x['semantic_peak_allocated_mib'] for x in rows),'semantic_peak_reserved_mib':max(x['semantic_peak_reserved_mib'] for x in rows),'text_peak_allocated_mib':max(x['text_peak_allocated_mib'] for x in rows),'text_peak_reserved_mib':max(x['text_peak_reserved_mib'] for x in rows),'full_three_query_batch':{'wall_s':fullbatch_duration,'queries_per_sec':3/fullbatch_duration,'DINO_calls':bt['DINO_calls'],'SAM_image_encodes':bt['sam_image_encodes']},'full_end_to_end_mean_s':float(np.mean([x['full_end_to_end_s'] for x in measured])),'cached_semantic_query_mean_s':float(np.mean([x['cached_semantic_query_s'] for x in measured])),'semantic_mean_s':float(np.mean([x['semantic_compute_s'] for x in measured])),'text_mean_s':float(np.mean([x['text_compute_s'] for x in measured])),'three_query_batch':{'wall_s':duration,'queries_per_sec':3/duration,'DINO_calls':bat['DINO_calls'],'SAM_image_encodes':bat['sam_image_encodes'],'note':'Batch throughput; not unary query latency.'},'device_total_used_utilization_csv':used,'rows':rows}
  (RUN/'performance.json').write_text(json.dumps(report,indent=2));print('PERFORMANCE_COMPLETE',flush=True)
 finally:
  for pipe,proc in [(pa,sa),(pb,sb)]:
   try:pipe.send(None)
   except (BrokenPipeError,EOFError):pass
   proc.join(timeout=20)
   if proc.is_alive():proc.terminate()
if __name__=='__main__':
 os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 try:main()
 except BaseException:
  import traceback
  (RUN/'performance_failure.txt').write_text(traceback.format_exc());raise
