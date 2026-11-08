"""Image-only frozen-I2 semantic inference. Never opens target GT."""
import os,sys,json,time,argparse,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path('/root/autodl-tmp/piers-main');EFF=ROOT/'third_party/effective_omnisam'
sys.path.insert(0,str(EFF))
import utils.sam2_bootstrap
import torch,numpy as np,cv2
from PIL import Image
import validation as v
from utils.tools import merge_width_scatter_np
from i1_runtime import verify_i2,install_fixed_memory,I2,sha

def main(a):
 verify_i2();torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(0);v.setup_seed(1234)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 man=json.loads(Path(a.manifest).read_text());images=man['images'][a.shard::a.shards]
 if a.limit:images=images[:a.limit]
 ns=argparse.Namespace(seed=1234,backbone='sam2_s',dataset='DensePASS',class_set=13,num_classes=13,
  use_mem_bank=True,num_maskmem=9,enable_scm=False,enable_bepl=False)
 model=v.build_sam_model(ns,torch.device('cuda'),ckpt_path=a.ckpt).eval()
 audit=install_fixed_memory(model);mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
 t0=time.perf_counter();torch.cuda.reset_peak_memory_stats();done=0
 for im in images:
  dest=out/(im['image_id']+'.npz')
  if dest.exists():done+=1;continue
  with Image.open(im['path']) as pil:
   rgb=np.asarray(pil.convert('RGB').resize((4096,1024),Image.Resampling.BICUBIC)).copy()
  tensor=torch.from_numpy(rgb).permute(2,0,1).contiguous().float().div(255)
  tensor=(tensor-mean)/std
  starts=list(range(0,4096-1024+1,384));logits=np.zeros((1,18,13,1024,1024),np.float32)
  begin=time.perf_counter()
  with torch.no_grad():
   for pas,xstarts in enumerate([starts,list(reversed(starts))]):
    model.memory_bank._init_output_dict_source()
    for f,x in enumerate(xstarts):
     prediction,*_=model(f,tensor[:, :, x:x+1024].unsqueeze(0).contiguous().cuda())
     logits[0,pas*9+f]=prediction[0].cpu().numpy()
   model.memory_bank._init_output_dict_source()
   summed=merge_width_scatter_np(logits,(1024,4096),1024,384,2,'cuda',True)[0]
  # Merge utility returns overlap sums, NOT probabilities. Normalize first.
  normal=summed/np.maximum(summed.sum(0,keepdims=True),1e-8)
  label=cv2.resize(summed.argmax(0).astype(np.uint8),(2048,400),interpolation=cv2.INTER_NEAREST)
  small=np.stack([cv2.resize(x,(2048,400),interpolation=cv2.INTER_NEAREST) for x in normal]).astype(np.float16)
  assert np.isfinite(small).all() and np.max(np.abs(small.astype(np.float32).sum(0)-1))<.002
  np.savez_compressed(dest,probs=small,label=label,image_id=im['image_id'])
  done+=1;torch.cuda.synchronize()
  state={'status':'running','done':done,'total':len(images),'elapsed_s':time.perf_counter()-t0,
   'last_image_s':time.perf_counter()-begin,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'audit':audit}
  (out/('progress_s'+str(a.shard)+'.json')).write_text(json.dumps(state,indent=2))
  print('SEMANTIC',a.shard,done,len(images),round(state['last_image_s'],2),im['image_id'],flush=True)
  del logits,summed,normal,small,tensor
 state={'status':'complete','images':len(images),'manifest':a.manifest,'manifest_sha256':sha(a.manifest),
  'checkpoint':a.ckpt,'checkpoint_sha256':sha(a.ckpt),'I2':I2,'audit':audit,'elapsed_s':time.perf_counter()-t0,
  'no_GT_read':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
 (out/('metadata_s'+str(a.shard)+'.json')).write_text(json.dumps(state,indent=2));print('SEMANTIC_COMPLETE',state['images'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
 ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--ckpt',default=I2['checkpoint_path']);a=ap.parse_args()
 try:main(a)
 except BaseException:
  Path(a.out).mkdir(parents=True,exist_ok=True);(Path(a.out)/('failure_s'+str(a.shard)+'.txt')).write_text(traceback.format_exc());raise
