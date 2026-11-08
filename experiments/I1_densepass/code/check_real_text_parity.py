import sys,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path('/root/autodl-tmp/piers-main');RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'third_party/grounded_sam2'))
from piers_ljj.workers.entry import _build_hf_backend
from piers_ljj.text.grounded_adapter import GroundedConfig
from piers_ljj.text.pipeline import run_erp_text
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
cfg=GroundedConfig(grounding_model_id='/root/autodl-tmp/weights/grounding-dino-tiny-hf',sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',sam2_ckpt='/root/autodl-tmp/weights/sam2.1_hiera_small.pt',device='cuda',dtype='bf16')
backend=_build_hf_backend(cfg,'cuda');man=json.loads((RUN/'train160.json').read_text());result=[]
for im in man['images'][:2]:
 folder=RUN/'text_train'/im['image_id'];meta=json.loads((folder/'metadata.json').read_text())
 with np.load(folder/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=2048).astype(bool)
 qs=[q for q in man['queries'] if q['image_id']==im['image_id']][:2]
 for q in qs:
  old=run_erp_text(Image.open(im['path']).convert('RGB'),q['query_id'],q['raw_text'],{'win_w':1024,'win_h':400,'stride_x':512,'y_start':0},cfg,backend_factory=lambda:backend)
  a=np.zeros((400,2048),bool);b=np.zeros_like(a)
  for mask in old['masks']:a|=mask
  for i,c in enumerate(meta['candidates']):
   if c['query_id']==q['query_id']:b|=masks[i]
  result.append({'image_id':im['image_id'],'query':q['raw_text'],'different_pixels':int((a!=b).sum()),'reference_candidates':len(old['masks'])})
  print('PARITY',result[-1],flush=True)
(RUN/'real_text_parity.json').write_text(json.dumps(result,indent=2));assert all(x['different_pixels']==0 for x in result),result
print('REAL_TEXT_PARITY_PASSED',flush=True)
