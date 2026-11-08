"""Exact official model/memory implementation comparison on real nine-frame sequence."""
import sys,os,json,importlib.util,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];V=ROOT/'third_party/effective_omnisam';sys.path.insert(0,str(V))
import utils.sam2_bootstrap
import torch,numpy as np,argparse
from PIL import Image
import validation as v
from i1_dense_runtime import install_fixed_memory

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);v.setup_seed(1234);work=R/'work/original_omnisam'
 sys.path.insert(0,str(V/'sam'));import sam_lora_image_encoder;sys.modules['sam.sam_lora_image_encoder']=sam_lora_image_encoder
 sys.path.insert(0,str(R/'work/official_deps'))
 official=module('official_reference_model',work/'model.py');memory=module('official_reference_memory',work/'memory_bank.py');official.MemoryBank=memory.MemoryBank
 decoder=module('official_reference_decoder',work/'model__modules__decoder_sam.py');lora=module('official_reference_lora',work/'sam__sam_lora_image_encoder.py')
 official.SegFormerHead=decoder.SegFormerHead;official.SegFormerHead_adaption=decoder.SegFormerHead_adaption;official.LoRA_Sam=lora.LoRA_Sam
 ns=argparse.Namespace(backbone='sam2_s',num_classes=13,num_maskmem=9)
 fork=v.build_sam_model(ns,torch.device('cuda'),ckpt_path='/root/autodl-tmp/weights/omnisam_s.pth').eval();install_fixed_memory(fork)
 sam=v.build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(V/'sam/checkpoints/sam2.1_hiera_small.pt'),device='cuda')
 orig=official.OmniSAM_adapt(sam,num_classes=13,num_maskmem=9).cuda().eval()
 state=torch.load('/root/autodl-tmp/weights/omnisam_s.pth',map_location='cpu',weights_only=True);state={k.removeprefix('module.'):x for k,x in state.items()};orig.load_state_dict(state,strict=True)
 path=json.loads((R/'val100.json').read_text())['images'][0]['path'];image=np.array(Image.open(path).convert('RGB').resize((4096,1024),Image.Resampling.BICUBIC));x=torch.from_numpy(image).permute(2,0,1).contiguous().float().div(255);x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
 rows=[]
 with torch.no_grad():
  for i,left in enumerate(range(0,3073,384)):
   t=x[:,:,left:left+1024].unsqueeze(0).contiguous().cuda();a=fork(i,t)[0];b=orig(i,t)[0];diff=(a-b).abs();rows.append({'frame':i,'max_logit_difference':float(diff.max()),'argmax_pixels_different':int((a.argmax(1)!=b.argmax(1)).sum())})
 result={'status':'passed' if all(x['argmax_pixels_different']==0 and x['max_logit_difference']<1e-5 for x in rows) else 'different','official_model_url':'https://raw.githubusercontent.com/Ding-Zhong/OmniSAM/main/model/model.py','official_memory_url':'https://raw.githubusercontent.com/Ding-Zhong/OmniSAM/main/model/memory_bank.py','strict_checkpoint_loading':True,'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [work/'model.py',work/'memory_bank.py',work/'model__modules__decoder_sam.py',work/'sam__sam_lora_image_encoder.py']},'scope':'One real9-patch sequence using downloaded official model, Memory, decoder and LoRA, strictly loaded same checkpoint; same installed SAM2 backbone runtime. Not proof of every training code equivalence.','rows':rows};(R/'reports/official_forward_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
