"""CUDA implementation of the SAME registered uncertainty and fusion formula."""
import numpy as np,torch
from seam_core import centrality,support,choose

def risk_cuda(base):
 p=torch.as_tensor(base,device='cuda',dtype=torch.float32);p=p/p.sum(0,keepdim=True).clamp_min(1e-8)
 entropy=-(p*p.clamp_min(1e-8).log()).sum(0)/np.log(p.shape[0]);top=p.topk(2,dim=0).values
 return (.5*entropy+.5*(1-(top[0]-top[1]))).cpu().numpy()

def select_cuda(base):return choose(risk_cuda(base),[1664,1728,1792,1856,1920],512,2,min_separation=128)

def fuse_cuda(base,base_label,extras,base_coverage=None):
 p=torch.as_tensor(base,device='cuda',dtype=torch.float32);p=p/p.sum(0,keepdim=True).clamp_min(1e-8)
 if not extras:return p.cpu().numpy(),base_label.copy(),{'B':0,'identity':True}
 C,H,W=p.shape;bw=torch.as_tensor(np.broadcast_to(base_coverage,(H,W)).copy(),device='cuda',dtype=torch.float32)
 num=p*bw;den=bw.clone();covered=torch.zeros(W,dtype=torch.bool,device='cuda')
 for s,ep in extras:
  e=torch.as_tensor(ep,device='cuda',dtype=torch.float32);e=e/e.sum(0,keepdim=True).clamp_min(1e-8);ww=e.shape[-1];cols=torch.as_tensor(support(s,ww,W),device='cuda',dtype=torch.long)
  confidence=(1+(e*e.clamp_min(1e-8).log()).sum(0)/np.log(C)).clamp(.05,1)
  w=torch.as_tensor(centrality(ww),device='cuda')[None,:]*confidence;num.index_add_(2,cols,e*w);den.index_add_(1,cols,w);covered[cols]=True
 out=num/den.clamp_min(1e-8);label=torch.as_tensor(base_label.copy(),device='cuda');prediction=out.argmax(0).to(label.dtype);label[:,covered]=prediction[:,covered]
 return out.cpu().numpy(),label.cpu().numpy(),{'B':len(extras),'identity':False}
