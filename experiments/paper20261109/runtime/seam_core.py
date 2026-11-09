"""Budgeted cross-seam selection and coverage-aware fusion. No GT or model imports."""
import numpy as np

def centrality(width):
 x=(np.arange(width,dtype=np.float32)+.5)/width
 return .25+.75*np.sin(np.pi*x)**2

def support(start,width,W):return (int(start)+np.arange(width))%W

def seam_mask(H=400,W=2048,band=128):
 a=np.zeros((H,W),bool);a[:,:band]=True;a[:,-band:]=True;return a

def semantic_risk(p):
 q=p.astype(np.float32);q/=np.maximum(q.sum(0,keepdims=True),1e-8)
 ent=-(q*np.log(np.maximum(q,1e-8))).sum(0)/np.log(q.shape[0])
 top=np.partition(q,-2,axis=0)[-2:];margin=top[-1]-top[-2]
 return .5*ent+.5*(1-margin)

def choose(risk,starts,width,B=2,band=128,min_separation=0):
 H,W=risk.shape;rr=np.asarray(risk,np.float32)*seam_mask(H,W,band)
 residual=rr.mean(0);quality=centrality(width);selected=[];scores=[]
 for k in range(min(B,len(starts))):
  choices=[]
  for s in starts:
   if s in selected or any(min((s-t)%W,(t-s)%W)<min_separation for t in selected):continue
   cols=support(s,width,W);choices.append((float((residual[cols]*quality).sum()),-abs(((s+width/2+W/2)%W)-W/2),-s,s))
  if not choices:break
  val,_,_,s=max(choices);selected.append(s);scores.append(val)
  cols=support(s,width,W);residual[cols]*=(1-quality)
 return selected,{'selected':selected,'marginal_scores':scores,'GT_reads':0}

def fuse(base,base_label,extras,base_coverage=None):
 p=base.astype(np.float32);p/=np.maximum(p.sum(0,keepdims=True),1e-8)
 if not extras:return p,base_label.copy(),{'B':0,'identity':True,'changed_pixels':0}
 C,H,W=p.shape
 bw=np.ones((H,W),np.float32) if base_coverage is None else np.broadcast_to(base_coverage,(H,W)).astype(np.float32).copy()
 num=p*bw;den=bw.copy();covered=np.zeros((H,W),bool)
 for s,ep in extras:
  e=ep.astype(np.float32);e/=np.maximum(e.sum(0,keepdims=True),1e-8);ww=e.shape[-1];cols=support(s,ww,W)
  conf=np.clip(1+((e*np.log(np.maximum(e,1e-8))).sum(0)/np.log(C)),.05,1)
  w=centrality(ww)[None,:]*conf
  num[:,:,cols]+=e*w;den[:,cols]+=w;covered[:,cols]=True
 out=num/np.maximum(den,1e-8);label=base_label.copy();label[covered]=out.argmax(0)[covered].astype(np.uint8)
 assert np.array_equal(label[~covered],base_label[~covered])
 return out,label,{'B':len(extras),'changed_pixels':int((label!=base_label).sum()),'covered_pixels':int(covered.sum()),'outside_coverage_changes':0}

def base_coverage():
 v=np.zeros(4096,np.float32)
 for s in range(0,3073,384):v[s:s+1024]+=2
 return v[::2][None,:]
