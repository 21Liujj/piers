"""Pure NumPy uncertainty-tempered semantic correction; no file I/O or GT."""
import numpy as np
K=13

def normalize_teacher(probs):
 p=np.array(probs,dtype=np.float32,copy=True)
 assert p.ndim==3 and p.shape[0]==K and np.isfinite(p).all() and p.min()>=0
 p/=np.maximum(p.sum(0,keepdims=True),1e-8)
 assert np.max(np.abs(p.sum(0)-1))<1e-5
 return p

def soft_correct(p,stored,proposal,reliability,eligible,upper=.85):
 assert p.shape==(K,*stored.shape) and proposal.shape==reliability.shape==eligible.shape==stored.shape
 assert np.isin(stored,np.arange(K)).all()
 out=stored.copy();rr,cc=np.nonzero(eligible)
 if not len(rr):return out,np.empty(0,np.float32),np.empty(0,np.uint8),0.
 cls=proposal[rr,cc].astype(np.int64);r=reliability[rr,cc].astype(np.float32)
 assert np.isin(cls,np.arange(K)).all() and np.isfinite(r).all() and (r>=0).all() and (r<=1).all()
 scores=p[:,rr,cc].copy();maxp=scores.max(0)
 alpha=r*np.clip((np.float32(upper)-maxp)/np.float32(upper-1/K),0.,1.)
 scores*=1-alpha[None,:];scores[cls,np.arange(len(cls))]+=alpha
 winner=scores.argmax(0).astype(np.uint8);out[rr,cc]=winner
 mass_error=float(np.max(np.abs(scores.sum(0)-1)))
 assert not (out[~eligible]!=stored[~eligible]).any() and mass_error<1e-5
 return out,alpha,winner,mass_error
