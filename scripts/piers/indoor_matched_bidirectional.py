# Indoor512 FoV port of published bidirectional cooperation; rules unchanged.
"""Auditable frozen-teacher cooperation, class13; open queries fall back exactly."""
import numpy as np,cv2
from piers_ljj.collaboration.reliability_gate import evaluate_candidate
from piers_ljj.text.assoc import associate

def morphology(mask,r,op='erode'):
 m=np.asarray(mask,np.uint8)
 if not r:return m.astype(bool)
 padded=np.pad(np.pad(m,((0,0),(r,r)),mode='wrap'),((r,r),(0,0)),mode='constant')
 kernel=np.ones((2*r+1,2*r+1),np.uint8)
 result=cv2.erode(padded,kernel) if op=='erode' else cv2.dilate(padded,kernel)
 return result[r:-r,r:-r].astype(bool)

def prepare(meta,masks,probs,qmeta):
 H,W=probs.shape[1:];valids={}
 for f in meta['fovs']:
  v=np.zeros((H,W),bool);cols=(f['x_start']+np.arange(f['win_w']))%W
  v[np.ix_(np.arange(f.get('y_start',0),f.get('y_start',0)+f['win_h']),cols)]=True;valids[f['view_id']]=v
 records=[]
 for i,c in enumerate(meta['candidates']):
  onto=qmeta.get(c['query_id'],{}).get('ontology') or {};cls=onto.get('map13');rel=onto.get('relation','unmapped')
  c={**c,'mask':masks[i],'valid':valids[c['fov_id']],'class':cls,'relation':rel}
  c['verdict']=evaluate_candidate(c,probs,c['valid']);records.append(c)
 return records

def text_outputs(records,probs,qids,*,alpha=.5,mode='gated_consensus'):
 H,W=probs.shape[1:];out={};stats={'agreed_groups':0,'fallback_groups':0,'changed_pixels':0};conf=probs.max(0);hard=probs.argmax(0)
 byq={qid:[] for qid in qids}
 for c in records:byq.setdefault(c['query_id'],[]).append(c)
 for qid,items in byq.items():
  union=np.zeros((H,W),bool)
  for c in items:union|=c['mask']
  result=np.zeros((H,W),bool)
  if mode=='independent' or not items:out[qid]=union;continue
  byid={c['id']:c for c in items}
  for group in associate(items):
   cs=[byid[x] for x in group['members']];base=np.zeros((H,W),bool)
   for c in cs:base|=c['mask']
   cls=cs[0]['class'];agreed=all(c['verdict']['state']=='agree' and float(c['sam_quality'])>=.8 for c in cs)
   if cls is None or (not agreed and mode not in ('no_semantic_consensus','ungated_semantic_consensus')):
    result|=base;stats['fallback_groups']+=1;continue
   cls=int(cls);stats['agreed_groups']+=1
   if mode=='score_only':guided=base
   elif mode=='boundary':
    edge=base&~morphology(base,2);ambiguous=conf<.8
    guided=base&~(edge&ambiguous&(probs[cls]<.2))
   else:
    numerator=np.zeros((H,W),np.float32);denominator=np.zeros((H,W),np.float32)
    for c in cs:
     weight=max(.01,float(c['detector_score'])*float(c['sam_quality']))
     if mode!='no_semantic_consensus':weight*=1+alpha*c['verdict']['support']
     numerator+=weight*c['mask'];denominator+=weight*c['valid']
    vote=numerator/np.maximum(denominator,1e-8)
    if mode!='no_semantic_consensus':vote+=alpha*(probs[cls]-.5)*morphology(base,2,'dilate')
    guided=(vote>=.5)&morphology(base,2,'dilate')
   # Explicit pixel conflict: restore the independent text result exactly.
   conflict=(conf>=.8)&(hard!=cls)
   if mode not in ('no_semantic_consensus','ungated_semantic_consensus'):guided[conflict]=base[conflict]
   result|=guided
  stats['changed_pixels']+=int((result!=union).sum());out[qid]=result
 return out,stats

def semantic_revision(base,probs,records,*,mode='gated',edge=2):
 base=np.asarray(base);H,W=base.shape;conf=probs.max(0);hard=probs.argmax(0)
 prop=np.full((H,W),-1,np.int16);cross=np.zeros((H,W),bool);bound=np.zeros((H,W),bool);blocked=np.zeros((H,W),bool)
 stats={'candidates':len(records),'accepted':0,'agree':0,'conflict':0,'uncertain':0,'blocked_conflicts':0}
 for c in records:stats[c['verdict']['state']]+=1
 for c in sorted(records,key=lambda c:float(c['sam_quality'])*float(c['detector_score']),reverse=True):
  cls=c['class']
  if cls is None or c['relation'] not in ('exact','coarse_compatible'):continue
  cls=int(cls)
  if mode!='naive' and float(c['sam_quality'])<.8 and not c.get('core_reliable',False):continue
  if mode=='gated' and c['verdict']['state']!='agree':continue
  mask=c['mask']&c['valid'];inner=mask if mode=='naive' else morphology(mask,edge)
  if not inner.any():continue
  stats['accepted']+=1
  if mode!='naive':
   conflict=(conf>=.8)&(hard!=cls)&inner;blocked|=conflict;inner=inner&~conflict
  cross|=inner&(prop>=0)&(prop!=cls)
  take=inner&(prop<0);prop[take]=cls
  bound|=mask&~inner
 if mode!='naive':prop[cross]=-1
 rev=base.copy();write=prop>=0;rev[write]=prop[write]
 changed=rev!=base
 stats.update(changed_pixels=int(changed.sum()),recovered_ignore=int((changed&(base==255)).sum()),written_pixels=int(write.sum()),blocked_conflicts=int(blocked.sum()),cross_class_conflicts=int(cross.sum()),coverage=float((rev!=255).mean()))
 return rev,stats

def confusion(pred,gt,n=13):
 valid=(gt>=0)&(gt<n);p=pred[valid].astype(np.int64);g=gt[valid].astype(np.int64)
 p=np.where((p>=0)&(p<n),p,n)
 return np.bincount(g*(n+1)+p,minlength=n*(n+1)).reshape(n,n+1)

def scores(hist):
 h=np.asarray(hist,dtype=np.float64);n=h.shape[0];tp=np.diag(h[:,:n]);den=h.sum(1)+h[:,:n].sum(0)-tp
 iou=np.divide(tp,den,out=np.full(n,np.nan),where=den>0)
 acc=np.divide(tp,h.sum(1),out=np.full(n,np.nan),where=h.sum(1)>0)
 return {'mIoU':float(np.nanmean(iou)*100),'mAcc':float(np.nanmean(acc)*100),'PixAcc':float(tp.sum()/max(h.sum(),1)*100),'per_class_iou_pct':(iou*100).tolist()}

def bounded_text_fusion(records,probs,qids,*,alpha=.5,gated=True):
 """Fuse only exact-taxonomy, agreeing text objects inside existing detector boxes.

 Coarse-compatible car/person and unmapped free text keep independent masks.
 Pixel conflict/low evidence restores the original text mask in gated mode.
 """
 H,W=probs.shape[1:];out={qid:np.zeros((H,W),bool) for qid in qids};hard=probs.argmax(0);conf=probs.max(0)
 stats={'accepted_candidates':0,'fallback_candidates':0,'added_pixels':0,'removed_pixels':0}
 for c in records:
  base=c['mask'];cls=c['class'];qid=c['query_id']
  eligible=cls is not None and c['relation']=='exact'
  if gated:eligible=eligible and c['verdict']['state']=='agree' and (float(c['sam_quality'])>=.8 or c.get('core_reliable',False))
  if not eligible:out[qid]|=base;stats['fallback_candidates']+=1;continue
  cls=int(cls);box=np.zeros((H,W),bool)
  segments=c.get('box_segs')
  if segments:
   for x1,y1,x2,y2 in segments:
    box[max(0,int(np.floor(y1))):min(H,int(np.ceil(y2))),max(0,int(np.floor(x1))):min(W,int(np.ceil(x2)))]=True
  else:
   # Training records retain FoV-local DINO boxes. The mask support's visible
   # FoV origin is recoverable from its explicit valid coverage.
   cols=np.flatnonzero(c['valid'].any(0));start=int(cols[0])
   if len(cols)>1 and np.any(np.diff(cols)>1):start=int(cols[np.argmax(np.diff(cols))+1])
   x1,y1,x2,y2=c['local_box'];xs=(start+np.arange(max(0,int(np.floor(x1))),min(512,int(np.ceil(x2)))))%W
   ys=np.arange(max(0,int(np.floor(y1))),min(H,int(np.ceil(y2))))
   box[np.ix_(ys,xs)]=True
  area=box&c['valid'];soft=(1-alpha)*np.where(base,.7,.3)+alpha*probs[cls]
  refined=base.copy();refined[area]=soft[area]>=.5
  if gated:
   fallback=((conf>=.8)&(hard!=cls))|(conf<.5)
   refined[fallback]=base[fallback]
  stats['accepted_candidates']+=1;stats['added_pixels']+=int((refined&~base).sum());stats['removed_pixels']+=int((base&~refined).sum());out[qid]|=refined
 return out,stats

def core_agreement(records,probs):
 """Positive interior correspondence can certify an oversegmented mask.

 Exact taxonomy only. Never converts an explicit conflict or missing class
 into agreement. Independent text score, SAM quality, location and frozen
 semantic core are all required; thresholds are fixed, not searched.
 """
 result=[]
 for original in records:
  c={**original,'verdict':dict(original['verdict'])};cls=c['class']
  if cls is not None and c['relation']=='exact' and c['verdict']['state']!='conflict':
   inner=morphology(c['mask']&c['valid'],2);area=int(inner.sum())
   core=int((inner&(probs[int(cls)]>=.8)).sum());fraction=core/max(area,1)
   reliable=core>=64 and fraction>=.1 and float(c['sam_quality'])>=.65 and float(c['detector_score'])>=.5
   if reliable:
    c['core_reliable']=True;c['core_pixels']=core;c['core_fraction']=fraction
    c['verdict']['state']='agree';c['verdict']['reason']='positive_interior_core_with_independent_text_and_mask_quality'
  result.append(c)
 return result
