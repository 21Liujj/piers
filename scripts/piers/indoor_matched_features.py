# Ported from experiments/paper20261109/runtime/feature_core.py.
# Only taxonomy size changes13->8; ontology map13 is a legacy class-id key.
"""Indoor8 port of the frozen semantic-text feature interface. No file I/O or labels.

Input arrays and cached candidate metadata only. Callers may apply this directly
to target13 caches; source/target GT, original-image crop y and image ids never
enter feature values. Each class/FoV contributes one union and max-quality maps.
"""
from dataclasses import dataclass
import numpy as np
import cv2

FEATURE_NAMES=[
 'teacher_p_class','teacher_max_prob','teacher_entropy_norm','teacher_top1_top2_margin','teacher_pred_equals_class',
 'view_SAM_max','view_DINO_max','view_log_candidate_area_max',
 'text_SAM_mean_views','text_SAM_max_views','text_DINO_mean_views','text_DINO_max_views',
 'text_log_candidate_area_mean_views','text_log_candidate_area_max_views',
 'distinct_FoV_votes','visible_FoV_count','distinct_votes_over_visible_views','view_mask_edge_distance_log_norm',
]+['class_'+str(c).zfill(2)+'_onehot' for c in range(8)]
TEACHER_ONLY_FEATURE_INDICES=list(range(5))+list(range(18,26))
assert len(FEATURE_NAMES)==26

@dataclass
class GroupMaps:
 class_id:int
 view_id:str
 mask:np.ndarray
 sam:np.ndarray
 dino:np.ndarray
 area:np.ndarray
 candidates:int=0

class FeatureContext:
 """In-memory image features; no GT path or GT array is accepted."""
 def __init__(self,meta,masks,teacher_probs,teacher_label,qmeta=None):
  probs=np.array(teacher_probs,dtype=np.float32,copy=True)
  assert probs.ndim==3 and probs.shape[0]==8 and np.isfinite(probs).all()
  assert probs.min()>=0
  probs/=np.maximum(probs.sum(0,keepdims=True),1e-8)
  self.probs=probs;self.H,self.W=probs.shape[1:]
  self.teacher_label=np.asarray(teacher_label).copy()
  assert self.teacher_label.shape==(self.H,self.W) and set(np.unique(self.teacher_label))<=set(range(8))
  self.maxp=probs.max(0)
  self.entropy=-(probs*np.log(np.maximum(probs,1e-8))).sum(0)/np.log(8.)
  top2=np.partition(probs,-2,axis=0)[-2:];self.margin=top2.max(0)-top2.min(0);del top2
  self.valids={};self.view_ids=[];self.wrap_x=False
  for f in meta['fovs']:
   vid=str(f['view_id']);assert vid not in self.valids
   h=int(f['win_h']);w=int(f['win_w']);x=int(f['x_start']);y=int(f.get('y_start',0))
   assert 0<=y and y+h<=self.H and 0<w<=self.W
   v=np.zeros((self.H,self.W),bool);cols=(x+np.arange(w))%self.W
   v[np.ix_(np.arange(y,y+h),cols)]=True;self.valids[vid]=v;self.view_ids.append(vid)
   self.wrap_x|=x<0 or x+w>self.W
  self.visible_count=np.stack(list(self.valids.values())).sum(0).astype(np.float32)
  query={q['query_id']:q for q in meta.get('queries',[])}
  if qmeta is not None:query.update(qmeta)
  masks=np.asarray(masks,dtype=bool);assert masks.shape==(len(meta['candidates']),self.H,self.W)
  self.groups={};self.skipped_unmapped=0;self.input_candidates=len(meta['candidates'])
  normalizer=float(np.log1p(self.H*self.W))
  for c,rawmask in zip(meta['candidates'],masks):
   onto=(query.get(c['query_id'],{}).get('ontology') or c.get('ontology') or {})
   cls=onto.get('map13');relation=onto.get('relation','unmapped')
   if cls is None or relation not in ['exact','coarse_compatible']:
    self.skipped_unmapped+=1;continue
   cls=int(cls)
   if cls<0 or cls>=8:self.skipped_unmapped+=1;continue
   vid=str(c['fov_id']);assert vid in self.valids
   mask=rawmask&self.valids[vid];area=int(mask.sum())
   if not area:continue
   key=(cls,vid)
   if key not in self.groups:
    self.groups[key]=GroupMaps(cls,vid,np.zeros((self.H,self.W),bool),np.zeros((self.H,self.W),np.float32),np.zeros((self.H,self.W),np.float32),np.zeros((self.H,self.W),np.float32))
   g=self.groups[key];g.candidates+=1;g.mask|=mask
   sam=float(np.clip(c['sam_quality'],0,1));dino=float(np.clip(c['detector_score'],0,1));logarea=float(np.log1p(area)/normalizer)
   g.sam[mask]=np.maximum(g.sam[mask],sam);g.dino[mask]=np.maximum(g.dino[mask],dino);g.area[mask]=np.maximum(g.area[mask],logarea)
  self.class_groups={cls:[g for (c,v),g in sorted(self.groups.items()) if c==cls] for cls in range(8)}

 def iter_groups(self):
  for key in sorted(self.groups):yield self.groups[key]

 def _edge_at(self,group,rows,cols):
  # Horizontal wrap padding makes target seam geometry consistent; source FoVs
  # are nonwrap and the source sampler excludes frame-edge single-view bands.
  m=group.mask.astype(np.uint8)
  if self.wrap_x:
   pad=min(self.H,self.W);x=np.pad(m,((0,0),(pad,pad)),mode='wrap');x=np.pad(x,((1,1),(0,0)),mode='constant')
   dist=cv2.distanceTransform(x,cv2.DIST_L2,3)[1:-1,pad:-pad]
  else:dist=cv2.distanceTransform(np.pad(m,((1,1),(1,1)),mode='constant'),cv2.DIST_L2,3)[1:-1,1:-1]
  return np.log1p(dist[rows,cols])/np.log1p(min(self.H,self.W))

 def features_at(self,class_id,view_id,rows,cols):
  rows=np.asarray(rows,dtype=np.int64);cols=np.asarray(cols,dtype=np.int64)
  assert rows.shape==cols.shape and rows.ndim==1
  group=self.groups[(int(class_id),str(view_id))]
  assert group.mask[rows,cols].all()
  n=len(rows);X=np.zeros((n,len(FEATURE_NAMES)),np.float32);cls=int(class_id)
  X[:,0]=self.probs[cls,rows,cols];X[:,1]=self.maxp[rows,cols];X[:,2]=self.entropy[rows,cols];X[:,3]=self.margin[rows,cols];X[:,4]=(self.teacher_label[rows,cols]==cls)
  X[:,5]=group.sam[rows,cols];X[:,6]=group.dino[rows,cols];X[:,7]=group.area[rows,cols]
  others=self.class_groups[cls];votes=np.stack([g.mask[rows,cols] for g in others]);counts=votes.sum(0).astype(np.float32);denom=np.maximum(counts,1)
  for mean_index,max_index,attr in [(8,9,'sam'),(10,11,'dino'),(12,13,'area')]:
   values=np.stack([getattr(g,attr)[rows,cols] for g in others]);X[:,mean_index]=values.sum(0)/denom;X[:,max_index]=values.max(0)
  visible=self.visible_count[rows,cols];X[:,14]=counts;X[:,15]=visible;X[:,16]=counts/np.maximum(visible,1);X[:,17]=self._edge_at(group,rows,cols)
  X[:,18+cls]=1
  assert X.shape==(n,26) and np.isfinite(X).all()
  return X


def build_context(meta,masks,teacher_probs,teacher_label,qmeta=None):
 return FeatureContext(meta,masks,teacher_probs,teacher_label,qmeta)

