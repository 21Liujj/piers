# Indoor geometry/taxonomy port of frozen fresh_core.py; gate thresholds unchanged.
"""Frozen GT-free proposal gate and shared reporting; no I/O or GT."""
import numpy as np
from indoor_matched_soft import normalize_teacher,soft_correct
def histogram(values):return np.bincount(values.astype(np.int64),minlength=8).tolist()

def proposals(ctx,models,protocol,batch_size=65536):
 assert ctx.probs.shape==(8,ctx.H,ctx.W)
 threshold=float(protocol['pixel_text_confidence_threshold']);nviews=int(protocol['distinct_FoV_votes_required'])
 ignore=float(protocol['baseline_ignore_confidence']);protect=float(protocol['teacher_high_conflict_protection'])
 votes=np.zeros((8,ctx.H,ctx.W),np.uint8);sums=np.zeros((8,ctx.H,ctx.W),np.float32)
 teacher_sums=np.zeros((8,ctx.H,ctx.W),np.float32);groups=[]
 base_ignore=ctx.maxp<ignore
 for group in ctx.iter_groups():
  cls=group.class_id;blocked=group.mask&base_ignore&(ctx.maxp>=protect)&(ctx.teacher_label!=cls)
  eligible=group.mask&base_ignore&~blocked;rr,cc=np.nonzero(eligible)
  accepted=0;Ptext_sum=Pteacher_sum=0.
  for start in range(0,len(rr),batch_size):
   rows,cols=rr[start:start+batch_size],cc[start:start+batch_size]
   X=ctx.features_at(cls,group.view_id,rows,cols);p=models.predict_np(X);keep=p[:,0].astype(np.float64)>=threshold
   r,c=rows[keep],cols[keep]
   votes[cls,r,c]+=1;sums[cls,r,c]+=p[keep,0];teacher_sums[cls,r,c]+=p[keep,1]
   accepted+=int(keep.sum());Ptext_sum+=float(p[keep,0].sum(dtype=np.float64));Pteacher_sum+=float(p[keep,1].sum(dtype=np.float64))
  groups.append({'class_id':cls,'view_id':group.view_id,'raw_union_pixels':int(group.mask.sum()),'high_teacher_conflict_blocked':int(blocked.sum()),'pixels_scored':len(rr),'Ptext_threshold_pixels':accepted,'mean_Ptext_pass':Ptext_sum/accepted if accepted else None,'mean_Pteacher_pass':Pteacher_sum/accepted if accepted else None})
 # Every competing calibrated per-view proposal counts as a conflict, even if
 # its class has only one FoV. This matches the previous all-class vote policy.
 competing=(votes>0).sum(0)>1;maxvotes=votes.max(0);top=votes.argmax(0)
 accepted=(maxvotes>=nviews)&~competing
 label=np.full((ctx.H,ctx.W),255,np.uint8);weight=np.zeros((ctx.H,ctx.W),np.float32)
 rr,cc=np.nonzero(accepted);cls=top[rr,cc];label[rr,cc]=cls
 weight[rr,cc]=sums[cls,rr,cc]/votes[cls,rr,cc]
 assert not (accepted&~base_ignore).any()
 assert not (accepted&(ctx.maxp>=protect)&(ctx.teacher_label!=label)).any()
 assert np.isfinite(weight).all() and weight.min()>=0 and weight.max()<=1
 assert not accepted.any() or float(weight[accepted].min())>=threshold-1e-6
 report={'groups':groups,'aux_pixels':len(rr),'coverage':len(rr)/(ctx.H*ctx.W),'baseline_ignore_pixels':int(base_ignore.sum()),'eligible_competing_class_pixels_rejected':int(competing.sum()),'two_view_pixels_before_cross_class_reject':int((maxvotes>=nviews).sum()),'class_hist':np.bincount(label[accepted],minlength=8).tolist(),'mean_weight':float(weight[accepted].mean()) if accepted.any() else 0.,'all_weights_are_calibrated_Ptext_only':True,'no_old_SAM_or_teacher_average_support_threshold':True}
 return {'label':label,'weight':weight,'votes':maxvotes,'proposal_class_count':(votes>0).sum(0).astype(np.uint8)},report

def arm_counts(stored,lab,w,votes,eligible,out,alpha=None,mass_error=0.,p=None):
 changed=out!=stored;same=eligible&(lab==stored);disagree=eligible&~same
 strong=(p.max(0)>=.8)&(stored!=lab) if p is not None else np.zeros_like(eligible)
 assert not (changed&~eligible).any() and not (changed&strong).any()
 row={'total_native_pixels':int(stored.size),'eligible_pixels':int(eligible.sum()),'eligible_fraction':float(eligible.mean()),'nominal_text_disagreement_pixels':int(disagree.sum()),'textsameclass_noop_pixels':int(same.sum()),'actual_changed_pixels':int(changed.sum()),'changed_to_text_pixels':int((changed&(out==lab)).sum()),'changed_to_neither_text_nor_original_pixels':int((changed&(out!=lab)).sum()),'unchanged_despite_text_disagreement_pixels':int((disagree&~changed).sum()),'outside_eligible_changes':int((changed&~eligible).sum()),'highconflict_protection_violations':int((changed&strong).sum()),'eligible_class_hist':histogram(lab[eligible]),'changed_to_class_hist':histogram(out[changed]),'changed_from_class_hist':histogram(stored[changed]),'eligible_votes_min':int(votes[eligible].min()) if eligible.any() else None,'eligible_votes_max':int(votes[eligible].max()) if eligible.any() else None,'posterior_mass_error':mass_error}
 if alpha is not None:
  row['alpha']={'min':float(alpha.min()) if len(alpha) else None,'max':float(alpha.max()) if len(alpha) else None,'mean':float(alpha.mean()) if len(alpha) else None,'sum':float(alpha.sum(dtype=np.float64)),'histogram_edges':np.linspace(0,1,11).tolist(),'histogram_counts':np.histogram(alpha,bins=np.linspace(0,1,11))[0].tolist()}
 return row
