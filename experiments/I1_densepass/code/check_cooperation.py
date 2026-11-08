import numpy as np,json
from pathlib import Path
from cooperation import morphology,prepare,text_outputs,semantic_revision,confusion,scores
H,W=24,48;v=np.ones((H,W),bool);m=np.zeros((H,W),bool);m[3:21,5:40]=True
q={'q':{'ontology':{'map13':12,'relation':'coarse_compatible'}}};meta={'fovs':[{'view_id':'v','x_start':0,'y_start':0,'win_w':W,'win_h':H}], 'candidates':[{'id':'c','query_id':'q','fov_id':'v','sam_quality':.95,'detector_score':.9}]}
p=np.zeros((13,H,W),np.float32);p[12]=.9;p[0]=.1;p[12,9:11,9:11]=.4;p[0,9:11,9:11]=.6;p[12,13:15,13:15]=.01;p[0,13:15,13:15]=.99
rec=prepare(meta,np.stack([m]),p,q);hard=p.argmax(0).astype(np.uint8);rev,stats=semantic_revision(hard,p,rec,mode='gated')
assert stats['changed_pixels']>0
assert np.array_equal(rev[13:15,13:15],hard[13:15,13:15])
unknown=[{**rec[0],'class':None,'relation':'unmapped','verdict':{'state':'uncertain','support':0}}]
t,_=text_outputs(unknown,p,['q']);assert np.array_equal(t['q'],m)
u,us=semantic_revision(hard,p,unknown,mode='gated');assert np.array_equal(u,hard)
confp=np.zeros_like(p);confp[0]=.99;confp[12]=.01;conf=prepare(meta,np.stack([m]),confp,q);t,_=text_outputs(conf,confp,['q']);assert np.array_equal(t['q'],m)
band=np.zeros((10,12),bool);band[:,[11,0,1]]=True;er=morphology(band,1);assert er[5,0]
h=confusion(np.full((2,2),255,np.uint8),np.zeros((2,2),np.uint8));assert scores(h)['mIoU']==0
print(json.dumps({'status':'passed','checks':['actual_nonzero_revision','confident_conflict_unchanged','unmapped_text_exact_fallback','unmapped_semantic_exact_fallback','conflict_text_exact_fallback','ERP_horizontal_wrap_morphology','prediction_ignore_counted_as_error'],'revision_stats':stats},indent=2))
