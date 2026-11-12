"""Prevent fabricated multi-view votes and protected-teacher overwrites."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/piers'))
from indoor_matched_features import build_context
from indoor_matched_gate import proposals
from indoor_matched_soft import soft_correct
from indoor_matched_finish import metric

GATE={'pixel_text_confidence_threshold':.9,'distinct_FoV_votes_required':2,
      'baseline_ignore_confidence':.85,'teacher_high_conflict_protection':.8}

class Reliable:
    def predict_np(self,X):return np.full((len(X),2),.95,np.float32)

def context(classes,views,confident=False):
    h,w=6,8;probs=np.full((8,h,w),.125,np.float32)
    if confident:probs[:]=.1/7;probs[0]=.9
    fovs=[{'view_id':v,'win_h':h,'win_w':w,'x_start':0} for v in sorted(set(views))]
    queries=[{'query_id':str(c),'ontology':{'map13':c,'relation':'exact'}} for c in set(classes)]
    candidates=[{'id':str(k),'query_id':str(c),'fov_id':v,'sam_quality':.95,'detector_score':.9}
                for k,(c,v) in enumerate(zip(classes,views))]
    return build_context({'fovs':fovs,'queries':queries,'candidates':candidates},
                         np.ones((len(classes),h,w),bool),probs,probs.argmax(0))

def test_duplicate_candidates_do_not_count_as_two_views():
    ctx=context([1,1],['v0','v0']);arrays,_=proposals(ctx,Reliable(),GATE)
    assert (arrays['label']==255).all()
    assert arrays['votes'].max()==1

def test_two_independent_views_can_correct_uncertain_semantics():
    ctx=context([1,1],['v0','v1']);arrays,_=proposals(ctx,Reliable(),GATE)
    eligible=arrays['label']!=255
    result,_,_,error=soft_correct(ctx.probs,ctx.teacher_label,arrays['label'],arrays['weight'],eligible)
    assert (result==1).all() and error<1e-5
    assert ctx.features_at(1,'v0',np.array([0]),np.array([0])).shape==(1,26)

def test_confident_teacher_and_cross_class_conflict_are_protected():
    for ctx in [context([1,1],['v0','v1'],True),context([1,1,2,2],['v0','v1','v0','v1'])]:
        arrays,_=proposals(ctx,Reliable(),GATE)
        assert (arrays['label']==255).all()

def test_accuracy_names_and_outside_predictions():
    cm=np.zeros((8,9),np.int64);cm[0,0]=1;cm[1,0]=2;cm[1,1]=1;cm[1,8]=1
    result=metric(cm)
    assert result['PixelAcc_pct']==40.
    assert result['mAcc_pct']==62.5
    assert result['per_class_gt_support']['chair']==4
    assert result['predictions_outside_label_set']==1
