"""Meaningful contracts for native requested-class evaluation, no model mocks."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/piers'))
from full_model_fusion import make_queries,requested_probabilities,source_channel

def test_new_categories_can_win_against_confident_source_head():
    p=np.zeros((13,1,3),np.float32);p[0]=1
    classes=['road','curb','truck','bus'];e=np.zeros((3,1,3),np.float32)
    e[0,0,0]=.9;e[1,0,1]=.9;e[2,0,2]=.9
    scores,pred=requested_probabilities(p,classes,e)
    assert pred.tolist()==[[1,2,3]]
    assert np.allclose(scores.sum(0),1)
    assert np.array_equal(p[0],np.ones((1,3)))

def test_original_novel_channels_are_zero_and_abstentions_explicit():
    p=np.zeros((13,1,2),np.float32);p[0]=1
    scores,pred=requested_probabilities(p,['curb','truck'])
    assert not scores.any() and (pred==255).all()

def test_query_list_preserves_synpass_native_novel_concepts():
    classes=['building','fence','other object','pedestrian','pole','road marking','road','sidewalk',
             'vegetation','vehicles','wall','traffic sign','sky','ground','bridge','railway track',
             'guardrail','traffic light','static object','moving object','water','terrain']
    queries,bank,out,channels=make_queries('opaque-image',classes)
    assert len(classes)==22 and len(out)==22 and len(bank)==14
    assert sum(c is None for c in channels)==9
    assert source_channel('truck') is None and source_channel('bus') is None
    assert channels[5] is None and channels[6]==0
    assert len({q['query_id'] for q in queries})==len(queries)

def test_requested_class_permutation_preserves_predictions():
    p=np.zeros((13,1,2),np.float32);p[0]=.2;p[1]=.8
    a,pa=requested_probabilities(p,['road','sidewalk','curb'],np.array([[[.9,.0]]],np.float32))
    b,pb=requested_probabilities(p,['curb','sidewalk','road'],np.array([[[.9,.0]]],np.float32))
    assert np.array_equal(a,b[[2,1,0]]) and pa.tolist()==[[2,1]] and pb.tolist()==[[0,1]]

def test_abstention_counts_as_miss_and_void_GT_is_ignored():
    from score_zero_shot_native import confusion,metrics
    h=confusion(np.array([[0,255,1]],np.uint8),np.array([[11,13,0]],np.uint8),[11,13],[0])
    assert h.tolist()==[[1,0,0],[0,0,1]]
    score=metrics(h)
    assert score['mIoU_pct']==50 and score['pixel_accuracy_pct']==50
    assert score['valid_GT_pixels']==2 and score['abstained_valid_pixels']==1

def test_native_synpass_IDs_do_not_merge_roadline_or_guardrail():
    from score_zero_shot_native import confusion
    gt=np.array([[1,6,7,16,17,22,0]],np.uint8)
    pred=np.array([[0,5,6,15,16,21,255]],np.uint8)
    h=confusion(pred,gt,list(range(1,23)),[0])
    assert h.shape==(22,23) and h.sum()==6 and np.diag(h[:,:22]).sum()==6

if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:test()
    print('PASSED',len(tests),'requested-category contracts')
