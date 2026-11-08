import numpy as np
from piers_ljj.collaboration.semantic_bridge import guide_and_union,compose_revision
from piers_ljj.collaboration.revision import build_revision

def square(x,y,size=6):
 m=np.zeros((16,16),bool);m[y:y+size,x:x+size]=True;return m

def test_scores_only_preserve_union_but_filter_consumes_kept_masks():
 a,b=square(0,0),square(10,10)
 prob=np.zeros((16,16),np.float32);prob[a]=0.9;prob[b]=0.01
 preds=[dict(id="a",detector_score=0.5),dict(id="b",detector_score=0.6)]
 args=(preds,[a,b],[np.ones_like(a),np.ones_like(b)],{12:prob},{"map13":12,"relation":"exact"},a.shape)
 soft,sm,_=guide_and_union(*args,mode="semantic_to_text")
 hard,hm,_=guide_and_union(*args,mode="semantic_to_text_hard",hard_thr=0.15)
 assert np.array_equal(sm,a|b)
 assert np.array_equal(hm,a)
 assert len([p for p in hard if p["kept"]])==1
 assert next(p for p in soft if p["id"]=="a")["rank_guided"]==0

def test_unmapped_fallback_never_expands_to_semantic_class():
 m=square(3,3)
 p=[dict(id="a",detector_score=0.6)]
 _,u,_=guide_and_union(p,[m],[np.ones_like(m)],{12:np.ones_like(m,dtype=float)},
                      {},m.shape,mode="semantic_to_text_hard")
 assert np.array_equal(u,m)
 _,u,_=guide_and_union(p,[m],[np.ones_like(m)],{12:np.ones_like(m,dtype=float)},
       {"map13":12,"relation":"coarse_compatible"},m.shape,mode="semantic_to_text")
 assert np.array_equal(u,m)

def test_consensus_disagreement_preserves_teacher():
 t=np.full((16,16),2,np.uint8)
 cs=[dict(mask=square(2,2),**{"class":12},relation="exact",sam_quality=0.95,support=0.1)]
 r=compose_revision(t,cs,semantic_threshold=0.5)
 assert np.array_equal(r["revised"],t)
 assert r["stats"]["skipped_semantic_disagreement"]==1

def test_cross_class_conflicts_keep_teacher_and_masks_are_disjoint():
 t=np.full((16,16),2,np.uint8)
 a,b=square(2,2,9),square(5,5,9)
 cs=[dict(mask=a,**{"class":12},relation="exact",sam_quality=0.95,support=0.9),
     dict(mask=b,**{"class":11},relation="exact",sam_quality=0.95,support=0.9)]
 r=compose_revision(t,cs,edge_width=1,semantic_threshold=0.5)
 assert r["conflict_mask"].any()
 assert np.array_equal(r["revised"][r["conflict_mask"]],t[r["conflict_mask"]])
 assert not np.any(r["revise_mask"]&r["ignore_mask"])
 assert np.array_equal(r["revised"][~(a|b)],t[~(a|b)])

def test_existing_revision_later_inner_clears_earlier_edge():
 a,b=square(1,1,9),square(5,1,9)
 r=build_revision((16,16),[dict(mask=a,quality=1),dict(mask=b,quality=1)],12,edge_width=1)
 assert not np.any(r["revise_mask"]&r["ignore_mask"])

def test_pixel_guard_keeps_confident_teacher_and_can_correct_uncertain_pixels():
 from piers_ljj.collaboration.semantic_bridge import guard_revision_pixels
 t=np.zeros((2,2),np.uint8)
 p0=np.array([[0.9,0.45],[0.9,0.45]],np.float32)
 p1=np.array([[0.05,0.4],[0.05,0.4]],np.float32)
 p2=1-p0-p1
 write=np.array([[1,1],[0,0]],bool);ignore=np.array([[0,0],[1,1]],bool)
 patch=dict(target_labels=np.ones((2,2),np.int16),revise_mask=write,ignore_mask=ignore,
            conflict_mask=np.zeros((2,2),bool),stats={})
 r=guard_revision_pixels(t,patch,{0:p0,1:p1,2:p2})
 assert np.array_equal(r["revised"],np.array([[0,1],[0,255]],np.uint8))
 assert r["stats"]["pixel_blocked_changes"]==1
 assert r["stats"]["confident_edges_preserved"]==1
