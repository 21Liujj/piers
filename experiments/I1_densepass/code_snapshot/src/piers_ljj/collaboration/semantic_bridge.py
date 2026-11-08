"""Inference-only semantic/text bridge; evaluation labels are never inputs."""
from pathlib import Path
import json,hashlib
import numpy as np
from .text_guide import apply_mode
from .revision import erode

class FrozenSemanticCache:
    def __init__(self,directory,manifest_path=None):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/"metadata.json").read_text())
        if self.meta.get("status")!="complete" or self.meta.get("class_space")!="densepass13@v1":
            raise ValueError("Incomplete or incompatible E0 cache")
        if manifest_path is not None:
            digest=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if digest!=self.meta["manifest_sha256"]:
                raise ValueError("E0 cache and query manifest differ")
        self.data=np.load(self.directory/"probs.npz",allow_pickle=False)
        self.ids={k for k in self.data.files if not k.startswith("__")}
        self.last_id=None;self.last_probs=None
    def image(self,image_id):
        if image_id not in self.ids:
            raise KeyError("E0 probability cache missing image "+image_id)
        if image_id!=self.last_id:
            arr=self.data[image_id].astype(np.float32)
            if arr.shape!=(13,400,2048) or not np.isfinite(arr).all():
                raise ValueError("Invalid teacher cache array")
            if arr.min()<0 or arr.max()>1 or np.max(np.abs(arr.sum(0)-1))>0.002:
                raise ValueError("Cache is not normalized class probabilities")
            self.last_id=image_id;self.last_probs={c:arr[c] for c in range(13)}
        return self.last_probs

def guide_and_union(predictions,masks,valids,probs,ontology,shape,
                    mode="none",alpha=0.5,hard_thr=0.15):
    if not (len(predictions)==len(masks)==len(valids)):
        raise ValueError("Candidate mask/metadata length mismatch")
    relation=(ontology or {}).get("relation","unmapped")
    cls=(ontology or {}).get("map13") if relation in {"exact","coarse_compatible"} else None
    items=[{**p,"mask":np.asarray(m,bool),"valid":np.asarray(v,bool),
            "class":cls,"relation":relation if cls is not None else "unmapped"}
           for p,m,v in zip(predictions,masks,valids)]
    if len({p["id"] for p in items})!=len(items):
        raise ValueError("Duplicate candidate ID")
    guided,log=apply_mode(items,probs,mode=mode,alpha=alpha,hard_thr=hard_thr)
    union=np.zeros(shape,bool)
    for item in guided:
        if item["kept"]:union|=item["mask"]
    clean=[{k:v for k,v in p.items() if k not in {"mask","valid"}} for p in guided]
    rank={p["id"]:i for i,p in enumerate(sorted(clean,key=lambda x:-x["score_guided"]))}
    for p in clean:p["rank_guided"]=rank[p["id"]]
    return clean,union,log

def compose_revision(teacher,candidates,qual_thr=0.8,edge_width=2,
                     semantic_threshold=None):
    """Candidate records carry predicted class, actual text mask, SAM quality and E0 support."""
    teacher=np.asarray(teacher)
    proposal=np.full(teacher.shape,-1,np.int16)
    conflict=np.zeros(teacher.shape,bool)
    edges=np.zeros(teacher.shape,bool)
    counts=dict(candidates=len(candidates),accepted=0,skipped_low_quality=0,
                skipped_unmapped=0,skipped_semantic_disagreement=0)
    for c in candidates:
        cls=c.get("class")
        if cls is None or c.get("relation") not in {"exact","coarse_compatible"}:
            counts["skipped_unmapped"]+=1;continue
        if float(c.get("sam_quality",0))<qual_thr:
            counts["skipped_low_quality"]+=1;continue
        if semantic_threshold is not None and float(c.get("support",0))<semantic_threshold:
            counts["skipped_semantic_disagreement"]+=1;continue
        mask=np.asarray(c["mask"],bool)
        if mask.shape!=teacher.shape:raise ValueError("Revision coordinates differ from teacher")
        inner=erode(mask,edge_width)
        conflict|=inner & (proposal>=0) & (proposal!=int(cls))
        proposal[inner & (proposal<0)]=int(cls)
        edges|=mask & ~inner
        counts["accepted"]+=1
    revise=(proposal>=0)&~conflict
    ignore=edges&~revise&~conflict
    revised=teacher.copy();revised[revise]=proposal[revise];revised[ignore]=255
    no_edge=teacher.copy();no_edge[revise]=proposal[revise]
    if np.any(revise&ignore):raise AssertionError("Overlapping write/ignore regions")
    if not np.array_equal(revised[~(revise|ignore)],teacher[~(revise|ignore)]):
        raise AssertionError("Pixels outside masks changed")
    if not np.array_equal(revised[conflict],teacher[conflict]):
        raise AssertionError("Cross-class conflict must keep the teacher")
    counts.update(written_pixels=int(revise.sum()),changed_pixels=int((revise&(teacher!=proposal)).sum()),
                  ignored_edge_pixels=int(ignore.sum()),conflict_pixels=int(conflict.sum()))
    return dict(revised=revised,no_edge_ignore=no_edge,revise_mask=revise,
                ignore_mask=ignore,target_labels=proposal,conflict_mask=conflict,stats=counts)

def guard_revision_pixels(teacher,patch,probs,confidence_threshold=0.8,target_ratio=0.5):
    """Keep confident teacher pixels; only let plausible text correct uncertain pixels."""
    teacher=np.asarray(teacher)
    confidence=np.maximum.reduce([np.asarray(p,np.float32) for p in probs.values()])
    proposal=patch["target_labels"]
    proposed=patch["revise_mask"]
    target_prob=np.zeros(teacher.shape,np.float32)
    for cls in np.unique(proposal[proposed]):
        m=proposed&(proposal==cls);target_prob[m]=np.asarray(probs[int(cls)])[m]
    uncertain=confidence<confidence_threshold
    write=proposed&((teacher==proposal)|(uncertain&(target_prob>=target_ratio*confidence)))
    ignore=patch["ignore_mask"]&uncertain
    revised=teacher.copy();revised[write]=proposal[write];revised[ignore]=255
    noedge=teacher.copy();noedge[write]=proposal[write]
    conflict=patch["conflict_mask"]
    assert not (write&ignore).any()
    assert np.array_equal(revised[~(write|ignore)],teacher[~(write|ignore)])
    assert np.array_equal(revised[conflict],teacher[conflict])
    stats={**patch["stats"],"written_pixels":int(write.sum()),
      "changed_pixels":int((write&(teacher!=proposal)).sum()),"ignored_edge_pixels":int(ignore.sum()),
      "pixel_blocked_changes":int((proposed&(teacher!=proposal)&~write).sum()),
      "confident_edges_preserved":int((patch["ignore_mask"]&~uncertain).sum())}
    return dict(revised=revised,no_edge_ignore=noedge,revise_mask=write,ignore_mask=ignore,
                target_labels=proposal,conflict_mask=conflict,stats=stats)
