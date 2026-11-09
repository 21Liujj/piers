"""GT-free adapter around recovered I1/I2/I3 and text novel-class evidence.

The frozen source semantic head is retained internally. Public outputs use the
caller-supplied category list. No dataset identity or target label is accepted.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np

SOURCE_CLASSES=('road','sidewalk','building','wall','fence','pole','traffic light',
                'traffic sign','vegetation','terrain','sky','person','car')
ALIASES={'pedestrian':'person','pedestrians':'person','people':'person','human':'person',
         'vehicle':'car','vehicles':'car','cars':'car','side walk':'sidewalk',
         'trafficlight':'traffic light','trafficsign':'traffic sign'}

def normalize(text):return ' '.join(text.strip().lower().replace('_',' ').split())

def source_channel(text):
    name=normalize(text);name=ALIASES.get(name,name)
    return SOURCE_CLASSES.index(name) if name in SOURCE_CLASSES else None

def make_queries(image_id,classes):
    names=tuple(normalize(x) for x in classes)
    if not names or len(names)>254 or len(set(names))!=len(names):
        raise ValueError('Supply 1..254 distinct category names')
    channels=[source_channel(x) for x in names]
    supported=[x for x in channels if x is not None]
    if len(supported)!=len(set(supported)):
        raise ValueError('Two requested categories map to the same source concept')
    bank=[]
    for i,name in enumerate(SOURCE_CLASSES):
        bank.append({'image_id':image_id,'query_id':image_id+':bank:'+str(i),
                     'raw_text':name,'task_type':'category_union','language':'en',
                     'ontology':{'map13':i,'relation':'coarse_compatible' if i in (11,12) else 'exact'}})
    bank.append({'image_id':image_id,'query_id':image_id+':bank:train','raw_text':'train',
                 'task_type':'category_union','language':'en','ontology':{'map13':None,'relation':'unmapped'}})
    all_queries=list(bank);output_queries=[]
    for i,(name,channel) in enumerate(zip(names,channels)):
        if channel is not None:q=bank[channel]
        elif name=='train':q=bank[-1]
        else:
            q={'image_id':image_id,'query_id':image_id+':requested:'+str(i),'raw_text':name,
               'task_type':'category_union','language':'en','ontology':{'map13':None,'relation':'unmapped'}}
            all_queries.append(q)
        output_queries.append(q['query_id'])
    return all_queries,[q['query_id'] for q in bank],output_queries,channels

def requested_probabilities(semantic13,classes,novel_evidence=None):
    """Same conditional competition for both methods, with explicit abstention.

    Unsupported original-head categories receive zero, never a fabricated
    channel. A new category gets mass only from independent text evidence.
    """
    p=np.asarray(semantic13,np.float32)
    if p.ndim!=3 or p.shape[0]!=13 or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Expected finite nonnegative source-head scores')
    channels=[source_channel(x) for x in classes];known=[i for i,c in enumerate(channels) if c is not None]
    novel=[i for i,c in enumerate(channels) if c is None]
    scores=np.zeros((len(classes),*p.shape[1:]),np.float32)
    if known:
        base=p[[channels[i] for i in known]];denom=base.sum(0,keepdims=True)
        scores[known]=np.divide(base,denom,out=np.zeros_like(base),where=denom>0)
    if novel_evidence is not None:
        e=np.asarray(novel_evidence,np.float32)
        if e.shape!=(len(novel),*p.shape[1:]) or not np.isfinite(e).all() or (e<0).any() or (e>1).any():
            raise ValueError('Novel evidence does not match requested taxonomy')
        if novel:
            if known:
                strength=e.max(0);scores[known]*=1-strength
                total=e.sum(0)
                scores[novel]=np.divide(e*strength,total,out=np.zeros_like(e),where=total>0)
            else:scores[novel]=e
    pred=scores.argmax(0).astype(np.uint8);pred[scores.max(0)<=0]=255
    return scores,pred

class FrozenCooperation:
    def __init__(self,repo,calibrator_dir):
        self.repo=Path(repo);self.calibrator_dir=Path(calibrator_dir)
        runtime=self.repo/'experiments/paper20261109/runtime'
        for p in [self.repo/'src',runtime]:
            sys.path.insert(0,str(p))
        from predictor import load_calibrator
        self.text=load_calibrator(self.calibrator_dir/'text_only.npz')
        self.teacher=load_calibrator(self.calibrator_dir/'teacher_only.npz')
        assert self.text.feature_indices.tolist()==list(range(5,31))
        self.gate={'pixel_text_confidence_threshold':.9,'distinct_FoV_votes_required':2,
                   'baseline_ignore_confidence':.85,'teacher_high_conflict_protection':.8}
        self.hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.calibrator_dir.glob('*.npz')}

    def apply(self,meta,masks,probs,stored,bank_ids,output_query_ids,classes):
        from feature_core import build_context
        from fresh_core import proposals,soft_correct,arm_counts
        from piers_ljj.collaboration.reliable_cooperation import prepare,bounded_text_fusion
        from piers_ljj.collaboration.open_vocab import build_novel_evidence,OpenVocabConfig
        qmeta={q['query_id']:q for q in meta['queries']};bank=set(bank_ids)
        take=[i for i,c in enumerate(meta['candidates']) if c['query_id'] in bank]
        semmeta={**meta,'queries':[q for q in meta['queries'] if q['query_id'] in bank],
                 'candidates':[meta['candidates'][i] for i in take]}
        ctx=build_context(semmeta,masks[np.asarray(take,np.int64)],probs,stored,qmeta)
        text,teacher=self.text,self.teacher
        class Pair:
            def predict_np(self,X):
                p=text.predict_np(X);p[:,1]=teacher.predict_np(X)[:,1];return p
        arrays,gate_audit=proposals(ctx,Pair(),self.gate)
        lab,weight=arrays['label'],arrays['weight'];eligible=(lab!=255)&(weight>0)
        corrected,alpha,winner,error=soft_correct(ctx.probs,stored,lab,weight,eligible)
        corrected_probs=ctx.probs.copy();rr,cc=np.nonzero(eligible)
        if len(rr):
            values=corrected_probs[:,rr,cc].copy()*(1-alpha[None,:])
            values[lab[rr,cc],np.arange(len(rr))]+=alpha
            assert np.array_equal(values.argmax(0),winner)
            corrected_probs[:,rr,cc]=values
        # One-shot semantic->text uses the original teacher, never the corrected
        # teacher. Guided masks are outputs and are never fed back into I1.
        records=prepare(meta,masks,ctx.probs,qmeta)
        guided_all,text_audit=bounded_text_fusion(records,ctx.probs,list(qmeta),alpha=.5,gated=True)
        guided={qid:guided_all[qid] for qid in output_query_ids}
        channels=[source_channel(x) for x in classes];novel_indices=[i for i,c in enumerate(channels) if c is None]
        name_by_qid={output_query_ids[i]:classes[i] for i in novel_indices};novel_candidates=[]
        for c in records:
            if c['query_id'] not in name_by_qid:continue
            novel_candidates.append({'id':c['id'],'class_name':name_by_qid[c['query_id']],
                'mask':c['mask'],'valid':c['valid'],'view_id':c['fov_id'],
                'detector_score':c['detector_score'],'sam_quality':c['sam_quality']})
        evidence,ov_audit=build_novel_evidence(novel_candidates,[classes[i] for i in novel_indices],stored.shape,OpenVocabConfig())
        scores,pred=requested_probabilities(corrected_probs,classes,evidence)
        audit={'I1_semantic':arm_counts(stored,lab,weight,arrays['votes'],eligible,corrected,alpha,error,ctx.probs),
               'I1_gate':gate_audit,'I1_text':text_audit,'open_vocabulary':ov_audit,
               'calibrator_sha256':self.hashes,'calibrator_origin':'Cityscapes_source_rebuild_20261109',
               'source_GT_or_target_GT_reads':0,'no_circular_evidence':True,
               'output_class_names':list(classes),'novel_class_names':[classes[i] for i in novel_indices]}
        return pred,audit,guided
