"""Original fixed sparse retrieval, with path resolution and explicit ablations.

Default parameters and tensor operations match the recovered Nov9 runtime.
Overrides are recorded by the experiment runner and never rewrite defaults.
"""
from pathlib import Path
import hashlib
from paper_paths import REPO as ROOT,memory_config
I2=memory_config()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_i2():
    for p,digest in I2['runtime_source_sha256'].items():
        assert sha(p)==digest,('Inference core changed',p)
    assert sha(I2['checkpoint_path'])==I2['checkpoint_sha256']
    return I2

def install_fixed_memory(model,stage='eval',overrides=None):
    cfg={**I2,**(overrides or {})}
    mb=model.memory_bank;histories={'source':{},'target':{}};orig=model.forward;reset=mb._init_output_dict_source
    def wrapped(frame_idx,x,mem_type='source',historical_pseudo_labels=None,**kw):
        hist=histories.setdefault(mem_type,{})
        merged=dict(hist)
        if historical_pseudo_labels:merged.update(historical_pseudo_labels)
        out=orig(frame_idx,x,mem_type=mem_type,historical_pseudo_labels=merged or None,**kw)
        hist[frame_idx]=out[0].argmax(dim=1).detach()
        audit['max_tokens']=max(audit['max_tokens'],int(getattr(mb,'last_effective_memory_length',0)))
        return out
    def clear():histories['source'].clear();return reset()
    model.forward=wrapped;mb._init_output_dict_source=clear
    if hasattr(mb,'_init_output_dict_target'):
        reset_target=mb._init_output_dict_target
        def clear_target():histories['target'].clear();return reset_target()
        mb._init_output_dict_target=clear_target
    mb.sparse_retriever.diagnostic_overlap=False
    mb.configure_sparse_retrieval(enabled=True,mode=cfg['mode'],keep_ratio=cfg['keep_ratio'],min_tokens=64,
        cache_enabled=False,fallback='feature_norm',fallback_full_when_uncertain=True,enable_in_train=True,
        enable_in_pseudo_label=True,boundary_reserve_ratio=.125,rare_class_ids=cfg['rare_class_ids'],
        rare_reserve_ratio=cfg['rare_reserve_ratio'],boundary_score_source=cfg['score'])
    mb.set_runtime_context(stage=stage,iter_idx=0)
    audit={'calls':0,'fallbacks':0,'max_tokens':0,'configuration':{k:cfg[k] for k in ['mode','score','keep_ratio','budget','rare_reserve_ratio']}}
    retrieve=mb.sparse_retriever.retrieve
    def checked(*a,**kw):
        result=retrieve(*a,**kw);stats=result[3];audit['calls']+=1;audit['fallbacks']+=int(stats.fallback_used)
        assert not stats.fallback_used and not stats.mask_shape_mismatch,str(stats)
        assert stats.selection_mode==cfg['mode'],stats.selection_mode
        if cfg['score']=='uncertainty':assert kw.get('uncertainty_map') is not None,'Missing uncertainty score'
        audit['max_tokens']=max(audit['max_tokens'],int(getattr(mb,'last_effective_memory_length',0)))
        return result
    mb.sparse_retriever.retrieve=checked
    return audit
