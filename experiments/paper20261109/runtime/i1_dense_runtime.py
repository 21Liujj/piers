"""Uncompressed control: same frozen engine, sparse retrieval disabled."""
from i1_runtime import verify_i2,sha,I2 as SEALED
I2={**SEALED,'mode':'dense','score':'none','budget':32768,'keep_ratio':1.}

def install_fixed_memory(model,stage='eval'):
    mb=model.memory_bank
    mb.configure_sparse_retrieval(enabled=False,mode='dense',keep_ratio=1.,min_tokens=64,cache_enabled=False,enable_in_train=False,enable_in_pseudo_label=False)
    mb.set_runtime_context(stage=stage,iter_idx=0);mb.sparse_retriever.diagnostic_overlap=False
    audit={'mode':'dense','calls':0,'fallbacks':0,'max_tokens':0};original=model.forward
    def forward(*args,**kwargs):
        result=original(*args,**kwargs);length=int(getattr(mb,'last_effective_memory_length',0))
        audit['max_tokens']=max(audit['max_tokens'],length);assert length<=32768
        return result
    model.forward=forward
    def no_sparse(*args,**kwargs):raise AssertionError('Dense control attempted sparse retrieval')
    mb.sparse_retriever.retrieve=no_sparse
    return audit
