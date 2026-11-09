"""Bounded eval-only caches BEFORE GroundingDINO vision/text fusion.

The multimodal encoder/decoder are recomputed for every original query.
This does not concatenate prompts, drop classes, or reuse fused encoder states.
"""
from collections import OrderedDict
import hashlib


def copy_containers(value):
    """HF appends positional levels to a returned list; never share that list."""
    if isinstance(value,list):return [copy_containers(x) for x in value]
    if isinstance(value,tuple):return tuple(copy_containers(x) for x in value)
    if isinstance(value,dict):
        items={k:copy_containers(v) for k,v in value.items()}
        return type(value)(**items) if hasattr(value,'to_tuple') else items
    return value


class PreFusionFeatureCache:
    def __init__(self,model,max_text_entries=128):
        self.model=model;self.max_text_entries=int(max_text_entries)
        if self.max_text_entries<1:raise ValueError('Text cache must be bounded and nonempty')
        self.image_key=None;self.text_key=None;self.visual={};self.text=OrderedDict();self.originals=[]
        self.counts={'vision_computes':0,'vision_hits':0,'text_computes':0,'text_hits':0,
                     'projection_computes':0,'projection_hits':0,'requests':0,'visual_invalidations':0}
        root=getattr(model,'model',model)
        for name in ['backbone','text_backbone','input_proj_vision']:
            if not hasattr(root,name):raise ValueError('Unsupported GroundingDINO structure: '+name)
        self._wrap(root.backbone,'vision')
        self._wrap(root.text_backbone,'text')
        for index,module in enumerate(root.input_proj_vision):self._wrap(module,'projection',index)

    def begin(self,pil_image,text):
        rgb=pil_image if pil_image.mode=='RGB' else pil_image.convert('RGB')
        key=(rgb.size,hashlib.sha256(rgb.tobytes()).digest())
        if key!=self.image_key:
            self.visual.clear();self.image_key=key;self.counts['visual_invalidations']+=1
        self.text_key=str(text);self.counts['requests']+=1

    def invalidate(self):
        self.visual.clear();self.text.clear();self.image_key=None;self.text_key=None

    def _wrap(self,module,kind,index=0):
        original=module.forward;self.originals.append((module,original))
        def forward(*args,**kwargs):
            import torch
            if module.training or torch.is_grad_enabled() or self.image_key is None:
                self.invalidate()
                return original(*args,**kwargs)
            # A cache instance belongs to fixed eval weights/device/dtype. The
            # caller must invalidate after changing any of those model states.
            if kind=='text':
                key=self.text_key
                if key in self.text:
                    self.counts['text_hits']+=1;self.text.move_to_end(key)
                    return copy_containers(self.text[key])
                result=original(*args,**kwargs);self.counts['text_computes']+=1
                self.text[key]=copy_containers(result)
                while len(self.text)>self.max_text_entries:self.text.popitem(last=False)
                return copy_containers(result)
            key=(kind,index)
            if key in self.visual:
                self.counts[kind+'_hits']+=1
                return copy_containers(self.visual[key])
            result=original(*args,**kwargs);self.counts[kind+'_computes']+=1
            self.visual[key]=copy_containers(result)
            return copy_containers(result)
        module.forward=forward

    def stats(self):
        return {**self.counts,'text_entries':len(self.text),'visual_entries':len(self.visual),
                'multimodal_encoder_outputs_cached':False,'query_words_or_thresholds_changed':False}

    def uninstall(self):
        for module,original in self.originals:module.forward=original
        self.originals.clear();self.invalidate()
