import torch
from PIL import Image
from types import SimpleNamespace
from piers_ljj.text.feature_cache import PreFusionFeatureCache

class Unit(torch.nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind;self.calls=0
    def forward(self,x):
        self.calls+=1
        return ([x],[x]) if self.kind=='vision' else x+1

def model():
    root=SimpleNamespace(backbone=Unit('vision').eval(),text_backbone=Unit('text').eval(),
                         input_proj_vision=[Unit('proj').eval()])
    return SimpleNamespace(model=root)

def test_reuses_only_prefusion_subgraphs_and_copies_mutable_lists():
    m=model();cache=PreFusionFeatureCache(m);image=Image.new('RGB',(4,4));x=torch.ones(1)
    with torch.no_grad():
        cache.begin(image,'car.');first=m.model.backbone(x);first[1].append(x)
        m.model.text_backbone(x);m.model.input_proj_vision[0](x)
        cache.begin(image,'bus.');second=m.model.backbone(x)
        m.model.text_backbone(x);m.model.input_proj_vision[0](x)
        cache.begin(image,'car.');m.model.text_backbone(x)
    assert len(second[1])==1
    assert m.model.backbone.calls==1 and m.model.text_backbone.calls==2
    assert m.model.input_proj_vision[0].calls==1
    assert cache.stats()['multimodal_encoder_outputs_cached'] is False

def test_image_content_change_invalidates_visual_cache_even_same_object():
    m=model();cache=PreFusionFeatureCache(m);image=Image.new('RGB',(4,4));x=torch.ones(1)
    with torch.no_grad():
        cache.begin(image,'car.');m.model.backbone(x)
        image.putpixel((0,0),(1,2,3));cache.begin(image,'car.');m.model.backbone(x)
    assert m.model.backbone.calls==2

def test_text_cache_is_bounded_and_training_or_grad_disables_reuse():
    m=model();cache=PreFusionFeatureCache(m,max_text_entries=2);image=Image.new('RGB',(4,4));x=torch.ones(1)
    with torch.no_grad():
        for word in ['a','b','c']:cache.begin(image,word);m.model.text_backbone(x)
    assert cache.stats()['text_entries']==2
    cache.begin(image,'c');m.model.text_backbone(x)
    assert cache.stats()['text_entries']==0
    cache.uninstall();assert not cache.originals
