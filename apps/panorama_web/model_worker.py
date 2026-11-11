"""Resident model processes; separate SAM2 namespaces exactly as paper runtime.

Extends the published inference output with guided query masks. It does not
modify frozen model weights, budgets, calibrators or category competition.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import write_json, read_json

def generic_text(row, folder, cfg, backend):
    import numpy as np
    from PIL import Image
    from piers_ljj.text.grounded_adapter import run_fov_text_branch
    from piers_ljj.geometry.erp import plan_fovs,crop_erp_wrap
    dino,sam=backend
    class Cache:
        def __init__(self):self.enc=None;self.key=None;self.encodes=0
        def set_image(self,pil):
            key=hashlib.sha256(np.asarray(pil).tobytes()).digest()
            if key!=self.key:self.enc=sam.set_image(pil);self.key=key;self.encodes+=1
            return self.enc
        def predict_boxes(self,enc,boxes):return sam.predict_boxes(enc,boxes)
    cache=Cache();queries=[{'image_id':row['image_id'],'query_id':row['image_id']+':generic:'+str(i),
                           'raw_text':name,'ontology':{'map13':None,'relation':'unmapped'}}
                          for i,name in enumerate(row['classes'])]
    with Image.open(row['path']) as src:rgb=np.asarray(src.convert('RGB'))
    h,w=rgb.shape[:2];fovs=plan_fovs(w,h,win_w=1024,win_h=h,stride_x=512,y_start=0)
    masks=[];candidates=[];calls=0
    for view in fovs:
        crop,_=crop_erp_wrap(rgb,view['x_start'],0,1024,h);pil=Image.fromarray(crop)
        for q in queries:
            out=run_fov_text_branch(pil,q['raw_text'],cfg,(dino,cache));calls+=1
            for k,c in enumerate(out['candidates']):
                m=np.zeros((h,w),bool);cols=(view['x_start']+np.arange(1024))%w
                m[:,cols]=np.asarray(c.mask,bool)
                candidates.append({'id':q['query_id']+':'+view['view_id']+':'+str(k),
                    'query_id':q['query_id'],'fov_id':view['view_id'],'detector_score':c.detector_score,
                    'sam_quality':c.sam_quality,'phrase':c.phrase,'local_box':list(c.box_xyxy),'mask_index':len(masks)})
                masks.append(m)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    packed=np.packbits(np.stack(masks),axis=2) if masks else np.zeros((0,h,(w+7)//8),np.uint8)
    np.savez_compressed(folder/'candidates.npz',masks_packed=packed,shape=np.array([h,w]))
    write_json(folder/'metadata.json',{'status':'complete','image_id':row['image_id'],'path':row['path'],
               'shape':[h,w],'fovs':fovs,'queries':queries,'candidates':candidates,'no_GT_read':True})
    return {'detector_calls':calls,'sam_image_encodes':cache.encodes,'candidates':len(candidates),
            'cross_image_prediction_cache':False,'scene':'indoor_generic_zero_shot'}

def serve(a):
    repo=Path(a.repo)
    sys.path[:0]=[str(repo/'src'),str(repo/'scripts/piers'),str(repo/'experiments/paper20261109/runtime')]
    from benchmark_complete_model_v3 import guard_labels,text_image
    guard_labels();begin=time.perf_counter()
    import torch
    if a.role=='text':
        sys.path.insert(0,str(repo/'third_party/grounded_sam2'))
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
        from piers_ljj.workers.entry import _build_hf_backend
        from piers_ljj.text.grounded_adapter import GroundedConfig
        from paper_paths import WEIGHTS
        cfg=GroundedConfig(grounding_model_id=str(WEIGHTS/'grounding-dino-tiny-hf'),
                          sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',
                          sam2_ckpt=str(WEIGHTS/'sam2.1_hiera_small.pt'),device='cuda',dtype='bf16',
                          cache_prefusion_features=False)
        backend=_build_hf_backend(cfg,'cuda')
    else:
        from full_model_semantic import SemanticEngine
        from full_model_fusion_v3 import FrozenCooperation,make_queries
        engine=SemanticEngine(repo,'i2',2)
        cooperation=FrozenCooperation(repo,repo/'assets/paper20261109/calibrators')
    torch.cuda.synchronize();control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    write_json(control/'READY.json',{'pid':os.getpid(),'role':a.role,'loading_seconds':time.perf_counter()-begin,
               'resident_allocated_mib':torch.cuda.memory_allocated()/2**20})
    handled=set()
    while True:
        for path in sorted(control.glob('request_*.json')):
            if path.name in handled:continue
            reply=control/('response_'+path.stem[8:]+'.json')
            try:
                request=read_json(path);op=request['operation']
                if op=='stop':write_json(reply,{'status':'stopped'});return
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                if a.role=='text':
                    if request['scene']=='street':result=text_image(request['row'],request['folder'],cfg,backend)
                    else:result=generic_text(request['row'],request['folder'],cfg,backend)
                else:
                    import numpy as np
                    from run_zero_shot_predictions import load_text
                    row=request['row'];meta,masks=load_text(Path(request['text_parent']),row,10)
                    probs,label,audit=engine.predict(row['path'])
                    _,bank,output,_=make_queries(row['image_id'],row['classes'])
                    pred,collab,guided=cooperation.apply(meta,masks,probs,label,bank,output,row['classes'])
                    folder=Path(request['folder']);folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/'semantic.npz',prediction=pred,
                        guided=np.stack([guided[q] for q in output]).astype(bool))
                    result={'semantic':audit,'cooperation':collab,'weights_updated':False,'GT_reads':0,
                            'source_checkpoint_sha256':engine.checkpoint_sha256}
                    del probs,label,masks,pred,guided
                torch.cuda.synchronize()
                result.update(status='complete',service_seconds=time.perf_counter()-started,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                write_json(reply,result);gc.collect()
            except BaseException:
                write_json(reply,{'status':'failed','traceback':traceback.format_exc()});raise
            handled.add(path.name)
        time.sleep(.025)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--control',required=True)
    p.add_argument('--role',choices=['text','semantic'],required=True);args=p.parse_args()
    try:serve(args)
    except BaseException:
        write_json(Path(args.control)/'FAILED.json',{'traceback':traceback.format_exc()});raise
