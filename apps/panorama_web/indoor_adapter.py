"""Use the verified indoor pipeline for web jobs without changing outdoor code.

Each SAM2 backend runs in its own process, as in the published web pipeline.
Only the uploaded image and the frozen inference protocol enter inference.
"""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES=['ceiling','chair','door','floor','sofa','table','wall','window']

def supported(classes):return bool(classes) and all(c in CLASSES for c in classes)

def run_adapted(folder,image,classes,jid,progress,check_cancel):
    repo=Path(os.environ.get('PIERS_INDOOR_REPO','/root/autodl-tmp/piers-main'))
    reference=repo/'runs/INDOOR-MATCHED-DINO-20261112'
    protocol=json.loads((reference/'protocol.json').read_text())
    checkpoint=Path(protocol['shared_checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=protocol['checkpoint_sha256']:
        raise RuntimeError('室内适配权重校验失败')
    root=folder/'indoor_matched';root.mkdir()
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'calibrators').mkdir()
    for name in ['text_only.npz','teacher_only.npz']:
        shutil.copy2(reference/'calibrators'/name,root/'calibrators'/name)
    canonical=image.resize((1024,512),Image.Resampling.BICUBIC)
    canonical.save(folder/'canonical.png')
    row={'image_id':jid,'path':str(folder/'canonical.png'),
         'input_sha256':hashlib.sha256((folder/'canonical.png').read_bytes()).hexdigest()}
    (root/'inference_heldout.json').write_text(json.dumps([row]))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    elapsed={};started=time.perf_counter()
    for action,stage,value,message in [
        ('semantic','semantic',.2,'运行室内适配语义模型与I2 I3'),
        ('text','text',.5,'运行Grounding DINO与SAM2文本分割'),
        ('fuse','semantic',.75,'进行室内可靠性协同融合')]:
        progress(stage=stage,progress=value,message=message);check_cancel()
        command=[sys.executable,'-u',str(repo/'scripts/piers/indoor_matched_run.py'),
                 action,'--out',str(root),'--split','heldout']
        if action=='semantic':command+=['--arm','full_core']
        begin=time.perf_counter()
        with (root/(action+'.log')).open('w') as log:
            process=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if process.returncode:
            raise RuntimeError('室内'+action+'推理失败：'+(root/(action+'.log')).read_text(errors='replace')[-1800:])
        elapsed[action]=time.perf_counter()-begin;check_cancel()
    text_folder=root/'text/heldout'/jid
    meta=json.loads((text_folder/'metadata.json').read_text())
    with np.load(text_folder/'candidates.npz') as z:
        masks=np.unpackbits(z['masks_packed'],axis=2,count=1024).astype(bool)
    with np.load(root/'full/heldout'/(jid+'_guided.npz')) as z:
        all_guided=np.unpackbits(z['masks_packed'],axis=2,count=1024).astype(bool)
    raw_prediction=np.load(root/'full/heldout'/(jid+'.npy'))
    indices=[CLASSES.index(c) for c in classes]
    prediction=np.full(raw_prediction.shape,255,np.uint8)
    for out_index,source_index in enumerate(indices):prediction[raw_prediction==source_index]=out_index
    queries={q['raw_text']:q for q in meta['queries']}
    requested_queries=[queries[c] for c in classes];wanted={q['query_id'] for q in requested_queries}
    take=[i for i,c in enumerate(meta['candidates']) if c['query_id'] in wanted]
    candidates=[{**meta['candidates'][i],'mask_index':k} for k,i in enumerate(take)]
    public_meta={**meta,'queries':requested_queries,'candidates':candidates}
    semantic=json.loads((root/'semantic/heldout/full_core'/(jid+'.json')).read_text())
    fusion=json.loads((root/'full/heldout'/(jid+'.json')).read_text())
    audit={'model':'full-indoor-matched-I1-I2-I3','indoor_adapted':True,
        'paper_release':'outdoor-v03-method+indoor-matched-20261112','I1':True,'I2':True,'I3':True,
        'CLIP_used':False,'checkpoint_sha256':semantic['checkpoint_sha256'],
        'lora_sha256':semantic['lora_sha256'],'semantic_forwards':20,'GT_reads':0,'weights_updated':False,
        'canonical_size':[1024,512],'output_size':list(image.size),
        'input_sha256':hashlib.sha256((folder/'input.png').read_bytes()).hexdigest(),
        'semantic':semantic,'cooperation':fusion,'text':{k:meta[k] for k in ['DINO_calls','SAM_image_encodes','elapsed_s']},
        'timing':{'model_loading_seconds':None,'text_seconds':elapsed['text'],
                  'semantic_seconds':elapsed['semantic']+elapsed['fuse'],
                  'server_processing_seconds':time.perf_counter()-started,
                  'warm_processing_seconds':None},
        'timing_scope':'Interactive subprocess pipeline with cold model loading and I/O; not isolated paper latency',
        'inference_identity':'Same Stage1 checkpoint, LoRA, full semantic engine, text queries and calibrators as verified373-image experiment',
        'source_fingerprints':{'indoor_adapter.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    return public_meta,masks[np.asarray(take,np.int64)],all_guided[np.asarray(indices,np.int64)],prediction,audit
