"""Isolated resident-model end-to-end benchmark, with genuine text inference.

Separate persistent vendor processes avoid the two incompatible SAM2 package
namespaces. Requests are SERIAL across both methods; no candidate or semantic
prediction cache substitutes for computation. The full method includes the
real DINO, SAM2, I1/I2/I3, category competition, IPC and prediction PNG writing.
Initial weight loading is reported separately and excluded from warm latency.
"""
from __future__ import annotations
import argparse, ctypes, gc, hashlib, json, os, subprocess, sys, threading, time, traceback
from pathlib import Path


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_labels():
    import builtins,io
    forbidden=('/gtFine/','/evaluation_gt/','/labels/','/labels_raw/','/labels_trainid/','/SynPASS/semantic/')
    def wrap(original):
        def checked(file,*args,**kwargs):
            if not isinstance(file,int) and any(x in str(Path(file).resolve()) for x in forbidden):
                raise RuntimeError('GT access forbidden in inference service')
            return original(file,*args,**kwargs)
        return checked
    builtins.open,io.open=wrap(builtins.open),wrap(io.open)


class DeviceMemorySampler:
    """Sample actual device-used memory every20ms on an otherwise idle GPU."""
    class Memory(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML init failed')
        self.handle=ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0),ctypes.byref(self.handle))!=0:
            raise RuntimeError('NVML GPU handle failed')
        self.peak=0;self.samples=0;self.stop=threading.Event()
    def sample(self):
        mem=self.Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(mem))!=0:
            raise RuntimeError('NVML memory query failed')
        self.peak=max(self.peak,int(mem.used));self.samples+=1
    def start(self):
        self.sample()
        def run():
            while not self.stop.wait(.02):self.sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set();self.thread.join();self.sample()
        self.lib.nvmlShutdown()
        return {'peak_device_used_mib_sampled':self.peak/2**20,'memory_samples':self.samples,'sampling_interval_s':.02}


def text_image(row, folder, cfg, backend):
    import numpy as np
    from PIL import Image
    from full_model_queries_v3 import make_queries
    from piers_ljj.text.grounded_adapter import run_fov_text_branch
    from piers_ljj.geometry.erp import plan_fovs,crop_erp_wrap
    class WithinImageCache:
        def __init__(self,base):self.base=base;self.key=None;self.enc=None;self.encodes=0;self.hits=0
        def set_image(self,pil):
            key=hashlib.sha256(np.asarray(pil.convert('RGB')).tobytes()).digest()
            if key!=self.key:self.enc=self.base.set_image(pil);self.key=key;self.encodes+=1
            else:self.hits+=1
            return self.enc
        def predict_boxes(self,enc,boxes):return self.base.predict_boxes(enc,boxes)
    dino,sam=backend;cache=WithinImageCache(sam)
    queries,_,_,_=make_queries(row['image_id'],row['classes'])
    with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
    height,width=rgb.shape[:2];fovs=plan_fovs(width,height,win_w=1024,win_h=height,stride_x=512,y_start=0)
    masks,candidates=[],[];calls=0
    for view in fovs:
        crop,_=crop_erp_wrap(rgb,view['x_start'],0,view['win_w'],height);pil=Image.fromarray(crop)
        for query in queries:
            result=run_fov_text_branch(pil,query['raw_text'],cfg,(dino,cache));calls+=1
            for k,cand in enumerate(result['candidates']):
                local=np.asarray(cand.mask,bool)
                if local.shape!=(height,1024):raise ValueError('Text-mask coordinates changed')
                mask=np.zeros((height,width),bool);cols=(view['x_start']+np.arange(1024))%width;mask[:,cols]=local
                candidates.append({'id':query['query_id']+':'+view['view_id']+':'+str(k),
                    'query_id':query['query_id'],'fov_id':view['view_id'],
                    'detector_score':cand.detector_score,'sam_quality':cand.sam_quality,'phrase':cand.phrase,
                    'local_box':list(cand.box_xyxy),'ontology':query.get('ontology'),'mask_index':len(masks)})
                masks.append(mask)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    packed=np.packbits(np.stack(masks),axis=2) if masks else np.zeros((0,height,(width+7)//8),np.uint8)
    np.savez_compressed(folder/'candidates.npz',masks_packed=packed,shape=np.array([height,width]))
    meta={'status':'complete','image_id':row['image_id'],'path':row['path'],'shape':[height,width],
          'fovs':fovs,'queries':queries,'candidates':candidates,'no_GT_read':True}
    write_json(folder/'metadata.json',meta)
    return {'detector_calls':calls,'sam_image_encodes':cache.encodes,'within_image_SAM_cache_hits':cache.hits,
            'candidates':len(candidates),'cross_image_candidate_cache_used':False,
            'prefusion_feature_cache':dino.cache_stats() if hasattr(dino,'cache_stats') else None}


def serve(a):
    guard_labels();repo=Path(a.repo);sys.path.insert(0,str(repo/'src'))
    control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if a.role=='text':
        sys.path.insert(0,str(repo/'third_party/grounded_sam2'))
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
        from piers_ljj.workers.entry import _build_hf_backend
        from piers_ljj.text.grounded_adapter import GroundedConfig
        sys.path.insert(0,str(repo/'experiments/paper20261109/runtime'))
        from paper_paths import WEIGHTS
        cfg=GroundedConfig(grounding_model_id=str(WEIGHTS/'grounding-dino-tiny-hf'),
            sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',sam2_ckpt=str(WEIGHTS/'sam2.1_hiera_small.pt'),
            device='cuda',dtype='bf16',cache_prefusion_features=os.environ.get('PIERS_TEXT_FEATURE_CACHE','0')=='1')
        backend=_build_hf_backend(cfg,'cuda')
    else:
        from full_model_semantic import SemanticEngine
        from full_model_fusion_v3 import FrozenCooperation,make_queries,requested_probabilities
        engine=SemanticEngine(repo,'dense' if a.role=='original' else 'i2',0 if a.role=='original' else 2)
        torch=engine.torch
        cooperation=FrozenCooperation(repo,a.calibrators) if a.role=='full' else None
        trace=[]
        def hook(module,args,kwargs):
            effective=int(kwargs['memory'].shape[0]);mb=engine.model.memory_bank
            before=int(mb.last_sparse_tokens_before) if a.role=='full' else effective
            trace.append({'before_sparse_selection':before,'attention_input_tokens':effective})
        engine.model.memory_bank.sam_model.memory_attention.register_forward_pre_hook(hook,with_kwargs=True)
    torch.cuda.synchronize()
    write_json(control/'READY.json',{'pid':os.getpid(),'role':a.role,'weight_loading_seconds':time.monotonic()-started,
        'resident_allocated_mib':torch.cuda.memory_allocated()/2**20})
    handled=set()
    while True:
        for request_path in sorted(control.glob('request_*.json')):
            if request_path.name in handled:continue
            request=json.loads(request_path.read_text());reply=control/('response_'+request_path.stem[8:]+'.json')
            try:
                operation=request['operation']
                if operation=='stop':write_json(reply,{'status':'stopped'});return
                if operation=='reset':
                    gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    result={'resident_allocated_mib':torch.cuda.memory_allocated()/2**20}
                else:
                    from run_zero_shot_predictions import validate_manifest,load_text
                    row=request['row'];validate_manifest({'images':[row]})
                    if sha(row['path'])!=row['input_sha256']:raise ValueError('Input SHA changed')
                    begin=time.perf_counter()
                    if a.role=='text':result=text_image(row,request['folder'],cfg,backend)
                    else:
                        import numpy as np,cv2
                        from PIL import Image
                        trace.clear()
                        if cooperation:meta,masks=load_text(Path(request['text_parent']),row,5)
                        probs,label,audit=engine.predict(row['path'])
                        if cooperation:
                            _,bank,queries,_=make_queries(row['image_id'],row['classes'])
                            pred,collab,_=cooperation.apply(meta,masks,probs,label,bank,queries,row['classes'])
                            verified_common13=None
                            if request.get('verify_common13',False):
                                from full_model_fusion import FrozenCooperation as V1
                                legacy,_,_=V1.apply(cooperation,meta,masks,probs,label,bank,queries,row['classes'])
                                if not np.array_equal(pred,legacy):raise AssertionError('Common13 path changed')
                                verified_common13=True
                        else:_,pred=requested_probabilities(probs,row['classes']);collab=None
                        width,height=row['original_size'];pred=cv2.resize(pred,(width,height),interpolation=cv2.INTER_NEAREST)
                        destination=Path(request['prediction']);destination.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(pred).save(destination)
                        result={'semantic':audit,'token_trace':list(trace),'prediction_sha256':sha(destination),
                                'common13_v1_v2_equal':verified_common13 if cooperation else None,
                                'classes':row['classes'],'target_GT_reads':0,'weights_updated':False}
                        del probs,label,pred
                        if cooperation:del masks,meta
                    torch.cuda.synchronize();result['service_seconds']=time.perf_counter()-begin
                result.update(status='complete',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                write_json(reply,result)
            except BaseException:
                write_json(reply,{'status':'failed','traceback':traceback.format_exc()});raise
            handled.add(request_path.name)
        time.sleep(.005)


class Service:
    def __init__(self,a,role):
        self.role=role;self.control=Path(a.out)/'control'/role
        if self.control.exists() and any(self.control.iterdir()):raise RuntimeError('Refuse stale service state: '+str(self.control))
        self.control.mkdir(parents=True,exist_ok=True);self.count=0
        log=(Path(a.out)/(role+'.log')).open('w')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--serve','--repo',a.repo,'--out',a.out,
                 '--calibrators',a.calibrators,'--role',role,'--control',str(self.control)]
        self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,cwd=Path(a.repo)/'third_party/grounded_sam2' if role=='text' else Path(a.repo))
        try:self.ready=self.wait(self.control/'READY.json',240)
        except BaseException:
            if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=15)
            raise
    def wait(self,path,timeout):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if self.process.poll() is not None:raise RuntimeError(self.role+' service exited; see its log')
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(.01)
        result=json.loads(path.read_text())
        if result.get('status')=='failed':raise RuntimeError(result['traceback'])
        return result
    def call(self,operation,**values):
        key=str(self.count).zfill(5);self.count+=1
        write_json(self.control/('request_'+key+'.json'),{'operation':operation,**values})
        return self.wait(self.control/('response_'+key+'.json'),600)
    def close(self):
        if self.process.poll() is None:
            try:self.call('stop');self.process.wait(timeout=15)
            except BaseException:self.process.terminate();self.process.wait(timeout=15)


def gpu_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()}


def parent(a):
    import numpy as np
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if gpu_pids():raise RuntimeError('GPU must be idle before isolated efficiency benchmark')
    plan=json.loads(Path(a.plan).read_text());write_json(out/'PROTOCOL.json',plan)
    measured=[]
    for method in ('original','full'):
        services=[]
        try:
            semantic=Service(a,method);services.append(semantic)
            text=Service(a,'text') if method=='full' else None
            if text:services.append(text)
            owned={s.process.pid for s in services}
            write_json(out/(method+'_INITIALIZATION.json'),{s.role:s.ready for s in services})
            trials=[(scenario['name'],scenario['images'][0],-1,True) for scenario in plan['scenarios']]
            trials += [(scenario['name'],row,repeat,False) for repeat in range(plan['repeats'])
                       for scenario in plan['scenarios'] for row in scenario['images']]
            for number,(scenario,row,repeat,warmup) in enumerate(trials):
                unexpected=gpu_pids()-owned
                if unexpected:raise RuntimeError('Unrelated GPU processes invalidate isolation: '+str(unexpected))
                for service in services:service.call('reset')
                key=method+'_'+str(number).zfill(3);folder=out/'trials'/key;folder.mkdir(parents=True,exist_ok=True)
                sampler=DeviceMemorySampler();sampler.start();start=time.perf_counter()
                text_result=text.call('predict',row=row,folder=str(folder/'text'/row['image_id'])) if text else None
                semantic_result=semantic.call('predict',row=row,text_parent=str(folder/'text'),prediction=str(folder/'prediction.png'),
                                              verify_common13=warmup and method=='full' and len(row['classes'])==13)
                elapsed=time.perf_counter()-start;memory=sampler.finish()
                result={'method':method,'scenario':scenario,'image_id':row['image_id'],'repeat':repeat,'warmup':warmup,
                        'input_sha256':row['input_sha256'],'end_to_end_seconds':elapsed,**memory,
                        'semantic_service':semantic_result,'text_service':text_result,
                        'isolated_from_other_GPU_jobs':True,'cached_candidate_reads':False}
                write_json(folder/'result.json',result)
                if not warmup:measured.append(result)
                write_json(out/'PROGRESS.json',{'method':method,'trial':number+1,'total_trials_per_method':len(trials),
                    'measured_trials_complete':len(measured),'last_seconds':elapsed,'last_scenario':scenario})
                print('BENCHMARK',method,number+1,len(trials),scenario,round(elapsed,3),'peakMiB',round(memory['peak_device_used_mib_sampled']),flush=True)
            write_json(out/(method+'_DONE.json'),{'status':'complete'})
        finally:
            for service in reversed(services):service.close()
    rows=[]
    for scenario in plan['scenarios']:
        for method in ('original','full'):
            samples=[r for r in measured if r['scenario']==scenario['name'] and r['method']==method]
            seconds=np.asarray([r['end_to_end_seconds'] for r in samples])
            rows.append({'scenario':scenario['name'],'method':method,'measurements':len(samples),
                'unique_images':len({r['image_id'] for r in samples}),'mean_seconds':float(seconds.mean()),
                'median_seconds':float(np.median(seconds)),'p95_seconds_descriptive_small_sample':float(np.quantile(seconds,.95)),
                'serial_images_per_second':float(1/seconds.mean()),
                'peak_device_used_mib_sampled':max(r['peak_device_used_mib_sampled'] for r in samples),
                'semantic_peak_allocated_mib':max(r['semantic_service']['peak_allocated_mib'] for r in samples),
                'semantic_peak_reserved_mib':max(r['semantic_service']['peak_reserved_mib'] for r in samples),
                'text_peak_allocated_mib':max((r['text_service']['peak_allocated_mib'] for r in samples if r['text_service']),default=0),
                'text_peak_reserved_mib':max((r['text_service']['peak_reserved_mib'] for r in samples if r['text_service']),default=0)})
    report={'status':'complete','rows':rows,'measured_trials':measured,
            'latency_scope':plan['latency_scope'],'limitations':plan['limitations']}
    write_json(out/'RESULTS.json',report);write_json(out/'COMPLETE.json',{'status':'complete','results':str(out/'RESULTS.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--calibrators',required=True);parser.add_argument('--plan');parser.add_argument('--serve',action='store_true')
    parser.add_argument('--role',choices=['original','full','text']);parser.add_argument('--control')
    args=parser.parse_args()
    try:serve(args) if args.serve else parent(args)
    except BaseException:
        write_json(Path(args.control if args.serve else args.out)/'FAILED.json',{'traceback':traceback.format_exc()});raise
