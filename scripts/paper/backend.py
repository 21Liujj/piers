"""Separate semantic/text processes, parameterized only by frozen variant."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/piers'),str(REPO/'experiments/paper20261109/runtime')]
from benchmark_complete_model_v3 import Service,serve as frozen_serve,write_json

def configuration(name):
    variants=json.loads((REPO/'configs/piers/paper20261109/matrix.json').read_text())['variants']
    if name not in variants:raise ValueError('Unknown variant: '+name)
    return variants[name]

def run_service(a):
    cfg=configuration(a.variant)
    os.environ['PIERS_TEXT_FEATURE_CACHE']='1' if cfg.get('text_feature_cache',False) else '0'
    if a.role!='text':
        import full_model_semantic
        base=full_model_semantic.SemanticEngine
        class VariantEngine(base):
            def __init__(self,repo,*args,**kwargs):
                super().__init__(repo,cfg['memory'],cfg['budget'],cfg.get('memory_config'),cfg.get('strategy','adaptive'))
        full_model_semantic.SemanticEngine=VariantEngine
    frozen_serve(a)

class PaperService(Service):
    def __init__(self,out,role,variant,calibrators=None):
        self.role=role;self.control=Path(out)/'control'/role;self.count=0
        if self.control.exists() and any(self.control.iterdir()):raise RuntimeError('Stale service control: '+str(self.control))
        self.control.mkdir(parents=True,exist_ok=True)
        self.log=(Path(out)/(role+'.log')).open('w')
        command=[sys.executable,'-u',__file__,'--serve','--repo',str(REPO),'--out',str(out),'--role',role,
                 '--variant',variant,'--calibrators',str(calibrators or REPO/'assets/paper20261109/calibrators'),'--control',str(self.control)]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        self.process=subprocess.Popen(command,cwd=REPO/'third_party/grounded_sam2' if role=='text' else REPO,
            env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:self.ready=self.wait(self.control/'READY.json',240)
        except BaseException:
            if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=15)
            raise

def worker(a):
    from benchmark_complete_model_v3 import guard_labels
    from run_zero_shot_predictions import validate_manifest,sha
    guard_labels();cfg=configuration(a.variant);manifest=json.loads(Path(a.manifest).read_text());validate_manifest(manifest)
    rows=manifest['images'][a.shard::a.shards]
    out=Path(a.out);role='text' if a.text_worker else ('full' if cfg['i1'] else 'original')
    service_name=('text' if a.text_worker else a.variant)+'_s'+str(a.shard)
    service=PaperService(out/'services'/service_name,role,a.variant)
    begin=time.monotonic()
    try:
        for n,row in enumerate(rows,1):
            if sha(row['path'])!=row['input_sha256']:raise ValueError('Input changed')
            if a.text_worker:
                result=service.call('predict',row=row,folder=str(out/'text'/row['image_id']))
            else:
                if cfg['i1']:
                    deadline=time.monotonic()+3600
                    while not (Path(a.text)/row['image_id']/'metadata.json').exists():
                        if list(out.glob('FAILED*')):raise RuntimeError('Other worker failed')
                        if time.monotonic()>deadline:raise TimeoutError('Text evidence not ready')
                        time.sleep(.2)
                folder=out/a.variant/row['image_id']
                result=service.call('predict',row=row,text_parent=a.text,prediction=str(folder/'prediction.png'),verify_common13=False)
                assert result['target_GT_reads']==0 and not result['weights_updated']
                assert result['semantic']['forward_calls']==18+cfg['budget']
                if cfg['memory']=='i2':assert result['semantic']['memory_audit']['fallbacks']==0
                write_json(folder/'audit.json',{'status':'complete','variant':a.variant,'classes':row['classes'],
                    'input_sha256':row['input_sha256'],'source_sha256':row['source_sha256'],
                    'prediction_sha256':result['prediction_sha256'],'service':result,'GT_reads':0,'new_training':False})
            write_json(out/('progress_'+('text' if a.text_worker else a.variant)+'_s'+str(a.shard)+'.json'),
                       {'done':n,'total':len(rows),'last_image':row['image_id'],'elapsed_s':time.monotonic()-begin})
            print('PAPER_PREDICTION',a.variant,role,a.shard,n,len(rows),row['image_id'],flush=True)
    finally:service.close()
    write_json(out/('DONE_'+('text' if a.text_worker else a.variant)+'_s'+str(a.shard)+'.json'),{'status':'complete','images':len(rows)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');p.add_argument('--repo',default=str(REPO))
    p.add_argument('--out',required=True);p.add_argument('--role',choices=['text','original','full']);p.add_argument('--control')
    p.add_argument('--calibrators');p.add_argument('--manifest');p.add_argument('--text',default='')
    p.add_argument('--variant',default='full');p.add_argument('--text-worker',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    try:run_service(a) if a.serve else worker(a)
    except BaseException:
        write_json(Path(a.out)/('FAILED_'+a.variant+'_'+str(a.shard)+'.json'),{'traceback':traceback.format_exc()});raise
