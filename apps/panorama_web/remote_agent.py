"""Private loopback GPU agent. Accessed through SSH, never a public GPU port."""
from __future__ import annotations
import argparse, atexit, base64, contextlib, fcntl, hashlib, io, json, os, queue
import signal, subprocess, sys, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from common import (VERSION,MAX_UPLOAD,valid_job_id,parse_classes,decode_image,canonical_image,
                    read_json,write_json,query_mapping,build_artifacts)
from gpu_guard import observed_gpu_pids

def gpu_pids():
    return observed_gpu_pids()

class ModelService:
    def __init__(self,repo,state,role,weights):
        self.control=state/('service_'+role+'_'+uuid.uuid4().hex[:10]);self.control.mkdir(parents=True)
        self.role=role;self.count=0;self.log=(self.control/'worker.log').open('w')
        env=dict(os.environ,PIERS_WEIGHTS_ROOT=str(weights),PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-u',str(Path(__file__).with_name('model_worker.py')),
                 '--repo',str(repo),'--control',str(self.control),'--role',role]
        self.process=subprocess.Popen(command,cwd=repo/'third_party/grounded_sam2' if role=='text' else repo,
            env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:self.ready=self.wait(self.control/'READY.json',300)
        except BaseException:self.close(force=True);raise
    def wait(self,path,timeout):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if self.process.poll() is not None:
                detail=(self.control/'worker.log').read_text(errors='replace')[-1800:]
                raise RuntimeError(self.role+'模型进程退出：'+detail)
            if time.monotonic()>deadline:raise TimeoutError(self.role+'模型处理超时')
            time.sleep(.05)
        result=read_json(path)
        if result.get('status')=='failed':raise RuntimeError(result['traceback'])
        return result
    def call(self,operation,**values):
        key=str(self.count).zfill(6);self.count+=1
        write_json(self.control/('request_'+key+'.json'),{'operation':operation,**values})
        return self.wait(self.control/('response_'+key+'.json'),1200)
    def close(self,force=False):
        if self.process.poll() is None:
            if not force:
                try:self.call('stop');self.process.wait(timeout=8)
                except BaseException:force=True
            if force:
                self.process.terminate()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.log.close()

class Agent:
    def __init__(self,args):
        self.args=args;self.repo=Path(args.repo).resolve();self.root=Path(args.state).resolve()
        self.jobs=self.root/'jobs';self.jobs.mkdir(parents=True,exist_ok=True)
        self.services={};self.lock_file=None;self.q=queue.Queue();self.cancelled=set();self.lock=threading.RLock()
        self.active=None;self.loading=False;self.last_use=time.monotonic();self.unload_requested=False;self.stopping=False
        self.last_error=None;self.started=time.time()
        for path in self.jobs.glob('*/status.json'):
            row=read_json(path)
            if row['status'] in ('queued','running','cancelling'):
                row.update(status='interrupted',stage='interrupted',message='服务已重启，请重新提交任务',finished=time.time())
                write_json(path,row)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();atexit.register(self.unload)
    def set(self,jid,**kw):
        with self.lock:
            path=self.jobs/jid/'status.json';row=read_json(path);row.update(kw);write_json(path,row)
    def unload(self):
        for s in list(self.services.values()):
            with contextlib.suppress(Exception):s.close()
        self.services.clear()
        if self.lock_file:
            fcntl.flock(self.lock_file,fcntl.LOCK_UN);self.lock_file.close();self.lock_file=None
        self.unload_requested=False
    def own_pids(self):
        owned={s.process.pid for s in list(self.services.values()) if s.process.poll() is None}
        # A worker loading weights has not yet been registered in services.
        child_file=Path('/proc/self/task')/str(self.thread.native_id)/'children'
        with contextlib.suppress(OSError):owned.update(int(x) for x in child_file.read_text().split())
        return owned
    def ready_gpu(self):
        external=gpu_pids()-self.own_pids()
        if external:
            if self.services:self.unload()
            return False
        if self.lock_file is None:
            lock_path=Path(self.args.gpu_lock);lock_path.parent.mkdir(parents=True,exist_ok=True)
            handle=lock_path.open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();return False
            self.lock_file=handle
        return True
    def model(self,role):
        if role not in self.services or self.services[role].process.poll() is not None:
            self.loading=True
            try:self.services[role]=ModelService(self.repo,self.root,role,Path(self.args.weights))
            finally:self.loading=False
        return self.services[role]
    def submit(self,data):
        jid=valid_job_id(data['id']);scene=data['scene'];classes=data['classes']
        if not isinstance(classes,list) or not classes or not all(isinstance(x,str) for x in classes):raise ValueError('类别列表无效')
        normalized,_=parse_classes(','.join(classes),scene)
        if classes!=normalized:raise ValueError('类别列表需要规范化且不能重复')
        if data.get('mode') not in ('text','semantic'):raise ValueError('任务模式无效')
        folder=self.jobs/jid
        if folder.exists():raise ValueError('任务已存在，请使用新的编号')
        image=decode_image(base64.b64decode(data['image'],validate=True))
        folder.mkdir();image.save(folder/'input.png')
        write_json(folder/'request.json',{'id':jid,'scene':scene,'classes':classes,'mode':data['mode']})
        write_json(folder/'status.json',{'id':jid,'status':'queued','stage':'queued','progress':0,
                    'message':'等待GPU处理','created':time.time(),'scene':scene,'mode':data['mode'],'classes':classes})
        self.q.put(jid);return self.status(jid)
    def status(self,jid):return read_json(self.jobs/valid_job_id(jid)/'status.json')
    def check_cancel(self,jid):
        if jid in self.cancelled:raise InterruptedError('任务已取消')
    def process(self,jid):
        folder=self.jobs/jid;data=read_json(folder/'request.json');scene=data['scene'];classes=data['classes']
        waiting=time.monotonic()
        while not self.ready_gpu():
            self.check_cancel(jid)
            self.set(jid,status='queued',stage='waiting_gpu',message='GPU正在运行其他实验，等待空闲')
            if time.monotonic()-waiting>1800:raise TimeoutError('等待GPU超过30分钟，请稍后重试')
            time.sleep(2)
        self.check_cancel(jid);queue_s=time.time()-self.status(jid)['created'];started=time.perf_counter()
        self.set(jid,status='running',stage='loading',progress=.05,message='准备模型；首次运行需要加载权重',started=time.time())
        from indoor_adapter import supported,run_adapted
        if scene=='indoor' and supported(classes):
            # Leave published street inference intact; use the verified indoor
            # implementation in independent backend processes.
            for service in list(self.services.values()):service.close()
            self.services.clear()
            with Image.open(folder/'input.png') as source:image=source.convert('RGB').copy()
            meta,masks,guided,prediction,audit=run_adapted(folder,image,classes,jid,
                lambda **values:self.set(jid,**values),lambda:self.check_cancel(jid))
            mapping={q['query_id']:i for i,q in enumerate(meta['queries'])}
            self.set(jid,stage='rendering',progress=.85,message='生成室内语义与实例分割结果')
            audit['timing']['queue_seconds']=queue_s
            result=build_artifacts(folder/'artifacts',image,classes,meta,masks,mapping,guided,prediction,data['mode'],audit)
            audit['timing']['server_processing_seconds']=time.perf_counter()-started
            write_json(folder/'artifacts/result.json',result)
            import zipfile
            with zipfile.ZipFile(folder/'artifacts/results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in result['files']:archive.write(folder/'artifacts'/name,name)
            self.check_cancel(jid)
            self.set(jid,status='succeeded',stage='complete',progress=1,message='分割完成',finished=time.time(),
                     instance_count=result['instance_count'],timing=audit['timing'])
            return
        load_start=time.perf_counter();text=self.model('text')
        semantic=self.model('semantic') if scene=='street' else None
        load_s=time.perf_counter()-load_start
        self.check_cancel(jid)
        with Image.open(folder/'input.png') as src:image=src.convert('RGB').copy()
        canon=canonical_image(image,scene);canon.save(folder/'canonical.png')
        input_sha=hashlib.sha256((folder/'canonical.png').read_bytes()).hexdigest()
        row={'image_id':jid,'path':str(folder/'canonical.png'),'width':canon.width,'height':canon.height,
             'original_size':list(canon.size),'classes':classes,'input_sha256':input_sha,
             'source_sha256':hashlib.sha256((folder/'input.png').read_bytes()).hexdigest()}
        text_folder=folder/'text'/jid
        self.set(jid,stage='text',progress=.2,message='识别文本目标并生成掩码')
        text_result=text.call('predict',row=row,folder=str(text_folder),scene=scene)
        self.check_cancel(jid)
        sem_result=None
        if semantic:
            self.set(jid,stage='semantic',progress=.6,message='运行完整I1/I2/I3语义分支与协同融合')
            sem_result=semantic.call('predict',row=row,text_parent=str(folder/'text'),folder=str(folder/'semantic'))
            self.check_cancel(jid)
            with np.load(folder/'semantic/semantic.npz',allow_pickle=False) as z:
                prediction=z['prediction'];guided=z['guided']
        meta=read_json(text_folder/'metadata.json')
        with np.load(text_folder/'candidates.npz',allow_pickle=False) as z:
            masks=np.unpackbits(z['masks_packed'],axis=2,count=canon.width).astype(bool)
        sys.path.insert(0,str(self.repo/'scripts/piers'))
        mapping=query_mapping(meta,classes,scene)
        if not semantic:
            guided=np.zeros((len(classes),canon.height,canon.width),bool)
            for n,c in enumerate(meta['candidates']):guided[mapping[c['query_id']]] |= masks[n]
            prediction=np.full((canon.height,canon.width),255,np.uint8)
            for idx in reversed(range(len(classes))):prediction[guided[idx]]=idx
        self.set(jid,stage='rendering',progress=.85,message='合并全景实例，生成可视化和下载文件')
        render_start=time.perf_counter()
        audit={'model':'full-v3-I1-I2-I3' if semantic else 'generic-grounded-SAM2-indoor',
               'indoor_adapted':False,'paper_release':'v0.3-paper-20261109','text_prefusion_cache':False,
               'I1':semantic is not None,'I2':semantic is not None,'I3':semantic is not None,
               'input_sha256':row['source_sha256'],'canonical_size':list(canon.size),'output_size':list(image.size),
               'GT_reads':0,'weights_updated':False,'text':text_result,'semantic':sem_result,
               'timing':{'queue_seconds':queue_s,'model_loading_seconds':load_s,
                         'text_seconds':text_result['service_seconds'],
                         'semantic_seconds':sem_result['service_seconds'] if sem_result else 0},
               'timing_scope':'Interactive service, not an isolated paper efficiency benchmark; includes rendering, excludes network',
               'source_fingerprints':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                    self.repo/'scripts/piers/full_model_semantic.py',self.repo/'scripts/piers/full_model_fusion_v3.py',
                    self.repo/'scripts/piers/full_model_queries_v3.py',Path(__file__).with_name('model_worker.py'),
                    Path(__file__).with_name('common.py')]}}
        result=build_artifacts(folder/'artifacts',image,classes,meta,masks,mapping,guided,prediction,data['mode'],audit)
        audit['timing'].update(render_seconds=time.perf_counter()-render_start,
                              server_processing_seconds=time.perf_counter()-started,
                              warm_processing_seconds=time.perf_counter()-started-load_s)
        # Repackage final timing after rendering. No inference is rerun.
        write_json(folder/'artifacts/result.json',result)
        import zipfile
        with zipfile.ZipFile(folder/'artifacts/results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in result['files']:z.write(folder/'artifacts'/name,name)
        self.check_cancel(jid)
        self.set(jid,status='succeeded',stage='complete',progress=1,message='分割完成',finished=time.time(),
                 instance_count=result['instance_count'],timing=audit['timing'])
    def run(self):
        while not self.stopping:
            try:jid=self.q.get(timeout=1)
            except queue.Empty:
                if self.unload_requested or (self.services and time.monotonic()-self.last_use>self.args.idle_ttl):self.unload()
                continue
            self.active=jid
            try:self.check_cancel(jid);self.process(jid)
            except InterruptedError:
                self.set(jid,status='cancelled',stage='cancelled',message='任务已取消',finished=time.time())
            except Exception as exc:
                traceback.print_exc();self.last_error=str(exc)[-1600:]
                self.set(jid,status='failed',stage='failed',message='处理失败，可查看错误详情后重试',
                         error=self.last_error,finished=time.time());self.unload()
            finally:
                self.active=None;self.last_use=time.monotonic();self.q.task_done()
                if self.unload_requested:self.unload()
    def health(self):
        stage=self.status(self.active)['stage'] if self.active else None
        try:external=gpu_pids()-self.own_pids()
        except Exception:external=set()
        return {'status':'ok','version':VERSION,'pid':os.getpid(),'active_job':self.active,
                'active_stage':stage,'other_gpu_processes':len(external),
                'queued':self.q.qsize(),'loading':self.loading,'models':list(self.services),
                'idle_release_seconds':self.args.idle_ttl,'gpu_lock':self.args.gpu_lock,
                'indoor_adapted':Path(os.environ.get('PIERS_INDOOR_REPO','/root/autodl-tmp/piers-main'),'runs/INDOOR-MATCHED-DINO-20261112/protocol.json').is_file(),
                'repo':str(self.repo),'last_error':self.last_error,
                'uptime_seconds':time.time()-self.started}

def handler_for(agent):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def send_json(self,code,value):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            try:
                parts=urlparse(self.path).path.strip('/').split('/')
                if parts==['health']:return self.send_json(200,agent.health())
                if len(parts)>=2 and parts[0]=='jobs':
                    jid=valid_job_id(parts[1])
                    if len(parts)==2:return self.send_json(200,agent.status(jid))
                    if len(parts)==3 and parts[2]=='result':return self.send_json(200,read_json(agent.jobs/jid/'artifacts/result.json'))
                    if len(parts)==4 and parts[2]=='files' and parts[3] in ('results.zip','result.json'):
                        path=agent.jobs/jid/'artifacts'/parts[3];raw=path.read_bytes();self.send_response(200)
                        self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(len(raw)))
                        self.end_headers();self.wfile.write(raw);return
                self.send_json(404,{'error':'not_found'})
            except FileNotFoundError:self.send_json(404,{'error':'任务或结果不存在'})
            except ValueError as exc:self.send_json(400,{'error':str(exc)})
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<0 or length>MAX_UPLOAD*2:return self.send_json(413,{'error':'请求过大'})
                data=json.loads(self.rfile.read(length) or b'{}');parts=self.path.strip('/').split('/')
                if parts==['jobs']:return self.send_json(202,agent.submit(data))
                if len(parts)==3 and parts[0]=='jobs' and parts[2]=='cancel':
                    jid=valid_job_id(parts[1]);row=agent.status(jid)
                    if row['status'] not in ('succeeded','failed','cancelled','interrupted'):
                        agent.cancelled.add(jid)
                        agent.set(jid,status='cancelling',message='当前计算阶段结束后停止')
                    return self.send_json(200,agent.status(jid))
                if parts==['api','models','unload']:
                    if agent.active:return self.send_json(409,{'error':'任务运行中，请先取消或等待完成'})
                    agent.unload_requested=True;return self.send_json(202,{'status':'unloading'})
                self.send_json(404,{'error':'not_found'})
            except (ValueError,KeyError) as exc:self.send_json(400,{'error':str(exc)})
            except Exception:
                traceback.print_exc();self.send_json(500,{'error':'服务器内部错误，请检查服务日志'})
    return Handler

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--state',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--port',type=int,default=8766)
    p.add_argument('--idle-ttl',type=int,default=90)
    p.add_argument('--gpu-lock',default='/root/autodl-tmp/.piers-gpu-exclusive.lock');args=p.parse_args()
    agent=Agent(args);server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(agent))
    def shutdown(*_):
        agent.stopping=True
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    try:server.serve_forever(poll_interval=.5)
    finally:agent.unload();server.server_close()
