"""PIERS local web app. Run one worker: python apps/panorama_web/app.py."""
from __future__ import annotations
import argparse,base64,contextlib,io,json,os,queue,sqlite3,sys,threading,time,uuid,zipfile
from contextlib import asynccontextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
from PIL import Image
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel,Field,SecretStr
from common import (VERSION,STREET_CLASSES,INDOOR_CLASSES,ZH,MAX_UPLOAD,decode_image,parse_classes,
                    read_json,write_json,score_uploaded_gt,valid_job_id)
from transport import GPUBridge

REPO=HERE.parents[1]
TERMINAL={'succeeded','failed','cancelled','interrupted'}
ARTIFACTS={'classes.png','semantic.png','overlay.jpg','instances.png','instance_layer.png',
           'instance_overlay.jpg','result.json','results.zip','accuracy.json'}

class Store:
    def __init__(self,home):
        self.home=Path(home);self.images=self.home/'images';self.jobs=self.home/'jobs'
        for p in (self.home,self.images,self.jobs):p.mkdir(parents=True,exist_ok=True)
        self.path=self.home/'workbench.sqlite3'
        with self.db() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY,name TEXT,width INT,height INT,created REAL,mime TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,image_id TEXT,scene TEXT,text TEXT,classes TEXT,mode TEXT,status TEXT,stage TEXT,message TEXT,progress REAL,created REAL,started REAL,finished REAL,error TEXT,elapsed REAL)')
    @contextlib.contextmanager
    def db(self):
        c=sqlite3.connect(self.path,timeout=20);c.row_factory=sqlite3.Row
        try:
            with c:yield c
        finally:c.close()
    def get_image(self,iid):
        valid_job_id(iid)
        with self.db() as c:row=c.execute('SELECT * FROM images WHERE id=?',(iid,)).fetchone()
        if not row:raise FileNotFoundError('图片不存在')
        return dict(row)
    def get_job(self,jid):
        valid_job_id(jid)
        with self.db() as c:row=c.execute('SELECT jobs.*,images.name AS image_name,images.width,images.height FROM jobs JOIN images ON jobs.image_id=images.id WHERE jobs.id=?',(jid,)).fetchone()
        if not row:raise FileNotFoundError('任务不存在')
        result=dict(row);result['classes']=json.loads(result['classes'])
        p=self.jobs/jid/'result.json'
        if result['status']=='succeeded' and p.exists():result['result']=read_json(p)
        if (self.jobs/jid/'accuracy.json').exists():result['accuracy']=read_json(self.jobs/jid/'accuracy.json')
        return result
    def update(self,jid,**values):
        if set(values)-{'status','stage','message','progress','started','finished','error','elapsed'}:raise ValueError('无效字段')
        with self.db() as c:c.execute('UPDATE jobs SET '+','.join(k+'=?' for k in values)+' WHERE id=?',(*values.values(),jid))

class Jobs:
    def __init__(self,store,bridge):
        self.store=store;self.bridge=bridge;self.q=queue.Queue();self.cancelled=set();self.active=None;self.stop=False
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop:
            try:jid=self.q.get(timeout=.5)
            except queue.Empty:continue
            self.active=jid;submitted=False;started=time.perf_counter()
            try:
                if jid in self.cancelled:raise InterruptedError()
                row=self.store.get_job(jid);self.store.update(jid,status='running',stage='uploading',message='将图片发送至GPU服务器',started=time.time(),progress=.02)
                if not self.bridge.health().get('connected'):self.bridge.connect(deploy=True)
                if jid in self.cancelled:raise InterruptedError()
                data=(self.store.images/row['image_id']/'original.bin').read_bytes()
                self.bridge.request('POST','/jobs',{'id':jid,'scene':row['scene'],'mode':row['mode'],
                    'classes':row['classes'],'image':base64.b64encode(data).decode()},timeout=90);submitted=True
                while True:
                    if jid in self.cancelled:self.bridge.request('POST','/jobs/'+jid+'/cancel',{})
                    remote=self.bridge.request('GET','/jobs/'+jid,timeout=20)
                    self.store.update(jid,status=remote['status'],stage=remote['stage'],message=remote['message'],
                        progress=remote.get('progress',0),error=remote.get('error'))
                    if remote['status'] in TERMINAL:break
                    if time.perf_counter()-started>2400:raise TimeoutError('本次任务超过40分钟，请检查GPU服务器')
                    time.sleep(.8)
                if remote['status']=='succeeded':
                    self.store.update(jid,status='running',stage='downloading',message='取回结果并保存到本机',progress=.96)
                    raw=self.bridge.request('GET','/jobs/'+jid+'/files/results.zip',timeout=90)
                    folder=self.store.jobs/jid;folder.mkdir(exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        if sum(x.file_size for x in z.infolist())>256*1024*1024:raise ValueError('结果文件过大')
                        if any(x.filename not in ARTIFACTS for x in z.infolist()):raise ValueError('结果压缩包包含意外文件')
                        for info in z.infolist():(folder/info.filename).write_bytes(z.read(info))
                    (folder/'results.zip').write_bytes(raw)
                    result=read_json(folder/'result.json')
                    if [c['name'] for c in result['classes']]!=row['classes']:raise ValueError('返回类别与本次请求不一致')
                    self.store.update(jid,status='succeeded',stage='complete',message='分割完成，结果已保存到本机',progress=1,
                                      finished=time.time(),elapsed=time.perf_counter()-started)
                else:self.store.update(jid,finished=time.time(),elapsed=time.perf_counter()-started)
            except InterruptedError:
                self.store.update(jid,status='cancelled',stage='cancelled',message='任务已取消',finished=time.time())
            except Exception as exc:
                if submitted:
                    with contextlib.suppress(Exception):self.bridge.request('POST','/jobs/'+jid+'/cancel',{})
                self.store.update(jid,status='failed',stage='failed',message='任务未完成，请检查连接或错误详情',
                                  error=self.bridge.sanitize_error(exc),finished=time.time(),elapsed=time.perf_counter()-started)
            finally:self.active=None;self.q.task_done()

class JobRequest(BaseModel):
    image_id:str
    scene:str='street'
    text:str=Field(default='',max_length=3200)

class ConnectionRequest(BaseModel):
    host:str
    port:int=Field(ge=1,le=65535)
    username:str='root'
    password:SecretStr=SecretStr('')
    repo:str|None=None
    python:str|None=None
    weights:str|None=None
    runtime:str|None=None

def create_app(home=None,bridge_override=None):
    home=Path(home or os.environ.get('PIERS_WEB_HOME',str(Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'PIERSWorkbench')))
    store=Store(home);bridge=bridge_override or GPUBridge(home);jobs=Jobs(store,bridge)
    @asynccontextmanager
    async def lifespan(app):
        with store.db() as c:c.execute("UPDATE jobs SET status='interrupted',stage='interrupted',message='本机服务已重启，请重新运行' WHERE status IN ('queued','running','cancelling')")
        jobs.thread.start()
        if not os.environ.get('PIERS_WEB_DISABLE_CONNECT'):
            def connect():
                with contextlib.suppress(Exception):bridge.connect(deploy=True)
            threading.Thread(target=connect,daemon=True).start()
        yield
        jobs.stop=True
        if jobs.active:jobs.cancelled.add(jobs.active)
        jobs.thread.join(timeout=3)
        bridge.close()
    app=FastAPI(title='全景图像分割系统',version=VERSION,lifespan=lifespan)
    app.state.store=store;app.state.bridge=bridge;app.state.jobs=jobs
    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse({'detail':str(exc)},status_code=400)
    @app.exception_handler(FileNotFoundError)
    async def missing(request,exc):return JSONResponse({'detail':str(exc)},status_code=404)
    @app.exception_handler(RequestValidationError)
    async def bad_request(request,exc):
        return JSONResponse({'detail':[{'loc':e['loc'],'msg':e['msg']} for e in exc.errors()]},status_code=422)
    @app.middleware('http')
    async def local_origin(request:Request,call_next):
        origin=request.headers.get('origin')
        if request.method in ('POST','PUT','DELETE') and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc!=request.headers.get('host'):return JSONResponse({'detail':'请求来源不匹配'},status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        return response
    @app.get('/api/health')
    def health():return {'status':'ok','version':VERSION,'active_job':jobs.active,'queued':jobs.q.qsize()}
    @app.get('/api/capabilities')
    def capabilities():
        return {'version':VERSION,'deployment':'local-ui-ssh-gpu','scenes':[
            {'id':'street','name':'室外街景','model':'完整模型 · I1 / I2 / I3','adapted':True,'classes':STREET_CLASSES},
            {'id':'indoor','name':'室内场景','model':'室内适配完整模型 · I1 / I2 / I3','adapted':True,'classes':INDOOR_CLASSES}],
            'semantic':True,'text':True,'instances':True,'labels_zh':ZH,'max_upload_mb':30,
            'notes':['无文本时使用场景默认类别表','实例通过跨视窗关联生成，未声称实例AP','精度指标需要标准标签']}
    @app.get('/api/connection')
    def get_connection():return {'config':bridge.public_config(),'health':bridge.health()}
    @app.post('/api/connection')
    def set_connection(value:ConnectionRequest):
        if jobs.active or jobs.q.qsize():raise HTTPException(409,'有任务正在运行，请完成或取消后更改连接')
        values=value.model_dump(exclude_none=True);values['password']=value.password.get_secret_value()
        try:return bridge.configure(values)
        except Exception as exc:raise HTTPException(400,bridge.sanitize_error(exc)) from exc
    @app.post('/api/connection/reconnect')
    def reconnect():
        try:return bridge.connect(deploy=True)
        except Exception as exc:raise HTTPException(503,bridge.sanitize_error(exc)) from exc
    @app.post('/api/models/unload')
    def unload():
        if jobs.active:raise HTTPException(409,'任务运行中，请完成后释放模型')
        try:return bridge.request('POST','/api/models/unload',{})
        except Exception as exc:raise HTTPException(503,bridge.sanitize_error(exc)) from exc
    @app.post('/api/images')
    async def upload_image(file:UploadFile=File(...)):
        raw=await file.read(MAX_UPLOAD+1);image=decode_image(raw);iid=uuid.uuid4().hex[:24]
        folder=store.images/iid;folder.mkdir();(folder/'original.bin').write_bytes(raw)
        preview=image.copy();preview.thumbnail((1800,900));preview.save(folder/'preview.jpg',quality=92)
        with Image.open(io.BytesIO(raw)) as im:mime=Image.MIME.get(im.format,'image/png')
        name=(file.filename or 'panorama').replace('\\','/').split('/')[-1][:180]
        with store.db() as c:c.execute('INSERT INTO images VALUES(?,?,?,?,?,?)',(iid,name,image.width,image.height,time.time(),mime))
        return {'id':iid,'name':name,'width':image.width,'height':image.height,'preview_url':'/api/images/'+iid+'/preview'}
    @app.get('/api/images/{iid}/{kind}')
    def image_file(iid:str,kind:str):
        row=store.get_image(iid)
        if kind=='preview':return FileResponse(store.images/iid/'preview.jpg',media_type='image/jpeg')
        if kind=='original':return FileResponse(store.images/iid/'original.bin',media_type=row['mime'])
        raise HTTPException(404,'图片不存在')
    @app.post('/api/jobs',status_code=202)
    def submit(value:JobRequest):
        store.get_image(value.image_id);classes,specified=parse_classes(value.text,value.scene)
        jid=uuid.uuid4().hex[:24];(store.jobs/jid).mkdir()
        with store.db() as c:c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (jid,value.image_id,value.scene,value.text,json.dumps(classes,ensure_ascii=False),'text' if specified else 'semantic',
             'queued','queued','等待处理',0,time.time(),None,None,None,None))
        jobs.q.put(jid);return store.get_job(jid)
    @app.get('/api/jobs')
    def list_jobs():
        with store.db() as c:rows=c.execute('SELECT jobs.*,images.name AS image_name FROM jobs JOIN images ON jobs.image_id=images.id ORDER BY created DESC LIMIT 100').fetchall()
        result=[]
        for r in rows:d=dict(r);d['classes']=json.loads(d['classes']);result.append(d)
        return {'jobs':result}
    @app.get('/api/jobs/{jid}')
    def get_job(jid:str):return store.get_job(jid)
    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid:str):
        row=store.get_job(jid)
        if row['status'] not in TERMINAL:
            jobs.cancelled.add(jid);store.update(jid,status='cancelling',message='正在取消，当前计算阶段结束后停止')
        return store.get_job(jid)
    @app.get('/api/jobs/{jid}/files/{name}')
    def artifact(jid:str,name:str):
        row=store.get_job(jid)
        if row['status']!='succeeded' or name not in ARTIFACTS or not (store.jobs/jid/name).is_file():raise HTTPException(404,'结果文件不存在')
        return FileResponse(store.jobs/jid/name,filename=name if name.endswith(('.zip','.json')) else None)
    @app.get('/api/jobs/{jid}/layer')
    def layer(jid:str,class_index:int|None=None,instance_id:int|None=None):
        row=store.get_job(jid)
        if row['status']!='succeeded':raise HTTPException(409,'结果尚未生成')
        if (class_index is None)==(instance_id is None):raise ValueError('请选择一个类别或实例')
        if class_index is not None:
            if not 0<=class_index<len(row['classes']):raise ValueError('类别编号超出范围')
            with Image.open(store.jobs/jid/'classes.png') as im:mask=np.asarray(im)==class_index
            color=row['result']['classes'][class_index]['color']
        else:
            item=next((r for r in row['result']['instances'] if r['id']==instance_id),None)
            if item is None:raise ValueError('实例不存在')
            with Image.open(store.jobs/jid/'instances.png') as im:mask=np.asarray(im)==instance_id
            color=item['color']
        rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=(*bytes.fromhex(color[1:]),255)
        out=io.BytesIO();Image.fromarray(rgba).save(out,format='PNG');return Response(out.getvalue(),media_type='image/png')
    @app.post('/api/jobs/{jid}/evaluate')
    async def evaluate(jid:str,file:UploadFile=File(...),encoding:str=Form('class_ids')):
        row=store.get_job(jid)
        if row['status']!='succeeded':raise HTTPException(409,'请先完成分割')
        raw=await file.read(MAX_UPLOAD+1)
        if len(raw)>MAX_UPLOAD:raise ValueError('标注文件超过30MB')
        try:
            with Image.open(io.BytesIO(raw)) as im:
                if im.format!='PNG':raise ValueError('请上传单通道 PNG 标注')
                if im.width*im.height>36_000_000:raise ValueError('标注图片过大')
                truth=np.asarray(im).copy()
        except OSError as exc:raise ValueError('无法读取标注图片') from exc
        with Image.open(store.jobs/jid/'classes.png') as im:prediction=np.asarray(im)
        report=score_uploaded_gt(prediction,truth,row['classes'],encoding)
        folder=store.jobs/jid;write_json(folder/'accuracy.json',report)
        result=read_json(folder/'result.json');result['accuracy']=report
        if 'accuracy.json' not in result['files']:result['files'].append('accuracy.json')
        write_json(folder/'result.json',result)
        with zipfile.ZipFile(folder/'results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in result['files']:z.write(folder/name,name)
        return report
    @app.get('/api/experiments')
    def experiments():
        root=REPO/'results/paper20261109';table=read_json(root/'complete_table.json')
        rows=list(table['rows'])
        indoor_path=REPO/'results/indoor_matched20261112/indoor_results.json'
        if indoor_path.exists():
            indoor=read_json(indoor_path)
            def metrics(value):
                return {'mIoU_pct':value['mIoU_pct'],'mAcc_pct':value['mAcc_pct'],
                    'pixel_accuracy_pct':value['PixelAcc_pct'],'FWIoU_pct':value['FWIoU_pct'],
                    'mDice_pct':value['mDice_pct'],
                    'per_class_IoU_pct':[value['per_class_iou_pct'][c] for c in INDOOR_CLASSES]}
            rows.append({'evaluation_space':'Stanford2D3D SPAN8 室内适配','images':indoor['n_images'],
                'classes':INDOOR_CLASSES,'original':metrics(indoor['arms']['original']),
                'full_final_v3':metrics(indoor['arms']['full']),
                'delta_mIoU_pp_95CI':indoor['paired_image_bootstrap_delta95CI_pp'],
                'training_condition':'Same shared indoor Stage1 checkpoint and LoRA; full I1 calibrated on training-area calib104',
                'CLIP_used':False})
        efficiency=[]
        for name in ('native_efficiency.json','densepass_efficiency.json'):
            data=read_json(root/name)
            efficiency.extend({**r,'source':name} for r in data['rows'])
        return {'release':table['release']+'+indoor-matched-20261112','created':table['created_at_CST'],'rows':rows,
                'limitations':table['limitations'],'efficiency':efficiency,
                'source_commit':'82d20b5b4ba24c42666dd8fee548d5c8618505c8',
                'notes':['室外使用冻结源域权重；室内两侧共享适配后的checkpoint和LoRA，上传任务不会改变这些指标',
                         '效率为RTX6000D、3张固定图片各2次测量，排除首次加载；完整系统当前更慢',
                         '室内适配已完成373张评测，mIoU37.64%到46.56%；当前LoRA未声称全局最优']}
    @app.get('/api/experiments/download')
    def download_experiments():
        return JSONResponse(experiments(),headers={'Content-Disposition':'attachment; filename="panorama_experiments_20261112.json"'})
    app.mount('/',StaticFiles(directory=str(HERE/'static'),html=True),name='workbench')
    return app

app=create_app()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8010);a=p.parse_args()
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=a.port,workers=1,log_level='info')
