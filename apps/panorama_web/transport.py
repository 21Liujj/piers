"""SSH transport. Passwords live only in process memory, never in a JSON file."""
from __future__ import annotations
import hashlib,http.client,json,os,shlex,socket,threading,time
from pathlib import Path,PurePosixPath
import paramiko
from common import VERSION,read_json,write_json

HERE=Path(__file__).resolve().parent
REMOTE_FILES=['common.py','model_worker.py','remote_agent.py','bootstrap_remote.py','gpu_guard.py']

class GPUBridge:
    def __init__(self,home):
        self.home=Path(home);self.config_path=self.home/'connection.json';self.host_keys=self.home/'known_hosts'
        self.config={'host':os.environ.get('PIERS_SSH_HOST',''),
            'port':int(os.environ.get('PIERS_SSH_PORT','22')),'username':os.environ.get('PIERS_SSH_USER','root'),
            'repo':os.environ.get('PIERS_REMOTE_REPO','/root/autodl-tmp/piers-web-frozen-v03'),
            'python':os.environ.get('PIERS_REMOTE_PYTHON','/root/autodl-tmp/envs/piers/bin/python'),
            'weights':os.environ.get('PIERS_REMOTE_WEIGHTS','/root/autodl-tmp/weights'),
            'runtime':os.environ.get('PIERS_REMOTE_RUNTIME','/root/autodl-tmp/piers-web-runtime'),
            'agent_port':int(os.environ.get('PIERS_AGENT_PORT','8766'))}
        if self.config_path.exists():self.config.update(read_json(self.config_path))
        self.password=os.environ.get('PIERS_SSH_PASSWORD','');self.client=None;self.lock=threading.RLock()
        self.health_cache={'connected':False,'message':'尚未连接GPU服务器','version':VERSION}
        self.last_health=0.
    def public_config(self):return {**self.config,'has_credentials':bool(self.password)}
    def sanitize_error(self,exc):
        s=str(exc)
        if self.password:s=s.replace(self.password,'[已隐藏]')
        return s[:1200]
    def close(self):
        with self.lock:
            if self.client:self.client.close()
            self.client=None
    def ssh(self):
        with self.lock:
            if self.client and self.client.get_transport() and self.client.get_transport().is_active():return self.client
            if not self.config['host']:raise RuntimeError('请先在连接设置中填写GPU服务器地址')
            local_keys=any((Path.home()/'.ssh'/name).exists() for name in ('id_ed25519','id_rsa','id_ecdsa'))
            if not self.password and not os.environ.get('PIERS_SSH_KEY') and not local_keys:raise RuntimeError('请先在连接设置中输入服务器密码')
            client=paramiko.SSHClient();client.load_system_host_keys()
            if self.host_keys.exists():client.load_host_keys(str(self.host_keys))
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.config['host'],port=self.config['port'],username=self.config['username'],
                           password=self.password or None,key_filename=os.environ.get('PIERS_SSH_KEY'),
                           timeout=15,auth_timeout=20,banner_timeout=20,allow_agent=True,look_for_keys=True)
            client.save_host_keys(str(self.host_keys));client.get_transport().set_keepalive(20)
            self.client=client;return client
    def request(self,method,path,data=None,timeout=25):
        client=self.ssh();transport=client.get_transport()
        channel=transport.open_channel('direct-tcpip',('127.0.0.1',self.config['agent_port']),('127.0.0.1',0),timeout=timeout)
        channel.settimeout(timeout);conn=http.client.HTTPConnection('127.0.0.1',self.config['agent_port'],timeout=timeout);conn.sock=channel
        raw=json.dumps(data,ensure_ascii=False).encode() if data is not None else None
        try:
            conn.request(method,path,body=raw,headers={'Content-Type':'application/json','Connection':'close'})
            response=conn.getresponse();body=response.read()
            if response.status>=400:
                try:message=json.loads(body).get('error',body.decode(errors='replace'))
                except ValueError:message=body.decode(errors='replace')
                raise RuntimeError(str(message))
            if response.getheader('Content-Type','').startswith('application/json'):return json.loads(body)
            return body
        finally:conn.close()
    def configure(self,values):
        allowed={'host','port','username','password','repo','python','weights','runtime','agent_port'}
        if set(values)-allowed:raise ValueError('未知的连接配置')
        with self.lock:
            candidate={**self.config,**{k:v for k,v in values.items() if k!='password'}}
            if not str(candidate['host']).strip() or any(x in candidate['host'] for x in '\r\n /'):raise ValueError('服务器地址无效')
            for key in ('port','agent_port'):
                candidate[key]=int(candidate[key])
                if not 1<=candidate[key]<=65535:raise ValueError('端口号必须在1到65535之间')
            for key in ('repo','python','weights','runtime'):
                value=candidate[key]
                if not isinstance(value,str) or not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts or '\x00' in value:
                    raise ValueError('服务器路径必须为绝对路径')
            self.close();self.config=candidate
            if values.get('password'):self.password=values['password']
            write_json(self.config_path,self.config)
        return self.connect(deploy=True)
    def deploy(self):
        client=self.ssh();remote=PurePosixPath(self.config['runtime']);code=remote/'code';state=remote/'state'
        sftp=client.open_sftp()
        def mkdir(path):
            for p in reversed([path,*path.parents]):
                if str(p)=='/':continue
                try:sftp.stat(str(p))
                except FileNotFoundError:sftp.mkdir(str(p))
        try:
            mkdir(code);mkdir(state)
            new_hash=hashlib.sha256(b''.join((HERE/n).read_bytes() for n in REMOTE_FILES)).hexdigest()
            try:
                with sftp.open(str(code/'manifest.json')) as f:old_hash=json.load(f).get('sha256')
            except FileNotFoundError:old_hash=None
            try:current=self.request('GET','/health',timeout=3)
            except Exception:current=None
            changed=old_hash!=new_hash
            if current and changed and current.get('active_job'):raise RuntimeError('网页任务运行中，请完成后更新服务')
            if changed:
                for name in REMOTE_FILES:
                    destination=str(code/name);temporary=destination+'.upload'
                    sftp.put(str(HERE/name),temporary)
                    try:sftp.posix_rename(temporary,destination)
                    except OSError:
                        try:sftp.remove(destination)
                        except FileNotFoundError:pass
                        sftp.rename(temporary,destination)
            command=[self.config['python'],str(code/'bootstrap_remote.py'),'--repo',self.config['repo'],
                     '--state',str(state),'--weights',self.config['weights'],'--port',str(self.config['agent_port'])]
            if changed:command.append('--restart')
            _,stdout,stderr=client.exec_command(shlex.join(command),timeout=40)
            result=stdout.read().decode();error=stderr.read().decode();exit_code=stdout.channel.recv_exit_status()
            if exit_code:raise RuntimeError(error[-1600:] or result[-1600:] or 'GPU服务启动失败')
            with sftp.open(str(code/'manifest.json'),'w') as f:f.write(json.dumps({'sha256':new_hash,'version':VERSION}))
        finally:sftp.close()
    def connect(self,deploy=False):
        try:
            if deploy:self.deploy()
            health=self.request('GET','/health',timeout=6)
            self.health_cache={'connected':True,**health,'message':'GPU服务器已连接'};self.last_health=time.time()
        except Exception as exc:
            self.health_cache={'connected':False,'message':self.sanitize_error(exc),'version':VERSION};self.last_health=time.time()
            raise
        return self.health_cache
    def health(self,force=False):
        if force or time.time()-self.last_health>8:
            try:self.connect()
            except Exception:pass
        return self.health_cache
