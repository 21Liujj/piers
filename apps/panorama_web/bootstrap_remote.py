"""Start/restart only this application's private agent, leaving experiments alone."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path

def probe(port):
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=2) as r:return json.load(r)
    except Exception:return None

def main(a):
    state=Path(a.state).resolve();state.mkdir(parents=True,exist_ok=True)
    current=probe(a.port)
    if current and not a.restart:print(json.dumps(current));return
    if current:
        if current.get('active_job'):raise RuntimeError('网页任务运行中，暂不更新服务')
        pid=int(current['pid']);cmd=Path('/proc')/str(pid)/'cmdline'
        if str(Path(__file__).with_name('remote_agent.py')).encode() not in cmd.read_bytes():raise RuntimeError('端口由其他服务占用')
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not probe(a.port):break
            time.sleep(.1)
        else:raise RuntimeError('旧网页服务未停止，未强制中断')
    log=(state/'agent.log').open('a')
    command=[sys.executable,'-u',str(Path(__file__).with_name('remote_agent.py')),
             '--repo',a.repo,'--state',str(state),'--weights',a.weights,'--port',str(a.port),
             '--gpu-lock',a.gpu_lock,'--idle-ttl',str(a.idle_ttl)]
    process=subprocess.Popen(command,cwd=Path(__file__).parent,stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (state/'agent.pid').write_text(str(process.pid))
    for _ in range(100):
        health=probe(a.port)
        if health:print(json.dumps(health));return
        if process.poll() is not None:raise RuntimeError('服务启动失败，请查看agent.log')
        time.sleep(.1)
    raise RuntimeError('网页GPU服务未就绪')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--state',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--port',type=int,default=8766)
    p.add_argument('--gpu-lock',default='/root/autodl-tmp/.piers-gpu-exclusive.lock')
    p.add_argument('--idle-ttl',type=int,default=90);p.add_argument('--restart',action='store_true')
    main(p.parse_args())
