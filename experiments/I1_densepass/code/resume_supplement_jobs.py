from pathlib import Path
import os,signal,json,sys,time
r=Path(__file__).resolve().parent
if '--watchdog' in sys.argv:
 start=time.monotonic()
 while time.monotonic()-start<7200 and not (r/'resume_supplements.flag').exists():time.sleep(20)
for j in json.loads((r/'paused_supplement_jobs.json').read_text()):
 p=Path('/proc')/str(j['pid'])/'cmdline'
 if p.exists() and p.read_bytes().hex()==j['cmdline_hex']:
  os.kill(j['pid'],signal.SIGCONT);print('RESUMED',j['pid'],flush=True)
(r/'supplement_jobs_resumed.json').write_text(json.dumps({'at_unix':time.time()}))
