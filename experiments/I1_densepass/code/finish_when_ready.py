from pathlib import Path
import time,subprocess,os
R=Path(__file__).resolve().parent;D=R/'dense_controls';deadline=time.time()+2400
while not all(p.exists() for p in [D/'trained/report.json',D/'cache_official/metadata_s0.json',D/'supplements_resumed.json']):
 if (D/'failure.txt').exists():raise RuntimeError((D/'failure.txt').read_text())
 if time.time()>deadline:raise TimeoutError('Final controls did not finish within40min')
 time.sleep(10)
subprocess.run(['/root/autodl-tmp/envs/piers/bin/python','-u',str(R/'finish_package.py')],env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),check=True)
