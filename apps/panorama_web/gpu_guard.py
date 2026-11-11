"""Conservative GPU occupancy detection, including restricted-container NVML.

Some cloud containers report no nvidia-smi compute PIDs while CUDA workers
are running. Check visible device file descriptors and known orchestration
commands as well, so the web service does not slip between experiment stages.
"""
import os,re,subprocess
from pathlib import Path

def orchestrates_gpu(argv):
    scripts={Path(x).name for x in argv if x.endswith('.py')}
    if 'reproduce.py' in scripts and 'run' in argv:return True
    if 'backend.py' in scripts and any(x in argv for x in ('--serve','--text-worker','--manifest')):return True
    return bool(scripts & {'benchmark_complete_model_v3.py','benchmark_text_feature_cache.py',
                          'run_zero_shot_predictions.py','run_i3_semantic.py'})

def observed_gpu_pids(proc_root=Path('/proc'),nvml_pids=None):
    if nvml_pids is None:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=8)
        if r.returncode:raise RuntimeError('无法查询GPU状态')
        nvml_pids={int(s.strip()) for s in r.stdout.splitlines() if s.strip().isdigit()}
    result=set(nvml_pids)
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if argv and Path(argv[0]).name=='nvidia-smi':continue
            if orchestrates_gpu(argv):result.add(int(proc.name));continue
            for fd in (proc/'fd').iterdir():
                try:device=os.readlink(fd)
                except OSError:continue
                if re.fullmatch(r'/dev/nvidia\d+',device):result.add(int(proc.name));break
        except (OSError,PermissionError):continue
    return result
