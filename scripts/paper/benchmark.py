"""Remap a recorded timing plan to this checkout's verified prepared inputs."""
from pathlib import Path
import argparse,json,os,sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/piers')]
from benchmark_complete_model_v3 import parent,write_json,sha

def materialize(plan,prepared):
    available={}
    for folder in prepared:
        folder=Path(folder)
        prep=json.loads((folder/'PREPARATION.json').read_text())
        if sha(folder/'manifest.json')!=prep['manifest_sha256']:raise ValueError('Prepared manifest changed')
        for row in json.loads((folder/'manifest.json').read_text())['images']:
            key=(row['image_id'],tuple(row['classes']))
            if key in available:raise ValueError('Duplicate prepared image/category protocol')
            available[key]=row
    mapped=json.loads(json.dumps(plan))
    for scenario in mapped['scenarios']:
        rows=[]
        for recorded in scenario['images']:
            key=(recorded['image_id'],tuple(recorded['classes']))
            if key not in available:raise ValueError('Required benchmark image not prepared: '+str(key))
            row=available[key]
            if row['input_sha256']!=recorded['input_sha256']:raise ValueError('Benchmark input bytes differ')
            if row['original_size']!=recorded['original_size']:raise ValueError('Benchmark geometry differs')
            rows.append(row)
        scenario['images']=rows
    return mapped

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True,nargs='+');p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh benchmark output directory')
    plan=materialize(json.loads(Path(a.plan).read_text()),a.prepared)
    out.mkdir(parents=True,exist_ok=True);write_json(out/'LOCAL_PLAN.json',plan)
    os.environ['PIERS_TEXT_FEATURE_CACHE']='0'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    parent(SimpleNamespace(repo=str(REPO),out=str(out),plan=str(out/'LOCAL_PLAN.json'),calibrators=str(REPO/'assets/paper20261109/calibrators')))

if __name__=='__main__':main()
