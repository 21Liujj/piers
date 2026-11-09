"""Portable roots for the audited outdoor inference runtime."""
from pathlib import Path
import json,os

REPO=Path(__file__).resolve().parents[3]
WEIGHTS=Path(os.environ.get('PIERS_WEIGHTS_ROOT',str(REPO/'assets/weights'))).expanduser().resolve()
DATA=Path(os.environ.get('PIERS_DATA_ROOT',str(REPO/'data'))).expanduser().resolve()
EXTRA=Path(os.environ.get('PIERS_DATA_EXTRA_ROOT',str(REPO/'data_extra'))).expanduser().resolve()
RUNS=Path(os.environ.get('PIERS_RUNS_ROOT',str(REPO/'runs'))).expanduser().resolve()
CALIBRATORS=REPO/'assets/paper20261109/calibrators'

def expand(value):
    if isinstance(value,str):
        for key,path in {'PIERS_REPO_ROOT':REPO,'PIERS_WEIGHTS_ROOT':WEIGHTS,'PIERS_DATA_ROOT':DATA,'PIERS_DATA_EXTRA_ROOT':EXTRA,'PIERS_RUNS_ROOT':RUNS}.items():
            value=value.replace('${'+key+'}',str(path))
        return value
    if isinstance(value,list):return [expand(x) for x in value]
    if isinstance(value,dict):return {expand(k):expand(v) for k,v in value.items()}
    return value

def memory_config():return expand(json.loads((REPO/'configs/piers/paper20261109/memory.json').read_text()))
