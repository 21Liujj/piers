"""Recompute published aggregate accuracy from recorded integer pixel counts."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/piers'))
from score_zero_shot_native import metrics

def verify():
    root=REPO/'results/paper20261109'
    manifest=json.loads((root/'ledger_manifest.json').read_text())
    for name,digest in manifest['file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Ledger changed: '+name)
    frozen=json.loads((root/'complete_table.json').read_text(encoding='utf-8'))
    output=[]
    for task,spec in manifest['tasks'].items():
        row=frozen['rows'][spec['table_row']]
        with np.load(root/spec['histograms'],allow_pickle=False) as data:
            if len(set(data['image_ids'].tolist()))!=spec['images']:raise ValueError('Repeated or missing image IDs')
            report={'task':task,'images':spec['images']}
            for variant in ['original','full']:
                h=data[variant]
                if h.shape!=(spec['images'],len(row['classes']),len(row['classes'])+1):raise ValueError('Invalid histogram dimensions')
                if not np.issubdtype(h.dtype,np.integer) or (h<0).any():raise ValueError('Invalid pixel counts')
                actual=metrics(h.sum(0));expected=row['full_final_v3' if variant=='full' else 'original']
                for key,value in actual.items():
                    if not np.allclose(np.asarray(value),np.asarray(expected[key]),rtol=0,atol=1e-9):
                        raise AssertionError((task,variant,key))
                report[variant]=actual
        report['delta_mIoU_pp']=report['full']['mIoU_pct']-report['original']['mIoU_pct'];output.append(report)
        print('VERIFIED',task,spec['images'],'mIoU',round(report['original']['mIoU_pct'],6),'->',round(report['full']['mIoU_pct'],6))
    print('ALL_FROZEN_LEDGERS_VERIFIED',len(output))
    return output
