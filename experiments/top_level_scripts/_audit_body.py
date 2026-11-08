import sys
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
from PIL import Image
import numpy as np, os
rd='/root/autodl-tmp/data_extra/WildPASS13/labels_raw'
cd='/root/autodl-tmp/data_extra/WildPASS13/labels_trainid'
bd='/root/autodl-tmp/data_extra/WildPASS13/labels_trainid.backup_20260511_175555'
fs=sorted(os.listdir(rd))[:20]
print('file | raw | cur | bak')
for f in fs:
    ra=np.array(Image.open(os.path.join(rd,f)))
    cu=np.array(Image.open(os.path.join(cd,f)))
    ba=np.array(Image.open(os.path.join(bd,f)))
    print(f, sorted(np.unique(ra).tolist()), sorted(np.unique(cu).tolist()), round(float((cu!=255).sum())/cu.size,3), sorted(np.unique(ba).tolist()), round(float((ba!=255).sum())/ba.size,3), flush=True)
# check where trainid came from: compare cur vs raw
print('done-audit')
