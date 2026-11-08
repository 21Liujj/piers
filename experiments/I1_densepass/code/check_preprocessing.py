import sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
root=Path('/root/autodl-tmp/piers-main');sys.path.insert(0,str(root/'third_party/effective_omnisam'))
from dataset.DensePASS import DensePASS13
ds=DensePASS13('/root/autodl-tmp/data/DensePASS','/root/autodl-tmp/dataset/densepass_list/val.txt',crop_size=(4096,1024),sliding_window=(384,1024),passes=2)
a,*_=ds[0]
pil=Image.open('/root/autodl-tmp/data/DensePASS/leftImg8bit/val/100_.png').convert('RGB').resize((4096,1024),Image.Resampling.BICUBIC)
b=torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).contiguous().float().div(255)
b=(b-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
print('input max difference',float((a[0]-b[:,:,:1024]).abs().max()),'frames',a.shape)
