import sys
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
from PIL import Image
import numpy as np, os, glob
# 1. raw source listing
for p in ["/root/autodl-tmp/OmniSAM-main/data_extra/raw/WildPASS/gtFine/val/cs",
          "/root/autodl-tmp/data_extra/WildPASS13/labels_raw",
          "/root/autodl-tmp/weights",
          "/root/autodl-tmp/piers-main/third_party/effective_omnisam/exp2"]:
    print("DIR:",p)
    try:
        fs=sorted(os.listdir(p))[:8]
        print(fs)
    except Exception as e:
        print("ERR",e)
# 2. check raw gtFine filename pattern + unique
rawsrc="/root/autodl-tmp/OmniSAM-main/data_extra/raw/WildPASS/gtFine/val/cs"
try:
    fs=sorted(os.listdir(rawsrc))[:3]
    print("rawsrc sample",fs)
    for f in fs[:2]:
        a=np.array(Image.open(os.path.join(rawsrc,f)))
        print(f, a.shape, sorted(np.unique(a).tolist())[:25])
except Exception as e:
    print("rawsrc err",e)
# 3. RGB check for 10.png row profile
from pathlib import Path
rgb=np.array(Image.open('/root/autodl-tmp/data_extra/WildPASS13/images/10.png'))
raw=np.array(Image.open('/root/autodl-tmp/data_extra/WildPASS13/labels_raw/10.png'))
print("10.png shapes",rgb.shape,raw.shape)
H=rgb.shape[0]
for y in [5,50,100,150,200,250,300,350,395]:
    print(f"y={y} rawuniq={sorted(np.unique(raw[y,::64]).tolist())} rgb={rgb[y,1024].tolist()}")
print("done")
