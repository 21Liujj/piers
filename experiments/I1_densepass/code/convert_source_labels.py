from pathlib import Path
import json,concurrent.futures,hashlib,time
import numpy as np
from PIL import Image
RUN=Path(__file__).resolve().parent;SRC=Path('/root/autodl-tmp/data/Cityscapes');OUT=RUN/'source_cityscapes'
LUT=np.full(256,255,np.uint8)
for i,c in enumerate([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]):LUT[c]=i

def convert(p):
 rel=p.relative_to(SRC/'gtFine');target=OUT/'gtFine'/str(rel).replace('_labelIds.png','_labelTrainIds.png');target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  raw=np.asarray(Image.open(p));assert raw.ndim==2
  lab=LUT[raw.astype(np.uint8)];Image.fromarray(lab).save(target,compress_level=1)
 return str(rel),hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 if not (OUT/'leftImg8bit').exists():(OUT/'leftImg8bit').symlink_to(SRC/'leftImg8bit',target_is_directory=True)
 files=sorted((SRC/'gtFine/train').rglob('*_gtFine_labelIds.png'));assert len(files)==2975
 start=time.perf_counter();results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,res in enumerate(pool.map(convert,files),1):
   results.append(res)
   if i%250==0:print('SOURCE_CONVERTED',i,flush=True)
 (OUT/'provenance.json').write_text(json.dumps({'raw_source':str(SRC),'images':len(results),'mapping':LUT.tolist(),'input_sha256':dict(results),'elapsed_s':time.perf_counter()-start},indent=2));print('SOURCE_COMPLETE',len(results),flush=True)
