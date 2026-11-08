from pathlib import Path
import json,argparse,numpy as np
from PIL import Image
from cooperation import confusion,scores
RUN=Path(__file__).resolve().parent
MAP=np.full(256,255,np.uint8);MAP[:19]=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
def score(cache,out):
 man=json.loads((RUN/'val100.json').read_text());hists=[]
 for im in man['images']:
  with np.load(Path(cache)/(im['image_id']+'.npz')) as z:pred=z['label']
  name=Path(im['path']).name.replace('_.png','_labelTrainIds.png');gt=MAP[np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+name))]
  hists.append(confusion(pred,gt))
 metrics=scores(np.stack(hists).sum(0));out=Path(out);out.mkdir(exist_ok=True,parents=True)
 np.savez_compressed(out/'per_image_hist.npz',hist=np.stack(hists));(out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();print(json.dumps(score(a.cache,a.out)))
