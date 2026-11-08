"""Development-only quality audit; GT is never passed to the revision function."""
from pathlib import Path
import json,concurrent.futures,numpy as np
from PIL import Image
from evaluate_frozen import RUN,MAN,Q,MAP,load_old
from cooperation import prepare,semantic_revision

def one(im):
 iid=im['image_id'];meta,masks=load_old(iid)
 with np.load(RUN/'cache_val_v2'/(iid+'.npz')) as z:hard=z['label'];p=z['probs'].astype(np.float32)
 p/=np.maximum(p.sum(0,keepdims=True),1e-8);base=hard.copy();base[p.max(0)<.85]=255
 records=prepare(meta,masks,p,Q);rev,stats=semantic_revision(base,p,records,mode='gated')
 name=Path(im['path']).name.replace('_.png','_labelTrainIds.png');gt=MAP[np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+name))]
 new=(base==255)&(rev!=255)&(gt!=255);same=new&(rev==hard);rows=[]
 for cls in range(13):
  m=new&(rev==cls);rows.append({'class':cls,'pixels':int(m.sum()),'correct':int((m&(rev==gt)).sum()),'teacher_hard_correct':int((m&(hard==gt)).sum())})
 valid=(base!=255)&(gt!=255)
 return {'image_id':iid,'baseline_valid':int(valid.sum()),'baseline_correct':int((valid&(base==gt)).sum()),'new_pixels':int(new.sum()),'new_correct':int((new&(rev==gt)).sum()),'teacher_hard_correct_on_new':int((new&(hard==gt)).sum()),'same_as_teacher_hard':int(same.sum()),'by_class':rows}
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,MAN['images']))
 totals={k:sum(x[k] for x in rows) for k in ['baseline_valid','baseline_correct','new_pixels','new_correct','teacher_hard_correct_on_new','same_as_teacher_hard']}
 totals['baseline_precision']=totals['baseline_correct']/max(totals['baseline_valid'],1);totals['recovered_precision']=totals['new_correct']/max(totals['new_pixels'],1);totals['teacher_hard_precision_on_recovered']=totals['teacher_hard_correct_on_new']/max(totals['new_pixels'],1)
 classes=[]
 for cls in range(13):
  pixels=sum(r['by_class'][cls]['pixels'] for r in rows);correct=sum(r['by_class'][cls]['correct'] for r in rows)
  classes.append({'class':cls,'pixels':pixels,'correct':correct,'precision':correct/max(pixels,1)})
 report={'status':'complete','scope':'DensePASS100 development-only pseudo quality diagnosis, not training-set correctness','totals':totals,'by_class':classes,'rows':rows}
 (RUN/'recovered_quality_dev.json').write_text(json.dumps(report,indent=2));print('RECOVERED_QUALITY',json.dumps(totals),flush=True)
