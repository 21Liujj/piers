from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
from cooperation import confusion,scores
ROOT=Path('/root/autodl-tmp/piers-main');RUN=Path(__file__).resolve().parent;REF=ROOT/'runs/I2-complete-20261106_203924/D100_ca_unc_1024'
MAP=np.full(256,255,np.uint8);MAP[:19]=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
config=json.loads((REF/'config.json').read_text());man=json.loads((RUN/'val100.json').read_text());stats={k:np.zeros((13,14),np.int64) for k in ['archived_pred_raw_GT','archived_pred_roundtrip_GT','current_pred_raw_GT','current_pred_roundtrip_GT']};changed_gt=0;changed_pred=0
for i,im in enumerate(man['images']):
 name=Path(im['path']).name;assert name==config['file_names'][i]
 raw=MAP[np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+name.replace('_.png','_labelTrainIds.png')))]
 up=np.asarray(Image.fromarray(raw).resize((4096,1024),Image.Resampling.NEAREST));legacy=cv2.resize(up,(2048,400),interpolation=cv2.INTER_NEAREST)
 old=np.asarray(Image.open(REF/'predictions'/(str(i).zfill(4)+'.png')))
 with np.load(RUN/'cache_val_v2'/(im['image_id']+'.npz')) as z:new=z['label']
 changed_gt+=int((raw!=legacy).sum());changed_pred+=int((old!=new).sum())
 for key,pred,gt in [('archived_pred_raw_GT',old,raw),('archived_pred_roundtrip_GT',old,legacy),('current_pred_raw_GT',new,raw),('current_pred_roundtrip_GT',new,legacy)]:stats[key]+=confusion(pred,gt)
report={'status':'complete','images':100,'GT_pixels_changed_by_legacy_resize_roundtrip':changed_gt,'prediction_pixels_different_between_archived_and_current':changed_pred,'metrics':{k:scores(h) for k,h in stats.items()},'legacy_alignment':'PIL nearest upsample to4096x1024, then cv2 INTER_NEAREST downsample to2048x400, matching sealed validation._align_gt_hw_to_pred_hw','archived_sealed_mIoU':json.loads((REF/'result.json').read_text())['metrics']['mIoU'],'current_policy':'All new comparisons use untouched original2048x400 GT after class mapping; no upsample-downsample roundtrip. The sealed I2 artifacts and source are preserved.'}
(RUN/'reports/evaluation_protocol_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='metrics'}));print('AUDITED_MIOU',{k:v['mIoU'] for k,v in report['metrics'].items()})
