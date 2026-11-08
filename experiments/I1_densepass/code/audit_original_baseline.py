from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
from cooperation import confusion,scores
from analyze_results import paired_hist_boot
R=Path(__file__).resolve().parent;ROOT=R.parents[1];REF=ROOT/'runs/I2-complete-20261106_203924/D100_dense'
MAP=np.full(256,255,np.uint8);MAP[:19]=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
config=json.loads((REF/'config.json').read_text());man=json.loads((R/'val100.json').read_text());rawhist=[];legacyhist=[];newhist=[]
for i,im in enumerate(man['images']):
 name=Path(im['path']).name;assert name==config['file_names'][i]
 raw=MAP[np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+name.replace('_.png','_labelTrainIds.png')))]
 up=np.asarray(Image.fromarray(raw).resize((4096,1024),Image.Resampling.NEAREST));legacy=cv2.resize(up,(2048,400),interpolation=cv2.INTER_NEAREST)
 pred=np.asarray(Image.open(REF/'predictions'/(str(i).zfill(4)+'.png')))
 new=np.asarray(Image.open(R/'final_frozen_predictions/S2'/(im['image_id']+'.png')))
 rawhist.append(confusion(pred,raw));legacyhist.append(confusion(pred,legacy));newhist.append(confusion(new,raw))
a=np.stack(rawhist);b=np.stack(newhist)
report={'status':'complete','baseline_definition':'Local same-checkpoint uncompressed dense-memory OmniSAM-S reproduction; fork provenance is not proof of exact untouched official code. Paper numeric result separately cited.','paper':{'url':'https://arxiv.org/html/2503.07098v2#S4.T3','OmniSAM_S_with_MA_mIoU':60.23,'OmniSAM_B_with_MA_mIoU':62.46,'note':'S checkpoint cannot be claimed as B model. Published numbers are references, not same-run paired comparison.'},'dense_original_GT':scores(a.sum(0)),'dense_legacy_GT':scores(np.stack(legacyhist).sum(0)),'joint_I1_I2_original_GT':scores(b.sum(0)),'joint_vs_dense_image_bootstrap':paired_hist_boot(a,b),'dense_reference_dir':str(REF),'checkpoint_sha256':config['checkpoint_sha256']}
for k in ['R0','R1']:
 report['trained_'+k+'_vs_dense_original_GT_pp']=json.loads((R/'reports/RESULTS.json').read_text())['trained_mean_mIoU'][k]-report['dense_original_GT']['mIoU']
np.savez_compressed(R/'reports/original_dense_hist.npz',hist=a)
(R/'reports/original_baseline_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['dense_original_GT','dense_legacy_GT','joint_I1_I2_original_GT']},indent=2));print('MIOU',{k:report[k]['mIoU'] for k in ['dense_original_GT','dense_legacy_GT','joint_I1_I2_original_GT']})
