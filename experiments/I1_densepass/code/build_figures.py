from pathlib import Path
import json,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from cooperation import scores
RUN=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/piers-main');OUT=RUN/'reports/figures';OUT.mkdir(parents=True,exist_ok=True)
MAN=json.loads((RUN/'val100.json').read_text());IMS={x['image_id']:x for x in MAN['images']};Q={x['query_id']:x for x in MAN['queries']}
FINE={'car':13,'person':11,'road':0,'vegetation':8,'sky':10,'train':16};MAP=np.full(256,255,np.uint8);MAP[:19]=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
OLD=json.loads((RUN/'frozen_eval/per_image.json').read_text());NEW=json.loads((RUN/'final_frozen_per_image.json').read_text())
BASE={x['query_id']:x for r in OLD for x in r['text_queries'] if x['method']=='T1'};FINAL={x['query_id']:x for r in NEW for x in r['text_queries'] if x['method']=='T2_final'}

def overlay(rgb,mask,color):
 x=rgb.copy().astype(float);x[mask]=.5*x[mask]+.5*np.array(color);return np.clip(x,0,255).astype(np.uint8)
def bbox_component(mask,pad=48):
 if not mask.any():return (0,0,2048,400)
 lab,n=ndimage.label(mask);sizes=np.bincount(lab.flatten());sizes[0]=0;y,x=np.where(lab==sizes.argmax())
 return max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(2048,int(x.max())+pad+1),min(400,int(y.max())+pad+1)
def query_mask(root,iid,qid):
 with np.load(root/(iid+'.npz')) as z:return np.unpackbits(z[qid],axis=1,count=2048).astype(bool)

positive=[qid for qid,x in FINAL.items() if x['positive']]
best=max(positive,key=lambda q:FINAL[q]['iou']-BASE[q]['iou']);worst=min(positive,key=lambda q:FINAL[q]['iou']-BASE[q]['iou'])
neg=[qid for qid,x in FINAL.items() if not x['positive']];badneg=max(neg,key=lambda q:FINAL[q]['pred_pixels']);empty=next((q for q in sorted(neg) if FINAL[q]['pred_pixels']==0),sorted(neg)[0]);small=min([q for q in positive if FINAL[q]['text']=='person' and FINAL[q]['gt_pixels']>=16],key=lambda q:FINAL[q]['gt_pixels'])
seam=None
for meta in sorted((ROOT/'runs/I1-collab-20261106_213607/t1_candidate_shards').glob('*/*/metadata.json')):
 m=json.loads(meta.read_text())
 for c in m['candidates']:
  q=c['query_id']
  if c.get('crosses') and q in FINAL and FINAL[q]['text'] in ['car','person']:
   seam=q;break
 if seam:break
if seam is None:seam=next(q for q in positive if FINAL[q]['text']=='road')
cases=[('positive_gain',best),('negative_gain_or_failure',worst),('verified_negative_false_positive',badneg),('verified_negative_empty',empty),('small_person',small),('wrapped_FoV_category',seam)]
case_records=[]
for name,qid in cases:
 q=Q[qid];iid=q['image_id'];im=IMS[iid];rgb=np.array(Image.open(im['path']).convert('RGB'));label=Path('/root/autodl-tmp/data/DensePASS/gtFine/val')/Path(im['path']).name.replace('_.png','_labelTrainIds.png');raw=np.asarray(Image.open(label));gt=raw==FINE[FINAL[qid]['text']]
 t1=query_mask(RUN/'frozen_predictions/T1',iid,qid);t2=query_mask(RUN/'final_frozen_predictions',iid,qid)
 roll=1024 if name=='wrapped_FoV_category' else 0
 if roll:rgb=np.roll(rgb,roll,1);gt=np.roll(gt,roll,1);t1=np.roll(t1,roll,1);t2=np.roll(t2,roll,1)
 focus=gt if name=='small_person' else (t1^t2)
 if not focus.any():focus=t2 if t2.any() else gt
 x1,y1,x2,y2=bbox_component(focus);sl=(slice(y1,y2),slice(x1,x2))
 fig=plt.figure(figsize=(14,6));gs=fig.add_gridspec(2,3,height_ratios=[1,2]);ax=fig.add_subplot(gs[0,:]);ax.imshow(rgb);ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='yellow',linewidth=1.5));ax.axis('off');ax.set_title(name+' | '+q['raw_text']+' | '+iid,fontsize=11)
 panes=[('Original19 category GT',gt,(30,220,50)),('T1 independent',t1,(250,90,50)),('T2 reliable fusion',t2,(30,140,250))]
 for j,(title,mask,color) in enumerate(panes):
  a=fig.add_subplot(gs[1,j]);a.imshow(overlay(rgb,mask,color)[sl]);a.axis('off');a.set_title(title)
 a1=BASE[qid]['iou'];a2=FINAL[qid]['iou'];footer='Verified negative category; no instance GT claim.' if a1 is None else f'Whole-query Mask IoU: T1={100*a1:.2f}%, T2={100*a2:.2f}%. Crops/roll are visualization only.'
 fig.text(.5,.02,footer,ha='center',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/(name+'.png'),dpi=130);plt.close(fig)
 case_records.append({'case':name,'query_id':qid,'image_id':iid,'visual_crop':[x1,y1,x2,y2],'display_roll_x':roll,'baseline':BASE[qid],'final':FINAL[qid]})
(RUN/'reports/cases.json').write_text(json.dumps(case_records,indent=2))
# Qualitative arbitrary-language masks, including parts and relations with no invented GT.
qual=json.loads((RUN/'free_text_qual.json').read_text());fig,axes=plt.subplots(3,4,figsize=(16,7))
for row,im in enumerate(qual['images']):
 rgb=np.array(Image.open(im['path']).convert('RGB'));qs=[q for q in qual['queries'] if q['image_id']==im['image_id']]
 for col,q in enumerate(qs):
  mask=np.asarray(Image.open(RUN/'free_text_outputs'/im['image_id']/(q['query_id']+'.png')))>0
  axes[row,col].imshow(overlay(rgb,mask,(40,170,250)));axes[row,col].set_title(q['raw_text'],fontsize=9);axes[row,col].axis('off')
fig.suptitle('Real free-text outputs: independent fallback for unmapped queries; qualitative only',fontsize=13);fig.tight_layout();fig.savefig(OUT/'free_text_qualitative.png',dpi=130);plt.close(fig)
# A source-free target training pseudo-label illustration: selected by recovery proxy, not GT.
train=json.loads((RUN/'train160.json').read_text());pro=json.loads((RUN/'pseudo_final/report.json').read_text());chosen=max(pro['rows'],key=lambda x:x['recovered_ignore']);im=next(x for x in train['images'] if x['image_id']==chosen['image_id']);rgb=np.array(Image.open(im['path']).convert('RGB'));rel=im['src_file'].replace('.jpg','_labelTrainIds.png');r0=np.asarray(Image.open(RUN/'pseudo_final/R0/epoch1/gtFine'/rel));r1=np.asarray(Image.open(RUN/'pseudo_final/R1/epoch1/gtFine'/rel));new=(r0==255)&(r1!=255)
fig,axes=plt.subplots(3,1,figsize=(14,8));axes[0].imshow(rgb);axes[0].set_title('Target TRAIN image; no target GT used: '+im['src_file']);axes[1].imshow(overlay(rgb,r0==255,(130,130,130)));axes[1].set_title('R0 ignored supervision (gray)');axes[2].imshow(overlay(rgb,new,(255,220,20)));axes[2].set_title('R1 recovered supervision (yellow); correctness unobserved on train')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'training_pseudo_recovery.png',dpi=130);plt.close(fig)
# Paired seed1234 semantic success/failure, explicitly labelled as one seed.
palette=np.array(json.loads((Path('/root/autodl-tmp/dataset/densepass_list/info.json')).read_text())['palette'],np.uint8)[list(range(12))+[13]]
def color(a):
 out=np.full((*a.shape,3),240,np.uint8);v=a<13;out[v]=palette[a[v]];return out
if (RUN/'final_runs/R1_seed1234/evaluation/per_image_hist.npz').exists():
 with np.load(RUN/'final_runs/R0_seed1234/evaluation/per_image_hist.npz') as z:h0=z['hist']
 with np.load(RUN/'final_runs/R1_seed1234/evaluation/per_image_hist.npz') as z:h1=z['hist']
 deltas=[scores(b)['mIoU']-scores(a)['mIoU'] for a,b in zip(h0,h1)]
 for label,index in [('semantic_training_best',int(np.argmax(deltas))),('semantic_training_worst',int(np.argmin(deltas)))]:
  im=MAN['images'][index];iid=im['image_id'];rgb=np.array(Image.open(im['path']).convert('RGB'));gn=Path(im['path']).name.replace('_.png','_labelTrainIds.png');gt=MAP[np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+gn))]
  with np.load(RUN/'cache_val_v2'/(iid+'.npz')) as z:s0=z['label']
  with np.load(RUN/'final_runs/R0_seed1234/cache'/(iid+'.npz')) as z:r0=z['label']
  with np.load(RUN/'final_runs/R1_seed1234/cache'/(iid+'.npz')) as z:r1=z['label']
  fig,axes=plt.subplots(5,1,figsize=(14,12));panes=[('Input',rgb),('Complete project13 GT',color(gt)),('Frozen S0',color(s0)),('Trained R0 (seed1234)',color(r0)),('Trained R1 (seed1234)',color(r1))]
  for ax,(title,pane) in zip(axes,panes):ax.imshow(pane);ax.set_title(title);ax.axis('off')
  fig.suptitle('Fixed-I2 seed1234 | '+label+' | '+iid+f' | per-image mIoU delta={deltas[index]:+.3f}pp');fig.tight_layout();fig.savefig(OUT/(label+'.png'),dpi=120);plt.close(fig)
# Complete measured training loss, not a proxy for evaluation accuracy.
fig,ax=plt.subplots(figsize=(10,4))
for v in ['R0','R1']:
 f=RUN/'final_runs'/(v+'_seed1234')/'training/updates.jsonl'
 if f.exists():
  rows=[json.loads(x) for x in f.read_text().splitlines()];ys=np.array([x['loss'] for x in rows]);smooth=np.convolve(ys,np.ones(31)/31,mode='valid');ax.plot(np.arange(len(smooth))+16,smooth,label=v)
ax.set(xlabel='Optimizer update',ylabel='UDA loss (31-update mean)',title='Fixed-I2 seed1234 continuation loss; accuracy evaluated separately');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'training_loss.png',dpi=150);plt.close(fig)
print('FIGURES_COMPLETE',len(list(OUT.glob('*.png'))))
