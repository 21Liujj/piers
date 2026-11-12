"""Publication figures from completed matched indoor experiment records."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

CLASSES=['ceiling','chair','door','floor','sofa','table','wall','window']
COLORS=['#91c8e4','#eb745b','#b79bd5','#b8cd82','#efb75a','#73c2b2','#c9c9c9','#6989c3']

def save(fig,target,name):
    fig.savefig(target/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(target/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def main(a):
    root=Path(a.run);target=root/'figures';target.mkdir(exist_ok=True)
    result=json.loads((root/'INDOOR_MATCHED_RESULTS.json').read_text())
    assert result['status']=='complete' and result['n_images']==373
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    original=result['arms']['original'];full=result['arms']['full']
    fig,ax=plt.subplots(figsize=(7.4,3.8));x=np.arange(8);width=.36
    bars1=ax.bar(x-width/2,[original['per_class_iou_pct'][c] for c in CLASSES],width,label='Original OmniSAM',color='#a5b8c8')
    bars2=ax.bar(x+width/2,[full['per_class_iou_pct'][c] for c in CLASSES],width,label='Full I1 + I2 + I3',color='#236795')
    ax.set_xticks(x,CLASSES);ax.set_ylim(0,110);ax.set_ylabel('IoU (%)')
    ax.set_title('Indoor SPAN8 adaptation   Same checkpoint and LoRA\n373 test panoramas',loc='left',pad=16,fontsize=10)
    ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.bar_label(bars1,fmt='%.1f',padding=3,fontsize=8)
    for left,right in zip(bars1,bars2):
        height=right.get_height()
        padding=15 if max(left.get_height(),height)>=10 and abs(left.get_height()-height)<6 else 3
        ax.annotate(f'{height:.1f}',(right.get_x()+right.get_width()/2,height),
                    xytext=(0,padding),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    save(fig,target,'indoor_per_class_iou')
    fig,ax=plt.subplots(figsize=(7.5,4.5));names=['Original OmniSAM','I2 + I3 core','Full I1 + I2 + I3']
    values=[result['arms'][k]['mIoU_pct'] for k in ['original','full_core','full']]
    bars=ax.bar(names,values,color=['#a5b8c8','#6b9cbb','#236795'],width=.58)
    ax.set_ylim(0,max(values)*1.3);ax.set_ylabel('mIoU (%)');ax.bar_label(bars,fmt='%.2f',padding=5)
    lo,hi=result['paired_image_bootstrap_delta95CI_pp'];delta=result['delta_full_minus_original_mIoU_pp']
    ax.set_title(f'Full minus original: {delta:+.2f} pp\nImage bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]',loc='left',fontsize=10,pad=16)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,target,'indoor_main_comparison')
    log=Path(a.training_log)
    records=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
    fig,ax=plt.subplots(figsize=(8,3.7));ax.plot([r['step'] for r in records],[r['loss'] for r in records],color='#236795',lw=1.7)
    ax.set_xlabel('Optimizer step');ax.set_ylabel('Training cross entropy')
    ax.set_title('Stage 1 indoor LoRA and decoder adaptation',loc='left',pad=12)
    ax.text(.99,.94,'Logged rolling mean; fit split only',ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#555555')
    ax.grid(alpha=.18);save(fig,target,'indoor_stage1_training_loss')
    rows=json.loads((root/'inference_heldout.json').read_text())
    labels={r['id']:r for r in [json.loads(s) for s in Path(a.gt_manifest).read_text().splitlines() if s.strip()]}
    palette=np.array([[int(c[i:i+2],16) for i in (1,3,5)] for c in COLORS],np.uint8)
    def colorize(mask):
        rgb=np.full((*mask.shape,3),36,np.uint8);valid=mask<8;rgb[valid]=palette[mask[valid]];return rgb
    # Fixed indices chosen independently of scores, not selected for success.
    selected=[0,124,248];fig,axs=plt.subplots(3,4,figsize=(10,4.8))
    for row_index,index in enumerate(selected):
        row=rows[index];iid=row['image_id'];rec=labels[iid]
        rgb=np.asarray(Image.open(row['path']).convert('RGB'))
        gt=np.asarray(Image.open(Path(a.gt_root)/rec['area']/rec['label']))
        with np.load(root/'semantic/heldout/original'/(iid+'.npz')) as z:pred=z['prediction']
        final=np.load(root/'full/heldout'/(iid+'.npy'))
        for col,image in enumerate([rgb,colorize(gt),colorize(pred),colorize(final)]):
            axs[row_index,col].imshow(image);axs[row_index,col].axis('off')
            if row_index==0:axs[row_index,col].set_title(['ERP image','Ground truth','Original OmniSAM','Full I1 + I2 + I3'][col],fontsize=12)
    legend=[Patch(color=color,label=label) for color,label in zip(COLORS,CLASSES)]
    legend.append(Patch(color='#242424',label='ignore'))
    fig.legend(handles=legend,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=11)
    fig.subplots_adjust(wspace=.02,hspace=.08,bottom=.14)
    save(fig,target,'indoor_fixed_examples')
    (target/'figure_provenance.json').write_text(json.dumps({'result_file':str(root/'INDOOR_MATCHED_RESULTS.json'),
        'qualitative_manifest_indices':selected,'selection':'fixed evenly spaced indices, independent of results',
        'training_curve':'original logged rolling CE, not interpolation or fabricated estimates',
        'classes':CLASSES,'colors':COLORS},indent=2))
    print('Saved four verified-data figures as PNG and SVG.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--training-log',required=True)
    p.add_argument('--gt-manifest',required=True);p.add_argument('--gt-root',default='/root/autodl-tmp/data_extra/SPAN8_PREP')
    main(p.parse_args())
