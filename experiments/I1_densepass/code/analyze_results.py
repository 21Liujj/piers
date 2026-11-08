"""Paired image bootstrap and consolidated accuracy tables (no model inference)."""
from pathlib import Path
from collections import defaultdict
import json,csv,numpy as np
from cooperation import scores
RUN=Path(__file__).resolve().parent

def vec_miou(h):
 n=h.shape[-2];tp=np.diagonal(h[...,:n],axis1=-2,axis2=-1);den=h.sum(-1)+h[...,:n].sum(-2)-tp
 iou=np.divide(tp,den,out=np.full_like(tp,np.nan,dtype=float),where=den>0)
 return np.nanmean(iou,axis=-1)*100

def paired_hist_boot(a,b,seed=20261107,n=5000):
 # a,b: seeds x images x classes x(classes+1). The same image draw is used in every seed.
 if a.ndim==3:a=a[None];b=b[None]
 rng=np.random.default_rng(seed);deltas=[];N=a.shape[1]
 for start in range(0,n,100):
  ix=rng.integers(0,N,size=(min(100,n-start),N));da=[]
  for s in range(a.shape[0]):
   ah=a[s][ix].sum(1);bh=b[s][ix].sum(1);da.append(vec_miou(bh)-vec_miou(ah))
  deltas.extend(np.mean(da,axis=0).tolist())
 point=float(np.mean(vec_miou(b.sum(1))-vec_miou(a.sum(1))))
 return {'delta_mIoU_pp':point,'image_bootstrap_95CI_pp':np.percentile(deltas,[2.5,97.5]).tolist(),'bootstrap_draws':n,'resampling_unit':'image (all queries/seed predictions kept together)','training_seed_population_uncertainty_estimated':False}

def text_cluster_boot(base_rows,new_rows):
 base={x['query_id']:x for x in base_rows};new={x['query_id']:x for x in new_rows};ids=sorted({x['image_id'] for x in base_rows});stats=[]
 for iid in ids:
  qs=[q for q,x in base.items() if x['image_id']==iid and x['positive']]
  stats.append([sum(base[q]['iou'] for q in qs),sum(new[q]['iou'] for q in qs),len(qs)])
 arr=np.array(stats);rng=np.random.default_rng(20261107);d=[]
 for _ in range(5000):
  x=arr[rng.integers(0,len(arr),len(arr))].sum(0);d.append(100*(x[1]-x[0])/max(x[2],1))
 point=100*(arr[:,1].sum()-arr[:,0].sum())/arr[:,2].sum()
 return {'positive_Mask_IoU_delta_pp':float(point),'image_bootstrap_95CI_pp':np.percentile(d,[2.5,97.5]).tolist(),'resampling_unit':'image, preserving all positive queries in each sampled image','bootstrap_draws':5000}

def main():
 out=RUN/'reports';out.mkdir(exist_ok=True)
 frozen=json.loads((RUN/'frozen_eval/report.json').read_text());final=json.loads((RUN/'final_frozen_report.json').read_text())
 oldrows=json.loads((RUN/'frozen_eval/per_image.json').read_text());newrows=json.loads((RUN/'final_frozen_per_image.json').read_text())
 baseq=[x for r in oldrows for x in r['text_queries'] if x['method']=='T1'];newq=[x for r in newrows for x in r['text_queries'] if x['method']=='T2_final']
 semantic={**frozen['semantic'],'S2_final':final['semantic_S2']};text={**frozen['text'],**final['text']}
 baseh=np.array([r['semantic_hist']['S0'] for r in oldrows]);newh=np.array([r['semantic_hist'] for r in newrows])
 bootstrap={'frozen_semantic_S2_vs_S0':paired_hist_boot(baseh,newh),'final_text_vs_T1':text_cluster_boot(baseq,newq)}
 consolidated={'status':'frozen_results_complete','protocol':json.loads((RUN/'protocol.json').read_text()),'final_method':json.loads((RUN/'final_method.json').read_text()),'semantic_frozen':semantic,'text':text,'bootstrap':bootstrap,'training_screen':None,'trained':None}
 if (RUN/'screens/report.json').exists():consolidated['training_screen']=json.loads((RUN/'screens/report.json').read_text())
 if (RUN/'final_runs/report.json').exists():
  trained=json.loads((RUN/'final_runs/report.json').read_text());consolidated['trained']=trained;consolidated['status']='all_accuracy_complete'
  aa=[];bb=[]
  for seed in [1234,4321]:
   with np.load(RUN/'final_runs'/('R0_seed'+str(seed))/'evaluation/per_image_hist.npz') as z:aa.append(z['hist'])
   with np.load(RUN/'final_runs'/('R1_seed'+str(seed))/'evaluation/per_image_hist.npz') as z:bb.append(z['hist'])
  consolidated['bootstrap']['trained_R1_vs_R0']=paired_hist_boot(np.stack(aa),np.stack(bb))
  consolidated['trained_seed_deltas']={str(seed):trained['metrics']['R1_seed'+str(seed)]['mIoU']-trained['metrics']['R0_seed'+str(seed)]['mIoU'] for seed in [1234,4321]}
  consolidated['trained_mean_mIoU']={v:float(np.mean([trained['metrics'][v+'_seed'+str(seed)]['mIoU'] for seed in [1234,4321]])) for v in ['R0','R1']}
 if (RUN/'allhard_runs/report.json').exists():
  diag=json.loads((RUN/'allhard_runs/report.json').read_text());consolidated['allhard_diagnostic']=diag
  with np.load(RUN/'allhard_runs/R0_seed1234/evaluation/per_image_hist.npz') as z:ah=z['hist']
  with np.load(RUN/'allhard_runs/R1_seed1234/evaluation/per_image_hist.npz') as z:bh=z['hist']
  consolidated['bootstrap']['allhard_single_seed_R1_vs_R0']=paired_hist_boot(ah,bh)
  consolidated['allhard_diagnostic']['scope']='Post-main diagnostic, one seed only; full hard teacher labels rather than confidence-masked labels. Does not replace main two-seed comparison.'
 consolidated['performance']=json.loads((RUN/'performance.json').read_text())
 consolidated['evaluation_protocol_audit']=json.loads((out/'evaluation_protocol_audit.json').read_text())
 consolidated['recovered_supervision_dev_audit']=json.loads((RUN/'recovered_quality_dev.json').read_text())
 consolidated['acceptance']=json.loads((out/'acceptance.json').read_text())
 consolidated['protocol']['training_budget']='Frozen before trained-model validation:160 target panoramas,160 sampled source panoramas,1440 actual optimizer updates per arm and seed; seeds1234,4321.'
 consolidated['status']='implementation_and_short_controlled_experiments_complete_no_training_gain'
 (out/'RESULTS.json').write_text(json.dumps(consolidated,indent=2))
 with (out/'semantic_frozen.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['method','mIoU','mAcc','PixAcc']+[f'IoU_class{i}' for i in range(13)])
  for method,m in semantic.items():w.writerow([method,m['mIoU'],m['mAcc'],m['PixAcc']]+m['per_class_iou_pct'])
 keys=['positive_mask_IoU','positive_mask_Dice','positive_recall','positive_queries','negative_queries','negative_FP_area16','negative_FPR_area16']
 with (out/'text_metrics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['method']+keys)
  for method,m in text.items():w.writerow([method]+[m.get(k) for k in keys])
 if consolidated['trained']:
  with (out/'training_metrics.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['method_seed','mIoU','mAcc','PixAcc']+[f'IoU_class{i}' for i in range(13)])
   for method,m in consolidated['trained']['metrics'].items():w.writerow([method,m['mIoU'],m['mAcc'],m['PixAcc']]+m['per_class_iou_pct'])
 print('ANALYSIS',json.dumps({'status':consolidated['status'],'bootstrap':bootstrap,'trained_mean':consolidated.get('trained_mean_mIoU'),'trained_seed_deltas':consolidated.get('trained_seed_deltas')}),flush=True)
if __name__=='__main__':main()
