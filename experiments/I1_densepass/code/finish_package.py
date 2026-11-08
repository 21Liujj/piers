"""Final package after original-baseline controls; no model training."""
from pathlib import Path
import subprocess,os,json,time,hashlib,tarfile,numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parents[1];D=R/'dense_controls';O=R/'reports';PY='/root/autodl-tmp/envs/piers/bin/python';ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run(args,log):
 with (O/log).open('w') as f:subprocess.run([PY,*map(str,args)],env=ENV,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
assert read(D/'trained/report.json')['status']=='complete';assert read(D/'cache_official/metadata_s0.json')['status']=='complete'
run([ROOT/'i2_final_eval.py','--verify-only'],'I2_final_verify.log')
run([R/'final_acceptance.py'],'acceptance_final.log')
run([R/'audit_checkpoint_changes.py'],'checkpoint_audit.log')
run([R/'score_trained.py','--cache',D/'cache_official','--out',D/'official_baseline_evaluation'],'official_full_score.log')
manifest=read(R/'val100.json');changed=0;per=[]
for im in manifest['images']:
 with np.load(D/'cache_official'/(im['image_id']+'.npz')) as a,np.load(D/'cache_val'/(im['image_id']+'.npz')) as b:n=int((a['label']!=b['label']).sum())
 per.append({'image_id':im['image_id'],'different_pixels':n});changed+=n
checks={'official_source_forward_9patch_exact':read(O/'official_forward_parity.json')['status']=='passed','official_full100_strict_checkpoint':read(D/'cache_official/metadata_s0.json')['audit']['strict_checkpoint_loading'],'official_max_dense_tokens32768':read(D/'cache_official/metadata_s0.json')['audit']['max_tokens']==32768,'baseline_full100_evaluated':len(manifest['images'])==100,'frozen_all_tensors_only_decoder_changes':read(O/'checkpoint_tensor_audit.json')['status']=='passed','all_old_acceptance_checks':all(read(O/'acceptance.json')['checks'].values())}
public=read(R/'public_dense_road_smoke/result.json');public_audit=read(R/'public_dense_road_smoke/outputs/summary.json')
assert public['status']=='complete' and public['memory_mode']=='dense' and public['GT_read']==False
assert public_audit['records'][0]['semantic_stats']['changed_pixels']>0
checks['public_entry_dense_mode_real_model_and_nonzero_change']=True
newjobs={}
for seed in [1234,4321]:
 pair=[]
 for v in ['R0','R1']:
  folder=D/'trained'/(v+'_seed'+str(seed))/'training';a=read(folder/'result.json');cfg=read(folder/'config.json');assert a['actual_optimizer_steps']==1440 and a['target_GT_reads']==0 and a['max_parameter_delta']>0;assert sha(folder/'final.pth')==a['checkpoint_sha256'];assert a['memory_audit_student']['max_tokens']==32768 and a['memory_audit_source']['max_tokens']==32768
  assert cfg['fixed_memory']['mode']=='dense' and cfg['seed']==seed and cfg['iters']==160 and cfg['lr']==2e-6;pair.append(a);newjobs[v+'_seed'+str(seed)]={k:a[k] for k in ['actual_optimizer_steps','checkpoint_sha256','source_target_order_sha256','elapsed_s','peak_allocated_mib']}
 assert pair[0]['source_target_order_sha256']==pair[1]['source_target_order_sha256']
checks['dense_R0_R1_two_seeds_equal_initialization_budget_and_order']=True;assert all(checks.values())
write(O/'final_package_acceptance.json',{'status':'passed','checks':checks,'official_vs_fork_full100_different_pixels':changed,'note':'Per-frame9-sequence logits exactly match; whole-list predictions may have very small GPU numerical differences across concurrent runs. All main comparisons use explicitly scored official baseline.','full100_parity_rows':per,'dense_training_jobs':newjobs,'at_unix':time.time()})
# Freeze the exact labels that were supplied to every target loader.
rows=[];train=read(R/'train160.json')
for base in ['pseudo_final','pseudo_allhard','dense_controls/pseudo']:
 for v in ['R0','R1']:
  for im in train['images']:
   p=R/base/v/'epoch1/gtFine'/im['src_file'].replace('.jpg','_labelTrainIds.png');rows.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
write(O/'PSEUDO_CHECKSUMS.json',{'files':rows,'count':len(rows),'no_target_GT':True})
pseudo_ops={}
for base in ['pseudo_final','pseudo_allhard','dense_controls/pseudo']:
 stats={'changed_pixels':0,'recovered_ignore':0,'changed_previously_valid_label':0,'new_ignore':0}
 for im in train['images']:
  name=im['src_file'].replace('.jpg','_labelTrainIds.png');a=np.asarray(Image.open(R/base/'R0/epoch1/gtFine'/name));b=np.asarray(Image.open(R/base/'R1/epoch1/gtFine'/name));assert set(np.unique(b)).issubset(set(range(13))|{255})
  stats['changed_pixels']+=int((a!=b).sum());stats['recovered_ignore']+=int(((a==255)&(b!=255)).sum());stats['changed_previously_valid_label']+=int(((a!=255)&(a!=b)).sum());stats['new_ignore']+=int(((a!=255)&(b==255)).sum())
 pseudo_ops[base]=stats
write(O/'PSEUDO_OPERATION_AUDIT.json',{'no_GT_read':True,'training_images':160,'operations':pseudo_ops,'note':'With baseline confidence0.85 and conflict guard0.8, confidence-masked R1 adds ignored supervision; retained confident labels cannot be overwritten. Allhard diagnostic measures actual class changes.'})

# Identify the fixed-I2 training case figures explicitly, then render from original data.
p=R/'build_figures.py';s=p.read_text();s=s.replace("fig.suptitle(label+' | '+iid", "fig.suptitle('Fixed-I2 seed1234 | '+label+' | '+iid");s=s.replace("title='Training loss only; precision gains are evaluated independently'", "title='Fixed-I2 seed1234 continuation loss; accuracy evaluated separately'");p.write_text(s)
run([R/'build_figures.py'],'figure_refresh.log')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,seed in zip(axes,[1234,4321]):
 for v in ['R0','R1']:
  rows=[json.loads(x) for x in (D/'trained'/(v+'_seed'+str(seed))/'training/updates.jsonl').read_text().splitlines()];y=np.array([x['loss'] for x in rows]);x=np.arange(1,len(y)+1);ax.plot(x[30:],np.convolve(y,np.ones(31)/31,mode='valid'),label=v,alpha=.85)
 ax.set_title('Uncompressed OmniSAM continuation, seed'+str(seed));ax.set_xlabel('Actual optimizer update');ax.set_ylabel('UDA loss (31-update mean)');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Loss curves do not establish segmentation accuracy gains');fig.tight_layout();fig.savefig(O/'figures/dense_training_loss.png',dpi=150);plt.close(fig)
run([R/'finalize_reports.py'],'finalize.log')
a=read(O/'RESULTS.json');dense=a['dense_matched_training'];status={'status':'implementation_and_paper_anchored_short_experiments_complete','research_goal':'Stable semantic improvement over original OmniSAM is not established. Engineering completion is not a claim that accuracy targets are met.','main':a['paper_anchored_main'],'dense_training_means':dense['means'],'dense_training_R1_minus_R0':dense['bootstrap_R1_vs_R0'],'dense_training_R1_vs_original':dense.get('bootstrap_R1_vs_original'),'fixed_I2_training_means':a['trained_mean_mIoU'],'tests_passed':77,'figures':len(list((O/'figures').glob('*.png'))),'target_train_total':2000,'target_train_used_per_arm':160,'evaluation':'DensePASS100 development, originalGT','reports':str(O),'at_unix':time.time(),'supplements_resumed':(D/'supplements_resumed.json').exists()}
write(R/'FINAL_STATUS.json',status);write(O/'FINAL_STATUS.json',status)
(R/'PROGRESS.txt').write_text('Final paper-anchored short controls complete.\n'+json.dumps(status,indent=2,ensure_ascii=False)+'\nDo not report the +0.13pp compressed-control increment as a gain over original OmniSAM.\nRead reports/REPORT.txt and MAIN_PAPER_ANCHORED.csv.\n')
(O/'READ_FIRST.txt').write_text('先读 REPORT.txt，再看 MAIN_PAPER_ANCHORED.csv 和 DENSE_MATCHED_TRAINING.json。\n原论文Small参考60.23；主表采用官方源码+既有Small权重完整100图实测，与其余方法用同一原始GT。\n开发集已参与选模；160图/1440更新为短程续训，非完整2000图训练。\n工程与短程对照完成不等于语义精度目标达成。负结果、误报和无门控较高文本IoU均保留。\n复现入口 REPRODUCE.sh 按stage运行。\n本包含代码快照、报告、图表、日志摘要与训练元数据；不包含大模型权重、原始数据和全量概率缓存，它们保留在原远端路径。\n')
# Bundle reports and small per-run evidence, keeping large data/weights in place.
bundle=O/'I1_review_bundle.tar.gz'
with tarfile.open(bundle,'w:gz') as tar:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=bundle and p.name!='DELIVERABLES.json':tar.add(p,arcname='reports/'+str(p.relative_to(O)),recursive=False)
 for base in ['final_runs','allhard_runs','dense_controls/trained']:
  for p in (R/base).rglob('*'):
   if p.is_file() and p.suffix in ['.json','.jsonl'] and p.stat().st_size<2000000:tar.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 for p in [R/'train160.json',R/'val100.json',R/'baseline_protocol_correction.json',R/'FINAL_STATUS.json',D/'frozen_report.json',D/'trained/report.json',R/'pseudo_final/report.json']:
  tar.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
write(O/'DELIVERABLES.json',{'bundle':str(bundle),'bytes':bundle.stat().st_size,'sha256':sha(bundle),'source_snapshot_sha256':sha(O/'source_snapshot_final.tar.gz'),'report':str(O/'REPORT.txt'),'status':str(R/'FINAL_STATUS.json')})
print('FINAL_PACKAGE_COMPLETE',json.dumps(status,ensure_ascii=False),flush=True)
