"""Build transparent paper-anchored evidence package; never runs training or reads GT for inference."""
from pathlib import Path
import json,csv,hashlib,time,datetime,tarfile,subprocess,os
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parents[1];O=R/'reports';D=R/'dense_controls';O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,data):Path(p).write_text(json.dumps(data,indent=2,ensure_ascii=False))
def fmt(x):return f'{x:.4f}'
from analyze_results import paired_hist_boot,text_cluster_boot
A=read(O/'RESULTS.json');P=read(R/'performance.json');B=read(O/'original_baseline_audit.json');C=read(D/'frozen_report.json');acc=read(O/'acceptance.json')
# Main table anchored to the same uncompressed local OmniSAM-S reproduction.
official_metric_path=D/'official_baseline_evaluation/metrics.json'
Official=read(official_metric_path) if official_metric_path.exists() else C['baseline']
with np.load(D/'frozen_hists.npz') as z:dense_i1_hist=z['I1_only']
if (D/'official_baseline_evaluation/per_image_hist.npz').exists():
 with np.load(D/'official_baseline_evaluation/per_image_hist.npz') as z:official_hist=z['hist']
 C={**C,'local_dense_control_baseline':C['baseline'],'baseline':Official,'semantic_bootstrap':paired_hist_boot(official_hist,dense_i1_hist)}
 joint_hist=np.array([x['semantic_hist'] for x in read(R/'final_frozen_per_image.json')]);B={**B,'joint_vs_dense_image_bootstrap':paired_hist_boot(official_hist,joint_hist)}
base=Official['mIoU'];main=[]
for name,metric in [('Original_OmniSAM_S',Official),('Original_plus_I1_only',C['I1_only']),('Original_plus_fixed_I2_only',A['semantic_frozen']['S0']),('Original_plus_I1_and_fixed_I2',A['semantic_frozen']['S2_final'])]:
 main.append({'method':name,'mIoU_pct':metric['mIoU'],'delta_vs_original_pp':metric['mIoU']-base,'stage':'frozen weights; original GT; development100'})
with (O/'MAIN_PAPER_ANCHORED.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(main[0]));w.writeheader();w.writerows(main)
# A paper model has no free-text branch. Compare only exactly shared closed categories as an explicit auxiliary projection baseline.
manifest=read(R/'val100.json');classes={'road':0,'vegetation':8,'sky':10};common={k:[] for k in ['Original_closed_category_projection','T1_independent_text','I1_only_text','I1_I2_text']}
for im in manifest['images']:
 iid=im['image_id'];raw=np.asarray(Image.open('/root/autodl-tmp/data/DensePASS/gtFine/val/'+Path(im['path']).name.replace('_.png','_labelTrainIds.png')));valid=raw!=255;sem=np.asarray(Image.open(D/'predictions'/iid/'baseline.png'))
 if (D/'cache_official'/(iid+'.npz')).exists():
  with np.load(D/'cache_official'/(iid+'.npz')) as z:sem=z['label']
 with np.load(R/'frozen_predictions/T1'/(iid+'.npz')) as t1,np.load(D/'predictions'/iid/'text.npz') as one,np.load(R/'final_frozen_predictions'/(iid+'.npz')) as joint:
  for q in manifest['queries']:
   key=q['raw_text'].strip().lower()
   if q['image_id']!=iid or key not in classes or q['task_type']!='category_union':continue
   cls=classes[key];g=(raw==cls)&valid;gs=int(g.sum());qid=q['query_id']
   masks={'Original_closed_category_projection':sem==cls,'T1_independent_text':np.unpackbits(t1[qid],axis=1,count=2048).astype(bool),'I1_only_text':np.unpackbits(one[qid],axis=1,count=2048).astype(bool),'I1_I2_text':np.unpackbits(joint[qid],axis=1,count=2048).astype(bool)}
   for method,mask in masks.items():
    p=mask&valid;ps=int(p.sum());inter=int((p&g).sum());common[method].append({'image_id':iid,'query_id':qid,'positive':gs>0,'iou':inter/max(gs+ps-inter,1) if gs else None,'dice':2*inter/max(gs+ps,1) if gs else None,'recall':inter/max(gs,1) if gs else None,'false_positive':bool(ps>=16) if not gs else None})
common_summary={}
for method,rows in common.items():
 pos=[x for x in rows if x['positive']];neg=[x for x in rows if not x['positive']]
 common_summary[method]={'queries':len(rows),'positive_queries':len(pos),'negative_queries':len(neg),'positive_Mask_IoU_pct':float(np.mean([x['iou'] for x in pos])*100),'positive_Dice_pct':float(np.mean([x['dice'] for x in pos])*100),'positive_recall_pct':float(np.mean([x['recall'] for x in pos])*100),'negative_FPR_area16':float(np.mean([x['false_positive'] for x in neg])) if neg else None}
write(O/'common_exact_category_projection.json',{'protocol':'Only road/vegetation/sky are exactly shared original categories. Original output projected by category ID, not a free-text capability claim; car/person coarse-merge mismatch and unsupported queries excluded. All originalGT valid pixels, same300 queries.','metrics':common_summary,'I1_vs_original_bootstrap':text_cluster_boot(common['Original_closed_category_projection'],common['I1_only_text'])})
trained_dense=None
if (D/'trained/report.json').exists():
 trained_dense=read(D/'trained/report.json');aa=[];bb=[]
 for seed in [1234,4321]:
  with np.load(D/'trained'/('R0_seed'+str(seed))/'evaluation/per_image_hist.npz') as z:aa.append(z['hist'])
  with np.load(D/'trained'/('R1_seed'+str(seed))/'evaluation/per_image_hist.npz') as z:bb.append(z['hist'])
 trained_dense['bootstrap_R1_vs_R0']=paired_hist_boot(np.stack(aa),np.stack(bb));trained_dense['means']={v:float(np.mean([trained_dense['metrics'][v+'_seed'+str(s)]['mIoU'] for s in [1234,4321]])) for v in ['R0','R1']}
 trained_dense['delta_R1_mean_vs_original_pp']=trained_dense['means']['R1']-base
 if 'official_hist' in globals():trained_dense['bootstrap_R1_vs_original']=paired_hist_boot(np.stack([official_hist,official_hist]),np.stack(bb))
 write(O/'DENSE_MATCHED_TRAINING.json',trained_dense)
 with (O/'dense_training_metrics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['method_seed','mIoU_pct','delta_vs_original_pp'])
  for name,m in trained_dense['metrics'].items():w.writerow([name,m['mIoU'],m['mIoU']-base])
alljobs={}
for p in R.rglob('result.json'):
 try:d=read(p)
 except Exception:continue
 if 'actual_optimizer_steps' in d:alljobs[str(p.relative_to(R))]={k:d.get(k) for k in ['actual_optimizer_steps','elapsed_s','checkpoint_sha256','source_target_order_sha256','peak_allocated_mib','target_GT_reads']}
ledger=[]
for p in R.rglob('metadata_s*.json'):
 try:d=read(p)
 except Exception:continue
 if d.get('status')=='complete' and 'elapsed_s' in d:ledger.append({'type':'semantic_process_completed_slice','artifact':str(p.relative_to(R)),'process_wall_s':d['elapsed_s']})
for p in R.rglob('done_s*.json'):
 try:d=read(p)
 except Exception:continue
 if d.get('status')=='complete' and 'elapsed_s' in d:ledger.append({'type':'text_process_completed_slice','artifact':str(p.relative_to(R)),'process_wall_s':d['elapsed_s']})
for p,d in alljobs.items():ledger.append({'type':'actual_training_process','artifact':p,'process_wall_s':d['elapsed_s']})
write(O/'runtime_ledger.json',{'sum_recorded_process_wall_hours':sum(x['process_wall_s'] for x in ledger)/3600,'meaning':'Sum of recorded successful process intervals, often concurrent. NOT GPU kernel hours, billing hours, or full historical total. Some failures/startup/preexisting T0/T1 cache generation are not accounted; resume slices may omit earlier work.','records':ledger,'training_jobs':alljobs})
# Record actual runtime package versions without model execution.
py='/root/autodl-tmp/envs/piers/bin/python'
env=subprocess.run([py,'-m','pip','freeze'],capture_output=True,text=True,timeout=60);(O/'environment_pip_freeze.txt').write_text(env.stdout)
A['paper_anchored_main']=main;A['official_code_baseline']={'metrics':Official,'metadata':read(D/'cache_official/metadata_s0.json') if (D/'cache_official/metadata_s0.json').exists() else None,'evaluation_dir':str(D/'official_baseline_evaluation')};A['paper_reference']=B['paper'];A['original_baseline_audit']=B;A['dense_I1_only_frozen']=C;A['dense_matched_training']=trained_dense;A['common_exact_category_projection']=common_summary
A['claim']='No robust semantic improvement over original OmniSAM established by frozen inference. Text refinement gains are relative to the independent text branch. Trained results must be judged separately.'
A['status']='paper_anchored_controls_complete' if trained_dense else 'dense_two_seed_training_pending'
write(O/'RESULTS.json',A)
# Exact text category breakdown; all categories use original19 labels.
old=read(R/'frozen_eval/per_image.json');new=read(R/'final_frozen_per_image.json');den=read(D/'frozen_per_image.json');trows=[]
for rec in old:
 for q in rec['text_queries']:
  if q['method'] in ['T0','T1']:trows.append(q)
for rec in new:
 for q in rec['text_queries']:
  if q['method']=='T2_final':trows.append({**q,'method':'I1_I2_text'})
for rec in den:
 for q in rec['text_queries']:trows.append({**q,'method':'I1_only_text'})
with (O/'text_per_category.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['method','category','positive_queries','negative_queries','positive_Mask_IoU_pct','recall_pct','negative_FP_count_area16'])
 for method in ['T0','T1','I1_only_text','I1_I2_text']:
  for cls in ['car','person','road','vegetation','sky','train']:
   rows=[q for q in trows if q['method']==method and q['text']==cls];pos=[x for x in rows if x['positive']];neg=[x for x in rows if not x['positive']]
   w.writerow([method,cls,len(pos),len(neg),100*np.mean([x['iou'] for x in pos]) if pos else '',100*np.mean([x['recall'] for x in pos]) if pos else '',sum(x['false_positive'] for x in neg)])
# Compact scientific evidence figure.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.5));labels=['Original','+I1','+I2','+I1+I2'];values=[x['mIoU_pct'] for x in main]
axes[0].bar(labels,values,color=['#777777','#0072b2','#e69f00','#009e73']);axes[0].set_ylim(0,100);axes[0].set_ylabel('Semantic mIoU (%)');axes[0].set_title('Same checkpoint and original GT')
for i,v in enumerate(values):axes[0].text(i,v+1,f'{v:.3f}',ha='center')
entries=[('I1 only vs Original',C['semantic_bootstrap']['delta_mIoU_pp'],C['semantic_bootstrap']['image_bootstrap_95CI_pp']),('I1+I2 vs Original',B['joint_vs_dense_image_bootstrap']['delta_mIoU_pp'],B['joint_vs_dense_image_bootstrap']['image_bootstrap_95CI_pp']),('Joint trained R1 vs R0',A['bootstrap']['trained_R1_vs_R0']['delta_mIoU_pp'],A['bootstrap']['trained_R1_vs_R0']['image_bootstrap_95CI_pp'])]
if trained_dense:
 t=trained_dense['bootstrap_R1_vs_R0'];entries.append(('Dense trained R1 vs R0',t['delta_mIoU_pp'],t['image_bootstrap_95CI_pp']))
for i,(name,point,ci) in enumerate(entries):axes[1].errorbar(point,i,xerr=[[max(0,point-ci[0])],[max(0,ci[1]-point)]],fmt='o',capsize=4)
axes[1].set_yticks(range(len(entries)),[x[0] for x in entries]);axes[1].axvline(0,color='gray',ls='--');axes[1].set_xlabel('Delta mIoU (percentage points), image bootstrap95% CI');axes[1].set_title('Uncertainty on the development images');fig.suptitle('DensePASS100 development results; short continuation controls');fig.tight_layout();fig.savefig(O/'figures/paper_anchored_overview.png',dpi=160);plt.close(fig)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat();text=[]
def line(s=''):text.append(str(s))
line('创新点一：实现、原始 OmniSAM 基线对齐与短程实验报告');line('生成时间：'+now);line('远端运行目录：'+str(R));line()
line('结论与适用范围')
line('实现与真实推理/续训链路已跑通。冻结推理对照尚无稳健证据证明语义分割超过原始 OmniSAM；续训相对初始Original的变化与R1相对等预算R0的额外贡献必须分别看。文本掩码有相对独立文本分支的改善，不能改写成原论文语义 mIoU 增益。')
line('所有 DensePASS100 已被历史选模和本次方法选择使用，属于开发集。以下 bootstrap 是开发图像抽样不确定性；不能视为独立测试或只有两个种子时的训练种子总体置信区间。')
line('目标训练全集已恢复2000张；本次为40城市每城4张、共160张的平衡短程续训，非全部2000张完整训练。');line()
line('1. 原论文与复现基线')
line('论文：https://arxiv.org/html/2503.07098v2#S4.T3；官方代码：https://github.com/Ding-Zhong/OmniSAM')
line('表3：OmniSAM-S w/MA=60.23；OmniSAM-B w/MA=62.46。现有ckpt为Small，不能混用Base的62.46。论文数字单独列为文献参考，不与复现预测混作配对实验。')
line('官方主model、Memory、decoder、LoRA已下载，使用原始ckpt严格加载，在真实9-patch序列与本地原始路径logits逐元素完全一致。完整100图官方源码评测另存official_baseline_evaluation。仅审计环境安装mmcv-lite到run/work隔离目录，没有修改主训练环境或I2代码。')
line('主表Original以下载的官方model/Memory/decoder/LoRA代码、既有Small ckpt严格加载、num_maskmem9、未压缩最大32768 tokens的完整100图实测为准，使用同一SAM2 runtime。其余分支为用户Effective-Omnisam代码；未压缩路径已与官方真实9patch逐元素对齐。无.git，保存源码快照与SHA，不能宣称复现论文完整训练过程或验证了权重公开下载来源。官方DensePASS19→13与原始GT评分规则已核对。')
line('此前60.17966是仅创新点二压缩基线，不是原始OmniSAM。旧60.4594复现分数使用fork里的GT缩放回采样；相同预测用未改动原始GT评分为60.29496。旧结果和创新点二封存代码均保留。')
line('所有新主对比统一未改动2048×400原始GT，映射13类并保留255。GT经旧PIL+cv2上采样/下采样改变579000像素，禁止跨口径直接相减。详见evaluation_protocol_audit.json。');line()
line('2. 主表：全部锚定原始 OmniSAM-S，同一GT口径，单位%')
for row in main:line(f"{row['method']}: mIoU={row['mIoU_pct']:.6f}, 相对Original={row['delta_vs_original_pp']:+.6f}个百分点")
line('仅I1相对Original的图像bootstrap95%CI：'+str(C['semantic_bootstrap']['image_bootstrap_95CI_pp']))
line('I1+I2相对Original的图像bootstrap95%CI：'+str(B['joint_vs_dense_image_bootstrap']['image_bootstrap_95CI_pp']))
line('两者区间均跨零；不能使用“显著超过原始OmniSAM”或“语义精度提升0.13”作为主结论。+0.13仅为I2压缩后基线上的增量。');line()
line('3. 文本任务：真实 Grounding DINO + SAM2，600类别查询')
for name,m in [('T0 ERP直接文本',A['text']['T0']),('T1 独立固定FoV文本',A['text']['T1']),('仅I1的门控文本',C['text']),('I1+I2的门控文本',A['text']['T2_final']),('I1+I2无门控文本消融',A['text']['T2_final_no_gate'])]:
 line(f"{name}: 正样本MaskIoU={100*m['positive_mask_IoU']:.4f}%, Dice={100*m['positive_mask_Dice']:.4f}%, recall={100*m['positive_recall']:.4f}%, 负查询FP={m['negative_FP_area16']}/{m['negative_queries']} (阈值16有效像素)")
line('478正查询、122经原始完整GT确认的负查询；car/person严格用原19类细分类GT，不用合并13类冒充；200条实例/指代表达缺少可信实例GT，仅做定性。')
line('仅I1文本相对T1的IoU增量及95%CI：'+str(C['text_vs_T1_bootstrap']))
line('原始OmniSAM没有任意文本输入，原论文未报告MaskIoU600。新增文本任务以T1为模块消融，同时另报共享road/vegetation/sky共300查询的Original闭集输出类别投影比较，避免把新任务分数伪装成超过论文。')
for method,m in common_summary.items():line(f"共享3类别辅助比较 {method}: MaskIoU={m['positive_Mask_IoU_pct']:.4f}%")
line('这一辅助对比仅适用于共享类别。不能据此声称具备任意文本或实例定位能力。原始闭集模型若更优同样保留。')
line('无门控文本IoU高于主门控方法；不能声称门控全指标最优。主门控保留冲突/未知类/细粒度查询的独立文本结果。负查询误报仍高，未得到解决；purple flying elephant等复合表达存在真实误检，见定性图。');line()
line('4. 续训结果，和推理融合分开报告')
line('均独立从omnisam_s.pth开始，不从R0接着训练R1，不从随机初始化开始。只训练已有语义decoder stage2；图像encoder、LoRA、DINO与文本SAM2冻结。AdamW lr2e-6、wd1e-4、AMP，batch1全景，每全景9patch，共160迭代/1440次实际optimizer.step。')
line('源域监督CE + 目标冻结伪标签CE + 原有prototype正则；160源图从2975源集确定性抽样，160目标图固定并按种子洗牌；R0/R1每种子源/目标顺序哈希相同。训练/生成伪标签不读目标GT；训练有强制读GT拦截。')
line('原始dense模型配对对照：')
if trained_dense:
 for name,m in trained_dense['metrics'].items():line(f"{name}: {m['mIoU']:.6f}% (相对未续训Original {m['mIoU']-base:+.6f}pp)")
 line('dense R1相对Original的图像bootstrap：'+str(trained_dense.get('bootstrap_R1_vs_original')))
 line('dense R0/R1均值：'+str(trained_dense['means']));line('dense R1-R0：'+str(trained_dense['bootstrap_R1_vs_R0']));line('dense R1均值相对Original：'+str(trained_dense['delta_R1_mean_vs_original_pp'])+'pp')
else:line('仍在运行，不能宣称结果完成。查看dense_controls/progress.json。')
line('固定I2下的先前模块对照：')
for name,m in A['trained']['metrics'].items():line(f"{name}: {m['mIoU']:.6f}%")
line('均值：'+str(A['trained_mean_mIoU'])+'；R1-R0：'+str(A['bootstrap']['trained_R1_vs_R0']))
line('其R1均值相对Original约-0.234pp；主配对相对R0约-0.0382pp，两个种子均为负。没有训练收益。')
line('去掉0.85置信掩码的单种子补充诊断：R0=60.05103、R1=60.02336，差-0.02767pp，区间跨零。它不能替代主两种子对照。20迭代/180更新的早期Rsingle/Rnaive筛查亦保留在screens，预算不同不能作为正式160迭代主结论。')
line('主R0先以0.85置信度保留标签；R1的0.8强冲突保护使已保留的高置信类别不被覆盖，因此主训练操作是增补255忽略区监督，不应夸称修复了高置信错标签。实际改动类型逐像素统计在PSEUDO_OPERATION_AUDIT.json；全hard标签修正另有单种子诊断。')
line('源域labelIds→trainIds转换只写run内source_cityscapes；原始标签不变。伪标签存储为目标13类ID0..12/255，已针对目标loader实际identity映射核对。')
line('I2联合方法的开发集新增监督正确率75.19%，原高置信监督92.60%，同一新增区域原teacher hard准确率60.45%。这说明修正并非全错，但新增监督噪声较高；这是开发集审计，不是无标签训练集准确率，也不能直接证明训练下降的因果。');line()
line('5. 实现与防自我强化机制')
line('真实独立raw_text传入DINO与SAM2；固定4个1024×400 FoV、stride512、ERP横向循环映射。原有语义模型4096×1024、9patch×正反两遍；RGB/BICUBIC/ImageNet归一化与基线一致。')
line('三态agree/conflict/uncertain；平均类别支持≥0.5、SAM质量≥0.8才允许协同；≥0.8强像素冲突阻止写入。语义→文本仅对exact taxonomy候选、原DINO boxes和有效FoV内做有界概率融合：alpha0.5，文本内/外先验0.7/0.3。')
line('coarse car/person、tire/leftmost/服饰/未知类等保持独立文本掩码。没有把tire映射为car，没有仅因语义漏检删除有效文本。语义回写使用独立raw SAM mask，经同一冻结teacher gate接受后腐蚀2像素，再进行一次回写；不把语义修正过的文本掩码当成独立证据反复回灌。')
line('创新点二完全固定：class_aware uncertainty1024、rare[3,4,6,7] reserve0.15、num_maskmem9；无重开发或调参。Original与I1-only对照禁用压缩是基线消融，不是重调创新点二。创新点三未开发。')
line('早期score-only gate不改变掩码；consensus trimming降低召回；core扩展仅约0.01pp不采用；均保留原始结果，不择优隐藏。');line()
line('6. 速度与资源：独立实测，RTX6000D')
line(f"常驻模型暖态全链路单图单road查询：{P['full_end_to_end_mean_s']:.3f}s；已有语义缓存的新查询：{P['cached_semantic_query_mean_s']:.3f}s；T1纯文本FoV：{P['T1_text_only_end_to_end_mean_s']:.3f}s。")
line(f"双模型启动{P['startup_both_resident_workers_s']:.3f}s；首图含启动{P['cold_first_end_to_end_s']:.3f}s。1张warmup+5张测量，属于小样本性能预试。")
line('三查询文本batch：'+str(P['three_query_batch'])+'；含语义的三查询batch：'+str(P['full_three_query_batch']))
line(f"语义worker peak allocated/reserved={P['semantic_peak_allocated_mib']:.1f}/{P['semantic_peak_reserved_mib']:.1f}MiB；文本worker={P['text_peak_allocated_mib']:.1f}/{P['text_peak_reserved_mib']:.1f}MiB。各进程独立峰值，不等同同一时刻总峰值。")
line('性能器使用常驻worker，完整暖态包含读图、真实语义+文本模型、IPC解码与协同，不含PNG文件落盘；缓存查询是在常驻性能器中测量。公开i1_run.py为逐次启动进程的批处理入口，新文本需新输出目录，其单次冷启动与共享计算smoke耗时不可冒充这项暖态延迟。')
line('全设备占用含其他暂停任务常驻上下文，不作为本方法显存；正式测量时无并行GPU计算。不能把新文本任务和原始闭集OmniSAM用单个延迟数字混成同功能加速。I2自身独立profile保持原封存结果。')
line('精确算费GPU小时无法从现有日志恢复；runtime_ledger.json报告已记录成功进程wall-time之和及其缺项，明确不是kernel-time或账单小时。');line()
line('7. 数据、模型、验证与交付索引')
line('DensePASS train2000原图、100原始完整GT全部哈希核对；目标train官方无GT，不需要将城市jpg与val的__q掩码配对。800二值query masks不冒充100张完整语义GT。')
line('OmniSAM ckpt SHA256 ec0c607a6247651cb8c5eb04e3c15a7ae4b067655c4ca6cb5b6ff9d80adaba85。数据来源/归档校验在data/DensePASS/recovery_20261107。')
line('checkpoint_tensor_audit.json逐张量核对初始权重与训练后checkpoint，证明除decoder外encoder/LoRA/Memory未更新。')
line('77个项目测试通过；7项真实修改/冲突/未知/ERP边界检查；真实2图×2查询缓存与原文本流程逐像素一致；公开入口2图16查询与独立评测逐像素一致；3图12自由文本回退逐像素一致。')
line('公开入口支持 --memory-mode dense（仅创新点一）和fixed_i2（联合，默认）。封存输入为2048x400，其他尺寸显式拒绝。接受raw_text不等于保证任意复合指代或多语言理解。')
line('公开入口：scripts/piers/i1_run.py；核心：src/piers_ljj/collaboration/reliable_cooperation.py；训练与完整实验脚本位于本run。')
line('reports/MAIN_PAPER_ANCHORED.csv：主表；RESULTS.json：全结果；text_per_category.csv：细分类文本；dense_training_metrics.csv：原始dense训练；training_metrics.csv：固定I2模块训练；acceptance.json：原验收；final_package_acceptance.json：最终附加验收。')
line('reports/figures：成功/失败/真实负例/小目标/跨FoV/自由文本/训练伪标签/训练最佳最差/loss/原始baseline总览；cases.json记录选择规则。图像crop只展示局部，指标仍按完整查询计算。')
line('reports/REPRODUCE.sh：按stage执行，不会默认重跑训练；environment_pip_freeze.txt、SOURCE_MANIFEST.json、source_snapshot_final.tar.gz保存环境和版本证据。无.git，不捏造commit。')
line('最终模型位置：dense_controls/trained/{R0,R1}_seed{1234,4321}/training/final.pth；固定I2模型：final_runs同结构。没有因R1负收益而把某个挑选ckpt伪称最终最优。');line()
line('8. 是否达到今天的研究目标')
line('已完成：数据硬阻塞解除、真实任意文本入口、双向一次协同与回退、伪标签实际非零修正、同预算真实续训、原始OmniSAM主基线对齐、原GT评分、消融与负结果、bootstrap、资源计时、案例和复现脚本。')
line('尚未建立：创新点一对原始OmniSAM的稳定语义精度贡献、修正伪标签相对等预算R0的额外训练收益、较低负查询误报、可靠复合指代能力、全2000训练预算/独立测试验证。续训可能相对初始Original略升，若未超过R0则不能归因于创新点一。不能把工程完成写成论文精度目标已达成，也不能靠换比较基线或将推理融合当成训练提升。')
(O/'REPORT.txt').write_text('\n'.join(text)+'\n')
# Immutable source index, including exact vendor code used by local reproduction.
paths=set()
for sub in ['src','scripts','tests','configs','third_party/effective_omnisam','third_party/grounded_sam2']:
 for p in (ROOT/sub).rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.yaml','.yml','.json','.toml','.txt'] and '__pycache__' not in p.parts and p.stat().st_size<3000000:paths.add(p)
for p in ROOT.glob('*.py'):paths.add(p)
for p in R.glob('*.py'):paths.add(p)
for p in (R/'compat').rglob('*.py'):paths.add(p)
for p in (R/'work/original_omnisam').glob('*'):
 if p.is_file() and p.suffix in ['.py','.json']:paths.add(p)
paths.add(O/'official_dependencies_requirements.txt')
for p in R.glob('*.json'):paths.add(p)
paths.add(O/'REPRODUCE.sh');paths.add(ROOT/'upstream_versions.json')
source={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)};weightpaths=[Path('/root/autodl-tmp/weights/omnisam_s.pth'),Path('/root/autodl-tmp/weights/sam2.1_hiera_small.pt')]+[p for p in Path('/root/autodl-tmp/weights/grounding-dino-tiny-hf').glob('*') if p.is_file()]
write(O/'SOURCE_MANIFEST.json',{'at':now,'git':'No.git; hashes and tar snapshot','source_sha256':source,'weights_sha256':{str(p):sha(p) for p in sorted(weightpaths)},'data_manifests':{p.name:sha(p) for p in [R/'train160.json',R/'val100.json']},'no_sealed_I2_modification':True})
with tarfile.open(O/'source_snapshot_final.tar.gz','w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print('REPORTS_WRITTEN',json.dumps({'status':A['status'],'main':main,'dense_training':trained_dense['means'] if trained_dense else None,'common_categories':common_summary,'source_files':len(paths)},ensure_ascii=False),flush=True)
