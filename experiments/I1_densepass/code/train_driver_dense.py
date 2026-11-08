"""Controlled continuation through existing UDA Engine/Trainer/WithMemStrategy."""
import os,sys,json,time,argparse,random,traceback,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/piers-main');RUN=Path(__file__).resolve().parent;EFF=ROOT/'third_party/effective_omnisam'
sys.path.insert(0,str(EFF));sys.path.insert(0,str(RUN/'compat'))
os.environ['WANDB_MODE']='disabled';os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch,numpy as np
from PIL import Image
from run_adaptation import TrainConfig
from adaptation.engine import Engine
import utils.build as builders
from i1_dense_runtime import I2,verify_i2,install_fixed_memory,sha

class LoggedDataset(torch.utils.data.Dataset):
 def __init__(self,base,records,kind):self.base=base;self.records=records;self.kind=kind
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  self.records.append({'kind':self.kind,'index':int(i),'name':self.base.files[i].get('name','')});return self.base[i]
 def __getattr__(self,key):return getattr(self.base,key)

class AuditCB:
 def __init__(self,out,engine,total):self.out=out;self.engine=engine;self.total=total;self.first=None;self.count=0
 def on_iter_end(self,global_step,loss,lr,metrics=None,**kw):
  assert np.isfinite(loss),('Nonfinite loss',loss)
  self.count+=1;row={'update':self.count,'global_step':int(global_step),'loss':float(loss),'lr':float(lr),'time':time.time()}
  for k in ['loss_sup_s','loss_unsup','loss_proto','semantic_decoder_grad_norm','loss_total','prediction_change_ratio']:
   if metrics and k in metrics:
    try:row[k]=float(metrics[k])
    except (TypeError,ValueError):pass
  with (self.out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  (self.out/'progress.json').write_text(json.dumps({'status':'training','updates':self.count,'total_updates':self.total,'last':row},indent=2))
  if self.count%9==0:print('TRAIN_UPDATES',self.count,self.total,round(loss,5),flush=True)

def main(a):
 verify_i2();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 assert not (out/'final.pth').exists(),'Refuse to overwrite completed training'
 # Read-only aliases fix historical Cityscapes capitalization and target layout.
 view=out/'data_view';view.mkdir(exist_ok=True)
 links={'cityscapes':RUN/'source_cityscapes','DensePASS':Path('/root/autodl-tmp/data/DensePASS')}
 for name,target in links.items():
  dest=view/name
  if not dest.exists():dest.symlink_to(target,target_is_directory=True)
 wp=view/'WildPASS';wp.mkdir(exist_ok=True)
 if not (wp/'leftImg8bit').exists():(wp/'leftImg8bit').symlink_to('/root/autodl-tmp/data/DensePASS/leftImg8bit/train',target_is_directory=True)
 args=TrainConfig();args.dataset='Cityscapes';args.class_set=13;args.backbone='sam2_s';args.data_root=str(view)
 args.seed=a.seed;args.batch_size=1;args.lr=a.lr;args.num_maskmem=9;args.amp=True
 args.tgt_ckpt=I2['checkpoint_path'];args.src_ckpt=I2['checkpoint_path'];args.pd_root=str(out/'engine')
 args.use_wandb=False;args.use_tensorboard=False;args.use_text_proto=False;args.text_train_stage='stage2'
 args.reuse_pl_dir=str(Path(a.pseudo).resolve());args.train_max_iters=a.iters;args.num_epochs=1;args.conf_thr=0.0
 args.plugins_cfg={};args.use_patch_scheduler=False;args.use_memory_compression=False;args.mem_compression_enabled=False
 args.enable_scm=False;args.enable_sgdrf=False;args.use_pl_update_controller=False;args.pseudo_label_refiner={'mode':'off'}
 args.resume_save_interval=0
 builders._num_workers=lambda n:0
 builders.SAM_CFGS['sam2_s']=('/root/autodl-tmp/weights/sam2.1_hiera_small.pt','/configs/sam2.1/sam2.1_hiera_s.yaml')
 os.chdir(EFF)
 # A hard guard makes target evaluation-label reads during training fail loudly.
 orig_open=Image.open
 def train_open(fp,*aa,**kw):
  if isinstance(fp,(str,os.PathLike)):
   resolved=str(Path(fp).resolve())
   if '/data/DensePASS/gtFine/val/' in resolved or '/data_extra/WildPASS13/labels_' in resolved:raise RuntimeError('Target evaluation GT access attempted: '+resolved)
  return orig_open(fp,*aa,**kw)
 Image.open=train_open
 engine=Engine(args);engine.setup();engine.is_warmup=True;engine.resume_save_disabled=True
 aud_t=install_fixed_memory(engine.model_t,'train');aud_s=install_fixed_memory(engine.model_s,'train')
 records=[];src_factory=engine.data_builder.build_src;tgt_factory=engine.data_builder.build_tgt_from_pd
 def logged_src():
  from dataset.Cityscapes import Cityscapes13
  base=Cityscapes13(str(RUN/'source_cityscapes'),str(EFF/'dataset/cityscapes_list/train.txt'),crop_size=(2048,1024),sliding_window=(128,1024),set='train')
  wrapped=LoggedDataset(base,records,'source');val=None
  return torch.utils.data.DataLoader(wrapped,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed)),val
 def logged_tgt(pd):
  loader,val=tgt_factory(pd);wrapped=LoggedDataset(loader.dataset,records,'target')
  return torch.utils.data.DataLoader(wrapped,batch_size=1,shuffle=a.shuffle_target,num_workers=0,generator=torch.Generator().manual_seed(a.seed+9000)),val
 engine.data_builder.build_src=logged_src;engine.data_builder.build_tgt_from_pd=logged_tgt
 optimizer=engine.trainer.optimizer;total=a.iters*9;warm=min(18,max(1,total//10))
 step_counter={'actual':0}
 def counted_step(*unused):step_counter['actual']+=1
 optimizer.register_step_post_hook(counted_step)
 # Both arms use the same explicit short-run schedule, not the 1800-step long-run warmup.
 schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min(1.,(n+1)/warm)*max(.2,1.-n/max(total,1)))
 for cb in engine.trainer.callbacks:
  if cb.__class__.__name__=='LRCB':cb.scheduler=schedule
 audit_cb=AuditCB(out,engine,total);engine.trainer.callbacks.append(audit_cb)
 from dataclasses import asdict
 config={'args':asdict(args),'iters':a.iters,'optimizer_updates':total,'warmup_updates':warm,'seed':a.seed,'lr':a.lr,'pseudo':str(a.pseudo),'fixed_memory':I2,'target_confidence':'frozen teacher mask encoded in supplied labels; online student threshold disabled identically in all arms','target_label_storage':'Project13 trainIds 0..12, ignore255; verified against target loader identity LUT','evaluation_gt_guard':True,'variant':a.variant,'shuffle_target':a.shuffle_target}
 (out/'config.json').write_text(json.dumps(config,indent=2,default=str));initial={n:x.detach().cpu().clone() for n,x in engine.model_t.named_parameters() if x.requires_grad}
 start=time.perf_counter();torch.cuda.reset_peak_memory_stats();engine.train(0);torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 delta={n:float((x.detach().cpu()-initial[n]).abs().max()) for n,x in engine.model_t.named_parameters() if n in initial}
 assert audit_cb.count==total,(audit_cb.count,total)
 assert step_counter['actual']==total,('AMP skipped optimizer steps',step_counter['actual'],total)
 assert max(delta.values())>0,'No parameter update'
 state=engine.model_t.state_dict();torch.save(state,out/'final.pth')
 (out/'data_order.json').write_text(json.dumps(records,indent=2))
 result={'status':'complete','variant':a.variant,'iters':a.iters,'optimizer_updates':audit_cb.count,'actual_optimizer_steps':step_counter['actual'],'elapsed_s':elapsed,'peak_allocated_mib':max(torch.cuda.max_memory_allocated()/2**20,float(engine.last_train_stats.get('gpu_mem_peak',0))*1024),'max_parameter_delta':max(delta.values()),'changed_parameter_tensors':sum(x>0 for x in delta.values()),'source_target_order_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),'checkpoint_sha256':sha(out/'final.pth'),'target_GT_reads':0,'memory_audit_student':aud_t,'memory_audit_source':aud_s,'training_stats':engine.last_train_stats}
 (out/'result.json').write_text(json.dumps(result,indent=2,default=str));print('TRAIN_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='training_stats'},default=str),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pseudo',required=True);ap.add_argument('--out',required=True);ap.add_argument('--iters',type=int,required=True);ap.add_argument('--lr',type=float,default=2e-6);ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--variant',default='R0');ap.add_argument('--shuffle-target',action='store_true');a=ap.parse_args()
 try:main(a)
 except BaseException:
  Path(a.out).mkdir(parents=True,exist_ok=True);(Path(a.out)/'failure.txt').write_text(traceback.format_exc());raise
