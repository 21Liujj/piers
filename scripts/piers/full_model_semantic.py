"""Persistent image-only semantic engine, equivalent to recovered I3 runtime.

The original 18 forwards, I2 memory wrapper, adaptive two extra forwards and
I3 fusion are preserved. Model loading is shared across images in a worker.
"""
from __future__ import annotations
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np

class SemanticEngine:
    def __init__(self,repo,memory='i2',budget=2,memory_config=None,strategy='adaptive'):
        self.repo=Path(repo);self.memory=memory;self.budget=int(budget)
        if memory not in ('dense','i2') or budget not in (0,1,2):raise ValueError('Unsupported main-model configuration')
        if strategy not in ('adaptive','fixed','uniform'):raise ValueError('Unknown seam strategy')
        self.strategy=strategy
        folder=self.repo/'experiments/paper20261109/runtime';sys.path.insert(0,str(folder))
        import run_i3_semantic as runtime
        self.rt=runtime;torch=runtime.torch;self.torch=torch;self.cv2=runtime.cv2
        runtime.verify_i2();torch.set_num_threads(1);torch.set_num_interop_threads(1)
        runtime.cv2.setNumThreads(0);runtime.v.setup_seed(1234);torch.backends.cudnn.benchmark=False
        ns=argparse.Namespace(seed=1234,backbone='sam2_s',dataset='DensePASS',class_set=13,
                             num_classes=13,use_mem_bank=True,num_maskmem=9,enable_scm=False,enable_bepl=False)
        self.model=runtime.v.build_sam_model(ns,torch.device('cuda'),ckpt_path=runtime.I2['checkpoint_path']).eval()
        if memory=='i2':
            from i1_runtime import install_fixed_memory
            self.audit=install_fixed_memory(self.model,overrides=memory_config)
        else:self.audit=runtime.dense_memory(self.model)
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]
        self.checkpoint_sha256=runtime.I2['checkpoint_sha256']

    def predict(self,image):
        from PIL import Image
        torch=self.torch;cv2=self.cv2;rt=self.rt;model=self.model
        with Image.open(image) as pil:
            if pil.size!=(2048,400):raise ValueError('Pass a canonical 2048x400 RGB image')
            rgb=np.asarray(pil.convert('RGB').resize((4096,1024),Image.Resampling.BICUBIC)).copy()
        tensor=((torch.from_numpy(rgb).permute(2,0,1).contiguous().float().div(255)-self.mean)/self.std).cuda()
        starts=list(range(0,3073,384));contexts=[];calls=0
        logs=np.zeros((1,18,13,1024,1024),np.float32)
        for key in ('calls','fallbacks','max_tokens'):self.audit[key]=0
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.no_grad():
            for pas,seq in enumerate((starts,list(reversed(starts)))):
                model.memory_bank._init_output_dict_source();labels={}
                for f,x in enumerate(seq):
                    pred,*_=model(f,tensor[:,:,x:x+1024].unsqueeze(0).contiguous());calls+=1
                    labels[f]=pred.argmax(1).detach();logs[0,pas*9+f]=pred[0].cpu().numpy()
                contexts.append(({k:v for k,v in model.memory_bank.output_dict_source['frame_outputs'].items() if k>0},
                                 {k:v for k,v in labels.items() if k>0}))
            model.memory_bank._init_output_dict_source()
            sums=rt.merge_width_scatter_np(logs,(1024,4096),1024,384,2,'cuda',True)[0]
            norm=sums/np.maximum(sums.sum(0,keepdims=True),1e-8)
            label=cv2.resize(sums.argmax(0).astype(np.uint8),(2048,400),interpolation=cv2.INTER_NEAREST)
            base=np.stack([cv2.resize(c,(2048,400),interpolation=cv2.INTER_NEAREST) for c in norm]).astype(np.float16)
            del logs,sums,norm
            plan=[]
            if self.budget:
                if self.strategy=='fixed':plan=[1792]*self.budget
                elif self.strategy=='uniform':plan=[1792] if self.budget==1 else [1664,1920]
                else:plan=rt.choose(rt.risk_cuda(base),[1664,1728,1792,1856,1920],512,self.budget,min_separation=128)[0]
            extras=[]
            for pas,x in enumerate(plan):
                context,labels=contexts[pas]
                model.memory_bank._init_output_dict_source()
                model.memory_bank.output_dict_source={'frame_outputs':dict(context)}
                crop=tensor.index_select(2,(x*2+torch.arange(1024,device='cuda'))%4096).unsqueeze(0).contiguous()
                pred,*_=model(9,crop,historical_pseudo_labels=labels);calls+=1
                probability=pred.softmax(1)[0].cpu().numpy()
                extras.append((x,np.stack([cv2.resize(c,(512,400),interpolation=cv2.INTER_NEAREST) for c in probability]).astype(np.float16)))
            model.memory_bank._init_output_dict_source()
            probability,result,fusion=rt.fuse_cuda(base,label,extras,rt.base_coverage())
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        assert calls==18+self.budget
        if self.budget==0:assert np.array_equal(result,label)
        report={'memory':self.memory,'I3_budget':self.budget,'forward_calls':calls,
                'I3_strategy':self.strategy,
                'I3_selected_windows':list(plan),'memory_audit':dict(self.audit),
                'I3_fusion':fusion,'checkpoint_sha256':self.checkpoint_sha256,
                'weights_updated':False,'ground_truth_reads':0,'elapsed_s':elapsed}
        return probability.astype(np.float16),result,report
