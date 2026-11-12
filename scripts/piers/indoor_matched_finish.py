"""True DINO/SAM2 I1, indoor-only calibration, fusion, and strict paired scoring."""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from indoor_matched_run import REPO, OLD, CLASSES, sha, write_json

def rows_for(root,split,limit=0):
    rows=json.loads((root/f'inference_{split}.json').read_text())
    return rows[:limit] if limit else rows

def text(a):
    root=Path(a.out);protocol=json.loads((root/'protocol.json').read_text())
    sys.path[:0]=[str(REPO/'src'),str(REPO/'third_party/grounded_sam2')]
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1234)
    from piers_ljj.text.grounded_adapter import GroundedConfig,run_fov_text_branch
    from piers_ljj.workers.entry import _build_hf_backend
    weights=Path('/root/autodl-tmp/weights')
    cfg=GroundedConfig(grounding_model_id=str(weights/'grounding-dino-tiny-hf'),
        sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml',sam2_ckpt=str(weights/'sam2.1_hiera_small.pt'),
        device='cuda',dtype='bf16',box_thr=.35,text_thr=.25)
    dino,sam=_build_hf_backend(cfg,'cuda')
    fingerprint={'protocol_sha256':sha(root/'protocol.json'),'text_source_sha256':sha(__file__),
        'DINO_sha256':sha(weights/'grounding-dino-tiny-hf/pytorch_model.bin'),
        'SAM2_sha256':sha(weights/'sam2.1_hiera_small.pt')}
    rows=rows_for(root,a.split,a.limit);target=root/'text'/a.split;target.mkdir(parents=True,exist_ok=True)
    begun=time.monotonic()
    for index,row in enumerate(rows):
        dest=target/row['image_id'];dest.mkdir(exist_ok=True)
        if (dest/'metadata.json').exists():
            assert json.loads((dest/'metadata.json').read_text())['fingerprint']==fingerprint
            continue
        assert sha(row['path'])==row['input_sha256']
        with Image.open(row['path']) as image:rgb=np.asarray(image.convert('RGB'))
        assert rgb.shape==(512,1024,3)
        queries=[{'image_id':row['image_id'],'query_id':row['image_id']+':indoor:'+str(c),
            'raw_text':name,'ontology':{'map13':c,'relation':'exact'}} for c,name in enumerate(CLASSES)]
        fovs=[{'view_id':f'v{k:03}','x_start':x,'y_start':0,'win_w':512,'win_h':512}
              for k,x in enumerate((0,256,512,768))]
        masks=[];candidates=[];torch.cuda.synchronize();started=time.perf_counter()
        class WithinView:
            def __init__(self):self.enc=None
            def set_image(self,pil):
                if self.enc is None:self.enc=sam.set_image(pil)
                return self.enc
            def predict_boxes(self,enc,boxes):return sam.predict_boxes(enc,boxes)
        with torch.no_grad():
            for fov in fovs:
                cols=(fov['x_start']+np.arange(512))%1024
                pil=Image.fromarray(rgb[:,cols]);cached=WithinView()
                for query in queries:
                    result=run_fov_text_branch(pil,query['raw_text'],cfg,(dino,cached))
                    for k,c in enumerate(result['candidates']):
                        local=np.asarray(c.mask,bool);assert local.shape==(512,512)
                        mask=np.zeros((512,1024),bool);mask[:,cols]=local
                        candidates.append({'id':query['query_id']+':'+fov['view_id']+':'+str(k),
                            'query_id':query['query_id'],'fov_id':fov['view_id'],
                            'detector_score':c.detector_score,'sam_quality':c.sam_quality,
                            'phrase':str(c.phrase),'local_box':list(c.box_xyxy),
                            'ontology':query['ontology'],'mask_index':len(masks)})
                        masks.append(mask)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        packed=np.packbits(np.stack(masks),axis=2) if masks else np.zeros((0,512,128),np.uint8)
        tmp=dest/'candidates.partial'
        with tmp.open('wb') as f:np.savez_compressed(f,masks_packed=packed,shape=np.array([512,1024]))
        tmp.replace(dest/'candidates.npz')
        write_json(dest/'metadata.json',{'status':'complete','fingerprint':fingerprint,
            'image_id':row['image_id'],'input_sha256':row['input_sha256'],'shape':[512,1024],
            'fovs':fovs,'queries':queries,'candidates':candidates,'no_GT_read':True,
            'DINO_calls':32,'SAM_image_encodes':4,'elapsed_s':elapsed,
            'source_method':'published DINO/SAM2, literal class queries, within-view SAM cache'})
        write_json(target/'progress.json',{'done':index+1,'total':len(rows),'elapsed_s':time.monotonic()-begun})
        print('TEXT',a.split,index+1,len(rows),len(candidates),round(elapsed,3),flush=True)
        del masks,packed;gc.collect()
    write_json(target/'DONE.json',{'n':len(rows),'elapsed_s':time.monotonic()-begun,'fingerprint':fingerprint})

def load_context(root,split,iid):
    from indoor_matched_features import build_context
    dest=root/'text'/split/iid
    meta=json.loads((dest/'metadata.json').read_text())
    with np.load(dest/'candidates.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=2,count=1024).astype(bool)
    with np.load(root/'semantic'/split/'full_core'/(iid+'.npz')) as z:
        probs=z['probabilities'].astype(np.float32);stored=z['prediction']
    return build_context(meta,masks,probs,stored),meta,masks,stored

def calibrate(a):
    import torch
    import torch.nn.functional as F
    from indoor_matched_features import FEATURE_NAMES,TEACHER_ONLY_FEATURE_INDICES
    from indoor_matched_predictor import ReliabilityCalibrator
    root=Path(a.out);protocol=json.loads((root/'protocol.json').read_text())
    truth={r['id']:r for r in [json.loads(s) for s in (OLD/'split_calib.jsonl').read_text().splitlines() if s.strip()]}
    rows=rows_for(root,'calib');rng=np.random.default_rng(20261107)
    xs=[];ys=[];ws=[];records=[]
    for index,row in enumerate(rows):
        ctx,_,_,stored=load_context(root,'calib',row['image_id']);r=truth[row['image_id']]
        gt=np.asarray(Image.open(Path('/root/autodl-tmp/data_extra/SPAN8_PREP')/r['area']/r['label']))
        assert gt.shape==(ctx.H,ctx.W)
        count=0
        for group in ctx.iter_groups():
            rr,cc=np.nonzero(group.mask&(gt<8));population=len(rr)
            if not population:continue
            selected=rng.choice(population,min(256,population),replace=False)
            rr,cc=rr[selected],cc[selected]
            xs.append(ctx.features_at(group.class_id,group.view_id,rr,cc))
            ys.append(np.stack([gt[rr,cc]==group.class_id,gt[rr,cc]==stored[rr,cc]],axis=1))
            ws.append(np.full(len(rr),population/len(rr),np.float64));count+=len(rr)
        records.append({'image_id':row['image_id'],'samples':count})
        print('CALIB_SAMPLES',index+1,len(rows),count,flush=True)
    X=np.concatenate(xs);Y=np.concatenate(ys).astype(np.float64);w=np.concatenate(ws)
    target=root/'calibrators';target.mkdir(exist_ok=True)
    np.savez_compressed(target/'calib_samples.npz',X=X,Y=Y,weight=w)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261107)
    w=w/w.sum();prevalence=(Y*w[:,None]).sum(0).clip(1e-6,1-1e-6)
    for name,indices in [('text_only',list(range(5,26))),('teacher_only',TEACHER_ONLY_FEATURE_INDICES)]:
        raw=X[:,indices].astype(np.float64);mean=(raw*w[:,None]).sum(0)
        scale=np.sqrt(((raw-mean)**2*w[:,None]).sum(0));scale[scale<1e-8]=1.
        design=torch.from_numpy((raw-mean)/scale);labels=torch.from_numpy(Y);weight=torch.from_numpy(w)
        coef=torch.zeros((len(indices),2),dtype=torch.float64,requires_grad=True)
        intercept=torch.tensor(np.log(prevalence/(1-prevalence)),requires_grad=True)
        optimizer=torch.optim.Adam([coef,intercept],lr=.05)
        begun=time.monotonic();trace=[]
        for step in range(1000):
            optimizer.zero_grad();logits=design@coef+intercept
            nll=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weight[:,None]).sum(0)
            penalty=.001*coef.square().sum()/2
            loss=nll.sum()+penalty;loss.backward();optimizer.step()
            if step==0 or (step+1)%100==0:
                trace.append({'step':step+1,'objective':float(loss.detach()),'nll':nll.detach().tolist()})
                print('CALIB_FIT',name,step+1,float(loss.detach()),flush=True)
        model=ReliabilityCalibrator(mean,scale,coef.detach().numpy(),intercept.detach().numpy(),indices,26,FEATURE_NAMES)
        model.save(target/(name+'.npz'),{'origin':'indoor training-area calibration104 only',
            'config':protocol['calibrator_fit'],'samples':len(X),'images':records,
            'heldout_GT_reads':0,'trace':trace,'elapsed_s':time.monotonic()-begun,
            'fit_only_weighted_population_normalization':True,'intercept_penalized':False,
            'feature_onehot':'SPAN8,8 classes; indoor feature dimension26'})
    write_json(target/'DONE.json',{'calibration_images':len(rows),'samples':len(X),
        'text_only_sha256':sha(target/'text_only.npz'),'teacher_only_sha256':sha(target/'teacher_only.npz'),
        'heldout_GT_reads':0})

def fuse_predictions(a):
    sys.path.insert(0,str(REPO/'src'))
    from indoor_matched_gate import proposals,arm_counts
    from indoor_matched_soft import soft_correct
    from indoor_matched_predictor import load_calibrator
    from indoor_matched_bidirectional import prepare,bounded_text_fusion
    root=Path(a.out);protocol=json.loads((root/'protocol.json').read_text())
    text_model=load_calibrator(root/'calibrators/text_only.npz')
    teacher=load_calibrator(root/'calibrators/teacher_only.npz')
    class Pair:
        def predict_np(self,X):
            p=text_model.predict_np(X);p[:,1]=teacher.predict_np(X)[:,1];return p
    target=root/'full'/a.split;target.mkdir(parents=True,exist_ok=True)
    rows=rows_for(root,a.split,a.limit)
    for index,row in enumerate(rows):
        iid=row['image_id'];ctx,meta,masks,stored=load_context(root,a.split,iid)
        begun=time.monotonic();arrays,gate=proposals(ctx,Pair(),protocol['I1_gate'])
        lab,weight=arrays['label'],arrays['weight'];eligible=(lab!=255)&(weight>0)
        pred,alpha,winner,error=soft_correct(ctx.probs,stored,lab,weight,eligible)
        qmeta={q['query_id']:q for q in meta['queries']}
        records=prepare(meta,masks,ctx.probs,qmeta)
        guided,text_audit=bounded_text_fusion(records,ctx.probs,list(qmeta),alpha=.5,gated=True)
        np.save(target/(iid+'.npy'),pred)
        # Record both directions without allowing guided masks to feed back.
        np.savez_compressed(target/(iid+'_guided.npz'),masks_packed=np.packbits(np.stack(list(guided.values())),axis=2))
        report=arm_counts(stored,lab,weight,arrays['votes'],eligible,pred,alpha,error,ctx.probs)
        report.update({'I1_gate':gate,'I1_text':text_audit,'no_circular_evidence':True,
            'heldout_GT_reads':0,'elapsed_s':time.monotonic()-begun,
            'calibrator_sha256':{n:sha(root/'calibrators'/(n+'.npz')) for n in ('text_only','teacher_only')}})
        write_json(target/(iid+'.json'),report)
        print('FUSE',index+1,len(rows),report['actual_changed_pixels'],flush=True)
    write_json(target/'DONE.json',{'n':len(rows)})

def metric(cm):
    h=cm.astype(np.float64);tp=np.diag(h[:,:8]);gs=h.sum(1);ps=h[:,:8].sum(0)
    iou=np.divide(tp,gs+ps-tp,out=np.full(8,np.nan),where=(gs+ps-tp)>0)
    acc=np.divide(tp,gs,out=np.full(8,np.nan),where=gs>0)
    dice=np.divide(2*tp,gs+ps,out=np.full(8,np.nan),where=(gs+ps)>0)
    return {'mIoU_pct':float(np.nanmean(iou)*100),'mAcc_pct':float(np.nanmean(acc)*100),
        'PixelAcc_pct':float(tp.sum()/h.sum()*100),'FWIoU_pct':float(np.nansum(gs/h.sum()*iou)*100),
        'mDice_pct':float(np.nanmean(dice)*100),'per_class_iou_pct':dict(zip(CLASSES,(iou*100).tolist())),
        'gt_valid_pixels':int(h.sum()),'correct_pixels':int(tp.sum()),
        'per_class_gt_support':dict(zip(CLASSES,gs.astype(int).tolist())),
        'per_class_pred_support':dict(zip(CLASSES,ps.astype(int).tolist())),
        'predictions_outside_label_set':int(h[:,8].sum()),'confusion_matrix':cm.tolist()}

def score(a):
    root=Path(a.out);protocol=json.loads((root/'protocol.json').read_text());rows=rows_for(root,a.split,a.limit)
    truth={r['id']:r for r in [json.loads(s) for s in (OLD/f'split_{a.split}.jsonl').read_text().splitlines() if s.strip()]}
    cms={arm:[] for arm in ('original','full_core','full')};latencies={arm:[] for arm in cms}
    for row in rows:
        iid=row['image_id'];r=truth[iid]
        gt=np.asarray(Image.open(Path('/root/autodl-tmp/data_extra/SPAN8_PREP')/r['area']/r['label'])).astype(np.int64)
        valid=(gt>=0)&(gt<8);g=gt[valid]
        identities=[]
        for arm in cms:
            if arm=='full':pred=np.load(root/'full'/a.split/(iid+'.npy'))
            else:
                folder=root/'semantic'/a.split/arm
                with np.load(folder/(iid+'.npz')) as z:pred=z['prediction']
                audit=json.loads((folder/(iid+'.json')).read_text());identities.append(audit)
                latencies[arm].append(audit['elapsed_s'])
            p=pred[valid].astype(np.int64);p=np.where((p>=0)&(p<8),p,8)
            cms[arm].append(np.bincount(g*9+p,minlength=72).reshape(8,9))
        assert identities[0]['checkpoint_sha256']==identities[1]['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert identities[0]['lora_sha256']==identities[1]['lora_sha256']
        text_meta=json.loads((root/'text'/a.split/iid/'metadata.json').read_text())
        fusion=json.loads((root/'full'/a.split/(iid+'.json')).read_text())
        latencies['full'].append(identities[1]['elapsed_s']+text_meta['elapsed_s']+fusion['elapsed_s'])
    results={'protocol':protocol,'n_images':len(rows),'split':a.split,'status':'complete' if len(rows)==373 else 'partial',
        'arms':{arm:metric(np.sum(h,axis=0)) for arm,h in cms.items()},
        'timing_note':'Sum of recorded compute stages; excludes load, intermediate cache I/O and orchestration. Not isolated end-to-end latency.',
        'mean_recorded_compute_s':{arm:float(np.mean(t)) for arm,t in latencies.items()}}
    results['delta_full_minus_original_mIoU_pp']=results['arms']['full']['mIoU_pct']-results['arms']['original']['mIoU_pct']
    rng=np.random.default_rng(20261112);paired=[]
    for _ in range(2000):
        idx=rng.integers(0,len(rows),len(rows))
        paired.append(metric(np.asarray(cms['full'])[idx].sum(0))['mIoU_pct']-metric(np.asarray(cms['original'])[idx].sum(0))['mIoU_pct'])
    results['paired_image_bootstrap_delta95CI_pp']=np.percentile(paired,[2.5,97.5]).tolist()
    results['CI_limit']='Image bootstrap; correlated same-room views and single heldout area limit generalization.'
    write_json(root/'INDOOR_MATCHED_RESULTS.json',results)
    np.savez_compressed(root/'paired_confusions.npz',**{k:np.stack(v) for k,v in cms.items()},image_ids=np.array([r['image_id'] for r in rows]))
    print(json.dumps(results,indent=2),flush=True)

def run(a):
    {'text':text,'calibrate':calibrate,'fuse':fuse_predictions,'score':score}[a.action](a)
