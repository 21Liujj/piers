"""S-adaptive GPU eval: base linear + on-demand cross-seam re-observe (I3).
True wrap windows, no fake full-width boxes. B=0 degrades to base.
Compares: base(linear) vs uniform_extra(B) vs adaptive_extra(B) vs fixed_scm.
Fusion restricted to seam bands + reliability gating (honest local fusion).
"""
import os, sys, json, time, argparse
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
sys.path.insert(0, '/root/autodl-tmp/piers-main/src')
import torch, numpy as np, cv2
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader
from utils.tools import merge_width_scatter_np
from utils.metrics import fast_hist, per_class_iu
from piers_ljj.seam.adaptive import plan_cross_seam_windows, adaptive_extras, score_uda, seam_bands, cost_report
from PIL import Image
from torchvision import transforms

BAND_W = 256

def run(tag, B=0, mode='adaptive', n_img=20):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=f'/root/autodl-tmp/piers-main/runs/{tag}/metrics.json', log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True, save_pred_debug=0, pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='', pred_gt_histogram_md='', timing_json=None, run_label=tag)
    os.makedirs(f'/root/autodl-tmp/piers-main/runs/{tag}', exist_ok=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    keep_files = val_loader.dataset.files[:n_img]
    val_loader.dataset.files = keep_files
    import torch.utils.data as tud
    vl = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    model.eval()
    mb = model.memory_bank
    mb.configure_sparse_retrieval(enabled=False)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    hist = np.zeros((num_classes, num_classes))
    seam_hist = np.zeros((num_classes, num_classes))
    W_full, H_full = 4096, 1024
    L = (W_full - 1024)//384 + 1
    n_extra_used, t_extra, fused_px, seam_px = 0, 0.0, 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, (image, label, _, names, ori) in enumerate(vl):
            image = image.to(device)
            nf = val_loader.dataset._get_frame_cnt()
            logits_maps = np.zeros((1, 2*nf, len(class_names), 1024, 1024))
            for fi in range(nf):
                o,_,_ = model(fi, image[:,fi]); logits_maps[:,fi]=o.cpu().numpy()
            mb._init_output_dict_source()
            for fi in range(nf):
                o,_,_ = model(fi, image[:,nf+fi]); logits_maps[:,nf+fi]=o.cpu().numpy()
            mb._init_output_dict_source()
            # single merge returning probs; pred = argmax
            probs_m = merge_width_scatter_np(logits_maps, orig_size=(1024, W_full), W_patch=1024, stride_w=384, passes=2, device='cuda', return_probs=True)[0]  # CxHxW
            base_pred = probs_m.argmax(axis=0).astype(np.int64)  # HxW (1024x4096)
            ent = -(probs_m*np.log(probs_m+1e-8)).sum(axis=0)/np.log(num_classes)
            bands = seam_bands(W_full, BAND_W)
            l0,l1 = bands['left']; r0,r1 = bands['right']
            seam_ent = float(ent[:, l0:l1].mean() + ent[:, r0:r1].mean())/2.0
            cands = plan_cross_seam_windows(W_full, 1024, 1024, 1024, y_start=0, n=2)
            scores = {}
            for cd in cands:
                touch = bool(((base_pred[:, r0:r1]==12).any() or (base_pred[:, l0:l1]==12).any()))
                scores[cd['view_id']] = score_uda(seam_ent, touch, 0.0)
            sel = [c['view_id'] for c in cands[:B]] if mode=='uniform' else adaptive_extras(scores, B)
            fused = base_pred.copy()
            if sel:
                te = time.perf_counter()
                # load full-res ERP from dataset file record (exact path, no glob guessing)
                img_path = keep_files[bi]['img'] if isinstance(keep_files[bi], dict) else None
                full = tf(Image.open(img_path).convert('RGB').resize((W_full,H_full), Image.BICUBIC)) if img_path else None
                for vid in sel:
                    cd = [c for c in cands if c['view_id']==vid][0]
                    xs = cd['x_start'] % W_full
                    idx = [(xs+i)%W_full for i in range(1024)]
                    crop = full[:, 0:1024, :][:,:, idx].unsqueeze(0).to(device)
                    mb._init_output_dict_source()
                    o,_,_ = model(0, crop)
                    mb._init_output_dict_source()
                    ep_log = o[0].float().cpu().numpy()  # Kxhxw
                    K, eh, ew = ep_log.shape
                    eprob = np.exp(ep_log-ep_log.max(axis=0,keepdims=True)); eprob/=eprob.sum(axis=0,keepdims=True)
                    ep = eprob.argmax(axis=0).astype(np.int64)
                    econf = eprob.max(axis=0)
                    if (eh,ew)!=(1024,1024):
                        ep = cv2.resize(ep.astype(np.uint8),(1024,1024),interpolation=cv2.INTER_NEAREST).astype(np.int64)
                        econf = cv2.resize(econf.astype(np.float32),(1024,1024),interpolation=cv2.INTER_LINEAR)
                    # fuse ONLY seam-band columns with reliability gate
                    for lx in range(1024):
                        gx=(xs+lx)%W_full
                        if not ((l0<=gx<l1) or (r0<=gx<r1)): continue
                        gate = (ent[:,gx]>0.6) & (econf[:,lx]>0.7)
                        n_take = int(gate.sum())
                        if n_take:
                            fused[gate,gx] = ep[gate,lx]
                            fused_px += n_take
                    n_extra_used += 1
                t_extra += time.perf_counter()-te
            ol = ori.numpy().astype(np.int64)
            if ol.ndim==3: ol=ol[0]
            gt = cv2.resize(ol.astype(np.uint8),(2048,400),interpolation=cv2.INTER_NEAREST)
            pr = cv2.resize(fused.astype(np.uint8),(2048,400),interpolation=cv2.INTER_NEAREST)
            hist += fast_hist(gt.flatten(), pr.flatten(), num_classes)
            # seam-band local hist at full res: band cols only
            band_mask = np.zeros((1024, W_full), bool); band_mask[:, l0:l1]=True; band_mask[:, r0:r1]=True
            # gt full-res: resize ol to 1024x4096 nearest
            gt_full = cv2.resize(ol.astype(np.uint8),(W_full,1024),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            seam_hist += fast_hist(gt_full[band_mask].flatten(), fused[band_mask].flatten().astype(np.int64), num_classes)
    ious = per_class_iu(hist); s_ious = per_class_iu(seam_hist)
    out = {'tag':tag,'mode':mode,'B':B,'n_img':n_img,'mIoU':round(float(np.nanmean(ious))*100.0,4),
           'seam_mIoU':round(float(np.nanmean(s_ious))*100.0,4),
           'eval_time_sec':round(time.perf_counter()-t0,2),'n_extra_used':n_extra_used,
           'extra_time_sec':round(t_extra,2),'fused_px_total':fused_px,
           'cost':cost_report(2*L,B,1024,1024)}
    print('RESULT '+json.dumps(out), flush=True)
    json.dump(out, open(f'/root/autodl-tmp/piers-main/runs/{tag}/summary.json','w'), indent=2)
if __name__=='__main__':
    import sys as s
    run(s.argv[1], B=int(s.argv[2]), mode=s.argv[3], n_img=int(s.argv[4]))
