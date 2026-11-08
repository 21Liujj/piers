"""E0 semantic probs dump for Innovation-1 T2 (frozen teacher cache).
Reuses eff_omnisam validation pipeline with E0-effective-start frozen settings:
  backbone sam2_s, DensePASS class_set 13, no SCM, use_mem_bank (num_maskmem 9),
  ckpt /root/autodl-tmp/weights/omnisam_s.pth, data_root /root/autodl-tmp/data.
Output: runs/E0-effective-start/semantics/probs.npz
  keys = manifest image_id, value = (5,400,2048) uint8 quantized probs for
  channels [0 road, 8 vegetation, 10 sky, 11 person, 12 car] (the only classes
  with exact/coarse_compatible queries in manifest.dev-calib-100).
  meta keys: __channels__, __class_names__, __ckpt__, __note__.
Probs merged with merge_width_scatter_np(return_probs=True) at 1024x4096,
then bilinear-resized to ERP eval grid 400x2048 (same grid as text masks).
"""
import sys, os, json, hashlib
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import numpy as np
import torch, cv2
from utils.tools import merge_width_scatter_np
import validation as V

P = '/root/autodl-tmp/piers-main'
CKPT = '/root/autodl-tmp/weights/omnisam_s.pth'
MAN = f'{P}/configs/piers/manifest.dev-calib-100.json'
OUTDIR = f'{P}/runs/E0-effective-start/semantics'
os.makedirs(OUTDIR, exist_ok=True)

class A: pass
args = A()
args.seed = 1234; args.backbone = 'sam2_s'; args.dataset = 'DensePASS'
args.class_set = 13; args.data_root = '/root/autodl-tmp/data'
args.batch_size = 1; args.use_mem_bank = True; args.num_maskmem = 9
args.enable_scm = False; args.scm_pad_width = 512
args.enable_bepl = False; args.bepl_seam_width = 128; args.bepl_strength = 0.2
args.stanford_stride_w = 256; args.stanford_legacy_area_crop = False
args.wildpass_layout = 'cityscapes'; args.wildpass_processed_root = ''
args.val_list = ''; args.wildpass_image_dir = 'images'; args.wildpass_label_dir = 'labels_trainid'
args.stanford_processed_root = ''; args.stanford_val_list = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}', flush=True)
man = json.load(open(MAN))
src2id = {im['src_file']: im['image_id'] for im in man['images']}
print(f'manifest: {len(man["images"])} images', flush=True)

val_loader, class_names, num_classes = V.build_dataset_and_loader(args)
print(f'classes={class_names} num={num_classes} batches={len(val_loader)}', flush=True)
args.num_classes = num_classes
model = V.build_sam_model(args, device, ckpt_path=CKPT).to(device)
model.eval()

CH = [0, 8, 10, 11, 12]
print(f'channels={CH} = {[class_names[c] for c in CH]}', flush=True)
out = {}
with torch.no_grad():
    for index, batch in enumerate(val_loader):
        image, label, _, names, ori_label = batch
        image = image.to(device)
        B = image.shape[0]; H, W = image.shape[-2:]
        num_frames = val_loader.dataset._get_frame_cnt()
        logits_maps = np.zeros((B, 2 * num_frames, len(class_names), H, W))
        for frame_idx in range(num_frames):
            frame_img = image[:, frame_idx]
            output, feat, feat_de = model(frame_idx, frame_img)
            logits_maps[:, frame_idx] = output.cpu().numpy()
        model.memory_bank._init_output_dict_source()
        for frame_idx in range(num_frames):
            frame_img = image[:, num_frames + frame_idx]
            output, _, _ = model(frame_idx, frame_img)
            logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()
        model.memory_bank._init_output_dict_source()
        probs = merge_width_scatter_np(logits_np=logits_maps, orig_size=(1024, 4096),
                                       W_patch=W, stride_w=384, passes=2,
                                       device=str(device), return_probs=True)  # (B,C,1024,4096)
        for i in range(B):
            name = names[i] if isinstance(names, (list, tuple)) else names
            if isinstance(name, bytes): name = name.decode()
            name = str(name)
            iid = src2id.get(name) or src2id.get('/' + name) or src2id.get(name.lstrip('/'))
            if iid is None:
                # fallback: match by stem number
                stem = name.replace('.png', '')
                cand = [v for k, v in src2id.items() if k.replace('.png', '').strip('/') == stem.strip('/')]
                iid = cand[0] if cand else name
            arr = probs[i]  # (13,1024,4096) float
            small = np.zeros((len(CH), 400, 2048), dtype=np.uint8)
            for ci, c in enumerate(CH):
                r = cv2.resize(arr[c], (2048, 400), interpolation=cv2.INTER_LINEAR)
                small[ci] = np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)
            out[iid] = small
            print(f'[{index+1}/{len(val_loader)}] {name} -> {iid} probs {probs[i].shape} saved {small.shape}', flush=True)

sha = hashlib.sha256(open(CKPT, 'rb').read()).hexdigest()[:16]
meta = {'channels': np.array(CH), 'class_names': np.array(class_names, dtype=object),
        'ckpt': np.array(CKPT, dtype=object), 'ckpt_sha256_prefix': np.array(sha, dtype=object)}
save_dict = {**out, '__channels__': meta['channels'], '__class_names__': meta['class_names'],
             '__ckpt__': meta['ckpt'], '__ckpt_sha256_prefix__': meta['ckpt_sha256_prefix']}
np.savez_compressed(os.path.join(OUTDIR, 'probs.npz'), **save_dict)
print(f'DONE: {len(out)} images -> {OUTDIR}/probs.npz', flush=True)
print(f'ckpt sha256 prefix: {sha}', flush=True)
