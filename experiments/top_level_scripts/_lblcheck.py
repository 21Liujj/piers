"""Diagnostic: what label values does the eval actually compare against?"""
import sys, os, numpy as np
sys.path.insert(0, '/root/autodl-tmp/eff_omni')
from PIL import Image

ROOT = '/root/autodl-tmp/data_extra/WildPASS13'
LBL = os.path.join(ROOT, 'labels_trainid')

from utils.wildpass_label_convert import convert_wildpass_raw_to_train13

for f in ['1.png', '10.png', '100.png']:
    raw = np.array(Image.open(os.path.join(LBL, f))).astype('int32')
    mapped, enc = convert_wildpass_raw_to_train13(raw)
    u_raw = np.unique(raw)
    u_map = np.unique(mapped)
    print(f'--- {f} ---')
    print('  raw   uniq:', u_raw[:20], 'n=', len(u_raw))
    print('  mapped uniq:', u_map[:20], 'n=', len(u_map), 'enc=', enc)
    # pixel fraction per class
    tot = raw.size
    for v in u_raw:
        if v == 255:
            continue
        frac = float((raw == v).sum()) / tot
        print(f'    class {int(v):3d}: {frac*100:6.2f}%')
