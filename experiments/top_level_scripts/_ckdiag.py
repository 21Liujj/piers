"""Diagnostic: strict-load the I1 checkpoint into OmniSAM_adapt and report gaps."""
import sys, types, torch
sys.path.insert(0, '/root/autodl-tmp/eff_omni')


def shim():
    try:
        import importlib.util
        if importlib.util.find_spec('model.modules.dynamic_fsmc') is not None:
            return
    except Exception:
        pass

    class DynamicFSMC:
        def __init__(self, *a, **k):
            raise RuntimeError('no fsmc weights in ckpt')

        def __call__(self, *a, **k):
            raise RuntimeError('no fsmc weights in ckpt')

    m = types.ModuleType('model.modules.dynamic_fsmc')
    m.DynamicFSMC = DynamicFSMC
    sys.modules['model.modules.dynamic_fsmc'] = m


shim()
import utils.sam2_bootstrap  # noqa
from model.model import OmniSAM_adapt
from utils.build import build_sam2, SAM_CFGS

CK = sys.argv[1] if len(sys.argv) > 1 else (
    '/root/autodl-tmp/piers-main/runs/I1-densepass-20261107_121457/'
    'allhard_runs/R0_seed1234/training/final.pth')

device = torch.device('cuda')
model_cfg, ckpt_name = SAM_CFGS['sam2_s']
cfg = model_cfg if model_cfg.endswith('.yaml') else (
    'configs/sam2.1/sam2.1_hiera_s.yaml')
sam = build_sam2(cfg, '/root/autodl-tmp/eff_omni/sam/checkpoints/sam2.1_hiera_small.pt',
                  device=device)
m = OmniSAM_adapt(sam, num_classes=13, num_maskmem=9).to(device)

raw = torch.load(CK, map_location='cpu')
if isinstance(raw, dict) and 'model' in raw and isinstance(raw['model'], dict):
    raw = raw['model']
msd = m.state_dict()
print('ckpt tensors :', len(raw))
print('model tensors :', len(msd))
print('ckpt prefixes:', sorted(set(k.split('.')[0] for k in raw)))
print('model prefixes:', sorted(set(k.split('.')[0] for k in msd)))

# exact key match report
missing = [k for k in msd if k not in raw]
unexpected = [k for k in raw if k not in msd]
shape_mismatch = [(k, tuple(raw[k].shape), tuple(msd[k].shape))
                  for k in msd if k in raw and raw[k].shape != msd[k].shape]
print('\nMISSING in ckpt :', len(missing))
for k in missing[:25]:
    print('   -', k, tuple(msd[k].shape))
print('\nUNEXPECTED in ckpt:', len(unexpected))
for k in unexpected[:25]:
    print('   +', k, tuple(raw[k].shape))
print('\nSHAPE MISMATCH:', len(shape_mismatch))
for k, a, b in shape_mismatch[:25]:
    print('   ~', k, 'ckpt', a, 'model', b)

# how many model params actually come from ckpt
loaded = sum(1 for k in msd if k in raw and raw[k].shape == msd[k].shape)
nparam_loaded = sum(msd[k].numel() for k in msd if k in raw and raw[k].shape == msd[k].shape)
nparam_total = sum(v.numel() for v in msd.values())
print(f'\nloaded tensors: {loaded}/{len(msd)}')
print(f'loaded params : {nparam_loaded:,} / {nparam_total:,} '
      f'({100.0 * nparam_loaded / nparam_total:.2f}%)')
