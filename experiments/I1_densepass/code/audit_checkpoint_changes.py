"""Prove decoder-only continuation by comparing every saved tensor to the initial checkpoint."""
from pathlib import Path
import torch,json,time
R=Path(__file__).resolve().parent
torch.set_num_threads(1)
a=torch.load('/root/autodl-tmp/weights/omnisam_s.pth',map_location='cpu',weights_only=True);a={k.removeprefix('module.'):v for k,v in a.items()};rows=[]
for base in ['final_runs','allhard_runs','dense_controls/trained']:
 for p in sorted((R/base).glob('*/training/final.pth')):
  b=torch.load(p,map_location='cpu',weights_only=True);missing=sorted(set(a)-set(b));extra=sorted(set(b)-set(a));changed=[k for k in a.keys()&b.keys() if not torch.equal(a[k],b[k])];frozen_changed=[k for k in changed if not k.startswith('decoder.')]
  assert not missing and not extra,(str(p),missing,extra);assert not frozen_changed,(str(p),frozen_changed);assert changed
  rows.append({'checkpoint':str(p.relative_to(R)),'tensors_compared':len(a),'changed_tensors':changed,'non_decoder_changed':frozen_changed,'max_decoder_abs_delta':max(float((a[k].float()-b[k].float()).abs().max()) for k in changed)});del b
report={'status':'passed','at_unix':time.time(),'method':'All saved tensors, including frozen encoder/LoRA/Memory weights and BN buffers, compared to original same checkpoint. Only decoder state may differ; decoder BN running stats are included.','checkpoints':rows}
(R/'reports/checkpoint_tensor_audit.json').write_text(json.dumps(report,indent=2));print('CHECKPOINT_TENSOR_AUDIT',len(rows),'passed',flush=True)
