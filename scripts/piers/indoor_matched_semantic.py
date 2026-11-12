"""Indoor port of the published semantic engine; both arms share all weights.

The indoor canvas changes, while the sequential memory schedule, probability
merge, sparse-memory policy and coverage-aware seam fusion retain the outdoor
formulas. No label files are read by this module.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from indoor_geometry import semantic_starts, i3_candidate_starts, INDOOR_W, INDOOR_H, INDOOR_WIN
from indoor_seam_i3 import choose, semantic_risk, fuse, base_coverage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lora_sha(state):
    digest = hashlib.sha256()
    keys = sorted(k for k in state if '.linear_a_' in k or '.linear_b_' in k)
    for key in keys:
        digest.update(key.encode()); digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest(), len(keys)


class IndoorSemanticEngine:
    def __init__(self, checkpoint, memory='dense', budget=0, rare_ids=(4, 7)):
        from indoor_eval_runner import build_model
        from indoor_memory_i2 import install_dense_memory
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.manual_seed(1234); np.random.seed(1234)
        torch.backends.cudnn.benchmark = False
        self.checkpoint = str(Path(checkpoint).resolve())
        self.checkpoint_sha256 = sha(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload.get('model', payload)
        assert tuple(state['decoder.linear_pred.weight'].shape) == (8, 256, 1, 1)
        self.lora_sha256, self.lora_tensors = lora_sha(state)
        self.model, _ = build_model('A0', self.checkpoint, 'cuda')
        loaded = self.model.state_dict()
        assert all(torch.equal(v.cpu(), loaded[k].cpu()) for k, v in state.items())
        self.model.requires_grad_(False).eval()
        self.memory, self.budget = memory, int(budget)
        assert memory in ('dense', 'i2') and budget in (0, 1, 2)
        self.audit = {'calls': 0, 'fallbacks': 0, 'max_tokens': 0}
        mb = self.model.memory_bank
        if memory == 'dense':
            install_dense_memory(self.model)
        else:
            mb.sparse_retriever.diagnostic_overlap = False
            mb.configure_sparse_retrieval(enabled=True, mode='class_aware', keep_ratio=.03125,
                min_tokens=64, cache_enabled=False, fallback='feature_norm',
                fallback_full_when_uncertain=True, enable_in_train=True,
                enable_in_pseudo_label=True, boundary_reserve_ratio=.125,
                rare_class_ids=list(rare_ids), rare_reserve_ratio=.15,
                boundary_score_source='uncertainty')
            mb.set_runtime_context(stage='eval', iter_idx=0)
            retrieve = mb.sparse_retriever.retrieve
            def checked(*args, **kwargs):
                result = retrieve(*args, **kwargs)
                self.audit['calls'] += 1
                self.audit['fallbacks'] += int(result[3].fallback_used)
                assert not result[3].fallback_used, str(result[3])
                assert result[3].selection_mode == 'class_aware'
                assert kwargs.get('uncertainty_map') is not None
                assert kwargs.get('rare_class_mask') is not None
                return result
            mb.sparse_retriever.retrieve = checked
        self.trace = []
        def hook(module, args, kwargs):
            length = int(kwargs['memory'].shape[0])
            self.audit['max_tokens'] = max(self.audit['max_tokens'], length)
            self.trace.append(length)
        mb.sam_model.memory_attention.register_forward_pre_hook(hook, with_kwargs=True)
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    @torch.no_grad()
    def predict(self, image):
        from utils.tools import merge_width_scatter_np
        model = self.model; mb = model.memory_bank
        with Image.open(image) as pil:
            assert pil.size == (INDOOR_W, INDOOR_H), pil.size
            rgb = np.asarray(pil.convert('RGB')).copy()
        tensor = ((torch.from_numpy(rgb).permute(2, 0, 1).float()/255-self.mean)/self.std).cuda()
        for key in self.audit: self.audit[key] = 0
        self.trace.clear()
        torch.cuda.synchronize(); started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        logs = np.empty((1, 18, 8, INDOOR_H, INDOOR_WIN), np.float32)
        contexts = []; calls = 0
        starts = semantic_starts()
        for pas, order in enumerate((starts, list(reversed(starts)))):
            mb._init_output_dict_source(); labels = {}
            for f, start in enumerate(order):
                out = model(f, tensor[:, :, start:start+INDOOR_WIN].unsqueeze(0).contiguous(),
                            historical_pseudo_labels=labels or None)
                pred = out[0] if isinstance(out, (tuple, list)) else out
                calls += 1; labels[f] = pred.argmax(1).detach()
                logs[0, pas*9+f] = pred[0].float().cpu().numpy()
            contexts.append(({k:v for k,v in mb.output_dict_source['frame_outputs'].items() if k>0},
                             {k:v for k,v in labels.items() if k>0}))
        mb._init_output_dict_source()
        sums = merge_width_scatter_np(logs, (INDOOR_H, INDOOR_W), INDOOR_WIN, 64, 2, 'cuda', True)[0]
        probs = sums/np.maximum(sums.sum(0, keepdims=True), 1e-8)
        label = sums.argmax(0).astype(np.uint8)
        # The outdoor engine stores the base distribution as float16 before I3.
        probs = probs.astype(np.float16)
        del logs, sums
        # Same five-candidate seam layout as outdoor: centre W-win/2,
        # candidates spaced win/8, and minimum separation win/4.
        candidates = [int((INDOOR_W-INDOOR_WIN//2+k*INDOOR_WIN//8)%INDOOR_W) for k in range(-2,3)]
        plan = choose(semantic_risk(probs), candidates, INDOOR_WIN,
                      self.budget, min_separation=INDOOR_WIN//4)[0] if self.budget else []
        extras = []
        for pas, start in enumerate(plan):
            context, labels = contexts[pas]
            mb._init_output_dict_source(); mb.output_dict_source = {'frame_outputs': dict(context)}
            cols = (start+torch.arange(INDOOR_WIN, device='cuda')) % INDOOR_W
            crop = tensor.index_select(2, cols).unsqueeze(0).contiguous()
            out = model(9, crop, historical_pseudo_labels=labels)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            calls += 1
            extras.append((start, pred.softmax(1)[0].cpu().numpy().astype(np.float16)))
        mb._init_output_dict_source()
        probs, result, fusion = fuse(probs, label, extras, base_coverage())
        assert calls == 18+self.budget
        if not self.budget: assert np.array_equal(label, result)
        torch.cuda.synchronize()
        report = {'checkpoint_sha256': self.checkpoint_sha256, 'lora_sha256': self.lora_sha256,
            'lora_tensors': self.lora_tensors, 'memory': self.memory, 'I3_budget': self.budget,
            'forward_calls': calls, 'I3_selected_windows': plan, 'I3_fusion': fusion,
            'memory_audit': dict(self.audit), 'attention_token_lengths': list(self.trace),
            'elapsed_s': time.perf_counter()-started,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'weights_updated': False, 'ground_truth_reads': 0,
            'merge': 'upstream_softmax_probability_merge', 'text_logit_branch': False}
        return probs.astype(np.float16), result, report
