"""Text-only novel-class evidence and competition with a frozen semantic head.

Detector/SAM scores are heuristic evidence, not calibrated class probabilities.
No GT, semantic veto, or recursively refined evidence is used here. Single-view
candidates remain in the audit even when a text-only consistency rule prevents
their emission. Uncovered novel pixels have ZERO evidence, never a fake prior.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OpenVocabConfig:
    detector_min: float = 0.25
    sam_quality_min: float = 0.5
    distinct_views_min: int = 2
    association_iou: float = 0.3


def build_novel_evidence(candidates, class_names, shape, config=OpenVocabConfig()):
    """Return (E,H,W) evidence and an audit; support is never read from teacher.

    Candidate schema: class_name, mask, valid, view_id, detector_score,
    sam_quality. Class association uses IoU on shared visible pixels. Within
    each view overlapping detections count once (maximum confidence), then
    distinct-view evidence combines as 1 - product(1 - confidence).
    """
    names = tuple(class_names)
    if len(set(names)) != len(names):
        raise ValueError('novel class names must be unique')
    if config.distinct_views_min < 1:
        raise ValueError('distinct_views_min must be positive')
    evidence = np.zeros((len(names), *shape), np.float32)
    logs, stats = [], {}
    for ci, name in enumerate(names):
        accepted = []
        rows = [c for c in candidates if c['class_name'] == name]
        for cand in rows:
            mask = np.asarray(cand['mask'], bool)
            valid = np.asarray(cand['valid'], bool)
            if mask.shape != tuple(shape) or valid.shape != tuple(shape):
                raise ValueError('candidate and semantic coordinates differ')
            det, quality = float(cand['detector_score']), float(cand['sam_quality'])
            if not np.isfinite([det, quality]).all():
                raise ValueError('nonfinite text confidence')
            passed = det >= config.detector_min and quality >= config.sam_quality_min and bool((mask & valid).any())
            entry = {'id': cand.get('id'), 'class_name': name,
                     'view_id': cand['view_id'], 'detector_score': det,
                     'sam_quality': quality, 'kept': passed,
                     'reason': 'text_only_confidence' if passed else 'text_quality_or_empty',
                     'teacher_veto': False, 'distinct_views': 1, 'emitted': False}
            logs.append(entry)
            if passed:
                accepted.append((cand, mask & valid, valid, entry))
        # Connect detections only through actual overlap from DISTINCT views.
        parent = list(range(len(accepted)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, (a, ma, va, _) in enumerate(accepted):
            for j in range(i):
                b, mb, vb, _ = accepted[j]
                if a['view_id'] == b['view_id']:
                    continue
                shared = va & vb
                inter = np.count_nonzero(ma & mb & shared)
                union = np.count_nonzero((ma | mb) & shared)
                if union and inter / union >= config.association_iou:
                    parent[find(i)] = find(j)
        groups = {}
        for i, row in enumerate(accepted):
            groups.setdefault(find(i), []).append(row)
        group_counts = []
        for group in groups.values():
            views = {row[0]['view_id'] for row in group}
            group_counts.append(len(views))
            eligible = len(views) >= config.distinct_views_min
            for _, _, _, log in group:
                log.update(distinct_views=len(views), emitted=eligible,
                           reason='ov_text_multiview' if eligible else 'ov_single_view_retained')
            if not eligible:
                continue
            # One vote per physical FoV: aliases and duplicate boxes never
            # masquerade as independent observations.
            miss = np.ones(shape, np.float32)
            for view in views:
                vote = np.zeros(shape, np.float32)
                for cand, mask, _, _ in group:
                    if cand['view_id'] == view:
                        score = np.clip(cand['detector_score'], 0, 1) * np.clip(cand['sam_quality'], 0, 1)
                        np.maximum(vote, mask * score, out=vote)
                miss *= 1 - vote
            np.maximum(evidence[ci], 1 - miss, out=evidence[ci])
        stats[name] = {'candidates': len(rows), 'text_quality_passed': len(accepted),
                       'groups': len(groups), 'group_distinct_views': group_counts,
                       'emitted_candidates': sum(x['emitted'] for x in logs if x['class_name'] == name),
                       'evidence_coverage_pct': float((evidence[ci] > 0).mean() * 100)}
    return evidence, {'classes': stats, 'candidates': logs,
                      'score_kind': 'uncalibrated_text_evidence',
                      'no_circular_evidence': True, 'semantic_veto': False}


def fuse_probability_space(base_probs, novel_evidence):
    """Allocate novel mass without asking the closed-set head to recognise it.

    For novel channels e: u=max(e), novel mass=u*e/sum(e), old mass=(1-u)*p.
    The returned C+E channels sum to one. Inputs are never modified. This is
    a fixed, auditable fusion rule, not learned probability calibration.
    """
    base = np.asarray(base_probs, np.float32)
    novel = np.asarray(novel_evidence, np.float32)
    if base.ndim != 3 or novel.ndim != 3 or base.shape[1:] != novel.shape[1:]:
        raise ValueError('expected aligned (classes,height,width) arrays')
    if not np.isfinite(base).all() or not np.isfinite(novel).all() or (base < 0).any() or (novel < 0).any() or (novel > 1).any():
        raise ValueError('invalid probability/evidence values')
    denom = base.sum(axis=0, keepdims=True)
    if (denom <= 0).any():
        raise ValueError('semantic probabilities must cover every pixel')
    base = base / denom
    strength = novel.max(axis=0, keepdims=True)
    novel_sum = novel.sum(axis=0, keepdims=True)
    novel_mass = np.divide(novel * strength, novel_sum, out=np.zeros_like(novel), where=novel_sum > 0)
    return np.concatenate([base * (1 - strength), novel_mass], axis=0)


WILDPASS_CLASSES = ('road', 'sidewalk', 'curb', 'crosswalk', 'vehicle', 'person', 'truck', 'bus')
WILDPASS_NOVEL = ('curb', 'crosswalk', 'truck', 'bus')


def wildpass_native_probabilities(semantic13, novel_evidence):
    """Assemble all EIGHT channels before argmax; never project hard labels.

    Known channels use the same taxonomy-conditioned four-channel competition
    as the legacy baseline. Vehicle is the old coarse channel; text channels
    can split truck/bus away from it. Raw-8 GT remains untouched.
    """
    semantic13 = np.asarray(semantic13)
    if semantic13.shape[0] != 13 or novel_evidence.shape[0] != 4:
        raise ValueError('WildPASS requires 13 semantic and 4 novel channels')
    combined = fuse_probability_space(semantic13[[0, 1, 12, 11]], novel_evidence)
    # combined: road, sidewalk, vehicle, person, curb, crosswalk, truck, bus
    return combined[[0, 1, 4, 5, 2, 3, 6, 7]]
