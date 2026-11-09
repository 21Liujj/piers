"""Version 2: text-evidence competition in a requested category space.

This module consumes no dataset identifier, class IDs or labels. It retains
all raw text detections in its audit. The closed semantic head cannot veto a
novel category. Text confidences are ranking evidence, not class posteriors.

Changes relative to v1:
* Correlated overlapping FoVs are averaged, not treated as independent trials.
* New classes compete with text evidence for the requested known classes.
* Residual nouns (other/static/moving object, ground) exclude independently
  text-supported specific concepts in the same requested category list.
"""
from __future__ import annotations

from dataclasses import asdict
import re
import numpy as np

from .open_vocab import OpenVocabConfig

VERSION = 'exclusive-text-competition-v2-20261109'


def normalize_name(name):
    return ' '.join(str(name).strip().lower().replace('_', ' ').split())


def residual_kind(name):
    """Language roles, independent of any benchmark's numerical labels."""
    name = normalize_name(name)
    if name in {'ground', 'ground surface'}:
        return 'ground_surface'
    if re.fullmatch(r'(other|miscellaneous|unclassified|unknown)( (object|objects|thing|things|stuff))?', name):
        return 'other'
    if re.fullmatch(r'(static|stationary|moving|dynamic) (object|objects|thing|things)', name):
        return 'generic_object'
    return None


def is_specific_surface(name):
    tokens = set(re.findall(r'[a-z]+', normalize_name(name)))
    return bool(tokens & {'road', 'roads', 'street', 'streets', 'sidewalk', 'sidewalks',
                          'pavement', 'pavements', 'footpath', 'crosswalk', 'crosswalks',
                          'curb', 'curbs', 'kerb', 'kerbs', 'terrain', 'soil', 'sand',
                          'grass', 'gravel', 'railway', 'railroad', 'water'})


def build_competitive_text_evidence(candidates, class_names, shape, config=OpenVocabConfig()):
    """Build aligned evidence for ALL requested classes, without semantic input.

    Association and text-quality thresholds are unchanged from v1. Within a
    physical FoV duplicates contribute once. A group's pixel score is the mean
    over its distinct FoVs that can see that pixel, including zero votes where
    a visible FoV's associated mask does not cover it. Repeated observations
    cannot increase evidence beyond the strongest actual observation.
    """
    names = tuple(class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Requested class names must be nonempty and unique')
    if len(shape) != 2 or any(int(x) < 1 for x in shape):
        raise ValueError('Expected positive height and width')
    if config.distinct_views_min < 1:
        raise ValueError('distinct_views_min must be positive')
    evidence = np.zeros((len(names), *shape), np.float32)
    logs, statistics = [], {}
    for ci, name in enumerate(names):
        accepted = []
        rows = [c for c in candidates if c['class_name'] == name]
        for cand in rows:
            mask, valid = np.asarray(cand['mask'], bool), np.asarray(cand['valid'], bool)
            if mask.shape != tuple(shape) or valid.shape != tuple(shape):
                raise ValueError('Candidate coordinates differ from the requested output')
            det, quality = float(cand['detector_score']), float(cand['sam_quality'])
            if not np.isfinite([det, quality]).all():
                raise ValueError('Nonfinite text confidence')
            keep = det >= config.detector_min and quality >= config.sam_quality_min and bool((mask & valid).any())
            log = {'id': cand.get('id'), 'class_name': name, 'view_id': cand['view_id'],
                   'detector_score': det, 'sam_quality': quality, 'kept': keep,
                   'emitted': False, 'distinct_views': 1, 'teacher_veto': False,
                   'reason': 'text_quality_pass' if keep else 'text_quality_or_empty'}
            logs.append(log)
            if keep:
                accepted.append((cand, mask & valid, valid, log))
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
        distinct_counts = []
        for group in groups.values():
            views = sorted({row[0]['view_id'] for row in group}, key=str)
            distinct_counts.append(len(views))
            eligible = len(views) >= config.distinct_views_min
            for _, _, _, log in group:
                log.update(distinct_views=len(views), emitted=eligible,
                           reason='text_multiview_mean' if eligible else 'single_view_retained')
            if not eligible:
                continue
            total = np.zeros(shape, np.float32)
            visible_count = np.zeros(shape, np.uint16)
            for view in views:
                vote, visible = np.zeros(shape, np.float32), np.zeros(shape, bool)
                for cand, mask, valid, _ in group:
                    if cand['view_id'] == view:
                        score = float(np.clip(cand['detector_score'], 0, 1) * np.clip(cand['sam_quality'], 0, 1))
                        np.maximum(vote, mask * score, out=vote)
                        visible |= valid
                total += vote
                visible_count += visible
            mean = np.divide(total, visible_count, out=np.zeros_like(total), where=visible_count > 0)
            np.maximum(evidence[ci], mean, out=evidence[ci])
        statistics[name] = {'candidates': len(rows), 'text_quality_passed': len(accepted),
                            'groups': len(groups), 'group_distinct_views': distinct_counts,
                            'emitted_candidates': sum(r[3]['emitted'] for r in accepted),
                            'evidence_coverage_pct': float(100 * (evidence[ci] > 0).mean())}
    return evidence, {'version': VERSION, 'config': asdict(config), 'classes': statistics,
                      'candidates': logs, 'view_aggregation': 'mean_of_visible_distinct_views',
                      'score_kind': 'uncalibrated_text_ranking_evidence',
                      'no_circular_evidence': True, 'semantic_veto': False}


def exclude_specific_from_residual(evidence, class_names):
    """Residual terms are complements of text-supported specific concepts.

    This resolves overlaps of language concepts, not a claim that the teacher
    recognises new categories. Nothing is suppressed using semantic logits.
    All input detections and pre-exclusion evidence remain auditable.
    """
    values = np.asarray(evidence, np.float32)
    if values.ndim != 3 or len(class_names) != values.shape[0]:
        raise ValueError('Class/evidence mismatch')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Invalid text evidence')
    out = values.copy()
    kinds = [residual_kind(c) for c in class_names]
    specific = [i for i, kind in enumerate(kinds) if kind is None]
    audit = []
    for i, kind in enumerate(kinds):
        if kind is None:
            continue
        competitors = [j for j in specific if kind != 'ground_surface' or is_specific_surface(class_names[j])]
        covered = np.zeros(values.shape[1:], bool)
        for j in competitors:
            covered |= values[j] > 0
        removed = (out[i] > 0) & covered
        out[i, covered] = 0
        audit.append({'class_name': class_names[i], 'language_role': kind,
                      'specific_competitors': [class_names[j] for j in competitors],
                      'excluded_pixels': int(removed.sum()),
                      'exclusion_evidence': 'independent_specific_text_masks', 'teacher_veto': False})
    return out, audit


def _labels_from_adjusted_text(known_labels, class_names, source_channels, adjusted, config):
    known = [i for i, c in enumerate(source_channels) if c is not None]
    novel = sorted([i for i, c in enumerate(source_channels) if c is None],
                   key=lambda i: normalize_name(class_names[i]))
    pred = np.asarray(known_labels, np.uint8).copy()
    if not novel:
        return pred
    best = np.asarray(novel)[adjusted[novel].argmax(0)]
    strength = adjusted[novel].max(0)
    if known:
        reference = np.maximum(adjusted[known].max(0), config.detector_min * config.sam_quality_min)
        emit = (strength > reference) | ((pred == 255) & (strength > 0))
    else:
        emit = strength > 0
    pred[emit] = best[emit]
    return pred


def compete_from_known_labels(known_labels, class_names, source_channels, text_evidence,
                              config=OpenVocabConfig()):
    """Exactly the label decision of the probability API, using cached labels.

    The known-head peak cancels in the ordinal novel/known comparison. This
    permits lossless reuse of a frozen common-class prediction, without
    inventing or reconstructing semantic probability maps.
    """
    labels = np.asarray(known_labels)
    text = np.asarray(text_evidence, np.float32)
    known = [i for i, c in enumerate(source_channels) if c is not None]
    if text.ndim != 3 or text.shape[0] != len(class_names) or labels.shape != text.shape[1:]:
        raise ValueError('Cached known labels and text coordinates differ')
    if not set(np.unique(labels).tolist()) <= set(known) | {255}:
        raise ValueError('Cached labels must contain only requested known classes')
    adjusted, exclusions = exclude_specific_from_residual(text, class_names)
    pred = _labels_from_adjusted_text(labels, class_names, source_channels, adjusted, config)
    return pred, {'version': VERSION, 'residual_exclusions': exclusions,
                  'competition': 'novel_text_vs_requested_known_text_on_same_score_scale',
                  'cached_label_equivalence': 'known_peak_cancels_exactly_in_ordinal_comparison',
                  'teacher_veto': False, 'no_circular_evidence': True,
                  'novel_pixel_share_pct': float(100 * np.isin(pred, [i for i,c in enumerate(source_channels) if c is None]).mean())}


def compete_requested_classes(semantic13, class_names, source_channels, text_evidence,
                              config=OpenVocabConfig()):
    """Ordinal novel/known comparison; the semantic head cannot veto novel text.

    Keep semantic ranking within the known classes. At a pixel, a novel score
    beats the leading known score exactly when its text evidence exceeds ALL
    requested known-class text evidence (or the fixed quality floor if known
    text has no support). This uses comparable detector/SAM units. The final
    normalisation provides nonnegative ranking weights, not calibrated odds.
    """
    p, text = np.asarray(semantic13, np.float32), np.asarray(text_evidence, np.float32)
    if p.ndim != 3 or p.shape[0] != 13 or text.shape != (len(class_names), *p.shape[1:]):
        raise ValueError('Expected aligned source head and requested text evidence')
    if len(source_channels) != len(class_names) or len(set(class_names)) != len(class_names):
        raise ValueError('Invalid requested class registry')
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid semantic scores')
    known = [i for i, c in enumerate(source_channels) if c is not None]
    novel = [i for i, c in enumerate(source_channels) if c is None]
    scores = np.zeros_like(text)
    if known:
        base = p[[source_channels[i] for i in known]]
        denom = base.sum(0, keepdims=True)
        scores[known] = np.divide(base, denom, out=np.zeros_like(base), where=denom > 0)
    adjusted, exclusions = exclude_specific_from_residual(text, class_names)
    if novel:
        if known:
            floor = config.detector_min * config.sam_quality_min
            if floor <= 0:
                raise ValueError('A positive source-independent quality floor is required')
            reference = np.maximum(adjusted[known].max(0), floor)
            known_peak = scores[known].max(0)
            # If the requested known head is empty, positive novel text can
            # still emit instead of being vetoed by a zero semantic score.
            scale = np.where(known_peak > 0, known_peak, 1.0)
            scores[novel] = (adjusted[novel] / reference[None]) * scale[None]
        else:
            scores[novel] = adjusted[novel]
    total = scores.sum(0, keepdims=True)
    scores = np.divide(scores, total, out=np.zeros_like(scores), where=total > 0)
    known_labels = np.full(p.shape[1:], 255, np.uint8)
    if known:
        order = np.asarray(sorted(known, key=lambda i: source_channels[i]))
        known_labels = order[scores[order].argmax(0)].astype(np.uint8)
        known_labels[scores[known].max(0) <= 0] = 255
    pred = _labels_from_adjusted_text(known_labels, class_names, source_channels, adjusted, config)
    audit = {'version': VERSION, 'residual_exclusions': exclusions,
             'competition': 'novel_text_vs_requested_known_text_on_same_score_scale',
             'score_kind': 'normalised_ranking_weights_not_calibrated_probabilities',
             'teacher_veto': False, 'no_circular_evidence': True,
             'novel_pixel_share_pct': float(100 * np.isin(pred, novel).mean())}
    return scores, pred, audit
