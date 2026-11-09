"""Frozen I1/I2/I3 plus versioned dataset-independent text competition.

Legacy v1 sources are retained verbatim. Only the final requested-class
competition and its text-evidence aggregation change in this release.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from full_model_fusion import (
    FrozenCooperation as LegacyCooperation, SOURCE_CLASSES, ALIASES,
    make_queries, normalize, source_channel, requested_probabilities,
)


class FrozenCooperation(LegacyCooperation):
    def apply(self, meta, masks, probs, stored, bank_ids, output_query_ids, classes):
        from feature_core import build_context
        from fresh_core import proposals, soft_correct, arm_counts
        from piers_ljj.collaboration.reliable_cooperation import prepare, bounded_text_fusion
        from piers_ljj.collaboration.open_vocab_competition import (
            VERSION, build_competitive_text_evidence, compete_requested_classes,
        )
        qmeta = {q['query_id']: q for q in meta['queries']}
        bank = set(bank_ids)
        take = [i for i, c in enumerate(meta['candidates']) if c['query_id'] in bank]
        semmeta = {**meta, 'queries': [q for q in meta['queries'] if q['query_id'] in bank],
                   'candidates': [meta['candidates'][i] for i in take]}
        ctx = build_context(semmeta, masks[np.asarray(take, np.int64)], probs, stored, qmeta)
        text, teacher = self.text, self.teacher

        class Pair:
            def predict_np(self, X):
                p = text.predict_np(X)
                p[:, 1] = teacher.predict_np(X)[:, 1]
                return p

        arrays, gate_audit = proposals(ctx, Pair(), self.gate)
        lab, weight = arrays['label'], arrays['weight']
        eligible = (lab != 255) & (weight > 0)
        corrected, alpha, winner, error = soft_correct(ctx.probs, stored, lab, weight, eligible)
        corrected_probs = ctx.probs.copy()
        rr, cc = np.nonzero(eligible)
        if len(rr):
            values = corrected_probs[:, rr, cc].copy() * (1 - alpha[None, :])
            values[lab[rr, cc], np.arange(len(rr))] += alpha
            assert np.array_equal(values.argmax(0), winner)
            corrected_probs[:, rr, cc] = values
        records = prepare(meta, masks, ctx.probs, qmeta)
        guided_all, text_audit = bounded_text_fusion(records, ctx.probs, list(qmeta), alpha=.5, gated=True)
        guided = {qid: guided_all[qid] for qid in output_query_ids}
        channels = [source_channel(x) for x in classes]
        novel = [i for i, c in enumerate(channels) if c is None]
        if not novel:
            # Preserve the existing common13 output EXACTLY, including ties.
            _, pred = requested_probabilities(corrected_probs, classes)
            ov_audit = {'version': VERSION, 'requested_novel_classes': 0,
                        'common_class_path': 'bitwise_legacy_requested_probabilities',
                        'semantic_veto': False, 'no_circular_evidence': True}
        else:
            name_by_qid = {output_query_ids[i]: classes[i] for i in range(len(classes))}
            candidates = [
                {'id': c['id'], 'class_name': name_by_qid[c['query_id']],
                 'mask': c['mask'], 'valid': c['valid'], 'view_id': c['fov_id'],
                 'detector_score': c['detector_score'], 'sam_quality': c['sam_quality']}
                for c in records if c['query_id'] in name_by_qid
            ]
            evidence, evidence_audit = build_competitive_text_evidence(candidates, classes, stored.shape)
            _, pred, competition_audit = compete_requested_classes(corrected_probs, classes, channels, evidence)
            ov_audit = {'version': VERSION, 'text_evidence': evidence_audit,
                        'competition': competition_audit, 'semantic_veto': False,
                        'no_circular_evidence': True}
        audit = {'I1_semantic': arm_counts(stored, lab, weight, arrays['votes'], eligible, corrected, alpha, error, ctx.probs),
                 'I1_gate': gate_audit, 'I1_text': text_audit, 'open_vocabulary': ov_audit,
                 'calibrator_sha256': self.hashes,
                 'calibrator_origin': 'Cityscapes_source_rebuild_20261109',
                 'source_GT_or_target_GT_reads': 0, 'no_circular_evidence': True,
                 'output_class_names': list(classes), 'novel_class_names': [classes[i] for i in novel],
                 'revision_status': 'post_diagnostic_v2_not_a_fresh_unseen_holdout'}
        return pred, audit, guided
