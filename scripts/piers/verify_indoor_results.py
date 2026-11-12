"""Recompute the archived indoor table on CPU; no image or GPU access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    root = REPO / 'results/indoor_matched20261112'
    result = json.loads((root / 'indoor_results.json').read_text(encoding='utf-8'))
    protocol = result['protocol']
    assert result['status'] == 'complete' and result['n_images'] == 373
    assert not protocol['stage2_CLIP_checkpoint_used']
    with np.load(root / 'paired_confusions.npz', allow_pickle=False) as ledger:
        ids = ledger['image_ids'].tolist()
        assert len(ids) == len(set(ids)) == 373
        expected_ids = [r['image_id'] for r in json.loads((root / 'inference_heldout.json').read_text())]
        assert ids == expected_ids
        supports = []
        for arm in ('original', 'full_core', 'full'):
            counts = ledger[arm]
            assert counts.shape == (373, 8, 9) and np.issubdtype(counts.dtype, np.integer)
            assert (counts >= 0).all()
            total = counts.sum(axis=0)
            recorded = result['arms'][arm]
            assert total.tolist() == recorded['confusion_matrix']
            tp = np.diag(total[:, :8]).astype(np.float64)
            gt = total.sum(1)
            pred = total[:, :8].sum(0)
            iou = tp / (gt + pred - tp)
            metrics = {'mIoU_pct': iou.mean() * 100,
                       'mAcc_pct': (tp / gt).mean() * 100,
                       'PixelAcc_pct': tp.sum() / total.sum() * 100,
                       'FWIoU_pct': (gt / total.sum() * iou).sum() * 100,
                       'mDice_pct': (2 * tp / (gt + pred)).mean() * 100}
            for name, value in metrics.items():
                assert abs(value - recorded[name]) < 1e-9, (arm, name)
            for name, value in zip(protocol['classes'], iou * 100):
                assert abs(value - recorded['per_class_iou_pct'][name]) < 1e-9
            assert int(total.sum()) == recorded['gt_valid_pixels'] == 155096619
            assert int(total[:, 8].sum()) == recorded['predictions_outside_label_set'] == 0
            supports.append(counts.sum(2))
        assert all(np.array_equal(supports[0], s) for s in supports[1:])
    for name, expected in protocol['source_sha256'].items():
        content = (REPO / 'scripts/piers' / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, ('source fingerprint', name)
    calibration = json.loads((root / 'calibrators/DONE.json').read_text())
    for name in ('teacher_only', 'text_only'):
        content = (root / 'calibrators' / (name + '.npz')).read_bytes()
        assert hashlib.sha256(content).hexdigest() == calibration[name + '_sha256']
    web = json.loads((root / 'web/verification.json').read_text(encoding='utf-8-sig'))
    assert web['status'] == 'passed' and web['real_web_jobs'] == 3
    assert web['pixel_equivalence_all_three'] and not web['CLIP_used']
    if args.checkpoint:
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as source:
            for block in iter(lambda: source.read(1 << 20), b''):
                digest.update(block)
        assert digest.hexdigest() == protocol['checkpoint_sha256']
    print(json.dumps({'status': 'passed', 'images': 373, 'arms': 3,
                      'metrics_recomputed': 5, 'per_class_IoU_recomputed': True,
                      'per_image_GT_support_equal': True, 'source_fingerprints': 'passed',
                      'calibrator_fingerprints': 'passed', 'web_examples': 3,
                      'checkpoint_checked': bool(args.checkpoint)}, indent=2))


if __name__ == '__main__':
    main()
