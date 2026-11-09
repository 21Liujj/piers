"""Semantic regression cases for the broad-category overwrite failure."""
import numpy as np
from piers_ljj.collaboration.open_vocab_competition import (
    build_competitive_text_evidence, compete_requested_classes,
    exclude_specific_from_residual,
    compete_from_known_labels,
)


def candidate(name, view, confidence=.8, mask=None):
    mask = np.ones((2, 3), bool) if mask is None else mask
    return dict(id=name+view, class_name=name, view_id=view,
                detector_score=confidence, sam_quality=1.,
                mask=mask, valid=np.ones((2, 3), bool))


def test_correlated_views_do_not_turn_moderate_support_into_near_certainty():
    two, _ = build_competitive_text_evidence([candidate('ground', v, .6) for v in ['a', 'b']], ['ground'], (2, 3))
    four, _ = build_competitive_text_evidence([candidate('ground', v, .6) for v in ['a', 'b', 'c', 'd']], ['ground'], (2, 3))
    assert np.allclose(two, .6) and np.allclose(four, two)


def test_ground_cannot_overwrite_independently_text_supported_road():
    p = np.zeros((13, 1, 2), np.float32); p[0] = 1.
    text = np.array([[[.4, 0]], [[.95, .95]]], np.float32)
    original = text.copy()
    scores, labels, audit = compete_requested_classes(p, ['road', 'ground'], [0, None], text)
    assert labels.tolist() == [[0, 1]]  # ground still works outside the specific road mask
    assert np.array_equal(text, original) and np.allclose(scores.sum(0), 1.)
    assert audit['residual_exclusions'][0]['excluded_pixels'] == 1


def test_generic_static_dynamic_and_other_cannot_absorb_specific_vehicle_text():
    p = np.zeros((13, 1, 1), np.float32); p[12] = 1.
    text = np.array([[[.35]], [[.8]], [[.9]], [[.7]]], np.float32)
    _, labels, audit = compete_requested_classes(p, ['vehicles', 'static object', 'moving object', 'other object'],
                                                 [12, None, None, None], text)
    assert labels.item() == 0 and len(audit['residual_exclusions']) == 3


def test_novel_object_can_override_a_confident_but_uninformed_closed_head():
    p = np.zeros((13, 1, 1), np.float32); p[0] = 1.
    text = np.array([[[.4]], [[.6]]], np.float32)
    _, labels, audit = compete_requested_classes(p, ['road', 'traffic cone'], [0, None], text)
    assert labels.item() == 1 and not audit['teacher_veto']


def test_novel_score_competes_against_comparable_known_text_evidence():
    p = np.zeros((13, 1, 2), np.float32); p[12] = 1.
    text = np.array([[[.7, .3]], [[.4, .6]]], np.float32)
    _, labels, _ = compete_requested_classes(p, ['car', 'truck'], [12, None], text)
    assert labels.tolist() == [[0, 1]]


def test_two_views_with_disjoint_instances_do_not_validate_each_other():
    left = np.zeros((2, 3), bool); left[:, 0] = True
    right = np.zeros((2, 3), bool); right[:, 2] = True
    evidence, audit = build_competitive_text_evidence([candidate('bus', 'a', mask=left), candidate('bus', 'b', mask=right)], ['bus'], (2, 3))
    assert not evidence.any() and all(c['kept'] and not c['emitted'] for c in audit['candidates'])


def test_ground_does_not_exclude_a_category_that_is_not_a_ground_surface():
    text = np.ones((2, 1, 1), np.float32)
    adjusted, _ = exclude_specific_from_residual(text, ['ground', 'bird'])
    assert adjusted[0].item() == 1.


def test_requested_class_permutation_preserves_ties_and_novel_labels():
    p = np.zeros((13, 1, 1), np.float32); p[12] = 1.
    names, channels = ['car', 'bus', 'truck'], [12, None, None]
    text = np.array([[[.2]], [[.6]], [[.6]]], np.float32)
    a, labels_a, _ = compete_requested_classes(p, names, channels, text)
    order = [2, 0, 1]
    b, labels_b, _ = compete_requested_classes(p, [names[i] for i in order], [channels[i] for i in order], text[order])
    assert names[labels_a.item()] == [names[i] for i in order][labels_b.item()] == 'bus'
    assert np.allclose(a, b[np.argsort(order)])


def test_no_text_preserves_known_head_and_no_known_or_text_abstains():
    p = np.zeros((13, 1, 2), np.float32); p[0] = .2; p[1] = .8
    _, labels, _ = compete_requested_classes(p, ['road', 'sidewalk', 'motorcycle'], [0, 1, None], np.zeros((3, 1, 2)))
    assert labels.tolist() == [[1, 1]]
    _, labels, _ = compete_requested_classes(p, ['motorcycle'], [None], np.zeros((1, 1, 2)))
    assert (labels == 255).all()


def test_label_cache_decision_matches_probability_decision_for_arbitrary_confidence():
    rng=np.random.default_rng(1234)
    names=['building','other object','road','ground','truck','car']
    channels=[2,None,0,None,None,12]
    for _ in range(20):
        p=rng.random((13,4,5),dtype=np.float32)
        text=rng.random((6,4,5),dtype=np.float32);text[text < .3]=0
        order=np.asarray(sorted([0,2,5],key=lambda i:channels[i]))
        known=order[p[[channels[i] for i in order]].argmax(0)].astype(np.uint8)
        _,direct,_=compete_requested_classes(p,names,channels,text)
        cached,_=compete_from_known_labels(known,names,channels,text)
        assert np.array_equal(direct,cached)


def test_exact_evidence_ties_preserve_known_prediction():
    p=np.zeros((13,1,1),np.float32);p[12]=1
    _,pred,_=compete_requested_classes(p,['car','bus'],[12,None],np.full((2,1,1),.4,np.float32))
    assert pred.item()==0


if __name__ == '__main__':
    tests = [value for key, value in list(globals().items()) if key.startswith('test_')]
    for test in tests:
        test()
    print('PASSED', len(tests), 'open-vocabulary competition regression cases')
