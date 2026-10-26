import numpy as np
from piers_ljj.evaluation.metrics import (confusion_matrix, scores_from_cm,
                                          map19_to_13, binary_scores)


def test_map19_to_13_train_ignored():
    ids = np.array([11, 12, 13, 14, 15, 16, 17, 18])  # person..bicycle
    assert map19_to_13(ids).tolist() == [11, 11, 12, 12, 12, 255, 12, 12]


def test_cm_miou_not_mean_of_perimage():
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    s = scores_from_cm(cm)
    # dataset-level: class0 iou=1/2, class1 iou=2/3 -> miou=7/12
    assert abs(s["miou"] - 7 / 12) < 1e-9
    assert abs(s["pixacc"] - 0.75) < 1e-9


def test_empty_pred_on_gt_counts_zero():
    r = binary_scores(np.zeros((4, 4), int), np.ones((4, 4), int))
    assert r["iou"] == 0.0 and r["dice"] == 0.0


def test_both_empty_is_reject_not_perfect():
    r = binary_scores(np.zeros((4, 4), int), np.zeros((4, 4), int))
    assert r["iou"] is None and r.get("correct_reject") is True


def test_no_gt_is_null():
    r = binary_scores(np.ones((4, 4), int), None)
    assert r["iou"] is None and "qualitative" in r["reason"]
