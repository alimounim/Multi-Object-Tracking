import numpy as np
import pytest

from mot.eval.hota import ALPHAS, combine, sequence_stats


def run(frames):
    return combine([sequence_stats(frames)])


def test_perfect():
    frames = [(np.array([1, 2]), np.array([5, 6]), np.eye(2)) for _ in range(4)]
    r = run(frames)
    assert r["hota"] == pytest.approx(1) and r["deta"] == pytest.approx(1) and r["assa"] == pytest.approx(1)


def test_id_switch_halves_assa():
    # one GT object over 2 frames, predicted with a different ID in each frame
    frames = [(np.array([1]), np.array([7]), np.ones((1, 1))), (np.array([1]), np.array([8]), np.ones((1, 1)))]
    r = run(frames)
    assert r["deta"] == pytest.approx(1)
    assert r["assa"] == pytest.approx(0.5)  # each match: 1 / (2 + 1 - 1)
    assert r["hota"] == pytest.approx(np.sqrt(0.5))


def test_partial_overlap_counts_only_low_alphas():
    # IoU 0.5 everywhere: a match for alpha <= 0.5 (10 of 19 thresholds), a miss + FP above
    frames = [(np.array([1]), np.array([1]), np.full((1, 1), 0.5)) for _ in range(3)]
    r = run(frames)
    n_ok = int((ALPHAS <= 0.5 + 1e-9).sum())
    assert n_ok == 10
    assert r["hota"] == pytest.approx(n_ok / len(ALPHAS))
    assert r["loca"] == pytest.approx(0.5 * n_ok / len(ALPHAS))


def test_no_predictions():
    frames = [(np.array([1]), np.array([], int), np.zeros((1, 0)))]
    r = run(frames)
    assert r["hota"] == 0 and r["deta"] == 0


def test_false_positives_lower_deta():
    # 1 GT matched perfectly + 1 extra prediction every frame: DetA = 1 / (1 + 0 + 1)
    frames = [(np.array([1]), np.array([1, 2]), np.array([[1.0, 0.0]])) for _ in range(5)]
    r = run(frames)
    assert r["deta"] == pytest.approx(0.5)
    assert r["assa"] == pytest.approx(1)
