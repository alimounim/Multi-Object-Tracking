import numpy as np

from mot.trackers.offline import Detections, track_cached, tracker_config


def moving_boxes(n_frames=10):
    """Two objects moving right, one detection each per frame."""
    frame, xyxy = [], []
    for f in range(1, n_frames + 1):
        for y in (100, 300):
            frame.append(f)
            xyxy.append([50 + 5 * f, y, 90 + 5 * f, y + 40])
    n = len(frame)
    return {"frame": np.array(frame), "xyxy": np.array(xyxy, float), "conf": np.full(n, 0.9), "cls": np.full(n, 4)}


def test_detections_indexing():
    d = Detections(np.array([[0, 0, 10, 20], [5, 5, 7, 9]], float), np.array([0.9, 0.2]), np.array([1, 4]))
    assert np.allclose(d.xywh[0], [5, 10, 10, 20])
    sub = d[d.conf > 0.5]
    assert len(sub) == 1 and sub.cls[0] == 1


def test_two_objects_keep_their_ids():
    rows = track_cached(moving_boxes(), tracker_config("bytetrack.yaml"), frames=range(1, 11))
    assert set(rows[:, 1].astype(int)) == {1, 2}
    for tid in (1, 2):  # each ID stays on one horizontal lane
        assert len(np.unique(np.round(rows[rows[:, 1] == tid, 3], -1))) == 1
    assert (rows[:, 7] == 4).all()


def test_min_conf_filters_everything():
    cache = moving_boxes()
    cache["conf"][:] = 0.05
    assert len(track_cached(cache, tracker_config("bytetrack.yaml"), range(1, 11), min_conf=0.1)) == 0
