import numpy as np
import pytest

from mot.datasets.visdrone import list_sequences, load_annotations
from mot.eval.visdrone_eval import evaluate, ignored_overlap

# two pedestrians (ids 1, 2) over 3 frames, plus an ignored region at (200, 200)
GT = """\
1,1,0,0,10,10,1,1,0,0
1,2,50,50,10,10,1,1,0,0
2,1,1,0,10,10,1,1,0,0
2,2,51,50,10,10,1,1,0,0
3,1,2,0,10,10,1,1,0,0
3,2,52,50,10,10,1,1,0,0
1,0,200,200,40,40,0,0,0,0
"""


def write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def dirs(tmp_path):
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    write(gt / "seq.txt", GT)
    return gt, pred


def test_load_annotations_pads_short_rows(tmp_path):
    a = load_annotations(write(tmp_path / "r.txt", "1,5,0,0,10,10,0.9,4\n"))
    assert a.shape == (1, 10)
    assert a[0, 7] == 4 and a[0, 9] == -1


def test_load_annotations_missing_file(tmp_path):
    assert load_annotations(tmp_path / "nope.txt").shape == (0, 10)


def test_list_sequences(tmp_path):
    (tmp_path / "sequences" / "s1").mkdir(parents=True)
    (tmp_path / "sequences" / "s1" / "0000001.jpg").touch()
    (tmp_path / "annotations").mkdir()
    write(tmp_path / "annotations" / "s1.txt", GT)
    [seq] = list_sequences(tmp_path)
    assert seq.name == "s1" and len(seq) == 1 and len(seq.annotations()) == 7


def test_ignored_overlap():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], float)
    regions = np.array([[0, 0, 10, 10]], float)
    assert np.allclose(ignored_overlap(boxes, regions), [1.0, 0.5])


def test_perfect_tracking(dirs):
    gt, pred = dirs
    write(pred / "seq.txt", "".join(l + "\n" for l in GT.splitlines() if not l.endswith(",0,0,0,0")))
    ped = evaluate(gt, pred).loc["pedestrian"]
    assert ped["mota"] == 1.0 and ped["idf1"] == 1.0 and ped["num_switches"] == 0


def test_id_switch_and_ignored_region(dirs):
    gt, pred = dirs
    rows = [l for l in GT.splitlines() if not l.endswith(",0,0,0,0")]
    rows[4] = "3,7,2,0,10,10,1,1,0,0"            # id 1 becomes id 7 in frame 3
    rows.append("1,9,205,205,10,10,0.9,1,-1,-1")  # inside ignored region: not a false positive
    write(pred / "seq.txt", "\n".join(rows) + "\n")
    ped = evaluate(gt, pred).loc["pedestrian"]
    assert ped["num_switches"] == 1
    assert ped["num_false_positives"] == 0


def test_all_row_pools_counts_over_classes(dirs):
    gt, pred = dirs
    peds = [l for l in GT.splitlines() if not l.endswith(",0,0,0,0")]
    write(gt / "seq.txt", GT + "1,3,100,100,20,20,1,4,0,0\n2,3,101,100,20,20,1,4,0,0\n")  # 2 car boxes, missed
    write(pred / "seq.txt", "\n".join(peds) + "\n")
    table = evaluate(gt, pred)
    assert table.loc["pedestrian", "mota"] == 1.0 and table.loc["car", "mota"] == 0.0
    assert table.loc["mean", "mota"] == pytest.approx(0.5)  # average of the two classes with ground truth
    assert table.loc["all", "mota"] == pytest.approx(0.75)  # 1 - 2 misses / 8 boxes
    assert table.loc["all", "num_misses"] == 2


def test_missing_prediction_file(dirs):
    gt, pred = dirs
    ped = evaluate(gt, pred).loc["pedestrian"]
    assert ped["recall"] == 0 and ped["num_misses"] == 6
