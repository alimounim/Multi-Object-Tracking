"""Drawing helpers for rendering VisDrone-format tracks (boxes, ID labels, trails)."""
from collections import defaultdict, deque

import cv2
import numpy as np

from mot.datasets.visdrone import CATEGORIES, IGNORED, load_annotations


def color(track_id):
    rng = np.random.default_rng(int(track_id))
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def load_by_frame(path):
    """Rows of a VisDrone results/annotation file grouped by frame, ignored regions dropped."""
    data = load_annotations(path)
    data = data[data[:, 7] != IGNORED]
    by_frame = defaultdict(list)
    for row in data:
        by_frame[int(row[0])].append(row)
    return by_frame


class TrackPainter:
    """Draws one frame's tracks; keeps a short trail of recent centers per track ID."""

    def __init__(self, frame_width, trail=30):
        self.trails = defaultdict(lambda: deque(maxlen=trail))
        self.thick = max(1, round(2 * frame_width / 1920))
        self.font = 0.5 * frame_width / 1920 + 0.2

    def draw(self, img, rows):
        for _, tid, x, y, w, h, _, cat, *_ in rows:
            c = color(tid)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(img, p1, p2, c, self.thick)
            label = f"{int(tid)} {CATEGORIES.get(int(cat), '')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), c, -1)
            cv2.putText(img, label, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, self.font, (0, 0, 0), 1,
                        cv2.LINE_AA)
            self.trails[tid].append((int(x + w / 2), int(y + h)))
        for tid in {row[1] for row in rows}:  # only tracks visible in this frame
            if len(self.trails[tid]) > 1:
                cv2.polylines(img, [np.array(self.trails[tid])], False, color(tid), self.thick)
        return img


def put_title(img, text, scale=1.0):
    """White text on a dark band at the top-left corner."""
    thick = max(1, round(2 * scale))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 20, th + base + 20), (0, 0, 0), -1)
    cv2.putText(img, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img
