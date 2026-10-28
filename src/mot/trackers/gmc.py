"""Resolution-independent camera-motion compensation (GMC) for the Ultralytics trackers.

Ultralytics' GMC always halves the frame before estimating camera motion, so its cost grows with the
input resolution (sparseOptFlow: about 7-10 ms/frame at 1360-1904 px wide, 39 ms at 4K). `fixed_width_gmc` instead
picks the smallest integer downscale factor per frame that makes the frame GMC works on at most
`width` pixels wide.
"""
import math
from contextlib import contextmanager

from ultralytics.trackers.utils.gmc import GMC


def downscale_for(frame_width, width):
    return max(1, math.ceil(frame_width / width))


@contextmanager
def fixed_width_gmc(width):
    """While active, every GMC.apply() runs on frames downscaled to at most `width` pixels wide."""
    if not width:
        yield
        return
    original = GMC.apply

    def apply(self, raw_frame, detections=None):
        self.downscale = downscale_for(raw_frame.shape[1], width)
        return original(self, raw_frame, detections)

    GMC.apply = apply
    try:
        yield
    finally:
        GMC.apply = original
