import cv2
import numpy as np
from ultralytics.trackers.utils.gmc import GMC

from mot.trackers.gmc import downscale_for, fixed_width_gmc


def textured_frame(h, w, seed=0):
    noise = np.random.default_rng(seed).integers(0, 256, (h // 8, w // 8), dtype=np.uint8)
    img = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_downscale_for():
    assert downscale_for(3840, 640) == 6
    assert downscale_for(1360, 640) == 3
    assert downscale_for(680, 640) == 2
    assert downscale_for(640, 640) == 1
    assert downscale_for(320, 640) == 1


def test_fixed_width_gmc_recovers_shift():
    big = textured_frame(1000, 1400)
    prev, cur = big[100:820, 100:1380], big[106:826, 88:1368]  # camera moves: content shifts by (+12, -6)
    gmc, original = GMC("sparseOptFlow"), GMC.apply
    with fixed_width_gmc(320):
        gmc.apply(prev)
        H = gmc.apply(cur)
    assert gmc.downscale == 4
    np.testing.assert_allclose(H[:, 2], [12, -6], atol=1.5)
    assert GMC.apply is original  # restored on exit


def test_fixed_width_gmc_disabled():
    gmc = GMC("sparseOptFlow")
    with fixed_width_gmc(None):
        gmc.apply(textured_frame(720, 1280))
    assert gmc.downscale == 2
