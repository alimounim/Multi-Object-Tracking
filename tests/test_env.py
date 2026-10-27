import torch


def test_imports():
    import cv2, motmetrics, ultralytics  # noqa: F401
    import mot  # noqa: F401


def test_cuda_available():
    assert torch.cuda.is_available()
