import numpy as np
from .default_window_selector import default_window_selector

backends = ["torch", "tensorflow", "base"]

def make_segmenter(backend: str = "base", *args, **kwargs):
    if backend not in backends:
        raise ValueError(f"Unsupported backend {backend}, availible: {backends}")

    if backend == "torch":
        from .SegmenterTorch import SegmenterTorch

        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from .SegmenterTensorFlow import SegmenterTensorFlow

        return SegmenterTensorFlow(*args, **kwargs)
