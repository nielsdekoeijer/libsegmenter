backends = ["torch", "tensorflow", "numpy", "octave"]


def make_segmenter(backend: str = "numpy", *args, **kwargs):
    if backend not in backends:
        raise ValueError(f"Unsupported backend {backend}, availible: {backends}")

    if backend == "torch":
        from libsegmenter.impl.SegmenterTorch import SegmenterTorch

        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.impl.SegmenterTensorFlow import SegmenterTensorFlow

        return SegmenterTensorFlow(*args, **kwargs)

    if backend == "numpy":
        from libsegmenter.impl.SegmenterNumpy import SegmenterNumpy

        return SegmenterNumpy(*args, **kwargs)

    if backend == "octave":
        from libsegmenter.impl.SegmenterOctave import SegmenterOctave

        return SegmenterOctave(*args, **kwargs)
