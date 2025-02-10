backends = ["torch", "tensorflow", "numpy", "octave"]

def make_segmenter(backend: str = "numpy", *args, **kwargs):
    if backend not in backends:
        raise ValueError(f"Unsupported backend {backend}, availible: {backends}")

    if backend == "torch":
        from libsegmenter.backends.SegmenterTorch import SegmenterTorch

        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.backends.SegmenterTensorFlow import SegmenterTensorFlow

        return SegmenterTensorFlow(*args, **kwargs)

    if backend == "numpy":
        from libsegmenter.backends.SegmenterNumpy import SegmenterNumpy

        return SegmenterNumpy(*args, **kwargs)

    if backend == "octave":
        from libsegmenter.backends.SegmenterOctave import SegmenterOctave

        return SegmenterOctave(*args, **kwargs)
