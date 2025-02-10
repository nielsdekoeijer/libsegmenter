backends = ["torch", "tensorflow", "numpy", "octave"]

def make_segmenter(backend: str = "numpy", *args, **kwargs):
    if backend not in backends:
        raise ValueError(f"Unsupported backend {backend}, availible: {backends}")

    if backend == "torch":
        from libsegmenter.impl.SegmenterTorch import SegmenterTorch

        raise NotImplemented
        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.impl.SegmenterTensorFlow import SegmenterTensorFlow

        raise NotImplemented
        return SegmenterTensorFlow(*args, **kwargs)

    if backend == "numpy":
        from libsegmenter.impl.SegmenterNumpy import SegmenterNumpy

        raise NotImplemented
        return SegmenterNumpy(*args, **kwargs)

    if backend == "octave":
        from libsegmenter.impl.SegmenterOctave import SegmenterOctave

        raise NotImplemented
        return SegmenterOctave(*args, **kwargs)

