BACKENDS = ["torch", "tensorflow", "numpy", "octave"]


def make_segmenter(backend: str = "numpy", *args, **kwargs):
    """
    Factory function to create a segmenter instance based on the backend.

    Args:
        backend (str): The backend to use. Supported: ["numpy", "torch", "tensorflow", "octave"]
        *args, **kwargs: Arguments to pass to the segmenter.

    Returns:
        Segmenter instance for the specified backend.

    Raises:
        ValueError: If an unsupported backend is specified.
        NotImplementedError: If the backend is not implemented.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend {backend}, availible: {backends}")

    if backend == "numpy":
        from libsegmenter.backends.SegmenterNumpy import SegmenterNumpy

        return SegmenterNumpy(*args, **kwargs)

    if backend == "torch":
        from libsegmenter.backends.SegmenterTorch import SegmenterTorch

        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.backends.SegmenterTensorFlow import SegmenterTensorFlow

        return SegmenterTensorFlow(*args, **kwargs)

    # if backend == "octave":
    #     from libsegmenter.backends.SegmenterOctave import SegmenterOctave
    #
    #     return SegmenterOctave(*args, **kwargs)

    raise NotImplementedError(f"The '{backend}' backend is not implemented yet.")
