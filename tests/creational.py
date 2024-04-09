import pytest
import torch
import tensorflow
import libsegmenter

"""
tests testing the creational pattern for the Segmenters
"""


@pytest.mark.parametrize(
    ("backend", "frame_size", "hop_size", "window", "kwargs", "should_throw"),
    [
        # mismatch between specified frame_size and window
        ("torch", 99, 50, libsegmenter.hamming(100), None, True),
        ("torch", 100, 50, libsegmenter.hamming(99), None, True),
        ("tensorflow", 99, 50, libsegmenter.hamming(100), None, True),
        ("tensorflow", 100, 50, libsegmenter.hamming(99), None, True),
        # modes
        ("torch", 100, 50, libsegmenter.hamming(100), {"mode": "asdf"}, True),
        ("torch", 100, 50, libsegmenter.hamming(100), {"mode": "ola"}, False),
        ("torch", 100, 50, libsegmenter.hamming(100), {"mode": "wola"}, False),
        ("tensorflow", 100, 50, libsegmenter.hamming(100), {"mode": "asdf"}, True),
        ("tensorflow", 100, 50, libsegmenter.hamming(100), {"mode": "ola"}, False),
        ("tensorflow", 100, 50, libsegmenter.hamming(100), {"mode": "wola"}, False),
        # hop_size > frame_size
        ("torch", 100, 101, libsegmenter.hamming(100), None, True),
        ("tensorflow", 100, 101, libsegmenter.hamming(100), None, True),
        # invalid window
        ("torch", 100, 23, libsegmenter.blackman(100), None, True),
        ("tensorflow", 100, 23, libsegmenter.blackman(100), None, True),
    ],
)
def test_creational(backend, frame_size, hop_size, window, kwargs, should_throw):
    try:
        segmenter = libsegmenter.make_segmenter(
            backend=backend,
            frame_size=frame_size,
            hop_size=hop_size,
            window=window,
            **kwargs
        )
    except:
        if not should_throw:
            assert False

        return

    # should throw
    if should_throw:
        assert False


@pytest.mark.parametrize(
    ("backend", "frame_size", "hop_size", "window"),
    [
        ("torch", 100, 50, torch.tensor(libsegmenter.hamming(100))),
    ],
)
def test_pytorch_tensor(backend, frame_size, hop_size, window):
    segmenter = libsegmenter.make_segmenter(
        backend=backend, frame_size=frame_size, hop_size=hop_size, window=window
    )
