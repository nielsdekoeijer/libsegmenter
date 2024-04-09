import pytest
import torch
import tensorflow
import libsegmenter
import numpy

"""
tests testing the creational pattern for the Segmenters
"""


@pytest.mark.parametrize(
    ("segmenter", "x"),
    [
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hamming(100),
            ),
            torch.randn((1000)),
        ),
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hamming(100),
            ),
            torch.randn((1, 1000)),
        ),
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hann(100),
            ),
            torch.randn((1000)),
        ),
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hann(100),
            ),
            torch.randn((1, 1000)),
        ),
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.bartlett(100),
            ),
            torch.randn((1000)),
        ),
        (
            libsegmenter.make_segmenter(
                backend="torch",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.bartlett(100),
            ),
            torch.randn((1, 1000)),
        ),
        # (libsegmenter.make_segmenter(backend="torch", frame_size=30, hop_size=10, window=libsegmenter.blackman(30)), torch.randn((300))), #BREAKS
        # (libsegmenter.make_segmenter(backend="torch", frame_size=30, hop_size=10, window=libsegmenter.blackman(30)), torch.randn((1, 300))), #BREAKS
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hamming(100),
            ),
            tensorflow.random.normal(shape=[1000]),
        ),
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hamming(100),
            ),
            tensorflow.random.normal(shape=[1, 1000]),
        ),
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hann(100),
            ),
            tensorflow.random.normal(shape=[1000]),
        ),
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.hann(100),
            ),
            tensorflow.random.normal(shape=[1, 1000]),
        ),
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.bartlett(100),
            ),
            tensorflow.random.normal(shape=[1000]),
        ),
        (
            libsegmenter.make_segmenter(
                backend="tensorflow",
                frame_size=100,
                hop_size=50,
                window=libsegmenter.bartlett(100),
            ),
            tensorflow.random.normal(shape=[1000]),
        ),
        # (libsegmenter.make_segmenter(backend="tensorflow", frame_size=30, hop_size=10, window=libsegmenter.blackman(30)), tensorflow.random.normal(shape=[300])), #BREAKS
        # (libsegmenter.make_segmenter(backend="tensorflow", frame_size=30, hop_size=10, window=libsegmenter.blackman(30)), tensorflow.random.normal(shape=[1, 300])), #BREAKS
    ],
)
def test_reconstruction(segmenter, x):
    X = segmenter.segment(x)
    y = segmenter.unsegment(X)

    y = numpy.array(y)
    x = numpy.array(x)
    assert y == pytest.approx(x)
