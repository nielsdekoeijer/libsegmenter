import pytest
import libsegmenter

"""
tests ensuring the `check_cola` and windowing functions included in the library are correct
"""


@pytest.mark.parametrize(
    ("window", "hop_size", "valid"),
    [
        (libsegmenter.bartlett(100), 50, True),
        (libsegmenter.bartlett(100), 25, True),
        (libsegmenter.bartlett(100), 50, True),
        (libsegmenter.bartlett(100), 51, False),
        (libsegmenter.blackman(99), 33, True),
        (libsegmenter.blackman(99), 34, False),
        (libsegmenter.hamming(100), 50, True),
        (libsegmenter.hamming(100), 25, True),
        (libsegmenter.hamming(100), 23, False),
        (libsegmenter.hann(100), 50, True),
        (libsegmenter.hann(100), 25, True),
        (libsegmenter.hann(100), 23, False),
    ],
)
def test_check_cola(window, hop_size, valid):
    assert libsegmenter.check_cola(window, hop_size)[0] == valid
