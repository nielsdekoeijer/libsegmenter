# Copyright (c) 2025 Niels de Koeijer, Martin Bo Møller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def hann(segment_size: int) -> np.ndarray:
    M = float(segment_size)
    indices = np.arange(segment_size)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * indices / M))


def hann50(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Hann window of the given size with 50% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A hann window with 50% overlap
    """

    assert (segment_size % 2 == 0, f"segment_size must be even, got {segment_size}")

    return hann(segment_size), segment_size // 2


def hann75(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Hann window of the given size with 75% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A hann window with 75% overlap
    """

    assert (
        segment_size % 4 == 0,
        f"segment_size must be modulus 4, got {segment_size}",
    )

    return hann(segment_size), segment_size // 4
