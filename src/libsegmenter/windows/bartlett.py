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

def bartlett(window_size: int) -> np.ndarray:
    M = window_size + 1.0
    indices = np.arange(window_size)
    return 1.0 - np.abs(-1.0 * (M - 1) / 2.0 + indices) * 2.0 / (M - 1.0)


def bartlett50(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Bartlett window of the given size with 75% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A bartlett window with 50% overlap
    """

    assert (segment_size % 2 == 0, f"segment_size must be even, got {segment_size}")

    return bartlett(segment_size), segment_size // 2


def bartlett75(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Bartlett window of the given size with 75% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A bartlett window with 75% overlap
    """

    assert (
        segment_size % 4 == 0,
        f"segment_size must be modulus 4, got {segment_size}",
    )

    return bartlett(segment_size), segment_size // 4
