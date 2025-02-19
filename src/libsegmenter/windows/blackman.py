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
from numpy.typing import NDArray, DTypeLike
from typing import Tuple, Any


def _blackman(window_size: int, dtype: DTypeLike = np.float32) -> NDArray[Any]:
    M = window_size + 1
    indices = np.arange(window_size, dtype=dtype)

    return (
        7938.0 / 18608.0
        - 9240.0 / 18608.0 * np.cos(2.0 * np.pi * indices / (M - 1), dtype=dtype)
        + 1430.0 / 18608.0 * np.cos(4.0 * np.pi * indices / (M - 1), dtype=dtype)
    )


def blackman67(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a Blackman window of the given size with a 2/3 overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A blackman window with a 2/3 overlap

    """
    assert segment_size % 3 == 0, f"segment_size must be modulus 3, got {segment_size}"

    return _blackman(segment_size, dtype=dtype), segment_size // 3
