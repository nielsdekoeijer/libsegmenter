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


def _kaiser(
    window_size: int, beta: float, dtype: DTypeLike = np.float32
) -> NDArray[Any]:
    M = window_size + 1.0
    m = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0, dtype=np.dtype(dtype).type)
    alpha = beta * np.sqrt(1 - (m / (M / 2)) ** 2.0, dtype=np.dtype(dtype).type)
    return np.divide(np.i0(alpha), np.i0(beta), dtype=np.dtype(dtype).type)


def _kaiser_hop_size(segment_size: int, beta: float) -> int:
    return int(np.floor(1.7 * (float(segment_size) - 1.0) / (beta + 1.0)))


def kaiser85(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a Kaiser window of the given size with 85% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A kaiser window with 85% overlap

    """
    beta = 10.0
    hop_size = _kaiser_hop_size(segment_size, beta)
    return _kaiser(segment_size, beta, dtype=dtype), hop_size


def kaiser82(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a Kaiser window of the given size with 82% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A kaiser window with 82% overlap

    """
    beta = 8.0
    hop_size = _kaiser_hop_size(segment_size, beta)
    return _kaiser(segment_size, beta, dtype=dtype), hop_size
