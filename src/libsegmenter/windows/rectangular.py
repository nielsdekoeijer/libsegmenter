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


def _rectangular(window_size: int, dtype: DTypeLike = np.float32) -> NDArray[Any]:
    return np.ones(window_size, dtype=np.dtype(dtype).type)


def rectangular50(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a rectangular window of the given size with 50% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A rectangular window with 50% overlap

    """
    return _rectangular(segment_size, dtype=dtype), segment_size // 2


def rectangular0(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a rectangular window of the given size with 0% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A rectangular window with 0% overlap

    """
    return _rectangular(segment_size, dtype=dtype), segment_size
