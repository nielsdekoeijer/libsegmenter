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


def _hann(
    window_size: int, indices: NDArray[Any], dtype: DTypeLike = np.float32
) -> NDArray[Any]:
    M = window_size
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * indices / M, dtype=np.dtype(dtype).type))


def hann50(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a Hann window of the given size with 50% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A hann window with 50% overlap

    """
    assert segment_size % 2 == 0, f"segment_size must be even, got {segment_size}"

    indices = np.arange(segment_size, dtype=np.dtype(dtype).type)
    return _hann(segment_size, indices, dtype=dtype), segment_size // 2


def hann75(
    segment_size: int, dtype: DTypeLike = np.float32
) -> Tuple[NDArray[Any], int]:
    """
    Generates a Hann window of the given size with 75% overlap.

    Args:
        segment_size (int): Size of the window to be created.
        dtype (np.dtype): The desired datatype of the window

    Returns:
        A hann window with 75% overlap

    """
    assert segment_size % 4 == 0, f"segment_size must be modulus 4, got {segment_size}"

    indices = np.arange(segment_size, dtype=np.dtype(dtype).type)
    return _hann(segment_size, indices, dtype=dtype), segment_size // 4


def asymmetricHannOla(
    segment_size: int,
    hop_size: int,
    synthesis_segment_size: int,
    dtype: DTypeLike = np.float32,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Generates an asymmetric Hann window pair with overlap-add reconstruction.

    Args:
        segment_size (int): Size of the window to be created.
        hop_size (int): DESCRIPTION PLACEHOLDER
        synthesis_segment_size (int): DESCRIPTION PLACEHOLDER
        dtype (np.dtype): The desired datatype of the window

    Returns:
        analysis_window, synthesis_window

    """
    assert synthesis_segment_size % 2 == 0, (
        f"synthesis_segment_size must be even, got {synthesis_segment_size}"
    )

    analysis_window = np.ones(segment_size, dtype=np.dtype(dtype).type)

    f1 = np.zeros(segment_size - synthesis_segment_size, dtype=np.dtype(dtype).type)
    indices = np.arange(synthesis_segment_size, dtype=np.dtype(dtype).type)
    f2 = _hann(synthesis_segment_size, indices, dtype=np.dtype(dtype).type)
    synthesis_window = np.concatenate((f1, f2))
    return analysis_window, synthesis_window


def asymmetricHannWola(
    segment_size: int,
    hop_size: int,
    synthesis_segment_size: int,
    dtype: DTypeLike = np.float32,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Generates an asymmetric Hann window pair with weighted overlap-add reconstruction.

    Args:
        segment_size (int): Size of the window to be created.
        hop_size (int): DESCRIPTION PLACEHOLDER
        synthesis_segment_size (int): DESCRIPTION PLACEHOLDER
        dtype (np.dtype): The desired datatype of the window

    Returns:
        analysis_window, synthesis_window

    """
    assert synthesis_segment_size % 2 == 0, (
        f"synthesis_segment_size must be even, got {synthesis_segment_size}"
    )
    M = synthesis_segment_size // 2
    KM = segment_size - M
    h1 = np.sqrt(
        _hann(
            2 * KM,
            np.arange(KM - M, dtype=np.dtype(dtype).type),
            dtype=np.dtype(dtype).type,
        )
    )
    h2 = np.sqrt(
        _hann(
            2 * KM,
            np.arange(KM - M, KM, dtype=np.dtype(dtype).type),
            dtype=np.dtype(dtype).type,
        )
    )
    h3 = np.sqrt(
        _hann(
            2 * M,
            np.arange(M, 2 * M, dtype=np.dtype(dtype).type),
            dtype=np.dtype(dtype).type,
        )
    )
    analysis_window = np.concatenate((h1, h2, h3))

    f1 = np.zeros(KM - M, dtype=np.dtype(dtype).type)
    f2a = _hann(
        2 * M, np.arange(M, dtype=np.dtype(dtype).type), dtype=np.dtype(dtype).type
    )
    f2b = np.sqrt(
        _hann(
            2 * KM,
            np.arange(KM - M, KM, dtype=np.dtype(dtype).type),
            dtype=np.dtype(dtype).type,
        )
    )
    f2 = np.divide(f2a, f2b)
    f3 = np.sqrt(
        _hann(
            2 * M,
            np.arange(M, 2 * M, dtype=np.dtype(dtype).type),
            dtype=np.dtype(dtype).type,
        )
    )
    synthesis_window = np.concatenate((f1, f2, f3))
    return analysis_window, synthesis_window
