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


def kaiser(segment_size: int) -> np.ndarray:
    M = np.float(window_length + 1.0)
    m = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0)
    window = np.i0(beta * np.sqrt(1 - (m / (M / 2)) ** 2.0)) / np.i0(beta)
    return window


def kaiser85(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Hann window of the given size with 85% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A kaiser window with 85% overlap
    """

    beta = 10.0
    return kaiser(segment_size), int(np.floor(1.7 * (np.float(segment_size) - 1.0) / (beta + 1.0)))


def kaiser82(segment_size: int) -> (np.ndarray, int):
    """
    Generates a Hann window of the given size with 82% overlap.

    Args:
        segment_size (int): Size of the window to be created.

    Returns:
        A kaiser window with 82% overlap
    """

    beta = 8.0
    return kaiser(segment_size), int(np.floor(1.7 * (np.float(segment_size) - 1.0) / (beta + 1.0)))
