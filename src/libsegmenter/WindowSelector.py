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
from numpy.typing import NDArray
from typing import TypeVar
from libsegmenter.Window import Window
from libsegmenter.util.check_cola import check_cola

T = TypeVar("T", bound=np.generic)


def _adapt_window(window: NDArray[T], hop_size: int, scheme: str) -> Window:
    # TODO: windows ALWAYS normalized
    # TODO: windows ALWAYS cola_checked
    is_cola, normalization, e = check_cola(window, hop_size)
    assert is_cola, f"specified window failed cola check with error {e}"

    window = np.divide(window, normalization)

    if scheme == "ola":
        return Window(hop_size, np.ones(window.shape), window)

    if scheme == "wola":
        window = np.sqrt(window)
        return Window(hop_size, window, window)

    if scheme == "analysis":
        return Window(hop_size, window, None)

    raise ValueError(f"The '{scheme}' scheme is not supported.")


def WindowSelector(window: str, scheme: str, segment_size: int) -> Window:
    """
    Selects and returns a specific window function based on the given parameters.

    This function retrieves a window function based on the `window` type, applies
    an adaptation based on `scheme`, and returns the corresponding `Window` object.

    Args:
        window (str): The type of window function to apply. Supported values include:
            [
             `bartlett50`,
             `bartlett75`,
             `blackman67`,
             `kaiser85`,
             `hamming50`,
             `hamming75`,
             `hann50`,
             `hann75`,
             `rectangular0`
            ]
        scheme (str): The adaptation scheme to use. Supported values:
            [
             `ola`,
             `wola`,
             `analysis`
            ]
        segment_size (int): The size of the segment/window.

    Returns:
        Window: A `Window` object containing the selected window function and its
            corresponding hop size.

    Raises:
        ValueError: If an unknown window type or scheme is provided.

    """
    if window == "bartlett50":
        from libsegmenter.windows.bartlett import bartlett50

        return _adapt_window(*bartlett50(segment_size), scheme)

    if window == "bartlett75":
        from libsegmenter.windows.bartlett import bartlett75

        return _adapt_window(*bartlett75(segment_size), scheme)

    if window == "blackman67":
        from libsegmenter.windows.blackman import blackman67

        return _adapt_window(*blackman67(segment_size), scheme)

    if window == "kaiser85":
        from libsegmenter.windows.kaiser import kaiser85

        return _adapt_window(*kaiser85(segment_size), scheme)

    if window == "hamming50":
        from libsegmenter.windows.hamming import hamming50

        return _adapt_window(*hamming50(segment_size), scheme)

    if window == "hamming75":
        from libsegmenter.windows.hamming import hamming75

        return _adapt_window(*hamming75(segment_size), scheme)

    if window == "hann50":
        from libsegmenter.windows.hann import hann50

        return _adapt_window(*hann50(segment_size), scheme)

    if window == "hann75":
        from libsegmenter.windows.hann import hann75

        return _adapt_window(*hann75(segment_size), scheme)

    if window == "rectangular0":
        from libsegmenter.windows.rectangular import rectangular0

        return _adapt_window(*rectangular0(segment_size), scheme)

    if window == "rectangular50":
        from libsegmenter.windows.rectangular import rectangular50

        return _adapt_window(*rectangular50(segment_size), scheme)

    raise ValueError(f"The '{window}' window is not known.")
