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
from typing import TypeVar
from libsegmenter.Window import Window
from libsegmenter.util.check_cola import check_cola

T = TypeVar("T", bound=np.generic)


def AsymmetricWindowSelector(
    scheme: str,
    analysis_segment_size: int,
    hop_size: int,
    synthesis_segment_size: int,
) -> Window:
    """
    Designs an asymmetric Hann window pair based on the given parameters.

    This function retrieves a window function based on the `window` type, applies
    an adaptation based on `scheme`, and returns the corresponding `Window` object.

    Args:
        scheme (str): The adaptation scheme to use. Supported values:
            [
             `ola`,
             `wola`,
            ]
        analysis_segment_size (int): The size of the segment / analysis_window.
        hop_size (int): The hop size used for segmentation.
        synthesis_segment_size (int): The non-zero size of the systhesis_window.

    Returns:
        Window: A `Window` object containing the selected window function and its
            corresponding hop size.

    Raises:
        ValueError: If an unknown window type or scheme is provided.

    """
    if analysis_segment_size % hop_size != 0:
        raise ValueError(
            "The analysis_segment_size must be integer divisible by hop_size."
            + f" Received analysis_segment_size = '{analysis_segment_size}' "
            + f" and hop_size = '{hop_size}'."
        )
    if synthesis_segment_size % hop_size != 0:
        raise ValueError(
            "The synthesis_segment_size must be integer divisible by hop_size."
            + f" Received synthesis_segment_size = '{synthesis_segment_size}' "
            + f" and hop_size = '{hop_size}'."
        )
    if scheme != "ola" and scheme != "wola":
        raise ValueError(f"The '{scheme}' scheme is not supported.")

    elif scheme == "ola":
        from libsegmenter.windows.hann import asymmetricHannOla

        windows = asymmetricHannOla(
            analysis_segment_size, hop_size, synthesis_segment_size
        )

    else:  # WOLA
        from libsegmenter.windows.hann import asymmetricHannWola

        windows = asymmetricHannWola(
            analysis_segment_size, hop_size, synthesis_segment_size
        )

    analysis_window = windows[0]
    synthesis_window = windows[1]
    window = np.multiply(analysis_window, synthesis_window)

    is_cola, normalization, e = check_cola(window, hop_size)
    assert is_cola, f"specified window failed cola check with error {e}"

    synthesis_window = np.divide(synthesis_window, normalization)

    return Window(hop_size, analysis_window, synthesis_window)
