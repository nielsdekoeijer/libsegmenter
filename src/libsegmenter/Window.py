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

T = TypeVar("T", bound=np.generic)


class Window:
    """
    A class representing a windowing scheme used in signal segmentation.

    Attributes:
        hop_size (int): The step size for shifting the window in the segmentation
            process.
        analysis_window (NDArray[T]): The window function used during the analysis
           phase.
        synthesis_window (NDArray[T]): The window function used during the synthesis
           phase.

    """

    def __init__(
        self,
        hop_size: int,
        analysis_window: NDArray[T],
        synthesis_window: NDArray[T] | None,
    ) -> None:
        """
        Initializes the Window instance with the specified hop size and windows.

        Args:
            hop_size (int): The step size for shifting the window in the segmentation
                process.
            analysis_window (NDArray[T]): The window function applied during analysis.
            synthesis_window (NDArray[T] | None): The window function applied during
                synthesis.

        """
        self.hop_size = hop_size
        self.analysis_window = analysis_window
        self.synthesis_window = synthesis_window
