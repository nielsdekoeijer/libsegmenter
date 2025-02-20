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
from typing import TypeVar, Any
from libsegmenter.transforms.spectrogram.SpectrogramNumpy import SpectrogramNumpy 

T = TypeVar("T", bound=np.generic)


class PhaseNumpy:
    """
    A class for computing magnitudes.

    Currently, the normalization cannot be controlled and is thus `backward` by default.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the PhaseNumpy instance."""
        self.spectrogram = SpectrogramNumpy(*args, **kwargs)

    def forward(self, x: NDArray[T]) -> NDArray[np.complex128]:
        """
        Converts segments into a magnitude.

        Args:
            x (NDArray[T]): Segments as generated by a Segmenter object.

        """
        tensor = self.spectrogram.forward(x)
        return np.arctan2(tensor.imag, tensor.real)
