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


class SpectrogramNumpy:
    """
    A class for computing spectrograms.

    Currently, the normalization for the fourier transform cannot be controlled and is
    thus `backward` by default.

    """

    def __init__(self) -> None:
        """Initializes the SpectrogramNumpy instance."""
        return

    def forward(self, x: NDArray[T]) -> NDArray[np.complex128]:
        """
        Converts segments into a spectrogram.

        Args:
            x (NDArray[T]): Segments as generated by a Segmenter object.

        """
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "Input segment size is expected to be even for a consistent definition "
                + "of the inverse real-valued FFT."
            )
        return np.fft.rfft(x, axis=-1, norm="backward")

    def inverse(self, y: NDArray[np.complex128]) -> NDArray[np.float64]:
        """
        Converts spectrogram into segments.

        Args:
            y (NDArray[np.complex128]): Spectrogram resulting from a `forward` pass.

        """
        return np.fft.irfft(y, axis=-1, norm="backward")
