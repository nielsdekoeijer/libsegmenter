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
from typing import TypeVar, Tuple, Any
from libsegmenter.transforms.spectrogram.SpectrogramNumpy import SpectrogramNumpy

T = TypeVar("T", bound=np.generic)


class MagnitudeNumpy:
    """
    A class for computing magnitude and phase spectra.

    Currently, the normalization for the fourier transform cannot be controlled and is
    thus `backward` by default.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MagnitudeNumpy instance."""
        self._spectrogram = SpectrogramNumpy(*args, **kwargs)

    def forward(self, x: NDArray[T]) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Converts segments into a magnitude and phase spectrogram.

        Args:
            x (NDArray[T]): Segments as generated by a Segmenter object.

        """
        tensor = self._spectrogram.forward(x)
        return np.abs(tensor), np.angle(tensor)

    def inverse(
        self, magnitude: NDArray[np.complex128], phase: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        """
        Converts magnitude / phase spectrogram into segments.

        Args:
            magnitude (NDArray[np.complex128]): Magnitude spectrogram resulting from a
                `forward` pass.
            phase (NDArray[np.complex128]): Phase spectrogram resulting from a
                `forward` pass.

        """
        return self._spectrogram.inverse(np.multiply(magnitude, np.exp(1j * phase)))
