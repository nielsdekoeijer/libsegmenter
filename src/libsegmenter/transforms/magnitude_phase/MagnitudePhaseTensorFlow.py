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

import tensorflow as tf
from typing import Any, Tuple
from libsegmenter.transforms.spectrogram.SpectrogramTensorFlow import (
    SpectrogramTensorFlow,
)


class MagnitudePhaseTensorFlow:
    """A class for computing magnitudes using TensorFlow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the MagnitudePhaseTensorFlow instance."""
        self._spectrogram = SpectrogramTensorFlow(*args, **kwargs)

    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Converts segments into a magnitude.

        Args:
            x (tf.Tensor): Segments as generated by a Segmenter object.

        Returns:
            tf.Tensor: MagnitudePhase representation.

        """
        tensor = self._spectrogram.forward(x)
        return tf.abs(tensor), tf.math.angle(tensor)  # pyright: ignore

    def inverse(self, magnitude: tf.Tensor, phase: tf.Tensor) -> tf.Tensor:
        """
        Converts magnitude / phase spectrogram into segments.

        Args:
            magnitude (Tensor): MagnitudePhase spectrogram resulting from a `forward`
                pass.
            phase (Tensor): Phase spectrogram resulting from a `forward` pass.

        """
        magnitude_complex = tf.cast(magnitude, dtype=tf.complex64)  # pyright: ignore
        phase_complex = tf.cast(phase, dtype=tf.complex64)  # pyright: ignore
        j = tf.complex(0.0, 1.0)  # pyright: ignore
        complex_exp = tf.exp(j * phase_complex)  # pyright: ignore

        return self._spectrogram.inverse(
            magnitude_complex * complex_exp  # pyright: ignore
        )
