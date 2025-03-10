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


class SpectrogramTensorFlow:
    """
    A class for computing spectrograms using TensorFlow.

    The normalization for the Fourier transform is `backward` by default.
    """

    def __init__(self) -> None:
        """Initializes the SpectrogramTensorFlow instance."""
        return

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Converts segments into a spectrogram.

        Args:
            x (tf.Tensor): Input segments.

        Returns:
            tf.Tensor: Spectrogram representation.

        """
        s = tf.shape(x).numpy()  # pyright: ignore
        if s[-1] % 2 != 0:  # pyright: ignore
            raise ValueError(
                "Input segment size is expected to be even for a consistent definition "
                + "of the inverse real-valued FFT."
            )
        return tf.signal.rfft(x)  # pyright: ignore

    def inverse(self, y: tf.Tensor) -> tf.Tensor:
        """
        Converts spectrogram into segments.

        Args:
            y (tf.Tensor): Spectrogram from a `forward` pass.

        Returns:
            tf.Tensor: Reconstructed segments.

        """
        return tf.signal.irfft(y)  # pyright: ignore
