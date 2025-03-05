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

from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window


class SegmenterTensorFlow(tf.keras.layers.Layer):
    """
    A TensorFlow-based segmenter for input data using windowing techniques.

    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.

    Attributes:
        window (Window): A class containing hop size, and windows.

    """

    def __init__(self, window: Window) -> None:
        """
        Initializes the SegmenterTensorFlow instance.

        Args:
            window (Window): A window object containing segmentation parameters.

        """
        super(SegmenterTensorFlow, self).__init__()  # type: ignore

        self.window = window

    def segment(self, x: tf.Tensor) -> tf.Tensor:
        """
        Segments the input tensor into overlapping windows.

        Args:
            x (tf.Tensor): Input tensor (1D or 2D).

        Returns:
            Segmented tensor of shape (batch_size, num_segments, segment_size).

        """
        if len(x.shape) not in {1, 2}:
            raise ValueError(
                f"Only supports 1D or 2D inputs, provided {len(x.shape)}D."
            )

        batch_size = x.shape[0] if len(x.shape) == 2 else None
        num_samples = x.shape[-1]

        if batch_size is None:
            x = tf.reshape(x, (1, -1))  # Convert to batch format

        num_segments = compute_num_segments(
            num_samples, self.window.hop_size, self.window.analysis_window.shape[-1]
        )

        if num_segments <= 0:
            raise ValueError(
                "Input signal is too short for segmentation with the given parameters."
            )

        # Pre-allocation
        X = tf.zeros(
            (
                batch_size if batch_size is not None else 1,
                num_segments,
                self.window.analysis_window.shape[-1],
            ),
            dtype=x.dtype,
        )

        # Windowing
        analysis_window = tf.convert_to_tensor(
            self.window.analysis_window, dtype=x.dtype
        )
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            X = tf.tensor_scatter_nd_update(
                X,
                [
                    [i, k, j]
                    for i in range(batch_size if batch_size is not None else 1)
                    for j in range(self.window.analysis_window.shape[-1])
                ],
                tf.reshape(
                    x[:, start_idx : start_idx + self.window.analysis_window.shape[-1]]
                    * analysis_window,
                    [-1],
                ),
            )

        return tf.squeeze(X, axis=0) if batch_size is None else X

    def unsegment(self, X: tf.Tensor) -> tf.Tensor:
        """
        Reconstructs the original signal from segmented data.

        Args:
            X (tf.Tensor): Segmented tensor (2D or 3D).

        Returns:
            Reconstructed 1D or 2D signal.

        """
        if self.window.synthesis_window is None:
            raise ValueError("Given windowing scheme does not support unsegmenting.")

        if len(X.shape) not in {2, 3}:
            raise ValueError(
                f"Only supports 2D or 3D inputs, provided {len(X.shape)}D."
            )

        batch_size = X.shape[0] if len(X.shape) == 3 else None
        num_segments = X.shape[-2]
        segment_size = X.shape[-1]

        if batch_size is None:
            X = tf.reshape(X, (1, num_segments, -1))  # Convert to batch format

        num_samples = compute_num_samples(
            num_segments, self.window.hop_size, segment_size
        )

        if num_samples <= 0:
            raise ValueError(
                "Invalid segment structure, possibly due to incorrect windowing "
                + "parameters."
            )

        # Allocate memory for the reconstructed signal
        x = tf.zeros(
            (batch_size if batch_size is not None else 1, num_samples), dtype=X.dtype
        )

        # Overlap-add method for reconstructing the original signal
        tf.convert_to_tensor(self.window.synthesis_window, dtype=X.dtype)

        for k in range(num_segments):
            tmpIdx = tf.reshape(
                tf.range(
                    k * self.window.hop_size, k * self.window.hop_size + segment_size
                ),
                shape=(segment_size, 1),
            )

            for b in range(batch_size if batch_size is not None else 1):
                idx = tf.concat([tf.fill((segment_size, 1), b), tmpIdx], axis=1)
                x = tf.tensor_scatter_nd_add(
                    x, idx, self.window.synthesis_window * X[b, k, :]
                )

        return tf.squeeze(x, axis=0) if batch_size is None else x
