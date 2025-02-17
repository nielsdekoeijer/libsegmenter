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
from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window


class SegmenterNumpy:
    """
    A class for segmenting and reconstructing input data using windowing techniques.
    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.

    Attributes:
        window (Window): A class containing hop size, segment size, and window functions.
    """

    def __init__(self, window: Window):
        """
        Initializes the SegmenterNumpy instance.

        Args:
            window (Window): A window object containing segmentation parameters.
        """
        self.window = window

    def segment(self, x: np.ndarray) -> np.ndarray:
        """
        Segments the input signal into overlapping windows using the provided window parameters.

        Args:
            x (np.ndarray): Input array, either 1D (single sequence) or 2D (batch of sequences).

        Returns:
            np.ndarray: Segmented data of shape (batch_size, num_segments, segment_size).

        Raises:
            ValueError: If types are incorrect.
            ValueError: If input dimensions are invalid.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input x must be a NumPy array.")

        if x.ndim not in {1, 2}:
            raise ValueError(f"Only supports 1D or 2D inputs, provided {x.ndim}D.")

        batch_size = x.shape[0] if x.ndim == 2 else None
        num_samples = x.shape[-1]

        if batch_size == None:
            x = x.reshape(1, -1)  # Convert to batch format for consistency

        num_segments = compute_num_segments(
            num_samples, self.window.hop_size, self.window.analysis_window.shape[-1]
        )

        if num_segments <= 0:
            raise ValueError(
                f"Input signal is too short for segmentation with the given hop size ({self.window.hop_size}) and segment size ({self.window.analysis_window.shape[-1]})."
            )

        # Pre-allocation
        X = np.zeros(
            (
                batch_size if batch_size != None else 1,
                num_segments,
                self.window.analysis_window.shape[-1],
            ),
            dtype=x.dtype,
        )

        # Windowing
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            X[:, k, :] = (
                x[:, start_idx : start_idx + self.window.analysis_window.shape[-1]]
                * self.window.analysis_window
            )

        return (
            X.squeeze(0) if batch_size == None else X
        )  # Remove batch dimension if needed

    def unsegment(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstructs the original signal from segmented data using synthesis windowing.

        Args:
            X (np.ndarray): Segmented data with shape (batch_size, num_segments, segment_size)
                            or (num_segments, segment_size) for a single sequence.

        Returns:
            np.ndarray: Reconstructed signal.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a NumPy array.")

        if X.ndim not in {2, 3}:
            raise ValueError(f"Only supports 2D or 3D inputs, provided {X.ndim}D.")

        batch_size = X.shape[0] if X.ndim == 3 else None
        num_segments = X.shape[-2]
        segment_size = X.shape[-1]

        if batch_size == None:
            X = X.reshape(1, num_segments, -1)  # Convert to batch format

        num_samples = compute_num_samples(
            num_segments, self.window.hop_size, segment_size
        )

        if num_samples <= 0:
            raise ValueError(
                "Invalid segment structure, possibly due to incorrect windowing parameters."
            )

        # Efficient NumPy array allocation
        x = np.zeros(
            (batch_size if batch_size != None else 1, num_samples), dtype=X.dtype
        )

        # Vectorized accumulation
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            x[:, start_idx : start_idx + segment_size] += (
                X[:, k, :] * self.window.synthesis_window
            )

        return (
            x.squeeze(0) if batch_size == None else x
        )  # Remove batch dimension if needed
