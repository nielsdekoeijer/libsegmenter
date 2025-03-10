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
from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window

T = TypeVar("T", bound=np.generic)


class SegmenterNumpy:
    """
    A class for segmenting and reconstructing input data using windowing techniques.

    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.

    Attributes:
        window (Window): A class containing hop size, segment size, and windows.

    """

    def __init__(self, window: Window) -> None:
        """
        Initializes the SegmenterNumpy instance.

        Args:
            window (Window): A window object containing segmentation parameters.

        """
        self.window = window

    def segment(self, x: NDArray[T]) -> NDArray[T]:
        """
        Segments the input signal into overlapping windows using the window parameters.

        Args:
            x (np.ndarray): Input array, either 1D (sequence) or 2D (batch).

        Returns:
            Segmented data of shape (batch_size, num_segments, segment_size).

        Raises:
            ValueError: If types are incorrect.
            ValueError: If input dimensions are invalid.

        """
        if x.ndim not in {1, 2}:
            raise ValueError(f"Only supports 1D or 2D inputs, provided {x.ndim}D.")

        batch_size = x.shape[0] if x.ndim == 2 else None
        num_samples = x.shape[-1]

        if batch_size is None:
            x = x.reshape(1, -1)  # Convert to batch format for consistency

        num_segments = compute_num_segments(
            num_samples, self.window.hop_size, self.window.analysis_window.shape[-1]
        )

        if num_segments <= 0:
            raise ValueError(
                "Input signal is too short for segmentation with the given num_samples "
                + f"({num_samples}), hop size "
                + f"({self.window.hop_size}) and segment size "
                + f"({self.window.analysis_window.shape[-1]})."
            )

        # Pre-allocation
        y = np.zeros(
            (
                batch_size if batch_size is not None else 1,
                num_segments,
                self.window.analysis_window.shape[-1],
            ),
            dtype=x.dtype,
        )

        # Windowing
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            y[:, k, :] = np.multiply(
                x[:, start_idx : start_idx + self.window.analysis_window.shape[-1]],
                self.window.analysis_window,
            )

        return y.squeeze(0) if batch_size is None else y

    def unsegment(self, y: NDArray[T]) -> NDArray[T]:
        """
        Reconstructs the original signal from segmented data using synthesis windowing.

        Args:
            y (np.ndarray): Segmented data with shape (batch_size, num_segments,
                            segment_size) or (num_segments, segment_size) for a single
                            sequence.

        Returns:
            Reconstructed signal.

        """
        if self.window.synthesis_window is None:
            raise ValueError("Given windowing scheme does not support unsegmenting.")

        if y.ndim not in {2, 3}:
            raise ValueError(f"Only supports 2D or 3D inputs, provided {y.ndim}D.")

        batch_size = y.shape[0] if y.ndim == 3 else None
        num_segments = y.shape[-2]
        segment_size = y.shape[-1]

        if batch_size is None:
            y = y.reshape(1, num_segments, -1)  # Convert to batch format

        num_samples = compute_num_samples(
            num_segments, self.window.hop_size, segment_size
        )

        if num_samples <= 0:
            raise ValueError(
                "Invalid segment structure, possibly due to incorrect windowing "
                + "parameters."
            )

        # Efficient numpy array allocation
        x = np.zeros(
            (batch_size if batch_size is not None else 1, num_samples), dtype=y.dtype
        )

        # Vectorized accumulation
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            x[:, start_idx : start_idx + segment_size] += np.multiply(
                y[:, k, :], self.window.synthesis_window
            )

        return x.squeeze(0) if batch_size is None else x
