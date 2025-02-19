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

import torch

from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window


class SegmenterTorch(torch.nn.Module):
    """
    A PyTorch-based segmenter for input data using windowing techniques.

    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.

    Attributes:
        window (Window): A class containing hop size and windows.

    """

    def __init__(self, window: Window) -> None:
        """
        Initializes the SegmenterTorch instance.

        Args:
            window (Window): A window object containing segmentation parameters.

        """
        super().__init__()  # type: ignore

        self.window = window

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segments the input tensor into overlapping windows.

        Args:
            x (torch.Tensor): Input tensor (1D or 2D).

        Returns:
            Segmented tensor of shape (batch_size, num_segments, segment_size).

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
                "Input signal is too short for segmentation with the given parameters."
            )

        # Pre-allocation
        y = torch.zeros(
            (
                batch_size if batch_size is not None else 1,
                num_segments,
                self.window.analysis_window.shape[-1],
            ),
            device=x.device,
            dtype=x.dtype,
        )

        # Windowing
        analysis_window = torch.tensor(
            self.window.analysis_window, device=x.device, dtype=x.dtype
        )
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            y[:, k, :] = (
                x[:, start_idx : start_idx + self.window.analysis_window.shape[-1]]
                * analysis_window
            )

        return (
            y.squeeze(0) if batch_size is None else y
        )  # Remove batch dimension if needed

    def unsegment(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original signal from segmented data.

        Args:
            y (torch.Tensor): Segmented tensor (2D or 3D).

        Returns:
            Reconstructed 1D or 2D signal.

        Raises:
            ValueError: If types are incorrect.
            ValueError: If input dimensions are invalid.

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

        # allocate memory for the reconstructed signal
        x = torch.zeros(
            (batch_size if batch_size is not None else 1, num_samples),
            device=y.device,
            dtype=y.dtype,
        )

        # overlap-add method for reconstructing the original signal
        synthesis_window = torch.tensor(
            self.window.synthesis_window, device=x.device, dtype=x.dtype
        )

        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            x[:, start_idx : start_idx + segment_size] += y[:, k, :] * synthesis_window

        return x.squeeze(0) if batch_size is None else x
