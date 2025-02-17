import torch

from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window


class SegmenterTorch(torch.nn.Module):
    """
    A PyTorch-based segmenter for input data using windowing techniques.
    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.

    Attributes:
        window (Window): A class containing hop size, segment size, and window functions.
    """

    def __init__(self, window: Window):
        """
        Initializes the SegmenterTorch instance.

        Args:
            window (Window): A window object containing segmentation parameters.
        """
        super().__init__()

        self.window = window

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segments the input tensor into overlapping windows.

        Args:
            x (torch.Tensor): Input tensor (1D or 2D).

        Returns:
            torch.Tensor: Segmented tensor of shape (batch_size, num_segments, segment_size).

        Raises:
            ValueError: If types are incorrect.
            ValueError: If input dimensions are invalid.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a Torch tensor.")

        if x.ndim not in {1, 2}:
            raise ValueError(f"Only supports 1D or 2D inputs, provided {x.ndim}D.")

        batch_size = x.shape[0] if x.ndim == 2 else None
        num_samples = x.shape[-1]

        if batch_size == None:
            x = x.reshape(1, -1)  # Convert to batch format for consistency

        num_segments = compute_num_segments(
            num_samples, self.window.hop_size, self.window.segment_size
        )

        if num_segments <= 0:
            raise ValueError(
                f"Input signal is too short for segmentation with the given parameters."
            )

        # Pre-allocation
        X = torch.zeros(
            (batch_size, num_segments, self.window.segment_size),
            device=x.device,
            dtype=x.dtype,
        )

        # Windowing
        analysis_window = torch.tensor(
            self.window.analysis_window, device=x.device, dtype=x.dtype
        )
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            X[:, k, :] = (
                x[:, start_idx : start_idx + self.window.segment_size] * analysis_window
            )

        return (
            X.squeeze(0) if batch_size == None else X
        )  # Remove batch dimension if needed

    def unsegment(self, X: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original signal from segmented data.

        Args:
            X (torch.Tensor): Segmented tensor (2D or 3D).

        Returns:
            torch.Tensor: Reconstructed 1D or 2D signal.

        Raises:
            ValueError: If types are incorrect.
            ValueError: If input dimensions are invalid.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input x must be a Torch tensor.")

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

        # allocate memory for the reconstructed signal
        x = torch.zeros(
            (batch_size if batch_size != None else 1, num_samples),
            device=X.device,
            dtype=X.dtype,
        )

        # overlap-add method for reconstructing the original signal
        synthesis_window = torch.tensor(
            self.window.synthesis_window, device=x.device, dtype=x.dtype
        )

        print(x.shape)
        print(synthesis_window.shape)
        print(X.shape)
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            x[:, start_idx : start_idx + segment_size] += X[:, k, :] * synthesis_window

        return x.squeeze(0) if batch_size == None else x
