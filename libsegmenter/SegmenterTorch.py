import torch
import numpy as np
from .bindings import check_cola


class SegmenterTorch(torch.nn.Module):
    def __init__(
        self,
        frame_size,
        hop_size,
        window,
        mode="wola",
        edge_correction=True,
        normalize_window=True,
        device=None,
        dtype=None,
    ):
        """
        A class for segmenting input data using windowing and hop size with support for WOLA and OLA modes.

        Attributes:
            frame_size (int): Size of each segment.
            hop_size (int): Hop size between segments.
            window (Tensor): Windowing function applied to each segment.
            mode (str): Either 'wola' or 'ola' for Weighted Overlap-Add or Overlap-Add method.
            edge_correction (bool): If True, apply edge correction to the first and last segments.
            normalize_window (bool): If True, normalize the windowing function.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}

        super(SegmenterTorch, self).__init__()
        self.hop_size = hop_size
        self.frame_size = frame_size
        if isinstance(window, np.ndarray):
            self.window = torch.tensor(window)
        elif isinstance(window, torch.Tensor):
            self.window = window.to(device)
        else:
            raise ValueError("provided window is not numpy ndarray nor pytorch tensor")

        # asserts to ensure correctness
        if self.frame_size % 2 != 0:
            raise ValueError("only even frame_size is supported")

        if self.hop_size > self.frame_size:
            raise ValueError("hop_size cannot be larger than frame_size")

        if self.window.shape[0] != self.frame_size:
            raise ValueError("specified window must have the same size as frame_size")

        if any(window < 0.0):
            raise ValueError("specified window contains negative values")

        if check_cola(self.window.cpu().numpy(), self.hop_size)[0] == False:
            raise ValueError(
                "specified window is not COLA, consider using `default_window_selector`"
            )

        # compute prewindow and postwindow
        self.prewindow = self.window.clone()
        self.postwindow = self.window.clone()
        i = self.hop_size
        for h_idx in range(1, self.frame_size // self.hop_size + 1):
            idx1_start = h_idx * self.hop_size
            idx1_end = self.frame_size
            idx2_start = 0
            idx2_end = self.frame_size - idx1_start
            self.prewindow[idx2_start:idx2_end] = (
                self.prewindow[idx2_start:idx2_end] + self.window[idx1_start:idx1_end]
            )
            self.postwindow[idx1_start:idx1_end] = (
                self.postwindow[idx1_start:idx1_end] + self.window[idx2_start:idx2_end]
            )

        # Perform normalization of window function
        if normalize_window:
            value = check_cola(self.window.cpu().numpy(), self.hop_size)
            normalization = value[1]
            self.window = self.window / normalization
            self.prewindow = self.prewindow / normalization
            self.postwindow = self.postwindow / normalization

        if not edge_correction:
            self.prewindow = self.window
            self.postwindow = self.window

        if mode == "wola":
            self.mode = mode
            self.window = torch.sqrt(self.window)
            self.prewindow = torch.sqrt(self.prewindow)
            self.postwindow = torch.sqrt(self.postwindow)
        elif mode == "ola":
            self.mode = mode
        else:
            raise ValueError(f"only support for model ola and wola")

    def _segment(self, x, compute_spectrogram=False):
        if (x.dim() == 2) and (x.shape[1] > 1):
            number_of_batch_elements = x.shape[0]
            number_of_samples = x.shape[1]
            batched = True
        elif x.dim() == 1:
            number_of_batch_elements = 1
            number_of_samples = x.shape[0]

            # convert to batched to simplify subsequent code
            batched = False
            x = x.unsqueeze(0)
        else:
            raise ValueError(
                f"only support for inputs with dimension 1 or 2, provided {x.dim()}"
            )

        number_of_segments = (
            (number_of_samples) // self.hop_size - self.frame_size // self.hop_size + 1
        )

        X = torch.zeros(
            (number_of_batch_elements, number_of_segments, self.frame_size),
            **self.factory_kwargs,
        )

        for b in range(number_of_batch_elements):
            if self.mode == "wola":
                k = 0
                X[:, k, :] = (
                    x[:, k * self.hop_size : k * self.hop_size + self.frame_size]
                    * self.prewindow
                )
                for k in range(1, number_of_segments - 1):
                    X[:, k, :] = (
                        x[
                            :,
                            k * self.hop_size : k * self.hop_size + self.frame_size,
                        ]
                        * self.window
                    )
                k = number_of_segments - 1
                X[:, k, :] = (
                    x[:, k * self.hop_size : k * self.hop_size + self.frame_size]
                    * self.postwindow
                )
            else:
                for k in range(number_of_segments):
                    X[:, k, :] = x[
                        :, k * self.hop_size : k * self.hop_size + self.frame_size
                    ]

        if compute_spectrogram:
            X = torch.fft.rfft(X)

        # torchaudio convention
        X = X.permute(0, 2, 1)

        if not batched:
            # convert back to not-batched
            X = X.squeeze(0)

        return X

    def _unsegment(self, X, compute_spectrogram=False):
        if X.dim() == 3:
            number_of_batch_elements = X.shape[0]
            number_of_segments = X.shape[2]
            batched = True

        elif X.dim() == 2:
            number_of_batch_elements = 1
            number_of_segments = X.shape[1]

            # convert to batched to simplify subsequent code
            batched = False
            X = X.unsqueeze(0)
        else:
            raise ValueError(
                f"only support for inputs with dimension 2 or 3, provided {X.dim()}"
            )

        # torchaudio convention
        X = X.permute(0, 2, 1)

        if compute_spectrogram:
            X = torch.fft.irfft(X)

        number_of_samples = (number_of_segments - 1) * self.hop_size + self.frame_size

        x = torch.zeros(
            (number_of_batch_elements, number_of_samples), **self.factory_kwargs
        )
        k = 0
        x[:, k * self.hop_size : k * self.hop_size + self.frame_size] += (
            self.prewindow * X[:, k, :]
        )
        for k in range(1, number_of_segments - 1):
            x[:, k * self.hop_size : k * self.hop_size + self.frame_size] += (
                self.window * X[:, k, :]
            )
        k = k + 1
        x[:, k * self.hop_size : k * self.hop_size + self.frame_size] += (
            self.postwindow * X[:, k, :]
        )

        if not batched:
            # convert back to not-batched
            x = x.squeeze(0)
        return x

    def segment(self, x):
        return self._segment(x)

    def unsegment(self, X):
        return self._unsegment(X)

    def spectrogram(self, x):
        X = self._segment(x, compute_spectrogram=True)

    def unspectrogram(self, X):
        return self._unsegment(X, compute_spectrogram=True)
