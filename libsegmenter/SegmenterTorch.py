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
            self.window = torch.tensor(window, device=device)
        elif isinstance(window, torch.Tensor):
            self.window = window.to(device)
        else:
            raise ValueError(
                "provided window is not numpy ndarray nor pytorch tensor")

        # asserts to ensure correctness
        if self.frame_size % 2 != 0:
            raise ValueError("only even frame_size is supported")

        if self.hop_size > self.frame_size:
            raise ValueError("hop_size cannot be larger than frame_size")

        if self.window.shape[0] != self.frame_size:
            raise ValueError(
                "specified window must have the same size as frame_size")

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
        if edge_correction:
            for h_idx in range(1, self.frame_size // self.hop_size + 1):
                idx1_start = h_idx * self.hop_size
                idx1_end = self.frame_size
                idx2_start = 0
                idx2_end = self.frame_size - idx1_start
                self.prewindow[idx2_start:idx2_end] = (
                    self.prewindow[idx2_start:idx2_end]
                    + self.window[idx1_start:idx1_end]
                )
                self.postwindow[idx1_start:idx1_end] = (
                    self.postwindow[idx1_start:idx1_end]
                    + self.window[idx2_start:idx2_end]
                )

        # Perform normalization of window function
        if normalize_window:
            value = check_cola(self.window.cpu().numpy(), self.hop_size)
            normalization = value[1]
            self.window = self.window / normalization
            self.prewindow = self.prewindow / normalization
            self.postwindow = self.postwindow / normalization

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
            (number_of_samples) // self.hop_size -
            self.frame_size // self.hop_size + 1
        )

        X = torch.zeros(
            (number_of_batch_elements, number_of_segments, self.frame_size),
            **self.factory_kwargs,
        )

        if self.mode == "wola":
            k = 0
            X[:, k, :] = (
                x[:, k * self.hop_size: k * self.hop_size + self.frame_size]
                * self.prewindow
            )
            for k in range(1, number_of_segments - 1):
                X[:, k, :] = (
                    x[
                        :,
                        k * self.hop_size: k * self.hop_size + self.frame_size,
                    ]
                    * self.window
                )
            k = number_of_segments - 1
            X[:, k, :] = (
                x[:, k * self.hop_size: k * self.hop_size + self.frame_size]
                * self.postwindow
            )
        else:
            for k in range(number_of_segments):
                X[:, k, :] = x[
                    :, k * self.hop_size: k * self.hop_size + self.frame_size
                ]

        if compute_spectrogram:
            X = torch.fft.rfft(X)

        if not batched:
            # convert back to not-batched
            X = X.squeeze(0)

        return X

    def _unsegment(self, X, compute_spectrogram=False):
        if X.dim() == 3:
            number_of_batch_elements = X.shape[0]
            number_of_segments = X.shape[1]
            batched = True

        elif X.dim() == 2:
            number_of_batch_elements = 1
            number_of_segments = X.shape[0]

            # convert to batched to simplify subsequent code
            batched = False
            X = X.unsqueeze(0)
        else:
            raise ValueError(
                f"only support for inputs with dimension 2 or 3, provided {X.dim()}"
            )

        if compute_spectrogram:
            X = torch.fft.irfft(X)

        number_of_samples = (number_of_segments - 1) * \
            self.hop_size + self.frame_size

        x = torch.zeros(
            (number_of_batch_elements, number_of_samples), **self.factory_kwargs
        )
        k = 0
        x[:, k * self.hop_size: k * self.hop_size + self.frame_size] += (
            self.prewindow * X[:, k, :]
        )
        for k in range(1, number_of_segments - 1):
            x[:, k * self.hop_size: k * self.hop_size + self.frame_size] += (
                self.window * X[:, k, :]
            )
        k = k + 1
        x[:, k * self.hop_size: k * self.hop_size + self.frame_size] += (
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
        return self._segment(x, compute_spectrogram=True)

    def unspectrogram(self, X):
        return self._unsegment(X, compute_spectrogram=True)

    def magnitude_spectrogram(self, spectrogram):
        return spectrogram.abs()

    def phase_spectrogram(self, spectrogram):
        return spectrogram.angle()

    def bpd_transform(self, phase_spectrogram):
        # Transform from phase "spectrogram" to baseband phase difference "spectrogram"
        if (phase_spectrogram.dim() == 2):
            number_of_batch_elements = 1
            number_of_segments = phase_spectrogram.shape[0]
            number_of_frequencies = phase_spectrogram.shape[1]
            batched = False
        elif (phase_spectrogram.dim() == 3):
            number_of_batch_elements = phase_spectrogram.shape[0]
            number_of_segments = phase_spectrogram.shape[1]
            number_of_frequencies = phase_spectrogram.shape[2]
            batched = True
        if (number_of_frequencies != self.frame_size//2+1):
            raise Exception("number_of_frequencies did not match self.frame_size//2+1")

        if (batched):
            bpd = torch.zeros(
                (number_of_batch_elements,
                 number_of_segments, number_of_frequencies),
                **self.factory_kwargs
            )
            modulation_factor = 2.0 * torch.pi * \
                torch.arange(0, self.frame_size//2 + 1)/self.frame_size * self.hop_size

            for bIdx in range(0, number_of_batch_elements):
                bpd[bIdx, 0, :] = torch.exp(1.0j*(
                    phase_spectrogram[bIdx, 0, :] - modulation_factor)).angle()
                for sIdx in range(1, number_of_segments):
                    bpd[bIdx, sIdx, :] = torch.exp(1.0j*(phase_spectrogram[bIdx, sIdx, :] -
                                               phase_spectrogram[bIdx, sIdx - 1, :] - modulation_factor)).angle()
        else:
            bpd = torch.zeros(
                (number_of_segments, number_of_frequencies),
                **self.factory_kwargs
            )
            modulation_factor = 2.0 * torch.pi * \
                torch.arange(0, self.frame_size//2 + 1)/self.frame_size * self.hop_size

            bpd[0, :] = torch.exp(1.0j*(
                phase_spectrogram[0, :] - modulation_factor)).angle()
            for sIdx in range(1, number_of_segments):
                bpd[sIdx, :] = torch.exp(1.0j*(phase_spectrogram[sIdx, :] -
                                            phase_spectrogram[sIdx - 1, :] - modulation_factor)).angle()
        return bpd

    def inverse_bpd_transform(self, bpd):
        # Transform from baseband phase difference "spectrogram" to phase "spectrogram"
        if (bpd.dim() == 2):
            number_of_batch_elements = 1
            number_of_segments = bpd.shape[0]
            number_of_frequencies = bpd.shape[1]
            batched = False
        elif (bpd.dim() == 3):
            number_of_batch_elements = bpd.shape[0]
            number_of_segments = bpd.shape[1]
            number_of_frequencies = bpd.shape[2]
            batched = True
        if (number_of_frequencies != self.frame_size//2+1):
            raise Exception("number_of_frequencies did not match self.frame_size//2+1")

        modulation_factor = 2.0 * torch.pi * \
            torch.arange(0, self.frame_size//2 + 1)/self.frame_size * self.hop_size
        if (batched):
            phase_spectrogram = torch.zeros(
                (number_of_batch_elements, number_of_segments, number_of_frequencies),
                **self.factory_kwargs
            )
            for bIdx in range(0, number_of_batch_elements):
                for fIdx in range(0, number_of_frequencies):
                    phase_spectrogram[bIdx, :, fIdx] = torch.exp(
                        1.0j*(torch.cumsum(bpd[bIdx, :, fIdx] + modulation_factor[fIdx], dim=0))).angle()
        else:
            phase_spectrogram = torch.zeros(
                (number_of_segments, number_of_frequencies),
                **self.factory_kwargs
            )
            for fIdx in range(0, number_of_frequencies):
                phase_spectrogram[:, fIdx] = torch.exp(
                    1.0j*(torch.cumsum(bpd[:, fIdx] + modulation_factor[fIdx], dim=0))).angle()
        return phase_spectrogram

    def assemble_spectrogram_magnitude_phase(self, magnitude_spectrogram, phase_spectrogram):
        tmp = magnitude_spectrogram *torch.exp(1.0j*phase_spectrogram)
        if (tmp.dim() == 3):
            tmp[:,:,0] = torch.real(tmp[:,:,0]) + 0j
            tmp[:,:,-1] = torch.real(tmp[:,:,-1]) + 0j
        elif (tmp.dim() == 2):
            tmp[:,0] = torch.real(tmp[:,0]) + 0j
            tmp[:,-1] = torch.real(tmp[:,-1]) + 0j
        return tmp

    def assemble_spectrogram_real_imag(self, real_spectrogram, imag_spectrogram):
        return real_spectrogram + 1.0j*imag_spectrogram
