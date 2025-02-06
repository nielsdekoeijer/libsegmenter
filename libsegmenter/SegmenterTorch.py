import torch
import numpy as np
from windowObjectNumpy import windowObjectNumpy


class SegmenterTorch(torch.nn.Module):
    def __init__(
        self,
        window_obj: windowObjectNumpy,
        device=None,
        dtype=None,
    ):
        """
        A class for segmenting input data using windowing and hop size with support for WOLA and OLA modes.

        Attributes:
            window (Tensor): Windowing function applied to each segment.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}

        super(SegmenterTorch, self).__init__()
        self.window_obj = window_obj
        self.analysis_window = torch.tensor(
            window_obj.analysis_window, device=device)
        self.synthesis_window = torch.tensor(
            window_obj.synthesis_window, device=device)

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
                f"only support for inputs with dimension 1 or 2, provided {
                    x.dim()}"
            )

        number_of_segments = (
            (number_of_samples) // self.window_obj.hop_size -
            self.window_obj.segment_size // self.window_obj.hop_size + 1
        )

        X = torch.zeros(
            (number_of_batch_elements, number_of_segments,
             self.window_obj.segment_size),
            **self.factory_kwargs,
        )

        for k in range(0, number_of_segments):
            X[:, k, :] = (x[:, k * self.window_obj.hop_size: k * self.window_obj.hop_size +
                          self.window_obj.segment_size, ] * self.analysis_window)

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
                f"only support for inputs with dimension 2 or 3, provided {
                    X.dim()}"
            )

        if compute_spectrogram:
            X = torch.fft.irfft(X)

        number_of_samples = (number_of_segments - 1) * \
            self.window_obj.hop_size + self.window_obj.segment_size

        x = torch.zeros(
            (number_of_batch_elements, number_of_samples), **self.factory_kwargs
        )
        for k in range(0, number_of_segments):
            x[:, k * self.window_obj.hop_size: k * self.window_obj.hop_size + self.window_obj.segment_size] += (
                self.synthesis_window * X[:, k, :]
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
        if (number_of_frequencies != self.window_obj.segment_size//2+1):
            raise Exception(
                "number_of_frequencies did not match self.frame_size//2+1")

        if (batched):
            bpd = torch.zeros(
                (number_of_batch_elements,
                 number_of_segments, number_of_frequencies),
                **self.factory_kwargs
            )
            modulation_factor = 2.0 * torch.pi * \
                torch.arange(0, self.window_obj.segment_size//2 + 1) / \
                self.window_obj.segment_size * self.window_obj.hop_size

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
                torch.arange(0, self.window_obj.segment_size//2 + 1) / \
                self.window_obj.segment_size * self.window_obj.hop_size

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
        if (number_of_frequencies != self.window_obj.segment_size//2+1):
            raise Exception(
                "number_of_frequencies did not match self.frame_size//2+1")

        modulation_factor = 2.0 * torch.pi * \
            torch.arange(0, self.window_obj.segment_size//2 + 1) / \
            self.window_obj.segment_size * self.window_obj.hop_size
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
        tmp = magnitude_spectrogram * torch.exp(1.0j*phase_spectrogram)
        if (tmp.dim() == 3):
            tmp[:, :, 0] = torch.real(tmp[:, :, 0]) + 0j
            tmp[:, :, -1] = torch.real(tmp[:, :, -1]) + 0j
        elif (tmp.dim() == 2):
            tmp[:, 0] = torch.real(tmp[:, 0]) + 0j
            tmp[:, -1] = torch.real(tmp[:, -1]) + 0j
        return tmp

    def assemble_spectrogram_real_imag(self, real_spectrogram, imag_spectrogram):
        return real_spectrogram + 1.0j*imag_spectrogram
