class SegmenterNumpy:
    def __init__(self, window_obj: windowObjectNumpy):
        """
        A class for segmenting input data using windowing and hop size with
        support for WOLA and OLA modes.

        Attributes:
        """
        self.window_obj = window_obj

    def segment(self, x):
        return self._segment(x)

    def unsegment(self, X):
        return self._unsegment(X)

    def spectrogram(self, x):
        return self._segment(x, compute_spectrogram=True)

    def unspectrogram(self, X):
        return self._unsegment(X, compute_spectrogram=True)

    def _segment(self, x, compute_spectrogram=False):
        if x.ndim == 2:
            number_of_batch_elements = x.shape[0]
            number_of_samples = x.shape[1]
            batched = True
        elif x.ndim == 1:
            number_of_batch_elements = 1
            number_of_samples = x.shape[0]

            # convert to batched to simplify subsequent code
            batched = False
            x = x.reshape((1, x.size))
        else:
            raise ValueError(
                f"only support for inputs with dimension 1 or 2, provided {x.ndim}"
            )

        number_of_segments = (
            (number_of_samples) // self.window_obj.hop_size
            - self.window_obj.segment_size // self.window_obj.hop_size
            + 1
        )

        X = np.zeros(
            (
                number_of_batch_elements,
                number_of_segments,
                self.window_obj.segment_size,
            ),
        )

        for k in range(number_of_segments):
            X[:, k, :] = (
                x[
                    :,
                    k * self.window_obj.hop_size : k * self.window_obj.hop_size
                    + self.window_obj.segment_size,
                ]
                * self.window_obj.analysis_window
            )

        if compute_spectrogram:
            X = np.fft.rfft(X)

        if not batched:
            # convert back to not-batched
            X = X.squeeze(0)

        return X

    def _unsegment(self, X, compute_spectrogram=False):
        if X.ndim == 3:
            number_of_batch_elements = X.shape[0]
            number_of_segments = X.shape[1]
            batched = True

        elif X.ndim == 2:
            number_of_batch_elements = 1
            number_of_segments = X.shape[0]

            # convert to batched to simplify subsequent code
            batched = False
            X = X.reshape((1, X.shape[0], X.shape[1]))
        else:
            raise ValueError(
                f"only support for inputs with dimension 2 or 3,provided {X.dim()}"
            )

        if compute_spectrogram:
            X = np.fft.irfft(X)

        number_of_samples = (
            number_of_segments - 1
        ) * self.window_obj.hop_size + self.window_obj.segment_size

        x = np.zeros((number_of_batch_elements, number_of_samples))
        for k in range(number_of_segments):
            x[
                :,
                k * self.window_obj.hop_size : k * self.window_obj.hop_size
                + self.window_obj.segment_size,
            ] += self.window_obj.synthesis_window * X[:, k, :]

        if not batched:
            # convert back to not-batched
            x = x.squeeze(0)
        return x

