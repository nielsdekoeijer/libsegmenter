import numpy as np
import tensorflow as tf
from .bindings import check_cola


class SegmenterTensorFlow(tf.Module):
    def __init__(
        self,
        frame_size,
        hop_size,
        window,
        mode="wola",
        edge_correction=True,
        normalize_window=True,
    ):
        """
        A class for segmenting input data using windowing and hop size with support for WOLA and OLA modes.

        Attributes:
            segment_size (int): Size of each segment.
            hop_size (int): Hop size between segments.
            window (Tensor): Windowing function applied to each segment.
            mode (str): Either 'wola' or 'ola' for Weighted Overlap-Add or Overlap-Add method.
            edge_correction (bool): If True, apply edge correction to the first and last segments.
            normalize_window (bool): If True, normalize the windowing function.
        """
        super(SegmenterTensorFlow, self).__init__()
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.window = window

        # asserts to ensure correctness
        if self.frame_size % 2 != 0:
            raise ValueError("only even frame_size is supported")

        if self.hop_size > self.frame_size:
            raise ValueError("hop_size cannot be larger than frame_size")

        if self.window.shape[0] != self.frame_size:
            raise ValueError("specified window must have the same size as frame_size")

        if any(window < 0.0):
            raise ValueError("specified window contains negative values")

        if check_cola(window, self.hop_size)[0] == False:
            raise ValueError(
                "specified window is not COLA, consider using `default_window_selector`"
            )

        # compute prewindow and postwindow
        prewindow = np.copy(window)
        for hIdx in range(1, self.frame_size // self.hop_size + 1):
            idx1Start = hIdx * self.hop_size
            idx1End = self.frame_size
            idx2Start = 0
            idx2End = self.frame_size - idx1Start
            prewindow[idx2Start:idx2End] = (
                prewindow[idx2Start:idx2End] + window[idx1Start:idx1End]
            )

        postwindow = np.copy(window)
        for hIdx in range(1, self.frame_size // self.hop_size + 1):
            idx1Start = hIdx * self.hop_size
            idx1End = self.frame_size
            idx2Start = 0
            idx2End = self.frame_size - idx1Start
            postwindow[idx1Start:idx1End] = (
                postwindow[idx1Start:idx1End] + window[idx2Start:idx2End]
            )

        # this is a tiny bit hacked, but it works well in practise
        if normalize_window:
            value = check_cola(window, hop_size)
            normalization = value[1]
            window = window / normalization
            prewindow = prewindow / normalization
            postwindow = postwindow / normalization

        if not edge_correction:
            prewindow = window
            postwindow = window

        if mode == "wola":
            self.mode = mode
            window = np.sqrt(window)
            prewindow = np.sqrt(prewindow)
            postwindow = np.sqrt(postwindow)
        elif mode == "ola":
            self.mode = mode
        else:
            raise ValueError(f"only support for mode ola and wola")

        self.window = tf.convert_to_tensor(window, dtype=tf.float32)
        self.prewindow = tf.convert_to_tensor(prewindow, dtype=tf.float32)
        self.postwindow = tf.convert_to_tensor(postwindow, dtype=tf.float32)

    def _segment(self, x, compute_spectrogram=False):
        if (tf.rank(x) == 2) and (x.shape[1] > 1):
            number_of_batch_elements = x.shape[0]
            number_of_samples = x.shape[1]
            batched = True
        elif tf.rank(x) == 1:
            number_of_samples = x.shape[0]
            number_of_batch_elements = 1

            # convert to batched to simplify subsequent code
            batched = False
            x = tf.expand_dims(x, axis=0)

        else:
            raise ValueError(
                f"only support for inputs with dimension 1 or 2, provided {len(x.shape)}"
            )

        number_of_frames = (
            (number_of_samples) // self.hop_size - self.frame_size // self.hop_size + 1
        )

        X = tf.zeros(
            shape=(number_of_batch_elements, number_of_frames, self.frame_size)
        )
        for b in range(number_of_batch_elements):
            if self.mode == "wola":
                k = 0
                tmp = tf.reshape(
                    x[b, k * self.hop_size : k * self.hop_size + self.frame_size],
                    shape=(1, 1, self.frame_size),
                )
                X = tf.tensor_scatter_nd_add(X, [[[b, k]]], tmp * self.prewindow)
                for k in range(1, number_of_frames - 1):
                    tmp = tf.reshape(
                        x[
                            b,
                            k * self.hop_size : k * self.hop_size + self.frame_size,
                        ],
                        shape=(1, 1, self.frame_size),
                    )
                    X = tf.tensor_scatter_nd_add(X, [[[b, k]]], tmp * self.window)
                k = number_of_frames - 1
                tmp = tf.reshape(
                    x[b, k * self.hop_size : k * self.hop_size + self.frame_size],
                    shape=(1, 1, self.frame_size),
                )
                X = tf.tensor_scatter_nd_add(X, [[[b, k]]], tmp * self.postwindow)
            else:
                for k in range(number_of_frames):
                    tmp = tf.reshape(
                        x[
                            b,
                            k * self.hop_size : k * self.hop_size + self.frame_size,
                        ],
                        shape=(1, 1, self.frame_size),
                    )
                    X = tf.tensor_scatter_nd_add(X, [[[b, k]]], tmp)

        if compute_spectrogram:
            X = tf.signal.rfft(X)

        X = tf.transpose(X, perm=[0, 2, 1])

        if not batched:
            # convert back to not-batched
            X = tf.squeeze(X, axis=0)

        return X

    def _unsegment(self, X, compute_spectrogram=False):
        if tf.rank(X) == 3:
            number_of_batch_elements = X.shape[0]
            number_of_frames = X.shape[2]
            batched = True
        elif tf.rank(X) == 2:
            number_of_frames = X.shape[1]
            number_of_batch_elements = 1

            # convert to batched to simplify subsequent code
            batched = False
            X = tf.expand_dims(X, axis=0)
        else:
            raise ValueError(
                f"only support for inputs with dimension 2 or 3, provided {len(X.shape)}"
            )
        number_of_samples = (number_of_frames - 1) * self.hop_size + self.frame_size

        X = tf.transpose(X, perm=[0, 2, 1])

        if compute_spectrogram:
            X = tf.signal.irfft(X)

        x = tf.zeros(shape=(number_of_batch_elements, number_of_samples))
        for b in range(number_of_batch_elements):
            k = 0
            tmpIdx = tf.reshape(
                tf.range(k * self.hop_size, k * self.hop_size + self.frame_size),
                shape=(self.frame_size, 1),
            )
            idx = tf.concat(
                [tf.constant(b, shape=(self.frame_size, 1)), tmpIdx], axis=1
            )
            x = tf.tensor_scatter_nd_add(x, idx, self.prewindow * X[b, k, :])
            for k in range(1, number_of_frames - 1):
                tmpIdx = tf.reshape(
                    tf.range(k * self.hop_size, k * self.hop_size + self.frame_size),
                    shape=(self.frame_size, 1),
                )
                idx = tf.concat(
                    [tf.constant(b, shape=(self.frame_size, 1)), tmpIdx], axis=1
                )
                x = tf.tensor_scatter_nd_add(x, idx, self.window * X[b, k, :])
            k = k + 1
            tmpIdx = tf.reshape(
                tf.range(k * self.hop_size, k * self.hop_size + self.frame_size),
                shape=(self.frame_size, 1),
            )
            idx = tf.concat(
                [tf.constant(b, shape=(self.frame_size, 1)), tmpIdx], axis=1
            )
            x = tf.tensor_scatter_nd_add(x, idx, self.postwindow * X[b, k, :])

        if not batched:
            # convert back to not-batched
            x = tf.squeeze(x, axis=0)

        return x

    def segment(self, x):
        return self._segment(x)

    def unsegment(self, X):
        return self._unsegment(X)

    def spectrogram(self, x):
        X = self._segment(x, compute_spectrogram=True)

    def unspectrogram(self, X):
        return self._unsegment(X, compute_spectrogram=True)
