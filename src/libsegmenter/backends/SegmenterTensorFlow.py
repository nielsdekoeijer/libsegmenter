import tensorflow as tf

from libsegmenter.backends.common import compute_num_segments, compute_num_samples
from libsegmenter.Window import Window


class SegmenterTensorFlow(tf.keras.layers.Layer):
    """
    A TensorFlow-based segmenter for input data using windowing techniques.
    Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.
    
    Attributes:
        window (Window): A class containing hop size, segment size, and window functions.
    """

    def __init__(self, window: Window):
        """
        Initializes the SegmenterTensorFlow instance.

        Args:
            window (Window): A window object containing segmentation parameters.
        """
        super(SegmenterTensorFlow, self).__init__()

        self.window = window

    def segment(self, x: tf.Tensor) -> tf.Tensor:
        """
        Segments the input tensor into overlapping windows.

        Args:
            x (tf.Tensor): Input tensor (1D or 2D).

        Returns:
            tf.Tensor: Segmented tensor of shape (batch_size, num_segments, segment_size).
        """
        if not isinstance(x, tf.Tensor):
            raise TypeError("Input x must be a TensorFlow tensor.")

        if len(x.shape) not in {1, 2}:
            raise ValueError(f"Only supports 1D or 2D inputs, provided {len(x.shape)}D.")

        batch_size = x.shape[0] if len(x.shape) == 2 else None
        num_samples = x.shape[-1]

        if batch_size == None:
            x = tf.reshape(x, (1, -1))  # Convert to batch format

        num_segments = compute_num_segments(
            num_samples, self.window.hop_size, self.window.segment_size
        )

        if num_segments <= 0:
            raise ValueError("Input signal is too short for segmentation with the given parameters.")

        # Pre-allocation
        X = tf.zeros((batch_size, num_segments, self.window.segment_size), dtype=x.dtype)

        # Windowing
        analysis_window = tf.convert_to_tensor(self.window.analysis_window, dtype=x.dtype)
        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            X = tf.tensor_scatter_nd_update(
                X,
                [[i, k, j] for i in range(batch_size) for j in range(self.window.segment_size)],
                tf.reshape(x[:, start_idx : start_idx + self.window.segment_size] * analysis_window, [-1]),
            )
        
        return tf.squeeze(X, axis=0) if batch_size == None else X

    def unsegment(self, X: tf.Tensor) -> tf.Tensor:
        """
        Reconstructs the original signal from segmented data.

        Args:
            X (tf.Tensor): Segmented tensor (2D or 3D).

        Returns:
            tf.Tensor: Reconstructed 1D or 2D signal.
        """
        if not isinstance(X, tf.Tensor):
            raise TypeError("Input X must be a TensorFlow tensor.")

        if len(X.shape) not in {2, 3}:
            raise ValueError(f"Only supports 2D or 3D inputs, provided {len(X.shape)}D.")

        batch_size = X.shape[0] if len(X.shape) == 3 else None
        num_segments = X.shape[-2]
        segment_size = X.shape[-1]

        if batch_size == None:
            X = tf.reshape(X, (1, num_segments, -1))  # Convert to batch format

        num_samples = compute_num_samples(
            num_segments, self.window.hop_size, segment_size
        )

        if num_samples <= 0:
            raise ValueError("Invalid segment structure, possibly due to incorrect windowing parameters.")

        # Allocate memory for the reconstructed signal
        x = tf.zeros((batch_size if batch_size is not None else 1, num_samples), dtype=X.dtype)

        # Overlap-add method for reconstructing the original signal
        synthesis_window = tf.convert_to_tensor(self.window.synthesis_window, dtype=X.dtype)

        for k in range(num_segments):
            start_idx = k * self.window.hop_size
            x = tf.tensor_scatter_nd_add(
                x,
                [[i, j] for i in range(batch_size) for j in range(segment_size)],
                tf.reshape(X[:, k, :] * synthesis_window, [-1]),
            )
        
        return tf.squeeze(x, axis=0) if batch_size == None else x

