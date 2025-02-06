import numpy as np
from check_cola import check_cola


class windowObjectNumpy():
    def __init__(
        self,
        segment_size,
        window_scheme,
        reconstruction_scheme,
        hop_size=0,
        synthesis_segment_size=0,
    ):
        """
        A class for designing a valid window object for the segmenter.
        Attributes:
            segment_size (int): Size of each segment after windowing.
            window_scheme (str): Type of window function and overlap.
            reconstruction_scheme (str): Reconstruction scheme (ola, wola, analysisonly)
            hop_size (int): Hop size to be used in asymmetric windowing schemes.
            synthesis_segment_size (int): Non-zero samples in the synthesis window (for asymmetric window schemes only).
        """
        self.segment_size = segment_size
        self.window_scheme = window_scheme
        self.reconstruction_scheme = reconstruction_scheme
        self.hop_size = hop_size
        self.synthesis_segment_size = synthesis_segment_size

        if hop_size == 0:
            # Symmetric windowing scheme
            if window_scheme == "hann50":
                # Hann window with 50% overlap
                if int(segment_size) % 2 != 0:
                    raise ValueError(
                        "Odd length Hann window at 50 percent overlap is not currently supported."
                    )
                else:
                    window = self._hann(segment_size)
                    hop_size = int(segment_size) // 2
            elif window_scheme == "hann75":
                # Hann window with 75% overlap
                if int(segment_size) % 4 != 0:
                    raise ValueError(
                        "For Hann windows with 75 percent overlap, the window_length is expected to be divisible by 4."
                    )
                else:
                    window = self._hann(segment_size)
                    hop_size = int(segment_size) // 4
            elif window_scheme == "rectangular0":
                # Rectangular window with 0% overlap
                window = np.ones(segment_size, dtype=np.float64)
                hop_size = int(segment_size)
            elif window_scheme == "rectangular50":
                # Rectangular window with 50% overlap
                window = np.ones(segment_size, dtype=np.float64)
                hop_size = int(segment_size) // 2
            else:
                raise ValueError("No valid window_name was provided.")

            # Normalize window function
            value = check_cola(window, hop_size)
            normalization = value[1]
            window = window / normalization
            self.hop_size = hop_size

            if reconstruction_scheme == "analysisonly":
                self.analysis_window = window
                self.synthesis_window = None
            elif reconstruction_scheme == "ola":
                self.analysis_window = np.ones(segment_size, dtype=np.float64)
                self.synthesis_window = window
            elif reconstruction_scheme == "wola":
                self.analysis_window = np.sqrt(window)
                self.synthesis_window = np.sqrt(window)
            else:
                raise ValueError(
                    f"only support for analysisonly, ola, and wola")

        else:
            # Asymmetric windowing scheme
            if int(segment_size) % int(hop_size) != int(0):
                raise ValueError(
                    "Segment_size is not integer divisible by hop_size")
            if int(synthesis_segment_size) % int(hop_size) != int(0):
                raise ValueError(
                    "Synthesis_segment_size is not integer divisible by hop_size")
            if segment_size < synthesis_segment_size:
                raise ValueError(
                    "The synthesis_segment_size is expected to be larger than the analysis_segment_size")
            if reconstruction_scheme == "ola":
                self.analysis_window = np.ones(segment_size, dtype=np.float64)
                self.synthesis_window = np.zeros(
                    segment_size, dtype=np.float64)
                self.synthesis_window[segment_size -
                                      synthesis_segment_size:] = hann(synthesis_segment_size)
            elif reconstruction_scheme == "wola":
                M = int(synthesis_segment_size) // 2
                KM = segment_size - M
                h1 = np.sqrt(self._inlineHann(
                    2*KM, np.arange(KM-M, dtype=np.float64)))
                h2 = np.sqrt(self._inlineHann(
                    2*KM, np.arange(KM, KM+M, dtype=np.float64)))
                h3 = np.sqrt(self._inlineHann(
                    2*M, np.arange(M, 2*M, dtype=np.float64)))
                analysis_window = np.concatenate((h1, h2, h3))

                f1 = np.zeros(KM-M, dtype=np.float64)
                f2 = self._inlineHann(2*M, np.arange(M, dtype=np.float64)) / np.sqrt(
                    self._inlineHann(2*KM, np.arange(KM, KM+M, dtype=np.float64)))
                f3 = np.sqrt(self._inlineHann(
                    2*M, np.arange(M, 2*M, dtype=np.float64)))
                synthesis_window = np.concatenate((f1, f2, f3))

                window = analysis_window * synthesis_window
                value = check_cola(window, hop_size)
                normalization = value[1]
                self.analysis_window = analysis_window
                self.synthesis_window = synthesis_window / normalization

    def _inlineHann(self, window_length, m):
        return 0.5*(1.0 - np.cos(2.0*np.pi*m/window_length))

    def _hann(self, window_length):
        m = np.arange(0, window_length, dtype=np.float64)
        window = 0.5*(1.0 - np.cos(2.0*np.pi*m/window_length))
        return window
