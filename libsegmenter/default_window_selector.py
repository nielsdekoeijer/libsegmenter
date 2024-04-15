import numpy as np

def kaiser(window_length: int, beta: float) -> np.array:
    """
    Note that the Kaiser window is not strictly speaking COLA compliant as it does not have harmonic nulls that can be tuned to the
    harmonics of the frame rate. As such it only offers approximate perfect reconstruction. However, by tuning the beta parameter,
    the reconstruction error can be controlled
    """
    M = np.float64(window_length + 1.0)
    m = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0, dtype=np.float64)
    window = np.i0(beta * np.sqrt(1 - (m / (M / 2)) ** 2.0)) / np.i0(beta)
    return window


def default_window_selector(window_name: str, window_length: int) -> np.array:
    if window_name == "bartlett50":
        # Bartlett (triangular) window with 50% overlap
        if int(window_length) % 2 != 0:
            raise ValueError(
                "Odd length Bartlett window with 50 percent overlap is not currently supported."
            )
        else:
            window = bartlett(window_length)
            hopSize = int(window_length) // 2
            return window, hopSize
    elif window_name == "bartlett75":
        # Bartlett window with 75% overlap
        if int(window_length) % 4 != 0:
            raise ValueError(
                "Bartlett windows with 75 percent overlap expects a window_length divisible by 4."
            )
        else:
            window = bartlett(window_length)
            hopSize = int(window_length) // 4
            return window, hopSize
    elif window_name == "blackman":
        # Blackman window with 2/3 overlap
        if int(window_length) % 3 != 0:
            raise ValueError(
                "The Blackman window currently only supports overlaps of 2/3."
            )
        else:
            window = blackman(window_length)
            hopSize = int(window_length) // 3
            return window, hopSize
    elif window_name == "kaiser82":
        # Kaiser window with beta = 8 and approx 82% overlap
        beta = 8.0
        window = kaiser(window_length, beta)
        hopSize = int(np.floor(1.7 * (np.float64(window_length) - 1.0) / (beta + 1.0)))
        return window, hopSize
    elif window_name == "kaiser85":
        # Kaiser window with beta = 10 and approx 85% overlap
        beta = 10.0
        window = kaiser(window_length, beta)
        hopSize = int(np.floor(1.7 * (np.float64(window_length) - 1.0) / (beta + 1.0)))
        return window, hopSize
    elif window_name == "hamming50":
        # Hamming window with 50% overlap
        if window_length % 2 != 0:
            raise ValueError(
                "Odd length Hamming window at 50 percent overlap is not currently supported."
            )
        else:
            window = hamming(window_length)
            hopSize = int(window_length) // 2
            return window, hopSize
    elif window_name == "hamming75":
        # Hamming window with 75% overlap
        if int(window_length) % 4 != 0:
            raise ValueError(
                "For Hamming windows with 75 percent overlay, the window_length is expected to be divisible by 4."
            )
        else:
            window = hamming(window_length)
            hopSize = int(window_length) // 4
            return window, hopSize
    elif window_name == "hann50":
        # Hann window with 50% overlap
        if int(window_length) % 2 != 0:
            raise ValueError(
                "Odd length Hann window at 50 percent overlap is not currently supported."
            )
        else:
            window = hann(window_length)
            hopSize = int(window_length) // 2
            return window, hopSize
    elif window_name == "hann75":
        # Hann window with 75% overlap
        if int(window_length) % 4 != 0:
            raise ValueError(
                "For Hann windows with 75 percent overlap, the window_length is expected to be divisible by 4."
            )
        else:
            window = hann(window_length)
            hopSize = int(window_length) // 4
            return window, hopSize
    elif window_name == "rectangular0":
        # Rectangular window with 0% overlap
        window = np.ones(window_length, dtype=np.float64)
        hopSize = int(window_length)
        return window, hopSize
    elif window_name == "rectangular50":
        # Rectangular window with 50% overlap
        window = np.ones(window_length, dtype=np.float64)
        hopSize = int(window_length) // 2
        return window, hopSize
    else:
        raise ValueError("No valid window_name was provided.")
