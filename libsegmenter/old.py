# DEPRICATED IN FAVOUR OF THE CPP VERSION, BUT KEPT FOR DOCUMENTATION
def bartlett(window_length: int) -> np.array:
    """
    Bartlett (triangular) window
    """
    M = np.float64(window_length + 1.0)
    m = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0, dtype=np.float64)
    window = 1 - abs(m) * 2.0 / (M - 1)
    return window


def blackman(window_length: int) -> np.array:
    """
    Provides COLA-compliant windows for hopSize = (M-1)/3 when M is odd and
    hopSize M/3 when M is even
    """
    M = np.float64(window_length + 1.0)
    m = np.arange(0, M - 1, dtype=np.float64) / (M - 1)
    window = (
        7938.0 / 18608.0
        - 9240.0 / 18608.0 * np.cos(2.0 * np.pi * m)
        + 1430.0 / 18608.0 * np.cos(4.0 * np.pi * m)
    )

    return window


def hamming(window_length: int) -> np.array:
    """
    Provides COLA-compliant windows for hopSize = M/2, M/4, ...
    """
    M = np.float64(window_length)
    alpha = 25.0 / 46.0
    beta = (1 - alpha) / 2.0
    window = alpha - 2 * beta * np.cos(
        2.0 * np.pi * np.arange(0, window_length, dtype=np.float64) / window_length
    )
    return window


def hann(window_length: int) -> np.array:
    """
    Provides COLA-compliant windows for hopSize = window_length/2,
    window_length/4, ...
    """
    M = np.float64(window_length)
    m = np.arange(0, M, dtype=np.float64)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * m / M))
    return window

def check_cola(window: np.array, hop_size: int, eps=1e-5) -> (bool, np.float64):
    window_length = window.size

    # Assuming the sampling frequency is 1
    frame_rate = 1.0 / np.float64(hop_size)
    N = 6 * window_length
    sp = np.sum(window) / np.float64(hop_size) * np.ones(N, dtype=np.float64)
    ubound = sp[0] * 1.0
    lbound = sp[0] * 1.0
    n = np.arange(0, N, dtype=np.float64)

    for k in range(1, hop_size):
        f = frame_rate * k
        csin = np.exp(1j * 2.0 * np.pi * np.float64(f) * n)

        # Find exact window transform at frequency f
        Wf = np.sum(window * np.conj(csin[0:window_length]))
        hum = Wf * csin  # contribution to OLA "hum"
        sp = sp + hum / np.float64(hop_size)  # Poisson summation into OLA

        # Update lower and upper bounds
        Wfb = np.abs(Wf)
        ubound = ubound + Wfb / np.float64(hop_size)
        lbound = lbound - Wfb / np.float64(hop_size)

    normalization_value = (ubound + lbound) / 2
    if (ubound - lbound) < eps:
        return True, normalization_value
    else:
        return False, normalization_value
