import numpy as np


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
