# Copyright (c) 2025 Niels de Koeijer, Martin Bo Møller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Tuple

T = TypeVar("T", bound=np.generic)


def check_cola(
    window: NDArray[T], hop_size: int, eps: float = 1e-5
) -> Tuple[bool, float, float]:
    """
    Checks the Constant Overlap Add (COLA) condition for a given window function.

    Args:
        window (NDArray[T]): The window samples.
        hop_size (int): The hop size between frames.
        eps (float): Tolerance for checking the COLA condition. Defaults to 1e-5.

    Returns:
        Tuple[bool, float, float]:
            A 3-tuple containing:
            (is_cola, normalization_value, epsilon)

    """
    factor = float(np.sum(window, dtype=np.float32)) / hop_size
    N = 6 * window.size

    # initialize accumulator
    sp = np.full(N, factor, dtype=np.complex128)
    ubound = sp[0].real
    lbound = sp[0].real

    # loop over partial shifts
    frame_rate = 1.0 / hop_size
    for k in range(1, hop_size):
        f = frame_rate * k

        # complex sinusoids
        csin = np.exp(1j * 2.0 * np.pi * f * np.arange(N))

        # frequency domain representation of window
        Wf = np.sum(window[:window.size] * np.conjugate(csin[:window.size]))
        sp += (Wf * csin) / hop_size
        Wfb = abs(Wf)
        ubound += Wfb / hop_size
        lbound -= Wfb / hop_size

    e = ubound - lbound
    return (e < eps, (ubound + lbound) / 2.0, e)
