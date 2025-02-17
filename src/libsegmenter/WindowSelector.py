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

from libsegmenter.Window import Window

def adapt_window(window: np.ndarray, hop_size: int, scheme: str) -> Window:
    # TODO: windows ALWAYS normalized
    # TODO: windows ALWAYS cola_checked

    if scheme == "ola":
        return Window(hop_size, np.ones(window.shape), window) 

    if scheme == "wola":
        window = np.sqrt(window)
        return Window(hop_size, window, window)

    if scheme == "analysis":
        return Window(hop_size, window, None)

    raise ValueError(f"The '{scheme}' scheme is not supported.")

def WindowSelector(window: str, scheme: str, segment_size: int) -> Window:
    if window == "bartlett50":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.bartlett50 import bartlett50
        return bartlett50(segment_size)

    if window == "bartlett75":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.bartlett75 import bartlett75
        return bartlett75(segment_size)

    if window == "blackman":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.blackman import blackman
        return blackman(segment_size)

    if window == "kaiser82":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.kaiser82 import kaiser82
        return kaiser82(segment_size)

    if window == "kaiser85":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.kaiser85 import kaiser85
        return kaiser85(segment_size)

    if window == "hamming50":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.hamming50 import hamming50
        return hamming50(segment_size)

    if window == "hamming75":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.hamming75 import hamming75
        return hamming75(segment_size)

    if window == "hann50":
        from libsegmenter.window.hann import hann50
        return adapt_window(*hann50(segment_size))

    if window == "hann75":
        from libsegmenter.window.hann import hann75
        return adapt_window(*hann75(segment_size))

    if window == "rectangular0":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.rectangular0 import rectangular0
        return rectangular0(segment_size)

    if window == "rectangular50":
        raise NotImplementedError(f"The '{window}' window is not implemented yet.")
        from libsegmenter.window.rectangular50 import rectangular50
        return rectangular50(segment_size)

    raise ValueError(f"The '{window}' window is not known.")
