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

from libsegmenter.Window import Window

def WindowSelector(scheme: str, segment_size: int) -> Window:
    if window_name == "bartlett50":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.bartlett50 import bartlett50
        return bartlett50(segment_size)

    if window_name == "bartlett75":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.bartlett75 import bartlett75
        return bartlett75(segment_size)

    if window_name == "blackman":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.blackman import blackman
        return blackman(segment_size)

    if window_name == "kaiser82":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.kaiser82 import kaiser82
        return kaiser82(segment_size)

    if window_name == "kaiser85":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.kaiser85 import kaiser85
        return kaiser85(segment_size)

    if window_name == "hamming50":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.hamming50 import hamming50
        return hamming50(segment_size)

    if window_name == "hamming75":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.hamming75 import hamming75
        return hamming75(segment_size)

    if window_name == "hann50":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.hann50 import hann50
        return hann50(segment_size)

    if window_name == "hann75":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.hann75 import hann75
        return hann75(segment_size)

    if window_name == "rectangular0":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.rectangular0 import rectangular0
        return rectangular0(segment_size)

    if window_name == "rectangular50":
        raise NotImplementedError(f"The '{scheme}' windowing scheme is not implemented yet.")
        from libsegmenter.window.rectangular50 import rectangular50
        return rectangular50(segment_size)

    raise ValueError(f"The '{scheme}' windowing scheme is not known.")
