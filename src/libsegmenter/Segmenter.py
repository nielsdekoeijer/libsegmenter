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

from typing import Any

BACKENDS = ["torch", "tensorflow", "numpy", "octave"]


def Segmenter(*args: Any, backend: str = "numpy", **kwargs: Any) -> Any:
    """
    Factory function to create a segmenter instance based on the specified backend.

    Args:
        backend (str, optional): The backend to use. Supported options:
            ["numpy", "torch", "tensorflow"]. Defaults to "numpy".
        *args (Any): Additional positional arguments to pass to the segmenter.
        **kwargs (Any): Additional keyword arguments to pass to the segmenter.

    Returns:
        An instance of the segmenter corresponding to the chosen backend.

    Raises:
        ValueError: If an unsupported backend or scheme is specified.
        NotImplementedError: If the backend or scheme is not implemented.

    """
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend {backend}, availible: {BACKENDS}")

    if backend == "numpy":
        from libsegmenter.backends.SegmenterNumpy import SegmenterNumpy

        return SegmenterNumpy(*args, **kwargs)

    if backend == "torch":
        from libsegmenter.backends.SegmenterTorch import SegmenterTorch

        return SegmenterTorch(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.backends.SegmenterTensorFlow import SegmenterTensorFlow

        return SegmenterTensorFlow(*args, **kwargs)

    raise NotImplementedError(f"The '{backend}' backend is not implemented yet.")
