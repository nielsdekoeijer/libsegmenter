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
from typing import TypeVar, Any

T = TypeVar("T", bound=np.generic)


def Spectrogram(backend: str = "numpy", *args: Any, **kwargs: Any) -> Any:
    """
    Factory function to create a spectrogram instance based on the specified backend.

    Args:
        backend (str, optional): The backend to use. Supported options:
            ["numpy", "torch", "tensorflow"]. Defaults to "numpy".
        *args (Any): Additional positional arguments to pass to the segmenter.
        **kwargs (Any): Additional keyword arguments to pass to the segmenter.

    Returns:
        An instance of the transform corresponding to the chosen backend.

    Raises:
        ValueError: If an unsupported backend is specified.
        NotImplementedError: If the backend is not implemented.

    """
    if backend == "numpy":
        from libsegmenter.transforms.spectrogram.SpectrogramNumpy import (
            SpectrogramNumpy,
        )

        return SpectrogramNumpy(*args, **kwargs)

    if backend == "tensorflow":
        from libsegmenter.transforms.spectrogram.SpectrogramTensorFlow import (
            SpectrogramTensorFlow,
        )

        return SpectrogramTensorFlow(*args, **kwargs)

    if backend == "torch":
        from libsegmenter.transforms.spectrogram.SpectrogramTorch import (
            SpectrogramTorch,
        )

        return SpectrogramTorch(*args, **kwargs)

    raise ValueError(f"The '{backend}' backend is not known.")
