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

import pytest
from hypothesis import given, settings, Phase
from hypothesis import strategies as st

import torch
import tensorflow
import itertools
import numpy as np

from libsegmenter.Segmenter import make_segmenter
from libsegmenter.Window import Window

BACKENDS = ["numpy", "torch", "tensorflow"]

# helper to convert whatever backend -> numpy
def as_numpy(x, backend):
    if backend == "torch":
        return x.numpy()
    if backend == "tensorflow":
        return x.numpy()

    return x

# helper to convert numpy -> whatever backend
def as_backend(x, backend):
    if backend == "torch":
        return torch.tensor(x)
    elif backend == "tensorflow":
        return tensorflow.convert_to_tensor(x, dtype=tensorflow.float32)

    return x

# runs randomized but reproduceably 100 times
@pytest.mark.parametrize("backendA, backendB", itertools.permutations(BACKENDS, 2))
@settings(max_examples=100, phases=[Phase.generate])
@given(
    segment_size=st.integers(min_value=8, max_value=64),
    hop_size=st.integers(min_value=1, max_value=63),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_consistency(backendA, backendB, segment_size, hop_size, seed):
    # reproducability
    np.random.seed(seed)

    # use random windows
    analysis_window = np.random.randn(segment_size)
    synthesis_window = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    # generate consistent input data
    x = np.random.randn(2, segment_size)

    # use factory function to create segmenters
    segA = make_segmenter(backendA, window)
    segB = make_segmenter(backendB, window)

    # convert input to correct backend format
    xA = as_backend(x, backendA)
    xB = as_backend(x, backendB)

    # segmentation and reconstruction
    sA, sB = segA.segment(xA), segB.segment(xB)
    rA, rB = segA.unsegment(sA), segB.unsegment(sB)

    sA, rA = as_numpy(sA, backendA), as_numpy(rA, backendA)
    sB, rB = as_numpy(sB, backendB), as_numpy(rB, backendB)

    # assertions with clearer error messages
    assert np.allclose(sA, sB, atol=1e-5)
    assert np.allclose(rA, rB, atol=1e-5)
