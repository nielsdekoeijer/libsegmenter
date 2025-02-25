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
import tensorflow as tf
import itertools
import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Literal

from libsegmenter.Segmenter import Segmenter
from libsegmenter.Window import Window
from libsegmenter.TransformSelector import TransformSelector

T = TypeVar("T", bound=np.generic)
BackendType = Literal["numpy", "torch", "tensorflow"]
TransformType = Literal["phase", "magnitude", "spectrogram"]

BACKENDS: list[BackendType] = ["numpy", "torch", "tensorflow"]
TRANSFORMS: list[TransformType] = ["phase", "magnitude", "spectrogram"]


def as_numpy(
    x: NDArray[T] | torch.Tensor | tf.Tensor, backend: BackendType
) -> NDArray[T]:
    if backend == "torch":
        return x.numpy() if isinstance(x, torch.Tensor) else np.array(x)  # pyright: ignore
    if backend == "tensorflow":
        return x.numpy() if isinstance(x, tf.Tensor) else np.array(x)  # pyright: ignore
    return x  # pyright: ignore


def as_backend(
    x: NDArray[T], backend: BackendType
) -> NDArray[T] | torch.Tensor | tf.Tensor:
    if backend == "torch":
        return torch.tensor(x, dtype=torch.float32)
    elif backend == "tensorflow":
        return tf.convert_to_tensor(x, dtype=tf.float32)  # pyright: ignore
    return x


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("backendA, backendB", itertools.permutations(BACKENDS, 2))
@settings(max_examples=100, phases=[Phase.generate])
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_consistency(
    batched: bool,
    backendA: BackendType,
    backendB: BackendType,
    segment_size: int,
    hop_size: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    analysis_window: NDArray[np.float64] = np.random.randn(segment_size)
    synthesis_window: NDArray[np.float64] = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x: NDArray[np.float64] = np.random.randn(2, segment_size)
    else:
        x: NDArray[np.float64] = np.random.randn(segment_size)

    segA = Segmenter(backendA, window)
    segB = Segmenter(backendB, window)

    xA = as_backend(x, backendA)
    xB = as_backend(x, backendB)

    sA, sB = segA.segment(xA), segB.segment(xB)
    rA, rB = segA.unsegment(sA), segB.unsegment(sB)

    sA, rA = as_numpy(sA, backendA), as_numpy(rA, backendA)
    sB, rB = as_numpy(sB, backendB), as_numpy(rB, backendB)

    assert np.allclose(sA, sB, atol=1e-5)
    assert np.allclose(rA, rB, atol=1e-5)


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("backendA, backendB", itertools.permutations(BACKENDS, 2))
@settings(max_examples=100, phases=[Phase.generate])
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_transform_forward_consistency(
    batched: bool,
    transform: TransformType,
    backendA: BackendType,
    backendB: BackendType,
    segment_size: int,
    hop_size: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    analysis_window: NDArray[np.float64] = np.random.randn(segment_size)
    synthesis_window: NDArray[np.float64] = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x: NDArray[np.float64] = np.random.randn(2, segment_size)
    else:
        x: NDArray[np.float64] = np.random.randn(segment_size)

    x: NDArray[np.float64] = np.random.randn(segment_size)

    segA = Segmenter(backendA, window)
    segB = Segmenter(backendB, window)

    traA = TransformSelector(transform=transform, backend=backendA)
    traB = TransformSelector(transform=transform, backend=backendB)

    xA = as_backend(x, backendA)
    xB = as_backend(x, backendB)

    sA, sB = segA.segment(xA), segB.segment(xB)
    tA, tB = traA.forward(sA), traB.forward(sB)

    assert np.allclose(tA, tB, atol=1e-5)


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("transform", ["spectrogram"])
@pytest.mark.parametrize("backendA, backendB", itertools.permutations(BACKENDS, 2))
@settings(max_examples=100, phases=[Phase.generate])
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_transform_roundtrip_consistency(
    batched: bool,
    transform: TransformType,
    backendA: BackendType,
    backendB: BackendType,
    segment_size: int,
    hop_size: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    analysis_window: NDArray[np.float64] = np.random.randn(segment_size)
    synthesis_window: NDArray[np.float64] = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x: NDArray[np.float64] = np.random.randn(2, segment_size)
    else:
        x: NDArray[np.float64] = np.random.randn(segment_size)

    x: NDArray[np.float64] = np.random.randn(segment_size)

    segA = Segmenter(backendA, window)
    segB = Segmenter(backendB, window)

    traA = TransformSelector(transform=transform, backend=backendA)
    traB = TransformSelector(transform=transform, backend=backendB)

    xA = as_backend(x, backendA)
    xB = as_backend(x, backendB)

    sA, sB = segA.segment(xA), segB.segment(xB)
    tA, tB = traA.forward(sA), traB.forward(sB)
    assert np.allclose(tA, tB, atol=1e-5)

    rA, rB = traA.inverse(tA), traB.inverse(tB)
    assert np.allclose(rA, rB, atol=1e-5)
