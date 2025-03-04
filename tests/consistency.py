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
import subprocess
import tempfile
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
TransformType = Literal["magnitude_phase", "spectrogram"]

BACKENDS: list[BackendType] = ["numpy", "torch", "tensorflow"]
TRANSFORMS: list[TransformType] = ["magnitude_phase", "spectrogram"]


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


def run_octave(code: str) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".m", mode="w", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ["octave", "--silent", "--eval", f"run('{tmp_file_path}')"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 1
    finally:
        subprocess.run(["rm", "-f", tmp_file_path])  # Ensure file is deleted


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("backendA, backendB", itertools.permutations(BACKENDS, 2))
@settings(max_examples=10, phases=[Phase.generate], deadline=None)
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    num_hops=st.integers(min_value=0, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_consistency(
    batched: bool,
    backendA: BackendType,
    backendB: BackendType,
    segment_size: int,
    hop_size: int,
    num_hops: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    analysis_window: NDArray[np.float64] = np.random.randn(segment_size)
    synthesis_window: NDArray[np.float64] = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x: NDArray[np.float64] = np.random.randn(2, segment_size + num_hops * hop_size)
    else:
        x: NDArray[np.float64] = np.random.randn(segment_size + num_hops * hop_size)

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
@settings(max_examples=10, phases=[Phase.generate])
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    num_hops=st.integers(min_value=0, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_transform_roundtrip_consistency(
    batched: bool,
    transform: TransformType,
    backendA: BackendType,
    backendB: BackendType,
    segment_size: int,
    hop_size: int,
    num_hops: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    analysis_window: NDArray[np.float64] = np.random.randn(segment_size)
    synthesis_window: NDArray[np.float64] = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x: NDArray[np.float64] = np.random.randn(2, segment_size + num_hops * hop_size)
    else:
        x: NDArray[np.float64] = np.random.randn(segment_size + num_hops * hop_size)

    segA = Segmenter(backendA, window)
    segB = Segmenter(backendB, window)

    traA = TransformSelector(transform=transform, backend=backendA)
    traB = TransformSelector(transform=transform, backend=backendB)

    xA = as_backend(x, backendA)
    xB = as_backend(x, backendB)

    sA, sB = segA.segment(xA), segB.segment(xB)

    tA = traA.forward(sA)
    tB = traB.forward(sB)

    if transform == "magnitude_phase":
        assert len(tA) == 2  # pyright: ignore
        assert len(tB) == 2  # pyright: ignore
        assert np.allclose(tA[0], tB[0], atol=1e-4)  # pyright: ignore
        assert np.allclose(
            np.cos(as_numpy(tA[1], backend=backendA)),
            np.cos(as_numpy(tB[1], backend=backendB)),
            atol=1e-4,
        )  # pyright: ignore
        rA, rB = traA.inverse(*tA), traB.inverse(*tB)
        assert np.allclose(rA, rB, atol=1e-4)
    else:
        print(
            np.max(
                np.abs(as_numpy(tA, backend=backendA) - as_numpy(tB, backend=backendB))
            )
        )
        rA, rB = traA.inverse(tA), traB.inverse(tB)
        print(
            np.max(
                np.abs(as_numpy(rA, backend=backendA) - as_numpy(rB, backend=backendB))
            )
        )


# we have a special case for octave
@pytest.mark.parametrize("batched", [True, False])
@settings(max_examples=1, phases=[Phase.generate], deadline=None)
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_consistency_octave(
    batched: bool,
    segment_size: int,
    hop_size: int,
    seed: int,
) -> None:
    backendA: BackendType = "numpy"
    backendB: str = "octave"

    _ = backendA
    _ = backendB
    _ = batched
    _ = segment_size
    _ = hop_size
    _ = seed
