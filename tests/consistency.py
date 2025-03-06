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

import scipy
import os
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

        if result.returncode:
            print(result.stdout)
            print(result.stderr)

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
            atol=1e-3,
        )  # pyright: ignore
        iA, iB = traA.inverse(*tA), traB.inverse(*tB)
        assert np.allclose(iA, iB, atol=1e-4)
    else:
        assert np.allclose(tA, tB, atol=1e-4)  # pyright: ignore
        iA, iB = traA.inverse(tA), traB.inverse(tB)
        assert np.allclose(iA, iB, atol=1e-4)


# we have a special case for octave
@pytest.mark.parametrize("batched", [True, False])
@settings(max_examples=10, phases=[Phase.generate], deadline=None)
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    num_hops=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_consistency_octave(
    batched: bool,
    segment_size: int,
    hop_size: int,
    num_hops: int,
    seed: int,
) -> None:
    backendA: BackendType = "numpy"

    np.random.seed(seed)

    analysis_window = np.random.randn(segment_size)
    synthesis_window = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x = np.random.randn(2, segment_size + num_hops * hop_size)
    else:
        x = np.random.randn(segment_size + num_hops * hop_size)

    segA = Segmenter(backendA, window)

    xA = as_backend(x, backendA)
    sA = segA.segment(xA)
    rA = segA.unsegment(sA)

    sA_np = as_numpy(sA, backendA)
    rA_np = as_numpy(rA, backendA)

    # Define temporary file path
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_file:
        tmp_mat_file = tmp_file.name  # Get the file path
    scipy.io.savemat(
        tmp_mat_file,
        {
            "ref_s": sA_np,
            "ref_r": rA_np,
            "window_analysis": analysis_window,
            "window_synthesis": synthesis_window,
            "inp": x,
        },
    )

    # Generate Octave script
    octave_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/libsegmenter/")
    )
    assert os.path.exists(octave_path)

    octave_code = f"""
    addpath(genpath('{octave_path}'));

    load('{tmp_mat_file}');  % Load ref_s and ref_r

    # convert to column vector
    window_analysis = window_analysis(:)
    window_synthesis = window_synthesis(:)

    window = Window({hop_size}, window_analysis, window_synthesis);
    seg = SegmenterOctave(window);

    seg_s = seg.segment(inp);
    seg_r = seg.unsegment(seg_s);

    disp(size(seg_s))
    disp(size(seg_r))
    disp(size(ref_s))
    disp(size(ref_r))

    if (all(abs(seg_s - ref_s) < 1e-5))
        exit(0);
    else
        exit(1);
    end

    if (all(abs(seg_r - ref_r) < 1e-5))
        exit(0);
    else
        exit(1);
    end
    """

    print(octave_code)

    assert not run_octave(octave_code)


# we have a special case for octave
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("transform", TRANSFORMS)
@settings(max_examples=10, phases=[Phase.generate], deadline=None)
@given(
    segment_size=st.integers(min_value=32, max_value=64),
    hop_size=st.integers(min_value=1, max_value=32),
    num_hops=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_segmenter_roundtrip_consistency_octave(
    batched: bool,
    transform: TransformType,
    segment_size: int,
    hop_size: int,
    num_hops: int,
    seed: int,
) -> None:
    backendA: BackendType = "numpy"

    np.random.seed(seed)

    analysis_window = np.random.randn(segment_size)
    synthesis_window = np.random.randn(segment_size)
    window = Window(hop_size, analysis_window, synthesis_window)

    if batched:
        x = np.random.randn(2, segment_size + num_hops * hop_size)
    else:
        x = np.random.randn(segment_size + num_hops * hop_size)

    segA = Segmenter(backendA, window)
    traA = TransformSelector(transform=transform, backend=backendA)

    if transform == "magnitude_phase":
        xA = as_backend(x, backendA)

        sA = segA.segment(xA)
        sA_np = as_numpy(sA, backendA)

        tA_mag, tA_pha = traA.forward(sA)
        tA_mag_np = as_numpy(tA_mag, backendA)
        tA_pha_np = as_numpy(tA_pha, backendA)

        iA = traA.inverse(tA_mag, tA_pha)
        iA_np = as_numpy(iA, backendA)

        # Define temporary file path
        tmp_mat_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "tmp_refs.mat")
        )
        scipy.io.savemat(
            tmp_mat_file,
            {
                "sA": sA_np,
                "tA_mag": tA_mag_np,
                "tA_pha": tA_pha_np,
                "iA": iA_np,
            },
        )
    else:
        xA = as_backend(x, backendA)

        sA = segA.segment(xA)
        sA_np = as_numpy(sA, backendA)

        tA = traA.forward(sA)
        tA_np = as_numpy(tA, backendA)

        iA = traA.inverse(tA)
        iA_np = as_numpy(iA, backendA)

        # Define temporary file path
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_file:
            tmp_mat_file = tmp_file.name  # Get the file path
        scipy.io.savemat(
            tmp_mat_file,
            {
                "sA": sA_np,
                "tA": tA_np,
                "iA": iA_np,
            },
        )


    # Generate Octave script
    octave_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/libsegmenter/")
    )
    assert os.path.exists(octave_path)

    octave_code = f"""
    addpath(genpath('{octave_path}'));

    load('{tmp_mat_file}');  % Load sA and ref_r

    if strcmp('{transform}', 'magnitude_phase')
        tra = MagnitudePhaseOctave();
    end

    if strcmp('{transform}', 'spectrogram')
        tra = SpectrogramOctave();
    end

    if strcmp('{transform}', 'magnitude_phase')
        [tB_mag, tB_pha] = tra.forward(sA);
        iB = tra.inverse(tA_mag, tA_pha);
    else
        tB = tra.forward(sA);
        iB = tra.inverse(tB);
    end

    if strcmp('{transform}', 'magnitude_phase')
        if (all(abs(tA_mag - tB_mag) < 1e-5))
            exit(0);
        else
            exit(1);
        end

        if (all(abs(tA_pha - tB_pha) < 1e-5))
            exit(0);
        else
            exit(1);
        end
    else
        if (all(abs(tA - tB) < 1e-5))
            exit(0);
        else
            exit(1);
        end
    end

    if (all(abs(iA - iB) < 1e-5))
        exit(0);
    else
        exit(1);
    end
    """

    print(octave_code)

    assert not run_octave(octave_code)


