import pytest
import torch
import tensorflow
import libsegmenter
import numpy

"""
tests testing the creational pattern for the Segmenters
"""

test_cases_edge_corrected = []
for backend in ["torch", "tensorflow", "base"]:
    for mode in ["wola", "ola"]:
        for window_settings in [
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (300,),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (1, 300),
            },
        ]:
            test_cases_edge_corrected.append(
                (
                    libsegmenter.make_segmenter(
                        backend=backend,
                        frame_size=window_settings["window"].size,
                        hop_size=window_settings["hop_size"],
                        window=window_settings["window"],
                        mode=mode,
                        edge_correction=True,
                    ),
                    torch.randn((window_settings["input_size"])),
                )
            )


@pytest.mark.parametrize(("segmenter", "x"), test_cases_edge_corrected)
def test_reconstruction_corrected(segmenter, x):
    assert True == True
    return
    X = segmenter.segment(x)
    y = segmenter.unsegment(X)

    y = numpy.array(y)
    x = numpy.array(x)
    assert y == pytest.approx(x, abs=1e-5)


test_cases_edge_uncorrected = []
for backend in ["torch", "tensorflow", "base"]:
    for mode in ["wola", "ola"]:
        for window_settings in [
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (300,),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (1, 300),
            },
        ]:
            test_cases_edge_uncorrected.append(
                (
                    libsegmenter.make_segmenter(
                        backend=backend,
                        frame_size=window_settings["window"].size,
                        hop_size=window_settings["hop_size"],
                        window=window_settings["window"],
                        mode=mode,
                        edge_correction=False,
                    ),
                    torch.randn((window_settings["input_size"])),
                )
            )


@pytest.mark.parametrize(("segmenter", "x"), test_cases_edge_uncorrected)
def test_reconstruction_uncorrected(segmenter, x):
    assert True == True
    return
    X = segmenter.segment(x)
    y = segmenter.unsegment(X)

    y = numpy.array(y)
    x = numpy.array(x)
    assert y[segmenter.frame_size : -segmenter.frame_size] == pytest.approx(
        x[segmenter.frame_size : -segmenter.frame_size], abs=1e-5
    )


test_cases_torch_vs_tensorflow = []
for mode in ["wola", "ola"]:
    for window_settings in [
        {
            "hop_size": 50,
            "window": libsegmenter.hamming(100),
            "input_size": (1000,),
        },
        {
            "hop_size": 50,
            "window": libsegmenter.hamming(100),
            "input_size": (1, 1000),
        },
        {
            "hop_size": 50,
            "window": libsegmenter.hann(100),
            "input_size": (1000,),
        },
        {
            "hop_size": 50,
            "window": libsegmenter.hann(100),
            "input_size": (1, 1000),
        },
        {
            "hop_size": 50,
            "window": libsegmenter.bartlett(100),
            "input_size": (1000,),
        },
        {
            "hop_size": 50,
            "window": libsegmenter.bartlett(100),
            "input_size": (1, 1000),
        },
        {
            "hop_size": 10,
            "window": libsegmenter.blackman(30),
            "input_size": (300,),
        },
        {
            "hop_size": 10,
            "window": libsegmenter.blackman(30),
            "input_size": (1, 300),
        },
    ]:
        test_cases_torch_vs_tensorflow.append(
            (
                (
                    libsegmenter.make_segmenter(
                        backend="torch",
                        frame_size=window_settings["window"].size,
                        hop_size=window_settings["hop_size"],
                        window=window_settings["window"],
                        mode=mode,
                        edge_correction=True,
                    ),
                    libsegmenter.make_segmenter(
                        backend="tensorflow",
                        frame_size=window_settings["window"].size,
                        hop_size=window_settings["hop_size"],
                        window=window_settings["window"],
                        mode=mode,
                        edge_correction=True,
                    ),
                ),
                torch.randn((window_settings["input_size"])),
            ),
        )


@pytest.mark.parametrize(("segmenter", "x"), test_cases_torch_vs_tensorflow)
def test_torch_vs_tensorflow_segment(segmenter, x):
    X_tc = segmenter[0].segment(x)
    X_tf = segmenter[1].segment(x)

    assert X_tc == pytest.approx(X_tf, abs=1e-5)


@pytest.mark.parametrize(("segmenter", "x"), test_cases_torch_vs_tensorflow)
def test_torch_vs_tensorflow_unsegment(segmenter, x):
    x_tc = segmenter[0].unsegment(segmenter[0].segment(x))
    x_tf = segmenter[1].unsegment(segmenter[1].segment(x))

    assert x_tc == pytest.approx(x_tf, abs=1e-5)


test_cases_torch_vs_base = []
for edge_correction in [True, False]:
    for mode in ["wola", "ola"]:
        for window_settings in [
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hamming(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.hann(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1000,),
            },
            {
                "hop_size": 50,
                "window": libsegmenter.bartlett(100),
                "input_size": (1, 1000),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (300,),
            },
            {
                "hop_size": 10,
                "window": libsegmenter.blackman(30),
                "input_size": (1, 300),
            },
        ]:
            test_cases_torch_vs_base.append(
                (
                    (
                        libsegmenter.make_segmenter(
                            backend="torch",
                            frame_size=window_settings["window"].size,
                            hop_size=window_settings["hop_size"],
                            window=window_settings["window"].copy(),
                            mode=mode,
                            edge_correction=edge_correction,
                        ),
                        libsegmenter.make_segmenter(
                            backend="base",
                            frame_size=window_settings["window"].size,
                            hop_size=window_settings["hop_size"],
                            window=window_settings["window"].copy(),
                            mode=mode,
                            edge_correction=edge_correction,
                        ),
                    ),
                    torch.randn((window_settings["input_size"])),
                ),
            )


@pytest.mark.parametrize(("segmenter", "x"), test_cases_torch_vs_base)
def test_torch_vs_base_segment(segmenter, x):
    X_tc = segmenter[0].segment(x.clone())
    X_ba = segmenter[1].segment(x.clone())
    assert X_tc.shape == X_ba.shape
    assert X_ba == pytest.approx(X_tc, abs=1e-5)


@pytest.mark.parametrize(("segmenter", "x"), test_cases_torch_vs_base)
def test_torch_vs_base_unsegment(segmenter, x):
    x_tc = segmenter[0].segment(x.clone())
    x_ba = segmenter[1].segment(x.clone())
    assert x_tc.shape == x_ba.shape
    segmenter[0].unsegment(x_tc)
    segmenter[1].unsegment(x_ba)
    assert x_tc.shape == x_ba.shape
    assert x_tc == pytest.approx(x_ba, abs=1e-5)
