/*
 * MIT License
 *
 * Copyright (c) 2023 [Your Name]
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ==============================================================================
 *
 * Segmenter.hpp
 * Single-header library for audio signal segmentation.
 *
 * ==============================================================================
 */

#pragma once

#include "fft.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>

template<typename T>
T PI = std::acos(T{-1});

/*
 * ==============================================================================
 *
 * Window functions
 *
 * ==============================================================================
 */

namespace segmenter {
template <typename T>
void populateBartlettWindow(T* vec, const std::size_t windowSize)
{
    const T M = windowSize + 1.0;
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = 1.0 - std::abs(-1.0 * (M - 1) / 2.0 + i) * 2.0 / (M - 1.0);
    }
}

template <typename T>
void populateBlackmanWindow(T* vec, const std::size_t windowSize)
{
    const T M = T(windowSize + 1);
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] =
            7938.0 / 18608.0 -
            9240.0 / 18608.0 * cos(2.0 * PI<T> * T(i) / T(M - 1)) +
            1430.0 / 18608.0 * cos(4.0 * PI<T> * T(i) / T(M - 1));
    }
}

template <typename T>
void populateHammingWindow(T* vec, const std::size_t windowSize)
{
    const T M = T(windowSize);
    const T alpha = 25.0 / 46.0;
    const T beta = (1.0 - alpha) / 2.0;
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = alpha - 2.0 * beta * cos(2.0 * PI<T> * T(i) / M);
    }
}

template <typename T>
void populateHannWindow(T* vec, const std::size_t windowSize)
{
    const T M = T(windowSize);
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = 0.5 * (1.0 - cos(2.0 * PI<T> * T(i) / M));
    }
}

template <typename T>
void populateRectangularWindow(T* vec, const std::size_t windowSize)
{
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = T(1);
    }
}

/*
 * ==============================================================================
 *
 * Constant OverLap Add (COLA) condition checking
 *
 * ==============================================================================
 */

template <typename T>
struct COLAResult {
    bool isCola;
    T normalizationValue;
};

template <typename T>
COLAResult<T> checkCola(const T* window, const std::size_t windowSize,
                        const std::size_t hopSize, const T eps = 1e-5)
{
    // NOTE: Allocates
    const T frameRate = 1.0 / T(hopSize);

    T factor = 0.0;
    for (std::size_t i = 0; i < windowSize; i++) {
        factor += window[i];
    }
    factor /= T(hopSize);

    const std::size_t N = 6 * windowSize;
    auto sp = std::make_unique<std::complex<T>[]>(N);
    for (std::size_t i = 0; i < N; i++) {
        sp[i] = factor;
    }
    T ubound = sp[0].real();
    T lbound = sp[0].real();

    auto csin = std::make_unique<std::complex<T>[]>(N);
    for (std::size_t k = 1; k < hopSize; k++) {
        const T f = frameRate * T(k);
        for (std::size_t n = 0; n < N; n++) {
            csin[n] =
                std::exp(std::complex<T>(0.0, 2.0 * PI<T> * f * n));
        }

        std::complex<T> Wf = 0.0;
        for (std::size_t n = 0; n < windowSize; n++) {
            Wf += window[n] * std::conj(csin[n]);
        }

        for (std::size_t n = 0; n < N; n++) {
            sp[n] += (Wf * csin[n]) / T(hopSize);
        }

        const T Wfb = abs(Wf);
        ubound += Wfb / T(hopSize);
        lbound -= Wfb / T(hopSize);
    }

    COLAResult<T> result;
    result.isCola = (ubound - lbound) < eps;
    result.normalizationValue = (ubound + lbound) / 2.0;

    return result;
}

/*
 * ==============================================================================
 *
 * Segmentation Class
 *
 * ==============================================================================
 */

enum class SegmenterMode {
    WOLA,
    OLA,
};

template <typename T>
class Segmenter {
    const std::size_t m_hopSize;
    const std::size_t m_halfSpectrumSize;
    const SegmenterMode m_mode;
    const bool m_edgeCorrection;
    const bool m_normalizeWindow;
    const std::size_t m_frameSize;
    std::unique_ptr<T[]> m_window;
    std::unique_ptr<T[]> m_preWindow;
    std::unique_ptr<T[]> m_postWindow;
    std::unique_ptr<std::complex<T>[]> m_fftFWTwiddleFactors;
    std::unique_ptr<std::complex<T>[]> m_fftBWTwiddleFactors;
    std::unique_ptr<std::complex<T>[]> m_scratch0;
    std::unique_ptr<std::complex<T>[]> m_scratch1;
    std::unique_ptr<T[]> m_intermediate;

  public:
    Segmenter(std::size_t frameSize, std::size_t hopSize, const T* window,
              const std::size_t windowSize,
              const SegmenterMode mode = SegmenterMode::WOLA,
              const bool edgeCorrection = true,
              const bool normalizeWindow = true)
        : m_frameSize(frameSize), m_hopSize(hopSize),
          m_halfSpectrumSize(frameSize / 2 + 1), m_mode(mode),
          m_edgeCorrection(edgeCorrection), m_normalizeWindow(normalizeWindow),
          m_window(std::move(std::make_unique<T[]>(windowSize))),
          m_preWindow(std::move(std::make_unique<T[]>(windowSize))),
          m_postWindow(std::move(std::make_unique<T[]>(windowSize)))
    {
        // NOTE: Allocates in constructor
        if (m_frameSize % 2 != 0) {
            throw std::runtime_error("only even frameSize is supported");
        }

        if (m_hopSize > m_frameSize) {
            throw std::runtime_error(
                "hopSize cannot be larger than frame_size");
        }

        if (windowSize != frameSize) {
            throw std::runtime_error(
                "specified window must have the same size as frame_size");
        }

        for (std::size_t i = 0; i < windowSize; i++) {
            if (window[i] < 0.0) {
                throw std::runtime_error(
                    "specified window contains negative values");
            }
        }

        auto cola = checkCola<T>(window, frameSize, m_hopSize, 1e-5);
        if (!cola.isCola) {
            throw std::runtime_error("specified window is not COLA compliant "
                                     "for the given hop size");
        }

        for (std::size_t i = 0; i < windowSize; i++) {
            m_window[i] = window[i];
        }

        for (std::size_t i = 0; i < windowSize; i++) {
            m_preWindow[i] = m_window[i];
        }
        for (std::size_t i = 0; i < windowSize; i++) {
            m_postWindow[i] = m_window[i];
        }

        // edge correction
        if (m_edgeCorrection) {
            for (std::size_t i = 1;
                 i < std::size_t(m_frameSize / m_hopSize) + 1; i++) {
                std::size_t idx1Start = i * m_hopSize;
                std::size_t idx1End = m_frameSize;
                std::size_t idx2Start = 0;
                std::size_t idx2End = m_frameSize - idx1Start;
                std::size_t range = idx2End - idx2Start;

                for (std::size_t n = 0; n < range; n++) {
                    m_preWindow[idx2Start + n] += m_window[idx1Start + n];
                }

                for (std::size_t n = 0; n < range; n++) {
                    m_postWindow[idx1Start + n] += m_window[idx2Start + n];
                }
            }
        }

        // Window normalization
        if (m_normalizeWindow) {
            for (std::size_t i = 0; i < windowSize; i++) {
                m_window[i] = m_window[i] / cola.normalizationValue;
            }
            for (std::size_t i = 0; i < windowSize; i++) {
                m_preWindow[i] = m_preWindow[i] / cola.normalizationValue;
            }
            for (std::size_t i = 0; i < windowSize; i++) {
                m_postWindow[i] = m_postWindow[i] / cola.normalizationValue;
            }
        }

        // WOLA normalization
        if (m_mode == SegmenterMode::WOLA) {
            for (std::size_t i = 0; i < windowSize; i++) {
                m_window[i] = sqrt(m_window[i]);
            }

            for (std::size_t i = 0; i < windowSize; i++) {
                m_preWindow[i] = sqrt(m_preWindow[i]);
            }

            for (std::size_t i = 0; i < windowSize; i++) {
                m_postWindow[i] = sqrt(m_postWindow[i]);
            }
        }

        // Setup fft for spectrogram
        fft::FFTSTATUS err;
        m_fftFWTwiddleFactors =
            std::make_unique<std::complex<T>[]>(m_frameSize);
        err = fft::populateRfftTwiddleFactorsForward<T>(
            m_frameSize, m_fftFWTwiddleFactors.get(), m_frameSize);
        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error occured in the creation of the fft "
                                     "forward twiddle factors");
        }

        m_fftBWTwiddleFactors =
            std::make_unique<std::complex<T>[]>(m_frameSize);
        err = fft::populateRfftTwiddleFactorsBackward<T>(
            m_frameSize, m_fftBWTwiddleFactors.get(), m_frameSize);
        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error occured in the creation of the fft "
                                     "backward twiddle factors");
        }

        m_scratch0 = std::make_unique<std::complex<T>[]>(m_halfSpectrumSize);
        m_scratch1 = std::make_unique<std::complex<T>[]>(m_halfSpectrumSize);
        m_intermediate = std::make_unique<T[]>(m_frameSize);
    }

    void getSegmentationShapeFromUnsegmented(
        const std::array<std::size_t, 2>& unsegmentedShape,
        std::array<std::size_t, 3>& segmentedShape)
    {
        if (unsegmentedShape[1] % m_hopSize != 0) {
            throw std::runtime_error("specified input shape is not a modulus "
                                     "of the specified hop size");
        }
        segmentedShape[0] = unsegmentedShape[0];
        segmentedShape[1] =
            (unsegmentedShape[1] / m_hopSize) - m_frameSize / m_hopSize + 1;
        segmentedShape[2] = m_frameSize;
    }

    void getSegmentationShapeFromSegmented(
        std::array<std::size_t, 2>& unsegmentedShape,
        const std::array<std::size_t, 3>& segmentedShape)
    {
        unsegmentedShape[0] = segmentedShape[0];
        unsegmentedShape[1] = (segmentedShape[1] - 1) * m_hopSize + m_frameSize;
        if (unsegmentedShape[1] % m_hopSize != 0) {
            throw std::runtime_error("specified input shape is not a modulus "
                                     "of the specified hop size");
        }
    }

    void validateSegmentationShape(
        const std::array<std::size_t, 2>& unsegmentedShape,
        const std::array<std::size_t, 3>& segmentedShape)
    {
        std::array<std::size_t, 3> expected_segmentedShape{};
        getSegmentationShapeFromUnsegmented(unsegmentedShape,
                                            expected_segmentedShape);
        if (segmentedShape[0] != expected_segmentedShape[0]) {
            throw std::runtime_error("input and output batch sizes different "
                                     "for given input shapes.");
        }
        if (segmentedShape[1] != expected_segmentedShape[1]) {
            throw std::runtime_error(
                "output frame count invalid for given input shape");
        }
        if (segmentedShape[2] != expected_segmentedShape[2]) {
            throw std::runtime_error(
                "output frame size invalid for configured frame size");
        }
    }

    void getSpectrogramShapeFromUnsegmented(
        const std::array<std::size_t, 2>& unsegmentedShape,
        std::array<std::size_t, 3>& segmentedShape)
    {
        if (unsegmentedShape[1] % m_hopSize != 0) {
            throw std::runtime_error("specified input shape is not a modulus "
                                     "of the specified hop size");
        }
        segmentedShape[0] = unsegmentedShape[0];
        segmentedShape[1] =
            (unsegmentedShape[1] / m_hopSize) - m_frameSize / m_hopSize + 1;
        segmentedShape[2] = m_halfSpectrumSize;
    }

    void getSpectrogramShapeFromSegmented(
        std::array<std::size_t, 2>& unsegmentedShape,
        const std::array<std::size_t, 3>& segmentedShape)
    {
        unsegmentedShape[0] = segmentedShape[0];
        unsegmentedShape[1] = (segmentedShape[1] - 1) * m_hopSize + m_frameSize;
        if (unsegmentedShape[1] % m_hopSize != 0) {
            throw std::runtime_error("specified input shape is not a modulus "
                                     "of the specified hop size");
        }
    }

    void
    validateSpectrogramShape(const std::array<std::size_t, 2>& unsegmentedShape,
                             const std::array<std::size_t, 3>& segmentedShape)
    {
        if (!(m_frameSize && !(m_frameSize & (m_frameSize - 1)))) {
            throw std::runtime_error(
                "given segmenter is configured to a non-radix 2 frame size, "
                "spectrogram is thus not supported");
        }
        std::array<std::size_t, 3> expected_segmentedShape{};
        getSpectrogramShapeFromUnsegmented(unsegmentedShape,
                                           expected_segmentedShape);
        if (segmentedShape[0] != expected_segmentedShape[0]) {
            throw std::runtime_error("input and output batch sizes different "
                                     "for given input shapes.");
        }
        if (segmentedShape[1] != expected_segmentedShape[1]) {
            throw std::runtime_error(
                "output frame count invalid for given input shape");
        }
        if (segmentedShape[2] != expected_segmentedShape[2]) {
            throw std::runtime_error(
                "output frame size invalid for configured frame size");
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void segment(const T* itensor, const std::array<std::size_t, 2>& ishape,
                 T* otensor, const std::array<std::size_t, 3>& oshape)
    {
        validateSegmentationShape(ishape, oshape);
        std::size_t batchCount = oshape[0];
        std::size_t frameCount = oshape[1];

        switch (m_mode) {
        case (SegmenterMode::WOLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                std::size_t j = 0;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    otensor[i * oshape[1] * oshape[2] + j * oshape[2] + k] =
                        m_preWindow[k] *
                        itensor[i * ishape[1] + j * m_hopSize + k];
                }
                for (j = 1; j < frameCount - 1; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        otensor[i * oshape[1] * oshape[2] + j * oshape[2] + k] =
                            m_window[k] *
                            itensor[i * ishape[1] + j * m_hopSize + k];
                    }
                }
                j = frameCount - 1;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    otensor[i * oshape[1] * oshape[2] + j * oshape[2] + k] =
                        m_postWindow[k] *
                        itensor[i * ishape[1] + j * m_hopSize + k];
                }
            }
            break;
        case (SegmenterMode::OLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                for (std::size_t j = 0; j < frameCount; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        otensor[i * oshape[1] * oshape[2] + j * oshape[2] + k] =
                            itensor[i * ishape[1] + j * m_hopSize + k];
                    }
                }
            }
            break;
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void unsegment(const T* itensor, const std::array<std::size_t, 3>& ishape,
                   T* otensor, const std::array<std::size_t, 2>& oshape)
    {
        validateSegmentationShape(oshape, ishape);
        std::size_t batchCount = ishape[0];
        std::size_t frameCount = ishape[1];

        for (std::size_t i = 0; i < batchCount; i++) {
            std::size_t j = 0;
            for (std::size_t k = 0; k < m_frameSize; k++) {
                otensor[i * oshape[1] + j * m_hopSize + k] +=
                    m_preWindow[k] *
                    itensor[i * ishape[1] * ishape[2] + j * ishape[2] + k];
            }

            for (j = 1; j < frameCount - 1; j++) {
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    otensor[i * oshape[1] + j * m_hopSize + k] +=
                        m_window[k] *
                        itensor[i * ishape[1] * ishape[2] + j * ishape[2] + k];
                }
            }

            j = frameCount - 1;
            for (std::size_t k = 0; k < m_frameSize; k++) {
                otensor[i * oshape[1] + j * m_hopSize + k] +=
                    m_postWindow[k] *
                    itensor[i * ishape[1] * ishape[2] + j * ishape[2] + k];
            }
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void spectrogram(const T* itensor, const std::array<std::size_t, 2>& ishape,
                     std::complex<T>* otensor,
                     const std::array<std::size_t, 3>& oshape)
    {
        validateSpectrogramShape(ishape, oshape);
        std::size_t batchCount = oshape[0];
        std::size_t frameCount = oshape[1];
        std::complex<T>* out;
        fft::FFTSTATUS err;
        switch (m_mode) {
        case (SegmenterMode::WOLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                std::size_t j = 0;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    m_intermediate[k] =
                        m_preWindow[k] *
                        itensor[i * ishape[1] + j * m_hopSize + k];
                }
                out = &otensor[i * oshape[1] * oshape[2] + j * oshape[2]];

                err = fft::performRfftForward<T>(
                    m_frameSize, m_fftFWTwiddleFactors.get(), m_frameSize,
                    m_intermediate.get(), m_frameSize, out, m_halfSpectrumSize,
                    m_scratch0.get(), m_halfSpectrumSize);

                if (err != fft::FFTSTATUS::OK) {
                    throw std::runtime_error("error in fft 1");
                }

                for (j = 1; j < frameCount - 1; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        m_intermediate[k] =
                            m_window[k] *
                            itensor[i * ishape[1] + j * m_hopSize + k];
                    }
                    out = &otensor[i * oshape[1] * oshape[2] + j * oshape[2]];
                    err = fft::performRfftForward<T>(
                        m_frameSize, m_fftFWTwiddleFactors.get(), m_frameSize,
                        m_intermediate.get(), m_frameSize, out,
                        m_halfSpectrumSize, m_scratch0.get(),
                        m_halfSpectrumSize);
                    if (err != fft::FFTSTATUS::OK) {
                        throw std::runtime_error("error in fft 2");
                    }
                }

                j = frameCount - 1;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    m_intermediate[k] =
                        m_postWindow[k] *
                        itensor[i * ishape[1] + j * m_hopSize + k];
                }
                out = &otensor[i * oshape[1] * oshape[2] + j * oshape[2]];
                err = fft::performRfftForward<T>(
                    m_frameSize, m_fftFWTwiddleFactors.get(), m_frameSize,
                    m_intermediate.get(), m_frameSize, out, m_halfSpectrumSize,
                    m_scratch0.get(), m_halfSpectrumSize);
                if (err != fft::FFTSTATUS::OK) {
                    throw std::runtime_error("error in fft 3");
                }
            }
            break;
        case (SegmenterMode::OLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                for (std::size_t j = 0; j < frameCount; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        m_intermediate[k] =
                            itensor[i * ishape[1] + j * m_hopSize + k];
                    }
                    if (j == 0) {
                        for (std::size_t k = 0; k < m_frameSize; k++) {
                            std::cout << m_intermediate[k] << std::endl;
                        }
                    }
                    out = &otensor[i * oshape[1] * oshape[2] + j * oshape[2]];
                    err = fft::performRfftForward<T>(
                        m_frameSize, m_fftFWTwiddleFactors.get(), m_frameSize,
                        m_intermediate.get(), m_frameSize, out,
                        m_halfSpectrumSize, m_scratch0.get(),
                        m_halfSpectrumSize);
                    if (j == 0) {
                        for (std::size_t k = 0; k < m_halfSpectrumSize; k++) {
                            std::cout << out[k] << std::endl;
                        }
                    }
                    if (err != fft::FFTSTATUS::OK) {
                        throw std::runtime_error("error in fft 4");
                    }
                }
            }
            break;
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void unspectrogram(const std::complex<T>* itensor,
                       const std::array<std::size_t, 3>& ishape, T* otensor,
                       const std::array<std::size_t, 2>& oshape)
    {
        validateSpectrogramShape(oshape, ishape);
        std::size_t batchCount = ishape[0];
        std::size_t frameCount = ishape[1];

        const std::complex<T>* in;
        for (std::size_t i = 0; i < batchCount; i++) {
            std::size_t j = 0;
            in = &itensor[i * ishape[1] * ishape[2] + j * ishape[2]];

            fft::performRfftBackward<T>(
                m_halfSpectrumSize, m_fftBWTwiddleFactors.get(),
                m_halfSpectrumSize, in, m_halfSpectrumSize,
                m_intermediate.get(), m_frameSize, m_scratch0.get(),
                m_scratch1.get(), m_halfSpectrumSize);
            for (std::size_t k = 0; k < m_frameSize; k++) {
                otensor[i * oshape[1] + j * m_hopSize + k] +=
                    m_preWindow[k] * m_intermediate[k];
            }

            for (j = 1; j < frameCount - 1; j++) {
                in = &itensor[i * ishape[1] * ishape[2] + j * ishape[2]];
                fft::performRfftBackward<T>(
                    m_halfSpectrumSize, m_fftBWTwiddleFactors.get(),
                    m_halfSpectrumSize, in, m_halfSpectrumSize,
                    m_intermediate.get(), m_frameSize, m_scratch0.get(),
                    m_scratch1.get(), m_halfSpectrumSize);
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    otensor[i * oshape[1] + j * m_hopSize + k] +=
                        m_window[k] * m_intermediate[k];
                }
            }

            j = frameCount - 1;
            in = &itensor[i * ishape[1] * ishape[2] + j * ishape[2]];
            fft::performRfftBackward<T>(
                m_halfSpectrumSize, m_fftBWTwiddleFactors.get(),
                m_halfSpectrumSize, in, m_halfSpectrumSize,
                m_intermediate.get(), m_frameSize, m_scratch0.get(),
                m_scratch1.get(), m_halfSpectrumSize);
            for (std::size_t k = 0; k < m_frameSize; k++) {
                otensor[i * oshape[1] + j * m_hopSize + k] +=
                    m_postWindow[k] * m_intermediate[k];
            }
        }
    }
};
} // namespace segmenter
