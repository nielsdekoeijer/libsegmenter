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

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "Cola.hpp"
#include "FftInterface.hpp"
#include "Helper.hpp"
#include "Mode.hpp"
#include "Parameters.hpp"
#include "Span.hpp"

namespace segmenter {

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
    std::unique_ptr<T[]> m_scratch;
    std::unique_ptr<FWRfft<T>> m_FWRfft;
    std::unique_ptr<BWRfft<T>> m_BWRfft;

    void applyEdgeCorrection()
    {
        for (std::size_t i = 1; i < std::size_t(m_frameSize / m_hopSize) + 1;
             i++) {
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

    void applyWOLANormalization()
    {
        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_window[i] = sqrt(m_window[i]);
        }

        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_preWindow[i] = sqrt(m_preWindow[i]);
        }

        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_postWindow[i] = sqrt(m_postWindow[i]);
        }
    }

    void applyWindowNormalization(const T factor)
    {
        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_window[i] = m_window[i] / factor;
        }
        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_preWindow[i] = m_preWindow[i] / factor;
        }
        for (std::size_t i = 0; i < m_frameSize; i++) {
            m_postWindow[i] = m_postWindow[i] / factor;
        }
    }

  public:
    SegmenterParameters<T> m_parameters;
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
          m_postWindow(std::move(std::make_unique<T[]>(windowSize))),
          m_scratch(std::move(std::make_unique<T[]>(windowSize))),
          m_parameters(std::move(std::make_unique<T[]>(10)), frameSize, hopSize,
                       mode, edgeCorrection, normalizeWindow)
    {
        // Validate inputs Allocates in constructor
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

        // Cola check
        auto cola = checkCola<T>(window, frameSize, m_hopSize, 1e-3);
        if (!cola.isCola) {
            throw std::runtime_error("specified window is not COLA compliant "
                                     "for the given hop size, yielded: " + std::to_string(cola.epsilon));
        }

        // Clone window
        for (std::size_t i = 0; i < windowSize; i++) {
            m_window[i] = window[i];
        }
        for (std::size_t i = 0; i < windowSize; i++) {
            m_preWindow[i] = window[i];
        }
        for (std::size_t i = 0; i < windowSize; i++) {
            m_postWindow[i] = window[i];
        }

        // Edge correction
        if (m_edgeCorrection) {
            applyEdgeCorrection();
        }

        // Window normalization
        if (m_normalizeWindow) {
            applyWindowNormalization(cola.normalizationValue);
        }

        // WOLA normalization
        if (m_mode == SegmenterMode::WOLA) {
            applyWOLANormalization();
        }

        // Setup fft for spectrogram
        m_FWRfft = std::make_unique<FWRfft<T>>(m_frameSize);
        m_BWRfft = std::make_unique<BWRfft<T>>(m_frameSize);
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void segment(const T* itensor, const std::array<std::size_t, 2>& ishape,
                 T* otensor, const std::array<std::size_t, 3>& oshape)
    {
        validateSegmentationShape<T>(m_parameters, ishape, oshape);
        auto iview = Span<T, 2>(itensor, ishape);
        auto oview = MutableSpan<T, 3>(otensor, oshape);
        std::size_t batchCount = oshape[0];
        std::size_t frameCount = oshape[1];

        switch (m_mode) {
        case (SegmenterMode::WOLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                std::size_t j = 0;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    oview(i, j, k) =
                        m_preWindow[k] * iview(i, j * m_hopSize + k);
                }
                for (j = 1; j < frameCount - 1; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        oview(i, j, k) =
                            m_window[k] * iview(i, j * m_hopSize + k);
                    }
                }
                j = frameCount - 1;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    oview(i, j, k) =
                        m_postWindow[k] * iview(i, j * m_hopSize + k);
                }
            }
            break;
        case (SegmenterMode::OLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                for (std::size_t j = 0; j < frameCount; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        oview(i, j, k) = iview(i, j * m_hopSize + k);
                    }
                }
            }
            break;
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void spectrogram(const T* itensor, const std::array<std::size_t, 2>& ishape,
                     std::complex<T>* otensor,
                     const std::array<std::size_t, 3>& oshape)
    {
        validateSpectrogramShape<T>(m_parameters, ishape, oshape);
        auto iview = Span<T, 2>(itensor, ishape);
        auto oview = MutableSpan<std::complex<T>, 3>(otensor, oshape);
        std::size_t batchCount = oshape[0];
        std::size_t frameCount = oshape[1];
        std::complex<T>* out;
        switch (m_mode) {
        case (SegmenterMode::WOLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                std::size_t j = 0;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    m_scratch[k] = m_preWindow[k] * iview(i, j * m_hopSize + k);
                }
                out = &oview(i, j, 0);

                m_FWRfft->process(m_scratch.get(), m_frameSize, out,
                                  m_halfSpectrumSize);
                for (j = 1; j < frameCount - 1; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        m_scratch[k] = m_window[k] * iview(i, j * m_hopSize + k);
                    }
                    out = &oview(i, j, 0);
                    m_FWRfft->process(m_scratch.get(), m_frameSize, out,
                                      m_halfSpectrumSize);
                }

                j = frameCount - 1;
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    m_scratch[k] = m_postWindow[k] * iview(i, j * m_hopSize + k);
                }
                out = &oview(i, j, 0);
                m_FWRfft->process(m_scratch.get(), m_frameSize, out,
                                  m_halfSpectrumSize);
            }
            break;
        case (SegmenterMode::OLA):
            for (std::size_t i = 0; i < batchCount; i++) {
                for (std::size_t j = 0; j < frameCount; j++) {
                    for (std::size_t k = 0; k < m_frameSize; k++) {
                        m_scratch[k] = iview(i, j * m_hopSize + k);
                    }
                    out = &oview(i, j, 0);
                    m_FWRfft->process(m_scratch.get(), m_frameSize, out,
                                      m_halfSpectrumSize);
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
        validateSegmentationShape<T>(m_parameters, oshape, ishape);
        auto iview = Span<T, 3>(itensor, ishape);
        auto oview = MutableSpan<T, 2>(otensor, oshape);
        std::size_t batchCount = ishape[0];
        std::size_t frameCount = ishape[1];

        for (std::size_t i = 0; i < batchCount; i++) {
            std::size_t j = 0;
            for (std::size_t k = 0; k < m_frameSize; k++) {
                oview(i, j * m_hopSize + k) += m_preWindow[k] * iview(i, j, k);
            }

            for (j = 1; j < frameCount - 1; j++) {
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    oview(i, j * m_hopSize + k) += m_window[k] * iview(i, j, k);
                }
            }

            j = frameCount - 1;
            for (std::size_t k = 0; k < m_frameSize; k++) {
                oview(i, j * m_hopSize + k) += m_postWindow[k] * iview(i, j, k);
            }
        }
    }

    // operates on contiguous data in right_layout / c-style row major
    // NOTE: we don't like this design. Unfortunately, mdspan requires C++23.
    void unspectrogram(const std::complex<T>* itensor,
                       const std::array<std::size_t, 3>& ishape, T* otensor,
                       const std::array<std::size_t, 2>& oshape)
    {
        validateSpectrogramShape<T>(m_parameters, oshape, ishape);
        auto iview = Span<std::complex<T>, 3>(itensor, ishape);
        auto oview = MutableSpan<T, 2>(otensor, oshape);
        std::size_t batchCount = ishape[0];
        std::size_t frameCount = ishape[1];

        const std::complex<T>* in;
        for (std::size_t i = 0; i < batchCount; i++) {
            std::size_t j = 0;
            in = &iview(i, j, 0);

            m_BWRfft->process(in, m_halfSpectrumSize, m_scratch.get(),
                              m_frameSize);
            for (std::size_t k = 0; k < m_frameSize; k++) {
                oview(i, j * m_hopSize + k) += m_preWindow[k] * m_scratch[k];
            }

            for (j = 1; j < frameCount - 1; j++) {
                in = &iview(i, j, 0);
                m_BWRfft->process(in, m_halfSpectrumSize, m_scratch.get(),
                                  m_frameSize);
                for (std::size_t k = 0; k < m_frameSize; k++) {
                    oview(i, j * m_hopSize + k) += m_window[k] * m_scratch[k];
                }
            }

            j = frameCount - 1;
            in = &iview(i, j, 0);
            m_BWRfft->process(in, m_halfSpectrumSize, m_scratch.get(),
                              m_frameSize);
            for (std::size_t k = 0; k < m_frameSize; k++) {
                oview(i, j * m_hopSize + k) += m_postWindow[k] * m_scratch[k];
            }
        }
    }
};
} // namespace segmenter
