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

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>

#define SEGMENTER_M_PI 3.14159265358979323846

/*
 * ==============================================================================
 *
 * Window functions
 *
 * ==============================================================================
 */

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
            9240.0 / 18608.0 * cos(2.0 * SEGMENTER_M_PI * T(i) / T(M - 1)) +
            1430.0 / 18608.0 * cos(4.0 * SEGMENTER_M_PI * T(i) / T(M - 1));
    }
}

template <typename T>
void populateHammingWindow(T* vec, const std::size_t windowSize)
{
    const T M = T(windowSize);
    const T alpha = 25.0 / 46.0;
    const T beta = (1.0 - alpha) / 2.0;
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = alpha - 2.0 * beta * cos(2.0 * SEGMENTER_M_PI * T(i) / M);
    }
}

template <typename T>
void populateHannWindow(T* vec, const std::size_t windowSize)
{
    const T M = T(windowSize);
    for (std::size_t i = 0; i < windowSize; i++) {
        vec[i] = 0.5 * (1.0 - cos(2.0 * SEGMENTER_M_PI * T(i) / M));
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
                std::exp(std::complex<T>(0.0, 2.0 * SEGMENTER_M_PI * f * n));
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

    return COLAResult<T>{
        .isCola = (ubound - lbound) < eps,
        .normalizationValue = (ubound + lbound) / 2.0,
    };
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
    const std::size_t m_frameSize;
    const std::size_t m_hopSize;
    const SegmenterMode m_mode;
    const bool m_edgeCorrection;
    const bool m_normalizeWindow;
    std::unique_ptr<T> m_window;
    std::unique_ptr<T> m_preWindow;
    std::unique_ptr<T> m_postWindow;

  public:
    Segmenter(std::size_t frameSize, std::size_t hopSize, T* window,
              const std::size_t windowSize,
              const SegmenterMode mode = SegmenterMode::WOLA,
              const bool edgeCorrection = true,
              const bool normalizeWindow = true)
        : m_frameSize(frameSize), m_hopSize(hopSize), m_mode(mode),
          m_edgeCorrection(edgeCorrection), m_normalizeWindow(normalizeWindow)
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

        auto cola = checkCola<T>(m_window, frameSize, m_hopSize);
        if (!cola.isCola) {
            throw std::runtime_error("specified window is not COLA compliant "
                                     "for the given hop size");
        }

        // Window normalization
        m_window = std::make_unique<T[windowSize]>();
        if (m_normalizeWindow) {
            for (std::size_t i = 0; i < windowSize; i++) {
                m_window[i] = window[i] / cola.normalizationValue;
            }
        } else {
            for (std::size_t i = 0; i < windowSize; i++) {
                m_window[i] = window[i];
            }
        }

        // edge correction
        m_preWindow = std::make_unique<T[windowSize]>();
        for (std::size_t i = 0; i < windowSize; i++) {
            m_preWindow[i] = m_window[i];
        }

        m_postWindow = std::make_unique<T[windowSize]>();
        for (std::size_t i = 0; i < windowSize; i++) {
            m_preWindow[i] = m_window[i];
        }
        if (m_edgeCorrection) {
            for (std::size_t i = 1; i < m_frameSize / m_hopSize + 1; i++) {
                std::size_t preWindowStart = i * m_hopSize;
                std::size_t preWindowEnd = m_frameSize;
                for (std::size_t n = preWindowStart; n < preWindowEnd; n++) {
                    m_preWindow[n] += m_window[n];
                }

                std::size_t postWindowStart = 0;
                std::size_t postWindowEnd = m_frameSize - preWindowStart;
                for (std::size_t n = postWindowStart; n < postWindowEnd; n++) {
                    m_postWindow[n] += m_window[n];
                }
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
    }

    void segment() {
    }
    void unsegment() {
    }
    void spectrogram() {
    }
    void unspectrogram() {
    }
};
