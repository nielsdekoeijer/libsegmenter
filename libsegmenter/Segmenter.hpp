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
#include <stdexcept>
#include <cstddef>
#include <memory>

#define SEGMENTER_M_PI 3.14159265358979323846

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

  public:
    Segmenter(std::size_t frameSize, std::size_t hopSize, T* window,
              const std::size_t windowSize,
              const SegmenterMode mode = SegmenterMode::WOLA,
              const bool edgeCorrection = true,
              const bool normalizeWindow = true)
        : m_frameSize(frameSize), m_hopSize(hopSize), m_mode(mode),
          m_edgeCorrection(edgeCorrection), m_normalizeWindow(normalizeWindow)
    {
        if (m_frameSize % 2 != 0) {
            throw std::runtime_error("only even frameSize is supported");
        }

        if (m_hopSize > m_frameSize) {
            throw std::runtime_error("hopSize cannot be larger than frame_size");
        }

        if (windowSize != frameSize) {
            throw std::runtime_error("specified window must have the same size as frame_size");
        }

        for (std::size_t i = 0; i < windowSize; i++) {
            if (window[i] < 0.0) {
                throw std::runtime_error("specified window contains negative values");
            }
        }
    }
};
