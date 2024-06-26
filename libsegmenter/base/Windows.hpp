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
 * Windows.hpp
 * Library containing some windows.
 *
 * ==============================================================================
 */

#pragma once
#include "Helper.hpp"
#include <cmath>
#include <cstddef>

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
        vec[i] = 7938.0 / 18608.0 -
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
} // namespace segmenter
