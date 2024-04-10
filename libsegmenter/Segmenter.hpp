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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <cstddef>
#include <cmath>

#define SEGMENTER_M_PI 3.14159265358979323846

template<typename T>
void populateBartlettWindow(T* vec, const std::size_t windowLength) {
    const T M = windowLength + 1.0;
    for(std::size_t i = 0; i < windowLength; i++) {
        vec[i] = 1.0 - std::abs(-1.0 * (M - 1) / 2.0 + i) * 2.0 / (M - 1.0);
    }
}

template<typename T>
void populateBlackmanWindow(T* vec, const std::size_t windowLength) {
    const T M = T(windowLength + 1);
    for(std::size_t i = 0; i < windowLength; i++) {
        vec[i] = 7938.0/18608.0 - 9240.0/18608.0 * cos(2.0 * SEGMENTER_M_PI * T(i) / T(M - 1)) + 1430.0/18608.0 * cos(4.0 * SEGMENTER_M_PI * T(i) / T(M - 1));
    }
}

template<typename T>
void populateHammingWindow(T* vec, const std::size_t windowLength) {
    const T M = T(windowLength);
    const T alpha = 25.0 / 46.0;
    const T beta = (1.0 - alpha) / 2.0;
    for(std::size_t i = 0; i < windowLength; i++) {
        vec[i] = alpha - 2.0 * beta * cos(
            2.0 * SEGMENTER_M_PI * T(i) / M
        );
    }
}

template<typename T>
void populateHannWindow(T* vec, const std::size_t windowLength) {
    const T M = T(windowLength);
    for(std::size_t i = 0; i < windowLength; i++) {
        vec[i] = 0.5 * (1.0 - cos(2.0 * SEGMENTER_M_PI * T(i) / M));
    }
}

template<typename T>
void populateRectangularWindow(T* vec, const std::size_t windowLength) {
    for(std::size_t i = 0; i < windowLength; i++) {
        vec[i] = T(1);
    }
}
