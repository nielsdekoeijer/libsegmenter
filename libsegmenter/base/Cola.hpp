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
 * Cola.hpp
 * Library for checking the cola condition.
 *
 * ==============================================================================
 */

#pragma once
#include "Helper.hpp"
#include <memory>
#include <complex>

namespace segmenter {

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
            csin[n] = std::exp(std::complex<T>(0.0, 2.0 * PI<T> * f * n));
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
}
