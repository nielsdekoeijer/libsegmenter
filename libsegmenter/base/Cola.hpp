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
    T epsilon;
};

template <typename T>
COLAResult<T> checkCola(const T* window, const std::size_t windowSize,
                        const std::size_t hopSize, const T eps = 1e-5)
{
    // NOTE: Allocates
    const double frameRate = 1.0 / double(hopSize);

    double factor = 0.0;
    for (std::size_t i = 0; i < windowSize; i++) {
        factor += double(window[i]);
    }
    factor /= double(hopSize);

    const std::size_t N = 6 * windowSize;
    auto sp = std::make_unique<std::complex<double>[]>(N);
    for (std::size_t i = 0; i < N; i++) {
        sp[i] = factor;
    }
    double ubound = sp[0].real();
    double lbound = sp[0].real();

    auto csin = std::make_unique<std::complex<double>[]>(N);
    for (std::size_t k = 1; k < hopSize; k++) {
        const double f = frameRate * double(k);
        for (std::size_t n = 0; n < N; n++) {
            csin[n] = std::exp(std::complex<double>(0.0, 2.0 * PI<double> * f * n));
        }

        std::complex<double> Wf = 0.0;
        for (std::size_t n = 0; n < windowSize; n++) {
            Wf += double(window[n]) * std::conj(csin[n]);
        }

        for (std::size_t n = 0; n < N; n++) {
            sp[n] += (Wf * csin[n]) / double(hopSize);
        }

        const double Wfb = abs(Wf);
        ubound += Wfb / double(hopSize);
        lbound -= Wfb / double(hopSize);
    }

    COLAResult<T> result;
    result.epsilon = (ubound - lbound);
    result.isCola = (ubound - lbound) < eps;
    result.normalizationValue = (ubound + lbound) / 2.0;

    return result;
}
}
