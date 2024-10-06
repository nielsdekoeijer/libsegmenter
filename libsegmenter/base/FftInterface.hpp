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
 * FftInterface.hpp
 * Abstraction over FFT used in the segmenter, allows for injection of own fft
 * library. By default we use the open source `split-radix-fft` fft library.
 * Note that this limits the buffer size to radix 2 currently. Also note that
 * this isn't a bad choice for efficiency.
 *
 * ==============================================================================
 */

#pragma once

// from split-radix-fft
#include "fft.hpp"

#include "Helper.hpp"
#include <complex>
#include <memory>

namespace segmenter {
template <typename T>
class FWRfft {
    std::size_t m_size;
    std::size_t m_halfSpectrumSize;
    std::unique_ptr<std::complex<T>[]> m_fftTwiddleFactors;
    std::unique_ptr<std::complex<T>[]> m_scratch0;

  public:
    FWRfft(const std::size_t size)
        : m_size(size), m_halfSpectrumSize(size / 2 + 1),
          m_fftTwiddleFactors(
              std::move(std::make_unique<std::complex<T>[]>(m_size))),
          m_scratch0(std::move(
              std::make_unique<std::complex<T>[]>(m_halfSpectrumSize)))
    {
        fft::FFTSTATUS err;
        err = fft::populateRfftTwiddleFactorsForward<T>(
            m_size, m_fftTwiddleFactors.get(), m_size);
        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error occured in the creation of the fft "
                                     "forward twiddle factors");
        }
    }

    void process(const T* in, const std::size_t inSize, std::complex<T>* out,
                 const std::size_t outSize)
    {
        fft::FFTSTATUS err = fft::performRfftForward<T>(
            m_size, m_fftTwiddleFactors.get(), m_size, in, inSize, out, outSize,
            m_scratch0.get(), m_halfSpectrumSize);

        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error in fft");
        }
    }
};

template <typename T>
class BWRfft {
    std::size_t m_size;
    std::size_t m_halfSpectrumSize;
    std::unique_ptr<std::complex<T>[]> m_fftTwiddleFactors;
    std::unique_ptr<std::complex<T>[]> m_scratch0;
    std::unique_ptr<std::complex<T>[]> m_scratch1;

  public:
    BWRfft(const std::size_t size)
        : m_size(size), m_halfSpectrumSize(size / 2 + 1),
          m_fftTwiddleFactors(std::make_unique<std::complex<T>[]>(size)),
          m_scratch0(std::make_unique<std::complex<T>[]>(m_halfSpectrumSize)),
          m_scratch1(std::make_unique<std::complex<T>[]>(m_halfSpectrumSize))
    {
        fft::FFTSTATUS err;
        err = fft::populateRfftTwiddleFactorsBackward<T>(
            m_size, m_fftTwiddleFactors.get(), m_size);
        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error occured in the creation of the fft "
                                     "forward twiddle factors");
        }
    }

    void process(const std::complex<T>* in, const std::size_t inSize, T* out,
                 const std::size_t outSize)
    {
        fft::FFTSTATUS err;
        err = fft::performRfftBackward<T>(
            m_size, m_fftTwiddleFactors.get(), m_size, in, inSize, out, outSize,
            m_scratch0.get(), m_scratch1.get(), m_halfSpectrumSize);
        if (err != fft::FFTSTATUS::OK) {
            throw std::runtime_error("error in fft");
        }
    }
};
} // namespace segmenter
