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
 * Span.hpp
 * Span implementation for use in C++17, should be replaced by mdspan when
 * generally supported in C++23. Essentially an unsafe non-owning view over a
 * contiguous bit of memory where the alignment is assumed to be right-hand 
 * / c-style / row-index.
 *
 * ==============================================================================
 */

#pragma once
#include <array>
#include <cstdlib>

namespace segmenter {

template <typename T, std::size_t dim>
struct Span {
    const T* data;
    const std::array<std::size_t, dim> shape;

    Span(const T* data, const std::array<std::size_t, dim> shape)
        : data(data), shape(shape)
    {
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        static_assert(sizeof...(args) == dim, "Dimension mismatch");
        std::size_t indices[] = {static_cast<std::size_t>(args)...};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < dim; ++i) {
            offset = offset * shape[i] + indices[i];
        }
        return data[offset];
    }
};

template <typename T, std::size_t dim>
struct MutableSpan {
    T* data;
    const std::array<std::size_t, dim> shape;

    MutableSpan(T* data, const std::array<std::size_t, dim> shape)
        : data(data), shape(shape)
    {
    }

    template<typename... Args>
    T& operator()(Args... args) const {
        static_assert(sizeof...(args) == dim, "Dimension mismatch");
        std::size_t indices[] = {static_cast<std::size_t>(args)...};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < dim; ++i) {
            offset = offset * shape[i] + indices[i];
        }
        return data[offset];
    }
};
} // namespace segmenter
