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
 * Parameters.hpp
 * Library describing the parameters for the segmenter + validation + helper
 * functions.
 *
 * ==============================================================================
 */

#pragma once
#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Mode.hpp"

namespace segmenter {
template <typename T>
struct SegmenterParameters {
    std::unique_ptr<T[]> window;
    std::size_t frameSize;
    std::size_t hopSize;
    SegmenterMode mode;
    bool edgeCorrection;
    bool normalizeWindow;

    SegmenterParameters(std::unique_ptr<T[]>&& window,
                        const std::size_t frameSize, const std::size_t hopSize,
                        const SegmenterMode mode, const bool edgeCorrection,
                        const bool normalizeWindow)
        : window(std::move(window)), frameSize(frameSize), hopSize(hopSize),
          mode(mode), edgeCorrection(edgeCorrection),
          normalizeWindow(normalizeWindow)
    {
    }
    SegmenterParameters() = default;
};

template <typename T>
void getSegmentationShapeFromUnsegmented(
    const SegmenterParameters<T>& parameters,
    const std::array<std::size_t, 2>& unsegmentedShape,
    std::array<std::size_t, 3>& segmentedShape)
{
    if (unsegmentedShape[1] % parameters.hopSize != 0) {
        throw std::runtime_error("specified input shape is not a modulus "
                                 "of the specified hop size");
    }
    segmentedShape[0] = unsegmentedShape[0];
    segmentedShape[1] = (unsegmentedShape[1] / parameters.hopSize) -
                        parameters.frameSize / parameters.hopSize + 1;
    segmentedShape[2] = parameters.frameSize;
}

template <typename T>
void getSegmentationShapeFromSegmented(
    const SegmenterParameters<T>& parameters,
    std::array<std::size_t, 2>& unsegmentedShape,
    const std::array<std::size_t, 3>& segmentedShape)
{
    unsegmentedShape[0] = segmentedShape[0];
    unsegmentedShape[1] =
        (segmentedShape[1] - 1) * parameters.hopSize + parameters.frameSize;
    if (unsegmentedShape[1] % parameters.hopSize != 0) {
        throw std::runtime_error("specified input shape is not a modulus "
                                 "of the specified hop size");
    }
}

template <typename T>
void validateSegmentationShape(
    const SegmenterParameters<T>& parameters,
    const std::array<std::size_t, 2>& unsegmentedShape,
    const std::array<std::size_t, 3>& segmentedShape)
{
    std::array<std::size_t, 3> expected_segmentedShape{};
    getSegmentationShapeFromUnsegmented<T>(parameters, unsegmentedShape,
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

template <typename T>
void getSpectrogramShapeFromUnsegmented(
    const SegmenterParameters<T>& parameters,
    const std::array<std::size_t, 2>& unsegmentedShape,
    std::array<std::size_t, 3>& segmentedShape)
{
    if (unsegmentedShape[1] % parameters.hopSize != 0) {
        throw std::runtime_error("specified input shape is not a modulus "
                                 "of the specified hop size");
    }
    segmentedShape[0] = unsegmentedShape[0];
    segmentedShape[1] = (unsegmentedShape[1] / parameters.hopSize) -
                        parameters.frameSize / parameters.hopSize + 1;
    segmentedShape[2] = parameters.frameSize / 2 + 1;
}

template <typename T>
void getSpectrogramShapeFromSegmented(
    const SegmenterParameters<T>& parameters,
    std::array<std::size_t, 2>& unsegmentedShape,
    const std::array<std::size_t, 3>& segmentedShape)
{
    unsegmentedShape[0] = segmentedShape[0];
    unsegmentedShape[1] =
        (segmentedShape[1] - 1) * parameters.hopSize + parameters.frameSize;
    if (unsegmentedShape[1] % parameters.hopSize != 0) {
        throw std::runtime_error("specified input shape is not a modulus "
                                 "of the specified hop size");
    }
}

template <typename T>
void validateSpectrogramShape(
    const SegmenterParameters<T>& parameters,
    const std::array<std::size_t, 2>& unsegmentedShape,
    const std::array<std::size_t, 3>& segmentedShape)
{
    if (!(parameters.frameSize &&
          !(parameters.frameSize & (parameters.frameSize - 1)))) {
        throw std::runtime_error(
            "given segmenter is configured to a non-radix 2 frame size, "
            "spectrogram is thus not supported");
    }
    std::array<std::size_t, 3> expected_segmentedShape{};
    getSpectrogramShapeFromUnsegmented<T>(parameters, unsegmentedShape,
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
} // namespace segmenter
