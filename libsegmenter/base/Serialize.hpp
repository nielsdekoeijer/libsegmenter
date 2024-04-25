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
 * Serialize.hpp
 * Optional library for loading and saving segmenter parameters, requires boost.
 *
 * ==============================================================================
 */

#pragma once
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <fstream>

#include "Mode.hpp"
#include "Parameters.hpp"

namespace segmenter {
template <typename T>
void saveSegmenterParameters(const std::string path,
                             const SegmenterParameters<T>& item)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for saving");
    }
    boost::archive::xml_oarchive xml_oa(ofs);
    xml_oa << BOOST_SERIALIZATION_NVP(item.frameSize) &
        BOOST_SERIALIZATION_NVP(item.hopSize) &
        BOOST_SERIALIZATION_NVP(item.window) &
        BOOST_SERIALIZATION_NVP(item.edgeCorrection) &
        BOOST_SERIALIZATION_NVP(item.normalizeWindow);
}

template <typename T>
SegmenterParameters<T> loadSegmenterParameters(const std::string path)
{
    SegmenterParameters<T> item;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for loading");
    }
    boost::archive::xml_iarchive xml_ia(ifs);
    xml_ia >> BOOST_SERIALIZATION_NVP(item.frameSize) &
        BOOST_SERIALIZATION_NVP(item.hopSize) &
        BOOST_SERIALIZATION_NVP(item.window) &
        BOOST_SERIALIZATION_NVP(item.edgeCorrection) &
        BOOST_SERIALIZATION_NVP(item.normalizeWindow);

    return item;
}
} // namespace segmenter
