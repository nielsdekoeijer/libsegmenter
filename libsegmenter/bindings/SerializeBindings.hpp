#pragma once

#include "Bindings.hpp"
#include "Serialize.hpp"

void py_saveSegmenterParameters(
    const std::string path,
    const segmenter::SegmenterParameters<DATATYPE>& item);
segmenter::SegmenterParameters<DATATYPE>
py_loadSegmenterParameters(const std::string path);
