#include "SerializeBindings.hpp"
#include "Serialize.hpp"

void py_saveSegmenterParameters(
    const std::string path,
    const segmenter::SegmenterParameters<DATATYPE>& item)
{
    segmenter::saveSegmenterParameters<DATATYPE>(path, item);
}
segmenter::SegmenterParameters<DATATYPE>
py_loadSegmenterParameters(const std::string path)
{
    return segmenter::loadSegmenterParameters<DATATYPE>(path);
}
