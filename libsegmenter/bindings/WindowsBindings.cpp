#include "Bindings.hpp"
#include "Windows.hpp"

PYARRAY py_bartlett(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    segmenter::populateBartlettWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_blackman(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    segmenter::populateBlackmanWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_hamming(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    segmenter::populateHammingWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_hann(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    segmenter::populateHannWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_rectangular(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    segmenter::populateRectangularWindow<DATATYPE>(ptr, size);
    return result;
}

