#include "ColaBindings.hpp"
#include "Cola.hpp"

py::tuple py_checkCola(PYARRAY window, std::size_t hopSize,
                       DATATYPE eps = static_cast<DATATYPE>(1e-5))
{

    auto buf = window.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input should be a 1-dimensional array");

    const DATATYPE* window_ptr = static_cast<DATATYPE*>(buf.ptr);
    std::size_t windowSize = buf.shape[0];

    auto result =
        segmenter::checkCola<DATATYPE>(window_ptr, windowSize, hopSize, eps);
    return py::make_tuple(result.isCola, result.normalizationValue);
}

