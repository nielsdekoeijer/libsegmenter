#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "Segmenter.hpp"
using DATATYPE = double;

py::array_t<DATATYPE> py_bartlett(size_t size)
{
    auto result =
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateBartlettWindow<DATATYPE>(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_blackman(size_t size)
{
    auto result =
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateBlackmanWindow<DATATYPE>(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_hamming(size_t size)
{
    auto result =
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateHammingWindow<DATATYPE>(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_hann(size_t size)
{
    auto result =
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateHannWindow<DATATYPE>(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_rectangular(size_t size)
{
    auto result =
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateRectangularWindow<DATATYPE>(ptr, size);
    return result;
}

py::tuple py_checkCola(
    py::array_t<DATATYPE, py::array::c_style | py::array::forcecast> window,
    std::size_t hopSize, DATATYPE eps = static_cast<DATATYPE>(1e-5))
{

    auto buf = window.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input should be a 1-dimensional array");

    const DATATYPE* window_ptr = static_cast<DATATYPE*>(buf.ptr);
    std::size_t windowSize = buf.shape[0];

    auto result = checkCola<DATATYPE>(window_ptr, windowSize, hopSize, eps);
    return py::make_tuple(result.isCola, result.normalizationValue);
}

PYBIND11_MODULE(bindings, m)
{
    m.def("bartlett", py_bartlett);
    m.def("blackman", py_blackman);
    m.def("hamming", py_hamming);
    m.def("hann", py_hann);
    m.def("rectangular", py_rectangular);
    m.def("check_cola", &py_checkCola,
          "Check the Constant Overlap-Add (COLA) condition for a window",
          py::arg("window"), py::arg("hop_size"), py::arg("eps") = 1e-5);
}
