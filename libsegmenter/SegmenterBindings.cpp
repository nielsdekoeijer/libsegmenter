#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "Segmenter.hpp"
using DATATYPE = double;

py::array_t<DATATYPE> py_bartlett(size_t size) {
    auto result = py::array_t<DATATYPE>(size);
    auto buf = result.request();
    DATATYPE *ptr = static_cast<DATATYPE *>(buf.ptr);
    populateBartlettWindow(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_blackman(size_t size) {
    auto result = py::array_t<DATATYPE>(size);
    auto buf = result.request();
    DATATYPE *ptr = static_cast<DATATYPE *>(buf.ptr);
    populateBlackmanWindow(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_hamming(size_t size) {
    auto result = py::array_t<DATATYPE>(size);
    auto buf = result.request();
    DATATYPE *ptr = static_cast<DATATYPE *>(buf.ptr);
    populateHammingWindow(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_hann(size_t size) {
    auto result = py::array_t<DATATYPE>(size);
    auto buf = result.request();
    DATATYPE *ptr = static_cast<DATATYPE *>(buf.ptr);
    populateHannWindow(ptr, size);
    return result;
}

py::array_t<DATATYPE> py_rectangular(size_t size) {
    auto result = py::array_t<DATATYPE>(size);
    auto buf = result.request();
    DATATYPE *ptr = static_cast<DATATYPE *>(buf.ptr);
    populateRectangularWindow(ptr, size);
    return result;
}

PYBIND11_MODULE(bindings, m) {
    m.def("bartlett", py_bartlett);
    m.def("blackman", py_blackman);
    m.def("hamming", py_hamming);
    m.def("hann", py_hann);
    m.def("rectangular", py_rectangular);
}
