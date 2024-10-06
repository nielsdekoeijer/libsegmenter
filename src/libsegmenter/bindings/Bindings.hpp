#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
using TPYARRAY = py::array_t<T, py::array::c_style | py::array::forcecast>;

using DATATYPE = double;
using PYARRAY = TPYARRAY<DATATYPE>;
using CPYARRAY = TPYARRAY<std::complex<DATATYPE>>;
