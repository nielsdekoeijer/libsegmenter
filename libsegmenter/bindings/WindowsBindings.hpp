#include "Bindings.hpp"
#include <cstdlib>

py::tuple py_checkCola(PYARRAY window, std::size_t hopSize,
                       DATATYPE eps = static_cast<DATATYPE>(1e-5));
