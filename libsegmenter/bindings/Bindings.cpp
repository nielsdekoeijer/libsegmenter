#include "Bindings.hpp"
#include "Windows.hpp"
#include <iostream>
#include <string>

#include "ColaBindings.hpp"
#include "SegmenterBindings.hpp"
#include "WindowsBindings.hpp"

PYBIND11_MODULE(bindings, m)
{
    // window functions
    m.def("bartlett", &py_bartlett);
    m.def("blackman", &py_blackman);
    m.def("hamming", &py_hamming);
    m.def("hann", &py_hann);
    m.def("rectangular", &py_rectangular);

    // cola
    m.def("check_cola", &py_checkCola,
          "Check the Constant Overlap-Add (COLA) condition for a window",
          py::arg("window"), py::arg("hop_size"), py::arg("eps") = 1e-5);

    // segmenter
    py::class_<py_Segmenter>(m, "Segmenter")
        .def(py::init<std::size_t, std::size_t, PYARRAY, std::string, bool,
                      bool>(),
             py::arg("frame_size"), py::arg("hop_size"), py::arg("window"),
             py::arg("mode") = "wola", py::arg("edge_correction") = true,
             py::arg("normalize_window") = true)
        .def("segment", &py_Segmenter::segment)
        .def("unsegment", &py_Segmenter::unsegment)
        .def("spectrogram", &py_Segmenter::spectrogram)
        .def("unspectrogram", &py_Segmenter::unspectrogram);
}
