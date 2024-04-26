#include "Bindings.hpp"
#include "Windows.hpp"
#include <iostream>
#include <string>

#include "ColaBindings.hpp"
#include "SegmenterBindings.hpp"
#include "SerializeBindings.hpp"
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

    // serialize
    using Params = segmenter::SegmenterParameters<DATATYPE>;
    py::class_<Params>(m, "SegmenterParameters")
        .def(py::init([](py::array_t<DATATYPE> window, std::size_t frameSize,
                         std::size_t hopSize, segmenter::SegmenterMode mode,
                         bool edgeCorrection, bool normalizeWindow) {
            py::buffer_info info = window.request();
            auto w = std::make_unique<DATATYPE[]>(info.size);
            std::memcpy(w.get(), info.ptr, info.size * sizeof(DATATYPE));

            return std::make_unique<Params>(std::move(w), frameSize, hopSize,
                                            mode, edgeCorrection,
                                            normalizeWindow);
        }))
        .def_readonly("frame_size", &Params::frameSize)
        .def_readonly("hop_size", &Params::hopSize)
        .def_readonly("mode", &Params::mode)
        .def_readonly("edge_correction", &Params::edgeCorrection)
        .def_readonly("normalize_window", &Params::normalizeWindow)
        .def("clone_window", [](const Params& self) {
            py::array_t<DATATYPE> result(self.frameSize);
            auto result_info = result.request();
            DATATYPE* result_ptr = static_cast<DATATYPE*>(result_info.ptr);
            if (self.window) {
                std::memcpy(result_ptr, self.window.get(),
                            self.frameSize * sizeof(DATATYPE));
            }
            return result;
        });
    m.def("save", &py_saveSegmenterParameters);
    m.def("load", &py_loadSegmenterParameters);
}
