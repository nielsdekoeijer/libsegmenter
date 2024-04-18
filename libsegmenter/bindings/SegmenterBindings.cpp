#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Segmenter.hpp"
#include <iostream>
#include <string>

template <typename T>
using TPYARRAY = py::array_t<T, py::array::c_style | py::array::forcecast>;

using DATATYPE = double;
using PYARRAY = TPYARRAY<DATATYPE>;
using CPYARRAY = TPYARRAY<std::complex<DATATYPE>>;

PYARRAY py_bartlett(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateBartlettWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_blackman(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateBlackmanWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_hamming(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateHammingWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_hann(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateHannWindow<DATATYPE>(ptr, size);
    return result;
}

PYARRAY py_rectangular(size_t size)
{
    auto result = PYARRAY(size);
    auto buf = result.request();

    DATATYPE* ptr = static_cast<DATATYPE*>(buf.ptr);
    populateRectangularWindow<DATATYPE>(ptr, size);
    return result;
}

py::tuple py_checkCola(PYARRAY window, std::size_t hopSize,
                       DATATYPE eps = static_cast<DATATYPE>(1e-5))
{

    auto buf = window.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input should be a 1-dimensional array");

    const DATATYPE* window_ptr = static_cast<DATATYPE*>(buf.ptr);
    std::size_t windowSize = buf.shape[0];

    auto result = checkCola<DATATYPE>(window_ptr, windowSize, hopSize, eps);
    return py::make_tuple(result.isCola, result.normalizationValue);
}

class py_Segmenter {
  private:
    std::unique_ptr<Segmenter<DATATYPE>> m_segmenter;

    SegmenterMode determineMode(const std::string modeString)
    {
        if (modeString == "wola") {
            return SegmenterMode::WOLA;
        } else if (modeString == "ola") {
            return SegmenterMode::OLA;
        } else {
            throw std::runtime_error("Mode neither 'wola' nor 'ola'");
        }
    }

    template <typename T>
    TPYARRAY<T> makePythonArray(T* data,
                                const std::array<std::size_t, 3>& shape)
    {
        std::size_t size = sizeof(T);
        std::array<std::size_t, 3> stride;
        stride[0] = size * shape[1] * shape[2];
        stride[1] = size * shape[2];
        stride[2] = size;

        py::capsule free_when_done(
            data, [](void* f) { delete[] reinterpret_cast<T*>(f); });

        return py::array(
            py::buffer_info(data,                               // ptr to data
                            size,                               // scalar size
                            py::format_descriptor<T>::format(), // python format
                            3,     // number of dimensions
                            shape, // output shape
                            stride // output stride
                            ),
            free_when_done // capsule will free the data when Python object is
                           // collected
        );
    }

    template <typename T>
    TPYARRAY<T> makePythonArray(T* data,
                                const std::array<std::size_t, 2>& shape)
    {
        std::size_t size = sizeof(T);
        std::array<std::size_t, 2> stride;
        stride[0] = size * shape[1];
        stride[1] = size;

        py::capsule free_when_done(
            data, [](void* f) { delete[] reinterpret_cast<T*>(f); });

        return py::array(
            py::buffer_info(data,                               // ptr to data
                            size,                               // scalar size
                            py::format_descriptor<T>::format(), // python format
                            2,     // number of dimensions
                            shape, // output shape
                            stride // output stride
                            ),
            free_when_done // capsule will free the data when Python object is
                           // collected
        );
    }

    template <typename T>
    TPYARRAY<T> makePythonArray(T* data,
                                const std::array<std::size_t, 1>& shape)
    {
        std::size_t size = sizeof(T);
        std::array<std::size_t, 1> stride;
        stride[0] = size;

        py::capsule free_when_done(
            data, [](void* f) { delete[] reinterpret_cast<T*>(f); });

        return py::array(
            py::buffer_info(data,                               // ptr to data
                            size,                               // scalar size
                            py::format_descriptor<T>::format(), // python format
                            1,     // number of dimensions
                            shape, // output shape
                            stride // output stride
                            ),
            free_when_done // capsule will free the data when Python object is
                           // collected
        );
    }

  public:
    // Constructor: Initializes the Segmenter with provided arguments
    py_Segmenter(std::size_t frameSize, std::size_t hopSize, PYARRAY window,
                 std::string modeString = "wola", bool edgeCorrection = true,
                 bool normalizeWindow = true)
    {
        // Determine wola mode based on string
        SegmenterMode mode = determineMode(modeString);

        // Obtain an array from numpy
        auto buf = window.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Window should be a 1-dimensional array");
        const DATATYPE* window_ptr = static_cast<DATATYPE*>(buf.ptr);
        std::size_t windowSize = buf.shape[0];

        // Create a new Segmenter and pass the arguments
        m_segmenter = std::make_unique<Segmenter<DATATYPE>>(
            frameSize, hopSize, window_ptr, windowSize, mode, edgeCorrection,
            normalizeWindow);
    }

    PYARRAY
    segment(const PYARRAY& arr)
    {
        auto buf = arr.request();

        // extract input size
        bool batched;
        std::array<std::size_t, 2> ishape{};
        if (buf.ndim == 1) {
            ishape[0] = 1;
            ishape[1] = buf.shape[0];
            batched = false;
        } else if (buf.ndim == 2) {
            ishape[0] = buf.shape[0];
            ishape[1] = buf.shape[1];
            batched = true;
        } else {
            throw std::runtime_error(
                "input should be a 1-dimensional or 2-dimensional array");
        }

        std::array<std::size_t, 3> oshape{};
        m_segmenter->getSegmentationShapeFromUnsegmented(ishape, oshape);

        // get input pointer
        DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
        std::unique_ptr<DATATYPE[]> optr(
            new DATATYPE[oshape[0] * oshape[1] * oshape[2]]);

        m_segmenter->segment(iptr, ishape, optr.get(), oshape);

        if (batched) {
            return makePythonArray(optr.release(), oshape);
        } else {
            return makePythonArray(optr.release(), std::array<std::size_t, 2>{
                                                       oshape[1], oshape[2]});
        }
    }

    PYARRAY
    unsegment(const PYARRAY& arr)
    {
        auto buf = arr.request();

        // extract input size
        bool batched;
        std::array<std::size_t, 3> ishape{};
        if (buf.ndim == 2) {
            ishape[0] = 1;
            ishape[1] = buf.shape[0];
            ishape[2] = buf.shape[1];
            batched = false;
        } else if (buf.ndim == 3) {
            ishape[0] = buf.shape[0];
            ishape[1] = buf.shape[1];
            ishape[2] = buf.shape[2];
            batched = true;
        } else {
            throw std::runtime_error(
                "input should be a 2-dimensional or 3-dimensional array");
        }

        std::array<std::size_t, 2> oshape{};
        m_segmenter->getSegmentationShapeFromSegmented(oshape, ishape);

        // get input pointer
        DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
        std::unique_ptr<DATATYPE[]> optr(new DATATYPE[oshape[0] * oshape[1]]);

        m_segmenter->unsegment(iptr, ishape, optr.get(), oshape);

        if (batched) {
            return makePythonArray(optr.release(), oshape);
        } else {
            return makePythonArray(optr.release(),
                                   std::array<std::size_t, 1>{oshape[1]});
        }
    }

    CPYARRAY
    spectrogram(const PYARRAY& arr)
    {
        auto buf = arr.request();

        // extract input size
        bool batched;
        std::array<std::size_t, 2> ishape{};
        if (buf.ndim == 1) {
            ishape[0] = 1;
            ishape[1] = buf.shape[0];
            batched = false;
        } else if (buf.ndim == 2) {
            ishape[0] = buf.shape[0];
            ishape[1] = buf.shape[1];
            batched = true;
        } else {
            throw std::runtime_error(
                "input should be a 1-dimensional or 2-dimensional array");
        }

        std::array<std::size_t, 3> oshape{};
        m_segmenter->getSpectrogramShapeFromUnsegmented(ishape, oshape);

        // get input pointer
        DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
        std::unique_ptr<std::complex<DATATYPE>[]> optr(
            new std::complex<DATATYPE>[oshape[0] * oshape[1] * oshape[2]]);
        if (optr.get() == nullptr) {
            throw std::runtime_error("nullptr");
        }

        m_segmenter->spectrogram(iptr, ishape, optr.get(), oshape);

        if (batched) {
            return makePythonArray<std::complex<DATATYPE>>(optr.release(),
                                                           oshape);
        } else {
            return makePythonArray<std::complex<DATATYPE>>(
                optr.release(),
                std::array<std::size_t, 2>{oshape[1], oshape[2]});
        }
    }

    PYARRAY unspectrogram(const CPYARRAY& arr)
    {
        auto buf = arr.request();

        // extract input size
        bool batched;
        std::array<std::size_t, 3> ishape{};
        if (buf.ndim == 2) {
            ishape[0] = 1;
            ishape[1] = buf.shape[0];
            ishape[2] = buf.shape[1];
            batched = false;
        } else if (buf.ndim == 3) {
            ishape[0] = buf.shape[0];
            ishape[1] = buf.shape[1];
            ishape[2] = buf.shape[2];
            batched = true;
        } else {
            throw std::runtime_error(
                "input should be a 2-dimensional or 3-dimensional array");
        }

        std::array<std::size_t, 2> oshape{};
        m_segmenter->getSpectrogramShapeFromSegmented(oshape, ishape);

        // get input pointer
        std::complex<DATATYPE>* iptr =
            static_cast<std::complex<DATATYPE>*>(buf.ptr);
        std::unique_ptr<DATATYPE[]> optr(new DATATYPE[oshape[0] * oshape[1]]);

        m_segmenter->unspectrogram(iptr, ishape, optr.get(), oshape);

        if (batched) {
            return makePythonArray(optr.release(), oshape);
        } else {
            return makePythonArray(optr.release(),
                                   std::array<std::size_t, 1>{oshape[1]});
        }
    }
};

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
