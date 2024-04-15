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
    T** make2DPointerView(T* ptr, const std::array<std::size_t, 2>& shape)
    {
        T** view = new T*[shape[0]];
        for (std::size_t i = 0; i < shape[0]; i++) {
            view[i] = ptr + i * shape[1];
        }

        return view;
    }

    template <typename T>
    T*** make3DPointerView(T* ptr, const std::array<std::size_t, 3>& shape)
    {
        T*** view = new T**[shape[0]];
        for (std::size_t i = 0; i < shape[0]; i++) {
            view[i] = make2DPointerView(ptr + i * (shape[1] * shape[2]),
                                        {shape[1], shape[2]});
        }

        return view;
    }

  public:
    // Constructor: Initializes the Segmenter with provided arguments
    py_Segmenter(
        std::size_t frameSize, std::size_t hopSize,
        py::array_t<DATATYPE, py::array::c_style | py::array::forcecast> window,
        std::string modeString, bool edgeCorrection, bool normalizeWindow)
    {
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

    py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>
    segment(const py::array_t<DATATYPE,
                              py::array::c_style | py::array::forcecast>& x)
    {
        auto buf = x.request();

        // determine batch size
        std::size_t batchSize, inputLength;
        if (buf.ndim == 1) {
            batchSize = 1;
            inputLength = buf.shape[0];
        } else if (buf.ndim != 2) {
            batchSize = buf.shape[0];
            inputLength = buf.shape[1];

        } else {
            throw std::runtime_error(
                "input should be a 1-dimensional or 2-dimensional array");
        }

        // get input pointer
        DATATYPE* inputPtr = static_cast<DATATYPE*>(buf.ptr);

        // make input pointer view
        std::array<std::size_t, 2> inputBatchShape{batchSize, inputLength};
        DATATYPE** inputBatchPtrView =
            make2DPointerView(inputPtr, inputBatchShape);

        // get output pointer
        std::size_t frameCount = m_segmenter->getFrameCount(inputLength);
        std::size_t frameSize = m_segmenter->getFrameSize();
        DATATYPE* outputPtr = new DATATYPE[batchSize * frameSize * frameCount];

        // make output pointer view
        std::array<std::size_t, 3> outputBatchShape{batchSize, frameCount,
                                                    frameSize};
        DATATYPE*** outputBatchPtrView =
            make3DPointerView(outputPtr, outputBatchShape);

        // segment
        m_segmenter->segment(inputBatchPtrView, inputBatchShape,
                             outputBatchPtrView, outputBatchShape);

        // wrap in pyarray
        return py::array(py::buffer_info(
            outputPtr, sizeof(DATATYPE),
            py::format_descriptor<DATATYPE>::format(), 3, outputBatchShape,
            {sizeof(DATATYPE) * outputBatchShape[1] * outputBatchShape[2],
             sizeof(DATATYPE) * outputBatchShape[2], sizeof(DATATYPE)}));
    }
    /*
    void unsegment(const py::array_t<DATATYPE, py::array::c_style |
                                                   py::array::forcecast>& X)
    {
        void; // m_segmenter->unsegment();
    }
    void spectrogram() { m_segmenter->spectrogram(); }
    void unspectrogram() { m_segmenter->unspectrogram(); }
    */
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
        .def(py::init<
             std::size_t, std::size_t,
             py::array_t<DATATYPE, py::array::c_style | py::array::forcecast>,
             std::string, bool, bool>())
        .def("segment", &py_Segmenter::segment);
}
