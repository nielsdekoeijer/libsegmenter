#include "SegmenterBindings.hpp"
#include "Mode.hpp"
#include "Segmenter.hpp"
#include "Span.hpp"

segmenter::SegmenterMode
py_Segmenter::determineMode(const std::string modeString)
{
    if (modeString == "wola") {
        return segmenter::SegmenterMode::WOLA;
    } else if (modeString == "ola") {
        return segmenter::SegmenterMode::OLA;
    } else {
        throw std::runtime_error("Mode neither 'wola' nor 'ola'");
    }
}

template <typename T>
TPYARRAY<T> makePythonArray(T* data, const std::array<std::size_t, 3>& shape)
{
    auto view = segmenter::Span<T, 3>(data, shape);
    std::size_t size = sizeof(T);
    auto array = TPYARRAY<T>({shape[0], shape[1], shape[2]});
    auto result = array.template mutable_unchecked<3>();

    for (std::size_t i = 0; i < shape[0]; i++) {
        for (std::size_t j = 0; j < shape[1]; j++) {
            for (std::size_t k = 0; k < shape[2]; k++) {
                result(i, j, k) = view(i, j, k);
            }
        }
    }

    return array;
}

template <typename T>
TPYARRAY<T> makePythonArray(T* data, const std::array<std::size_t, 2>& shape)
{
    auto view = segmenter::Span<T, 2>(data, shape);
    std::size_t size = sizeof(T);
    auto array = TPYARRAY<T>({shape[0], shape[1]});
    auto result = array.template mutable_unchecked<2>();
    for (std::size_t i = 0; i < shape[0]; i++) {
        for (std::size_t j = 0; j < shape[1]; j++) {
            result(i, j) = view(i, j);
        }
    }

    return array;
}

template <typename T>
TPYARRAY<T> makePythonArray(T* data, const std::array<std::size_t, 1>& shape)
{
    auto view = segmenter::Span<T, 1>(data, shape);
    std::size_t size = sizeof(T);
    std::array<std::size_t, 1> stride;
    stride[0] = size;

    auto array = TPYARRAY<T>(shape[0]);
    auto result = array.template mutable_unchecked<1>();
    for (std::size_t i = 0; i < shape[0]; i++) {
        result(i) = view(i); 
    }

    return array;
}

// Constructor: Initializes the Segmenter with provided arguments
py_Segmenter::py_Segmenter(std::size_t frameSize, std::size_t hopSize,
                           PYARRAY window, std::string modeString = "wola",
                           bool edgeCorrection = true,
                           bool normalizeWindow = true)
{
    // Determine wola mode based on string
    segmenter::SegmenterMode mode = determineMode(modeString);

    // Obtain an array from numpy
    auto buf = window.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Window should be a 1-dimensional array");
    const DATATYPE* window_ptr = static_cast<DATATYPE*>(buf.ptr);
    std::size_t windowSize = buf.shape[0];

    // Create a new Segmenter and pass the arguments
    m_segmenter = std::make_unique<segmenter::Segmenter<DATATYPE>>(
        frameSize, hopSize, window_ptr, windowSize, mode, edgeCorrection,
        normalizeWindow);
}

PYARRAY py_Segmenter::segment(const PYARRAY& arr)
{
    auto buf = arr.request();

    // Extract input size
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
    segmenter::getSegmentationShapeFromUnsegmented<DATATYPE>(
        m_segmenter->m_parameters, ishape, oshape);

    // Get input pointer
    DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
    std::unique_ptr<DATATYPE[]> optr(
        new DATATYPE[oshape[0] * oshape[1] * oshape[2]]);
    m_segmenter->segment(iptr, ishape, optr.get(), oshape);

    if (batched) {
        return makePythonArray(optr.get(), oshape);
    } else {
        return makePythonArray(
            optr.get(), std::array<std::size_t, 2>{oshape[1], oshape[2]});
    }
}

PYARRAY py_Segmenter::unsegment(const PYARRAY& arr)
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
    segmenter::getSegmentationShapeFromSegmented<DATATYPE>(
        m_segmenter->m_parameters, oshape, ishape);

    // get input pointer
    DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
    std::unique_ptr<DATATYPE[]> optr(new DATATYPE[oshape[0] * oshape[1]]);

    m_segmenter->unsegment(iptr, ishape, optr.get(), oshape);

    if (batched) {
        return makePythonArray(optr.get(), oshape);
    } else {
        return makePythonArray(optr.get(),
                               std::array<std::size_t, 1>{oshape[1]});
    }
}

CPYARRAY py_Segmenter::spectrogram(const PYARRAY& arr)
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
    segmenter::getSpectrogramShapeFromUnsegmented<DATATYPE>(
        m_segmenter->m_parameters, ishape, oshape);

    // get input pointer
    DATATYPE* iptr = static_cast<DATATYPE*>(buf.ptr);
    std::unique_ptr<std::complex<DATATYPE>[]> optr(
        new std::complex<DATATYPE>[oshape[0] * oshape[1] * oshape[2]]);
    if (optr.get() == nullptr) {
        throw std::runtime_error("nullptr");
    }

    m_segmenter->spectrogram(iptr, ishape, optr.get(), oshape);

    if (batched) {
        return makePythonArray<std::complex<DATATYPE>>(optr.get(), oshape);
    } else {
        return makePythonArray<std::complex<DATATYPE>>(
            optr.get(), std::array<std::size_t, 2>{oshape[1], oshape[2]});
    }
}

PYARRAY py_Segmenter::unspectrogram(const CPYARRAY& arr)
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
    segmenter::getSpectrogramShapeFromSegmented<DATATYPE>(
        m_segmenter->m_parameters, oshape, ishape);

    // get input pointer
    std::complex<DATATYPE>* iptr =
        static_cast<std::complex<DATATYPE>*>(buf.ptr);
    std::unique_ptr<DATATYPE[]> optr(new DATATYPE[oshape[0] * oshape[1]]);

    m_segmenter->unspectrogram(iptr, ishape, optr.get(), oshape);

    if (batched) {
        return makePythonArray(optr.get(), oshape);
    } else {
        return makePythonArray(optr.get(),
                               std::array<std::size_t, 1>{oshape[1]});
    }
}
