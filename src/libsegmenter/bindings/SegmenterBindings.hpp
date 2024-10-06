#pragma once
#include "Bindings.hpp"
#include "Segmenter.hpp"
#include <memory>

class py_Segmenter {
  private:
    std::unique_ptr<segmenter::Segmenter<DATATYPE>> m_segmenter;
    segmenter::SegmenterMode determineMode(const std::string modeString);

  public:
    PYARRAY segment(const PYARRAY& arr);
    PYARRAY unsegment(const PYARRAY& arr);
    CPYARRAY spectrogram(const PYARRAY& arr);
    PYARRAY unspectrogram(const CPYARRAY& arr);
    py_Segmenter(std::size_t frameSize, std::size_t hopSize, PYARRAY window,
                 std::string modeString, bool edgeCorrection,
                 bool normalizeWindow);
};
