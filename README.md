# Segmenter
A small library intended to provide helper functions for block-based processing in Python.
The main idea is to help the user choose a combination of window function and hop size, which satisfy the constant-overlap-add (COLA) condition, i.e., 
if the processing does not modify the blocks, the act of segmenting and un-segmenting the input audio data should be perfectly reconstructing (with some potential latency introduced by the system).

The library currently supports two different modes of operation
 - "OLA": Overlap-add, where a rectangular window is applied to the input frames, and the specified window is applied to the output frames prior to reconstruction. This mode is intended for block-based processing in the time-domain, where the purposed of the overlapping windows is to interpolate the discontinuities between adjacent frames prior to reconstruction.
 - "WOLA": Weighted overlap-add, where a square-root (COLA)-window is applied to both the input frame and output frame. This mode is intended for processing in the frequency domain along the lines of Short-time Fourier Transform (STFT) processing.

The primary use-case for the library is to support machine learning tasks, which has led to a number of options which are designed to ease training tasks.
 - The default choice `edge_compensation == True`, means that the first and last window in both segmenter and un-segmenter are modified to compensate for windows outside the input data overlapping into the chunk of data being considered by the segmenter. Without this compensation, there would be a loss of energy at the beginning and end of the segmented data.
 - The segmenter is implemented in both TensorFlow and PyTorch to support multiple machine learning tasks. We also provide a C++ implementation that is identical to the pytorch and tensorflow ones that eases model deployemnet.

## Installation

## Use
Before use, the user will have to create a segmenter object using the `libsegmenter.make_segmenter()` function. The `make_segmenter()` function takes the following input arguments:
 - `backend`: `torch`
 - `frame_size`:
 - `hop_size`:
 - `window`: 
 - `mode`: (`ola`), `wola`. Choose either overlap-add (rectangular window applied at segmentation, and chosen window at unsegmentation), or windowed overlap-add (square root of chosen window applied at segmentation and unsegmentation).
 - `edge_correction`: (`True`), `False`.
 - `gain_normalization`: (`True`), `False`.
The segmenter will accept input audio signals of the shape `[number_of_batch_elements, number_of_samples]` where `number_of_batch_elements` are individual channels or audio files and `number_of_samples` are the number of samples in each batch element. The output of the segmenter is of shape `[number_of_batch_elements, number_of_segments, frame_size]`, where `number_of_segments` will ignore the remaining samples if the `number_of_samples` does not match an integer number of segments with the chosen overlap.
