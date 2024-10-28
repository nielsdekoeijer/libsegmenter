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
Simply install from PyPi:
```bash
python3 -m pip install libsegmenter
```
Or build from source
```bash
python3 -m pip install .
```

## Development
We have tests validating parity between pytorch, tensorflow, and C++ backends. Run them with:
```bash
pip install -e .  # build the python bindings in project root (NOTE: also run on changing C++)
python3 -m pytest # run in project root!
```
Please note again that rebuilding the python bindings locally in the root as given above is neccessary when developing.
This is because the development environment (from which `pytest` is run) uses the `libsegmenter` folder in the project root.
This folder will not contain the bindings by default, and they will not be updated on `pip install .`. 
As such, you are advised to rerun `pip install -e .` whilst developing to ensure the changes are correctly reflected in the unit tests.

## Use
### Example
The general workflow is as follows:
```python
# utilize the window helper to create a window of 512 samples
window, hop_size = libsegmenter.default_window_selector(
    "hann75",
    512,
)

# create the segmenter object
self.segmenter = libsegmenter.make_segmenter(
    backend="torch",
    frame_size=512,
    hop_size=hop_size,
    window=window,
    mode="wola",
    edge_correction=False,
    normalize_window=True
)

# segment input: (batch, nsamples) => (batch, ntotalframes, nframe)
X = segmenter.segment(x)
```
### In Detail
Before use, the user will have to create a segmenter object using the `libsegmenter.make_segmenter()` function. The `make_segmenter()` function takes the following input arguments:
 - `backend`: (`"base"`), `"torch"`, `"tensorflow"`. The choise of backend for the segmentation `"base"` is a c++ implementation with python bindings, `"torch"` for pytorch, and `"tensorflow"` for a tensor flow implementation.
 - `frame_size`: The length of each segment in samples.
 - `hop_size`: The number of samples the window is displaced between adjacent segments.
 - `window`: An array specifying the window function. Note this should have the length `frame_size` and be COLA-compliant with the chosen `hop_size`.
 - `mode`: (`"ola"`), `"wola"`. Choose either overlap-add (rectangular window applied at segmentation, and chosen window at unsegmentation), or windowed overlap-add (square root of chosen window applied at segmentation and unsegmentation).
 - `normalize_window`: (`True`), `False`. 
 - `edge_correction`: (`True`), `False`.

Note that upon creation, it will be checked whether the chosen window and hop_size are COLA compliant and will throw an error if this is not the case (if the choice of window and hop_size would lead to errors even if the audio is not changed between being segmented and unsegmented). To ease the use of the library the function `default_window_selector(window_name, window_length)` is introduced. This function will return a window of the requested length and supply a valid hop_size (if this is available for the chosen window length). The function currently supports the following default window types:
 - `"bartlett50"`: Triangular window with 50% overlap.
 - `"bartlett75"`: Triangular window with 75% overlap.
 - `"blackman"`: Blackman window with 2/3 ovelap.
 - `"hamming50"`: Hamming window with 50% overlap.
 - `"hamming75"`: Hamming windown with 75% overlap.
 - `"hann50"`: Hann window with 50% overlap.
 - `"hann75"`: Hann window with 75% overlap.
 - `"rectangular0"`: Rectangular window with 0% overlap.
 - `"rectangular50"`: Rectangular window with 50% overlap.
Please note that the choice of a suitable window for a given application is entirely left to the user.

The segmenter will accept input audio signals of the shape `[number_of_batch_elements, number_of_samples]` where `number_of_batch_elements` are individual channels or audio files and `number_of_samples` are the number of samples in each batch element. The output of the segmenter is of shape `[number_of_batch_elements, number_of_segments, frame_size]`, where `number_of_segments` will ignore the remaining samples if the `number_of_samples` does not match an integer number of segments with the chosen overlap.
