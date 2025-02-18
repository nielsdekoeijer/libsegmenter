# Segmenter
A small library intended to provide helper functions for block-based processing 
in Python. 

Find out more by exploring the code or reading [the docs](https://nielsdekoeijer.github.io/libsegmenter/).

## About
The main idea is to help the user choose a combination of window function and 
hop size, which satisfy the constant-overlap-add (COLA) condition, i.e., 
if the processing does not modify the blocks, the act of segmenting and 
un-segmenting the input audio data should be perfectly reconstructing 
(with some potential latency introduced by the system).

The library currently supports three  different modes of operation

- Overlap-Add (`ola`), where a rectangular window is applied to the input 
    frames, and the specified window is applied to the output frames prior to 
    reconstruction. This mode is intended for block-based processing in the 
    time-domain, where the purposed of the overlapping windows is to 
    interpolate the discontinuities between adjacent frames prior to 
    reconstruction.

- Weighted Overlap-Add (`wola`), where a square-root (COLA)-window is applied 
    to both the input frame and output frame. This mode is intended for 
    processing in the frequency domain along the lines of Short-time Fourier 
    Transform (STFT) processing.

- Analysis (`analysis`), where a window is applied to the input frames and
    disables computing output frames. Useful to obtain spectrograms.

The primary use-case for the library is to support machine learning tasks, 
which has led to a number of options which are designed to ease training tasks.
The segmenter is implemented in both TensorFlow and PyTorch to support multiple 
machine learning tasks. 

Recently, we have upgraded the library to version 1.0. This deprecated the 
C++ backend for now to simplify development. That being said, the general design
has been simplified so implementing your own backend (and verifying it with our
unit tests) should not be infeasible.

## Installation
Simply install from PyPi:
```bash
pip install libsegmenter
```

## Example
TODO

## Development
Install uv (pip replacement):
```bash
# install for linux / mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# install for windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install the development packages:
```bash
uv venv
source .venv/bin/activate
uv sync --dev
```

Add licenses:
```bash
addlicense -c "Niels de Koeijer, Martin Bo Møller" -l mit -y 2025 .
```

Serve docs locally:
```bash
mkdocs serve
```
