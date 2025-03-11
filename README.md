# The `libsegmenter` audio segmentation library
A small library intended to provide helper functions for block-based processing 
in Python. 

Find out more by exploring the code or reading 
[the docs](https://nielsdekoeijer.github.io/libsegmenter/).

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

## A word of caution
Note that segmentation is a **destructive operation** in the sense that we do not 
provide any pre and post windows. This means that the first and last couple of samples
of your audio post subsequently `segment`-ing and `unsegment`-ing are going to be 
windowed, thus different than what you started out with. 
This is something to take into account when training.

## Installation
Simply install from PyPi:
```bash
# base version
pip install libsegmenter

# with torch
pip install libsegmenter[torch]

# with tensorflow
pip install libsegmenter[tensorflow]
```

## Example
To make a segmenter with a specific window:
```python
import libsegmenter as seg
segmenter = seg.Segmenter(seg.WindowSelector("hann75", "ola", 1024), backend="torch")
```

With an asymetric window:
```python
import libsegmenter as seg
segmenter = seg.Segmenter(seg.AsymmetricWindowSelector("ola", 1024, 128, 2048), backend="torch")
```

Use various supported transforms:
```python
import libsegmenter as seg
segmenter = seg.Segmenter(seg.WindowSelector("hann75", "ola", 1024), backend="torch")
transform = seg.TransformSelector(transform="spectrogram", backend="torch")
X = transform.forward(segmenter.segment(x))
x = transform.inverse(x)
```

## Development
### Installing python
Install `uv` (pip replacement):
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

### Linting
We require everything to be fully typed. We enforce that by having 100% clearance on pyright:
```bash
uv run pyright
uv run ruff check
uv run ruff format
```

### Licenses
The project is licensed under MIT.
Add licenses using the `addlicense` tool found [here](https://github.com/google/addlicense):
```bash
addlicense -c "Niels de Koeijer, Martin Bo Møller" -l mit -y 2025 -ignore *.m
```

### Documentation
Docs are mainly automatically generated and described with docstrings.
To host the docs locally run:
```bash
mkdocs serve
```
They are automatically rebuilt on push to main.
