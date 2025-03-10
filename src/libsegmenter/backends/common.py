# Copyright (c) 2025 Niels de Koeijer, Martin Bo Møller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def compute_num_segments(num_samples: int, hop_size: int, segment_size: int) -> int:
    """
    Compute how many segments can be derived.

    Args:
        num_samples (int): Total samples in the dataset.
        hop_size (int): The step size for segment shifting.
        segment_size (int): Number of samples in one segment.

    Returns:
        int: Total number of segments.

    """
    if segment_size % hop_size > 0:
        return (num_samples // hop_size) - (segment_size // hop_size)
    else:
        return (num_samples // hop_size) - (segment_size // hop_size) + 1


def compute_num_samples(num_segments: int, hop_size: int, segment_size: int) -> int:
    """
    Compute the total number of samples from the number of segments.

    Args:
        num_segments (int): Total segments.
        hop_size (int): The step size for segment shifting.
        segment_size (int): Number of samples in one segment.

    Returns:
        int: Total number of samples.

    """
    return (num_segments - 1) * hop_size + segment_size
