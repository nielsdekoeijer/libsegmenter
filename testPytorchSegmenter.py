import matplotlib.pyplot as plt
import numpy as np
import checkCola as cola
import defaultWindowSelector as winSelector
import segmenter
import torch

window_size = 32
val = winSelector.defaultWindowSelector("hann75", window_size)
window = torch.from_numpy(val[0])
hop_size = val[1]

signal_length = 4*window_size
batch_size = 3
input = torch.ones((batch_size, signal_length))

test_segmenter = segmenter.Segmenter(window_size, hop_size, window, "wola", False, True)
frames = test_segmenter.segment(input)
output = test_segmenter.unsegment(frames)

b_idx = 2
fig, ax = plt.subplots()
ax.plot(input[b_idx,:].numpy())
ax.plot(output[b_idx,:].numpy())
ax.grid()
fig.savefig("testTorch.png")
