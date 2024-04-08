import matplotlib.pyplot as plt
import numpy as np
import checkCola as cola
import defaultWindowSelector as winSelector
import segmenterTensorFlow as segmenter
import tensorflow as tf

windowSize = 32
val = winSelector.defaultWindowSelector("hann75",windowSize)
window = val[0]
hopSize = val[1]
window = np.float32(window)


signalLength = 4*windowSize
batchSize = 3
input = tf.ones(shape=(batchSize, signalLength))

testSegmenter = segmenter.Segmenter(windowSize, hopSize, window, "wola")
frames = testSegmenter.segment(input)
output = testSegmenter.unsegment(frames)

bIdx = 2
fig, ax = plt.subplots()
ax.plot(input[bIdx,:].numpy())
ax.plot(output[bIdx,:].numpy())
ax.grid()
fig.savefig("test.png")


