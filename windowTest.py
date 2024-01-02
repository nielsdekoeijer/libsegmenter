import checkCola as cola
import defaultWindowSelector as winSelector

val = winSelector.defaultWindowSelector("hann50",32)
window = val[0]
hopSize = val[1]

testFlag = cola.checkCola(window, hopSize)

print(window)
print(testFlag)