import checkCola as cola
import defaultWindowSelector as winSelector

val = winSelector.defaultWindowSelector("hamming50",32)
window = val[0]
hopSize = val[1]

colaVal = cola.checkCola(window, hopSize)
testFlag = colaVal[0]
normalizationValue = colaVal[1]

print(window)
print(testFlag)
print(normalizationValue)