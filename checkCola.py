import numpy as np
#import matplotlib as plt

def checkCola(window, hopSize):
    windowLength = window.size
    frameRate = 1/np.float32(hopSize) # Assuming the sampling frequency is 1
    N = 6*windowLength
    sp = np.sum(window)/np.float32(hopSize) * np.ones(N, dtype=np.float32)
    ubound = sp[0]*1.0
    lbound = sp[0]*1.0
    n = np.arange(0,N, dtype=np.float32)
    for k in range(1,hopSize):
        f = frameRate*k
        csin = np.exp(1j*2.0*np.pi*np.float32(f)*n)
        # Find exact window transform at frequency f
        Wf = np.sum(window * np.conj(csin[0:windowLength]))
        hum = Wf*csin # contribution to OLA "hum"
        sp = sp + hum/np.float32(hopSize) # Poisson summation into OLA
        # Update lower and upper bounds
        Wfb = np.abs(Wf)
        ubound = ubound + Wfb/np.float32(hopSize)
        lbound = lbound - Wfb/np.float32(hopSize)

    if (ubound-lbound) < 1e-5:
        return True
    else:
        return False
    

