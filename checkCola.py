import numpy as np
import matplotlib.pyplot as plt

def checkCola(window, hopSize):
    windowLength = window.size
    frameRate = 1./np.float64(hopSize) # Assuming the sampling frequency is 1
    N = 6*windowLength
    sp = np.sum(window)/np.float64(hopSize) * np.ones(N, dtype=np.float64)
    ubound = sp[0]*1.0
    lbound = sp[0]*1.0
    n = np.arange(0,N, dtype=np.float64)

    plotLength = 12*windowLength
    windowSpectrum = np.fft.fft(window, plotLength)
    windowSpectrum = windowSpectrum / np.abs(windowSpectrum[0])
    normFrequency = np.arange(0, plotLength, dtype=np.float64) / np.float64(plotLength)
    plotLim = np.ones(2, dtype=np.float64)
    plotLim[0] = -80.0
    plotLim[1] = 10.0
    fig, ax = plt.subplots()
    ax.plot(normFrequency, 20*np.log10(np.abs(windowSpectrum)))

    ax.plot(1.0/np.float64(2.0*hopSize)*np.ones(2, dtype=np.float64), plotLim, dashes=[6, 2])


    for k in range(1,hopSize):
        f = frameRate*k
        csin = np.exp(1j*2.0*np.pi*np.float64(f)*n)
        # Find exact window transform at frequency f
        Wf = np.sum(window * np.conj(csin[0:windowLength]))
        hum = Wf*csin # contribution to OLA "hum"
        sp = sp + hum/np.float64(hopSize) # Poisson summation into OLA
        # Update lower and upper bounds
        Wfb = np.abs(Wf)
        ubound = ubound + Wfb/np.float64(hopSize)
        lbound = lbound - Wfb/np.float64(hopSize)
        ax.plot(np.float64(k)/np.float64(hopSize)*np.ones(2, dtype=np.float64), plotLim, dashes=[6, 2], color='k')

    ax.set(xlabel='Normalized frequency', ylabel='Magnitude [dB]')
    ax.set_ylim(plotLim[0], plotLim[1])
    ax.set_xlim(0, 0.5)
    ax.grid()
    fig.savefig("test.png")
    print(20*np.log10(ubound-lbound))
    if (ubound-lbound) < 1e-5:
        return True
    else:
        return False
    

