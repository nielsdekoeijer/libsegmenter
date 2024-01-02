import numpy as np

def bartlett(windowLength):
    # Bartlett (triangular) window
    M = np.float32(windowLength + 1.0)
    m = np.arange(-(M-1)/2.0, (M-1)/2.0, dtype=np.float32)
    window = 1 - abs(m)*2.0/(M-1)
    return window

def blackman(windowLength):
    # Provides COLA-compliant windows for hopSize = (M-1)/3 when M is odd and hopSize M/3 when M is even
    M = np.float32(windowLength + 1.0)
    m = np.arange(0,M-1, dtype=np.float32)/(M-1)
    window = 0.42 - 0.5*np.cos(2.0*np.pi*m) + 0.08*np.cos(4.0*np.pi*m)
    return window

def hamming(windowLength):
    # Provides COLA-compliant windows for hopSize = M/2, M/4, ...
    M = np.float32(windowLength)
    alpha = 25.0/46.0
    beta = (1-alpha)/2.0
    window = alpha - 2*beta*np.cos(2.0*np.pi*np.arange(0,windowLength, dtype=np.float32)/windowLength)
    return window

def hann(windowLength):
    # Provides COLA-compliant windows for hopSize = windowLength/2, windowLength/4, ...
    M = np.float32(windowLength)
    m = np.arange(0,M, dtype=np.float32)
    window = 0.5*(1.0 - np.cos(2.0*np.pi*m/M))
    return window

def kaiser(windowLength, beta):
    # Note that the Kaiser window is not strictly speaking COLA compliant as it does not have harmonic nulls that can be tuned to the 
    # harmonics of the frame rate. As such it only offers approximate perfect reconstruction. However, by tuning the beta parameter, 
    # the reconstruction error can be controlled
    M = np.float32(windowLength + 1.0)
    m = np.arange(-(M-1)/2.0, (M-1)/2.0, dtype=np.float32)
    window = np.i0(beta*np.sqrt(1 - (m/(M/2))**2.0)) / np.i0(beta)
    return window

def defaultWindowSelector(windowName, windowLength):
    match windowName:
        case "bartlett50":
            # Bartlett (triangular) window with 50% overlap
            if (int(windowLength)%2 != 0):
                raise ValueError("Odd length Bartlett window with 50 percent overlap is not currently supported.")
            else:
                window = bartlett(windowLength)
                hopSize = int(windowLength)//2
                return window, hopSize
        case "bartlett75":
            # Bartlett window with 75% overlap
            if (int(windowLength)%4 != 0):
                raise ValueError("Bartlett windows with 75 percent overlap expects a windowLength divisible by 4.")
            else:
                window = bartlett(windowLength)
                hopSize = int(windowLength)//4
                return window, hopSize
        case "blackman":
            # Blackman window with 2/3 overlap
            if (int(windowLength)%3 != 0):
                raise ValueError("The Blackman window currently only supports overlaps of 2/3.")
            else:
                window = blackman(windowLength)
                hopSize = int(windowLength)//3
                return window, hopSize
        case "kaiser82":
            # Kaiser window with beta = 8 and approx 82% overlap
            beta = 8.0
            window = kaiser(windowLength, beta)
            hopSize = int(np.floor(1.7*(np.float32(windowLength)-1.0)/(beta + 1.0)))
            return window, hopSize
        case "kaiser85":
            # Kaiser window with beta = 10 and approx 85% overlap
            beta = 10.0
            window = kaiser(windowLength, beta)
            hopSize = int(np.floor(1.7*(np.float32(windowLength)-1.0)/(beta + 1.0)))
            return window, hopSize
        case "hamming50":
            # Hamming window with 50% overlap
            if (windowLength%2 != 0):
                raise ValueError("Odd length Hamming window at 50 percent overlap is not currently supported.")
            else:
                window = hamming(windowLength)
                hopSize = int(windowLength)//2
                return window, hopSize
        case "hamming75":
            # Hamming window with 75% overlap
            if (int(windowLength)%4 != 0):
                raise ValueError("For Hamming windows with 75 percent overlay, the windowLength is expected to be divisible by 4.")
            else:
                window = hamming(windowLength)
                hopSize = int(windowLength)//4
                return window, hopSize
        case "hann50":
            # Hann window with 50% overlap
            if (int(windowLength)%2 != 0):
                raise ValueError("Odd length Hann window at 50 percent overlap is not currently supported.")
            else:
                window = hann(windowLength)
                hopSize = int(windowLength)//2
                return window, hopSize
        case "hann75":
            # Hann window with 75% overlap
            if (int(windowLength)%4 != 0):
                raise ValueError("For Hann windows with 75 percent overlap, the windowLength is expected to be divisible by 4.")
            else:
                window = hann(windowLength)
                hopSize = int(windowLength)//4
                return window, hopSize
        case "rectangular0":
            # Rectangular window with 0% overlap
            window = np.ones(windowLength, dtype=np.float32)
            hopSize = int(windowLength)
            return window, hopSize
        case "rectangular50":
            # Rectangular window with 50% overlap
            window = np.ones(windowLength, dtype=np.float32)
            hopSize = int(windowLength)//2
        case _:
            raise ValueError("No valid windowName was provided.")