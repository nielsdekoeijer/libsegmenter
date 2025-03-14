function window = blackmanWindow(windowSize, indices)
    if min(indices) < 0
        error(['The smallest index cannot be lower than 0. Received min(indices) = ' int2str(min(indices))] );
    end
    if max(indices) > windowSize-1
        error(['The largest index cannot be larger than window size - 1. Received max(indices) = ' int2str(max(indices)) ]);
    end
    window = 0.42 - 0.5 * cos(2*pi* indices/windowSize) + 0.08*cos(4*pi*indices/windowSize);
end
