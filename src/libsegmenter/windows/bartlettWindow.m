function window = bartlettWindow(windowSize, indices)
    if min(indices) < 0
        error(['The smallest index cannot be lower than 0. Received min(indices) = ' int2str(min(indices))] );
    end
    if max(indices) > windowSize-1
        error(['The largest index cannot be larger than window size - 1. Received max(indices) = ' int2str(max(indices)) ]);
    end
    m = indices - windowSize/2;
    window = 1 - abs(m)*2/windowSize;
end
