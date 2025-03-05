function [window, hopSize] = hann50(segmentSize)
    % Generates a Hann window of the given size with 50% overlap.
    %
    % Args:
    %   segmentSize (double): Size of the window to be created.
    %
    % Returns:
    %   A Hann window with 50% overlap
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received size(segmentSize) = ' int2str(size(segmentSize)) ]);
    end
    if mod(segmentSize,2)
        error(['Segment size must be even, got ' int2str(segmentSize) ]);
    end
    indices = (0:segmentSize-1)';
    window = hannWindow(segmentSize, indices);
    hopSize = segmentSize/2;
end
