function [window, hopSize] = hann75(segmentSize)
    % Generates a Hann window of the given size with 75% overlap.
    %
    % Args:
    %   segmentSize (double): Size of the window to be created.
    %
    % Returns:
    %   A Hann window with 75% overlap
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received size(segmentSize) = ' int2str(size(segmentSize))] );
    end
    if mod(segmentSize,4)
        error(['Segment size must be integer divisible by 4, got ' int2str(segmentSize) ]);
    end
    indices = (0:segmentSize-1)';
    window = hannWindow(segmentSize, indices);
    hopSize = segmentSize/4;
end
