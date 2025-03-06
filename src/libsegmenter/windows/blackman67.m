function [window, hopSize] = blackman67(segmentSize)
    % Generates a Blackman window of the given size with 2/3 overlap.
    %
    % Args:
    %   segmentSize (double): Size of the window to be created.
    %
    % Returns:
    %   A Blackman window with 2/3 overlap
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received size(segmentSize) = ' int2str(size(segmentSize))] );
    end
    if mod(segmentSize,3)
        error(['Segment size must be integer divisible by 3, got ' int2str(segmentSize) ]);
    end
    indices = (0:segmentSize-1)';
    window = blackmanWindow(segmentSize, indices);
    hopSize = segmentSize/3;
end
