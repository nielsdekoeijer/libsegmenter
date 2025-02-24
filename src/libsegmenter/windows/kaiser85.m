function [window, hopSize] = kaiser85(segmentSize)
    % Generates a Kaiser window of the given size with approx 85% overlap.
    %
    % Args:
    %   segmentSize (double): Size of the window to be created.
    %
    % Returns:
    %   A Kaiser window with approx 85% overlap
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received size(segmentSize) = ' int2str(size(segmentSize))] );
    end
    indices = (0:segmentSize-1)';
    beta = 10;
    window = _kaiser(segmentSize, beta, indices);
    hopSize = floor(1.7*segmentSize/(beta+1));
end
