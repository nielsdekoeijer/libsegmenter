function windowObj = WindowSelector(windowType, scheme, segmentSize)
    % Selects and returns a specific window function based on the given parameters.
    %
    % This function retrieves a window function based on the `windowType`, applies and adaptation based on `scheme`, and returns the corresponding `Window` object.
    %
    % Args:
    %   windowType (str): The type of window function to apply. Supported values include:
    %       [
    %        `bartlett50`,
    %        `bartlett75`,
    %        `blackman67`,
    %        `kaiser85`,
    %        `hamming50`,
    %        `hamming75`,
    %        `hann50`,
    %        `hann75`,
    %        `rectangular0`,
    %        `rectangular50`
    %       ]
    %   scheme (str): The adaptation scheme to use. Supported values:
    %       [
    %        `analysis`,
    %        `ola`,
    %        `wola`
    %       ]
    %   segmentSize (double): The size of the segment/window.
    %
    % Returns:
    %   window: A `Window` object containing the selected window function and its corresponding hop size.
    %
    % Throws:
    %   error: If an unknown window type or scheme is provided.
    if ~ischar(windowType)
        error('The windowType argument must be a string.');
    end
    if ~ischar(scheme)
        error('The desired scheme argument must be a string.');
    end
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received ' int2str(segmentSize)]);
    end
    switch windowType
        case 'bartlett50'
            [window, hopSize] = bartlett50(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'bartlett75'
            [window, hopSize] = bartlett75(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'blackman67'
            [window, hopSize] = blackman67(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'kaiser85'
            [window, hopSize] = kaiser85(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'hamming50'
            [window, hopSize] = hamming50(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'hamming75'
            [window, hopSize] = hamming75(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'hann50'
            [window, hopSize] = hann50(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'hann75'
            [window, hopSize] = hann75(segmentSize);
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'rectangular0'
            window = ones(segmentSize, 1);
            hopSize = segmentSize;
            windowObj = adaptWindow(window, hopSize, scheme);
        case 'rectangular50'
            window = ones(segmentSize,1);
            hopSize = segmentSize/2;
            windowObj = adaptWindow(window, hopSize, scheme);
        otherwise
            error(['The windowType (' windowType ') is not know']);
    end
end

function window = adaptWindow(window, hopSize, scheme)
    if size(window,1) ~= prod(size(window))
        error('The input window must be a column vector.');
    end
    if length(hopSize) ~= 1
        error(['The hop size must be a scalar. Received ' int2str(hopSize)]);
    end
    if ~ischar(scheme)
        error('The scheme argument must be a string.');
    end
    [isCola, normalization] = checkCola(window, hopSize);
    if ~isCola
        error('Specified window and hop size failed cola check');
    end
    window = window/normalization;
    segmentSize = length(window);
    switch scheme
        case 'analysis'
            window = Window(hopSize, window, []);
        case 'ola'
            window = Window(hopSize, ones(segmentSize,1), window);
        case 'wola'
            window = Window(hopSize, sqrt(window), sqrt(window));
        otherwise
            error(['The scheme ' scheme ' is not supported.']);
    end
end
