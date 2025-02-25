function windowObj = AsymmetricWindowSelector(scheme, segmentSize, hopSize, synthesisSegmentSize)
    % Creates asymmetric Hann windows for analysis and synthesis based on the given parameters.
    %
    % This function retrieves a window function based on the `windowType`, applies and adaptation based on `scheme`, and returns the corresponding `Window` object.
    %
    % Args:
    %   scheme (str): The adaptation scheme to use. Supported values:
    %       [
    %        `analysis`,
    %        `ola`,
    %        `wola`
    %       ]
    %   segmentSize (double): The size of the segment / analysisWindow.
    %   hopSize (double): The hopSize used in the segmentation.
    %   synthesisSegmentSize (double): The non-zero part of the synthesisWindow.
    %
    % Returns:
    %   window: A `Window` object containing the selected window function and its corresponding hop size.
    %
    % Throws:
    %   error: If an unknown window type or scheme is provided.
##    if ~ischar(windowType)
##        error('The windowType argument must be a string.');
##    end
    if ~ischar(scheme)
        error('The desired scheme argument must be a string.');
    end
    if length(segmentSize) ~= 1
        error(['The segment size must be a scalar. Received ' int2str(segmentSize)]);
    end
    if length(hopSize) ~= 1
        error(['The hop size must be a scalar. Received ' int2str(hopSize)]);
    end
    if length(synthesisSegmentSize) ~= 1
        error(['The synthesis segment size must be scalar. Received ' int2str(synthesisSegmentSize)]);
    end
    if mod(segmentSize, hopSize) ~= 0
        error(['Segment size must be integer divisible with the hop size. Received segment size = ' int2str(segmentSize) ' and hop size = ' int2str(hopSize) ]);
    end
    if mod(synthesisSegmentSize, hopSize) ~= 0
        error(['Synthesis segment size must be integer divisible with the hop size. Received synthesis segment size = ' int2str(synthesisSegmentSize) ' and hop size = ' int2str(hopSize)]);
    end
    if segmentSize < synthesisSegmentSize
        error(['The synthesis segement size is expected to be smaller than the segment size. Received segment size = ' int2str(segmentSize) ' and synthesis segment size = ' int2str(synthesisSegmentSize) ]);
    end

    switch scheme
        case 'ola'
            analysisWindow = ones(segmentSize,1);
            f1 = zeros(segmentSize - synthesisSegmentSize, 1);
            f2 = _hann(synthesisSegmentSize);
            synthesisWindow = [f1; f2];
        case 'wola'
            M = synthesisSegmentSize/2;
            KM = segmentSize-M;
            h1 = sqrt(_hann(2*KM, 0:KM - M -1)).';
            h2 = sqrt(_hann(2*KM, (0:M-1) + KM - M)).';
            h3 = sqrt(_hann(2*M, (0:M-1) + M)).';
            analysisWindow = [h1; h2; h3];

            f1 = zeros(KM-M, 1);
            f2 = _hann(2*M, (0:M-1)).' ./ sqrt(_hann(2*KM, (0:M-1) + KM - M).');
            f3 = sqrt(_hann(2*M, (0:M-1) + M)).';
            synthesisWindow = [f1; f2; f3];
        otherwise
            error(['The scheme ' scheme ' is not supported.']);
    end
    window = analysisWindow .* synthesisWindow;
    [isCola, normalization] = checkCola(window, hopSize);
    if ~isCola
        error('Specified window and hop size failed cola check');
    end
    synthesisWindow = synthesisWindow/normalization;
    windowObj = Window(hopSize, analysisWindow, synthesisWindow);
end

