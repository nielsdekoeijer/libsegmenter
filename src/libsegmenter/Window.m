classdef Window < handle
    %WINDOW
    % A class representing a windowing scheme used in signal segmentation.
    %
    % Attributes:
    %   segmentSize (double): The size of the segment / windows.
    %   hopSize (double): The step size for shifting the window in the segmentation process.
    %   analysisWindow (1D array): The window function used during the analysis phase.
    %   synthesisWindow (1D array): The window function used during the synthesis phase.
    properties
        segmentSize = 0;
        hopSize = 0;
        analysisWindow = [];
        synthesisWindow = [];
    end

    methods
        function obj = Window(hopSize, analysisWindow, synthesisWindow)
            % Initializes the Window instance with the specified segment size, hop size, and windows.
            %
            % Args:
            %   hopSize (double): The step size for shifting the window in the segmentation process.
            %   analysisWindow (1D array): The window function applied during analysis.
            %   synthesisWindow (1D array): The window function applied during synthesis.
            if length(hopSize) ~= 1
                error(['Hop size must be a scalar. Received ' int2str(hopSize)]);
            end
            if size(analysisWindow,1) ~= prod(size(analysisWindow))
                error(['Analysis window must be a column vector. Received size(analysisWindow) = ' int2str(size(analysisWindow)) ]);
            end
            if size(synthesisWindow,1) ~= prod(size(synthesisWindow))
                error(['Synthesis window must be a column vector. Received size(synthesisWindow) = ' int2str(size(synthesisWindow)) ]);
            end
            if ~isempty(synthesisWindow)
                if length(synthesisWindow) ~= length(analysisWindow)
                    error(['The analysis window length (' int2str(length(analysisWindow)) ') is not equal the the length of the synthesis window (' int2str(length(synthesisWindow)) ')' ]);
                end
            end
            segmentSize = length(analysisWindow);
            if hopSize > segmentSize || hopSize <= 0
                error(['The hop size (' int2str(hopSize) ') must be a value between 1 and the segment size (' int2str(segmentSize) ')']);
            end
            obj.segmentSize = segmentSize;
            obj.hopSize = hopSize;
            obj.analysisWindow = analysisWindow;
            obj.synthesisWindow = synthesisWindow;
        end
    end
end
