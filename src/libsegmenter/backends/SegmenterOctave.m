classdef SegmenterOctave < handle
    %SEGMENTEROCTAVE
    % A class for segmenting and reconstructing input data using windowing techniques.
    %
    % Supports Weighted Overlap-Add (WOLA) and Overlap-Add (OLA) methods.
    %
    % Attributes:
    %   window (Window): A class containing hop size, analysis and synthesis windows.

    properties
        window = {};
    end

    methods
        function obj = SegmenterOctave(window)
            % Initializes the Segmenter instance.
            %
            % Args:
            %   window (Window): A window object containing segmentation parameters.
            obj.window = window;
        end

        function output = segment(obj, input)
            % Segments the input signal into overlapping windows using the window parameters.
            %
            % Args:
            %   input (2D array): Input array, either 1D (sequence) or 2D (batch).
            %
            % Returns:
            %   Segmented data of size [batchSize, numSegments, segmentSize]
            %
            % Throws:
            %   error: If input dimensions are invalid
            if length(size(input)) > 2
                error(['The input dimensions cannot be larger than 2. Received length(size(input)) = ' int2str(length(size(input))) ]);
            end
            if numel(input) == length(input)
                if size(input,1) ~= 1
                    error(['The expected input dimension is 1 x numberOfSamples, received size(input) = ' int2str(size(input))]);
                end
                batchSize = 1;
                numSamples = length(input);
            else
                batchSize = size(input, 1);
                numSamples = size(input, 2);
            end
            numSegments = floor(numSamples / obj.window.hopSize) - ceil(obj.window.segmentSize / obj.window.hopSize) + 1;
            if numSegments <= 0
                error(['Input signal is too short for segmentation with the given hop size (' int2str(obj.window.hopSize) ') and segmentSize (' int2str(obj.window.segmentSize) ')']);
            end
            output = zeros(batchSize, numSegments, obj.window.segmentSize);
            for bIdx = 0:batchSize-1
                for sIdx = 0:numSegments-1
                    output(bIdx+1,sIdx+1,:) = input(bIdx+1,sIdx*obj.window.hopSize + 1 : sIdx*obj.window.hopSize + obj.window.segmentSize) .* obj.window.analysisWindow';
                end
            end
            if batchSize == 1
              output = squeeze(output);
            end
        end

        function output = unsegment(obj, input)
            % Reconstructs the original signal from segmented data using synthesis windowing.
            %
            % Args:
            %   input (3D array): Segmented data with size [batchSize, numSegments, segmentSize] or [numSegments, segmentSize] for a single sequence.
            %
            % Returns:
            %   Reconstructed signal.
            if length(size(input)) > 3 || length(size(input)) < 2
                error(['The input dimensions must be between 2 and 3. Received length(size(input)) = ' int2str(length(size(input))) ]);
            end
            dim = length(size(input));
            if dim == 3
                batchSize = size(input,1);
                numSegments = size(input,2);
                checkSegmentSize = size(input,3);
            elseif dim == 2
                batchSize = 1;
                numSegments = size(input, 1);
                checkSegmentSize = size(input, 2);
            end
            if checkSegmentSize ~= obj.window.segmentSize
                error('Input error: segmentSize of input data does not match the segmentSize of the segmenter.')
            end

            if isempty(obj.window.synthesisWindow)
                error('Given windowing scheme does not support unsegmenting.');
            end
            numSamples = numSegments * obj.window.hopSize + obj.window.segmentSize - obj.window.hopSize;
            output = zeros(batchSize, numSamples);
            if dim == 2
                for sIdx = 0:numSegments-1
                    output(sIdx*obj.window.hopSize + 1 : sIdx*obj.window.hopSize + obj.window.segmentSize) = ...
                        output(sIdx*obj.window.hopSize + 1 : sIdx*obj.window.hopSize + obj.window.segmentSize) + ...
                        input(sIdx+1, :) .* obj.window.synthesisWindow';
                end
            else
                for bIdx = 0:batchSize-1
                    for sIdx = 0:numSegments-1
                        output(bIdx+1, sIdx*obj.window.hopSize + 1 : sIdx*obj.window.hopSize + obj.window.segmentSize) = ...
                            output(bIdx+1, sIdx*obj.window.hopSize + 1 : sIdx*obj.window.hopSize + obj.window.segmentSize) + ...
                            reshape(input(bIdx+1, sIdx+1, :), 1, obj.window.segmentSize) .* obj.window.synthesisWindow';
                    end
                end
            end
        end
    end
end


