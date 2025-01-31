classdef segmenter < handle
    %SEGMENTER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        windowObj = {};
    end
    
    methods
        function obj = segmenter(windowObj)
            arguments (Input)
                windowObj (1,1) windowObject
            end
            arguments (Output)
                obj (1,1) segmenter
            end
            obj.windowObj = windowObj;
        end
        
        function output = segment(obj, input)
            % Divide input into windowed segments according to the settings
            % in obj.windowObj
            % 
            % size(input) = [numberOfBatchElements, numberOfSamples]
            %
            % size(output) = [numberOfBatchElements, numberOfSegments,
            % segmentSize]
            arguments (Input)
                obj (1,1) segmenter
                input (:,:) double
            end
            arguments (Output)
                output (:,:) double
            end
            if isscalar(size(input))
                numberOfBatches = 1;
                numberOfSamples = length(input);
            else
                numberOfBatches = size(input, 1);
                numberOfSamples = size(input, 2);
            end
            numberOfSegments = floor(numberOfSamples / obj.windowObj.hopSize) - floor(obj.windowObj.segmentSize / obj.windowObj.hopSize) + 1;
            output = zeros(numberOfBatches, numberOfSegments, obj.windowObj.segmentSize);
            for bIdx = 0:numberOfBatches-1
                for sIdx = 0:numberOfSegments-1
                    output(bIdx+1,sIdx+1,:) = input(bIdx+1,sIdx*obj.windowObj.hopSize + 1 : sIdx*obj.windowObj.hopSize + obj.windowObj.segmentSize) .* obj.windowObj.analysisWindow';
                end 
            end
        end

        function output = unsegment(obj, input)
            arguments (Input)
                obj (1,1) segmenter
                input (:,:,:) double
            end
            arguments (Output)
                output (:,1) double
            end
            dim = length(size(input));
            if dim == 3
                numberOfBatchElements = size(input,1);
                numberOfSegments = size(input,2);
                checkSegmentSize = size(input,3);
            elseif dim == 2
                numberOfBatchElements = 1;
                numberOfSegments = size(input, 1);
                checkSegmentSize = size(input, 2);
            end
            if checkSegmentSize ~= obj.windowObj.segmentSize
                error('Input error: segmentSize of input data does not match the segmentSize of the segmenter.')
            end

            if isempty(obj.windowObj.synthesisWindow)
                error('The synthesisWindow is empty. This indicates that a reconstructionScheme = analysisOnly was chosen during construction, which does not support unSegment (synthesis) operation.');
            end
            numberOfSamples = numberOfSegments * obj.windowObj.hopSize + obj.windowObj.segmentSize - obj.windowObj.hopSize;
            output = zeros(numberOfBatchElements, numberOfSamples);
            if dim == 2
                for sIdx = 0:numberOfSegments-1
                    output(sIdx*obj.windowObj.hopSize + 1 : sIdx*obj.windowObj.hopSize + obj.windowObj.segmentSize) = ...
                        output(sIdx*obj.windowObj.hopSize + 1 : sIdx*obj.windowObj.hopSize + obj.windowObj.segmentSize) + ...
                        input(sIdx+1, :) .* obj.windowObj.synthesisWindow';
                end 
            else
                for bIdx = 0:numberOfBatchElements
                    for sIdx = 0:numberOfSegments-1
                        output(bIdx+1, sIdx*obj.windowObj.hopSize + 1 : sIdx*obj.windowObj.hopSize + obj.windowObj.segmentSize) = ...
                            output(bIdx+1, sIdx*obj.windowObj.hopSize + 1 : sIdx*obj.windowObj.hopSize + obj.windowObj.segmentSize) + ...
                            input(bIdx+1, sIdx+1, :) .* obj.windowObj.synthesisWindow';
                    end 
                end
            end
        end
    end
end

