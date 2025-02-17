// Copyright (c) 2025 Niels de Koeijer, Martin Bo Møller
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

classdef windowObject < handle
    %WINDOWOBJECT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        segmentSize = 0;
        hopSize = 0;
        analysisWindow = [];
        synthesisWindow = [];
    end
    
    methods
        function obj = windowObject(segmentSize, windowScheme, reconstructionScheme, varargin)
            %WINDOWOBJECT Construct an instance of this class
            %   Detailed explanation goes here
            % arguments (Input)
            %     segmentSize (1,1) double
            %     windowScheme (1,1) string
            %     reconstructionScheme (1,1) string
            %     varargin
            % end
            arguments (Output)
                obj (1,1) windowObject
            end
            obj.segmentSize = segmentSize;

            if isempty(varargin)
                switch lower(windowScheme)
                    case 'hann50'
                        if mod(segmentSize, 2)
                            error('For Hann window with 50% overlap, the windowLength should be divisible by 2.');
                        end
                        window = obj.hannWindow(segmentSize);
                        obj.hopSize = segmentSize/2;
                        
                    case 'hann75'
                        if mod(segmentSize, 4)
                            error('For Hann window with 75% overlap, the windowLength should be divisible by 4.');
                        end
                        window = obj.hannWindow(segmentSize);
                        obj.hopSize = segmentSize/4;
                    
                    otherwise 
                        error(['Specified window scheme "' windowScheme '" is not currently supported.'])
                end
    
                % Normalize the window scheme
                [checkPassed, lowerBound, upperBound] = colaCheck(window, obj.hopSize);
                if ~checkPassed
                    error('COLA Condition not satisfied for the windowing scheme');
                end
                window = window/mean([lowerBound, upperBound]);
    
                switch lower(reconstructionScheme)
                    case 'analysisonly'
                        obj.analysisWindow = window;
                        obj.synthesisWindow = [];
                    case 'ola'
                        obj.analysisWindow = obj.rectangularWindow(segmentSize);
                        obj.synthesisWindow = window;
                    case 'wola'
                        obj.analysisWindow = sqrt(window);
                        obj.synthesisWindow = sqrt(window);
                    otherwise
                        error(['Specified reconstruction scheme "' reconstructionScheme '" is not currently supported.']);
                end
            else
                hopSize = varargin{1};
                synthesisSegmentSize = varargin{2};
                if mod(segmentSize, hopSize) ~= 0
                    error('SegmentSize is not integer divisible by hopSize');
                end
                if mod(synthesisSegmentSize, hopSize)
                    error('The synthesisSegmentSize is not integer divisible by the hopSize');
                end
                if segmentSize < synthesisSegmentSize
                    error('The synthesisSegmentSize is expected to be larger than the analysisSegmentSize');
                end
                obj.hopSize = hopSize;

                switch lower(reconstructionScheme)
                    case 'ola'
                        obj.analysisWindow = rectangularWindow(segmentSize);
                        f1 = zeros(segmentSize - synthesisSegmentSize,1);
                        f2 = hannWindow(synthesisSegmentSize);
                        window = [f1; f2];
                        [checkPassed, lowerBound, upperBound] = colaCheck(window, hopSize);
                        window = window / mean([lowerBound, upperBound]);
                        obj.synthesisWindow = window;

                    case 'wola'
                        inlineHann = @(L, n) 0.5*(1-cos(2*pi*n/L));
                        M = synthesisSegmentSize/2;
                        KM = segmentSize - M;
                        h1 = sqrt(inlineHann(2*KM,0:KM - M - 1 ))';
                        h2 = sqrt(inlineHann(2*KM, (0:M-1) + KM - M))';
                        h3 = sqrt(inlineHann(2*M, (0:M-1) + M))';
                        analysisWindow = [h1; h2; h3];

                        f1 = zeros(KM-M, 1);
                        f2 = inlineHann(2*M, (0:M-1))' ./ sqrt(inlineHann(2*KM, (0:M-1) + KM - M)');
                        f3 = sqrt(inlineHann(2*M, (0:M-1) + M)');
                        synthesisWindow = [f1; f2; f3];

                        window = analysisWindow .* synthesisWindow;
                        [checkPassed, lowerBound, upperBound] = colaCheck(window, hopSize);
                        normalizationConstant = (mean([lowerBound, upperBound]));
                        obj.analysisWindow = analysisWindow;
                        obj.synthesisWindow = synthesisWindow / normalizationConstant;
                    otherwise
                        error(['Specified reconstruction scheme "' reconstructionScheme '" is not currently supported when specifying additional input variables.'])
                end
                if ~checkPassed
                    error('COLA Condition not satisfied for the windowing scheme');
                end
            end
        end
        
        %% Window implementations
        function window = hannWindow(obj, windowLength)
            arguments (Input)
                obj (1,1) windowObject
                windowLength (1,1) double
            end
            arguments (Output)
                window (:,1) double
            end
            m = (0:windowLength-1)';
            window = 0.5*(1 - cos(2*pi*m/(windowLength)));
        end

        function window = rectangularWindow(obj, windowLength)
            arguments (Input)
                obj (1,1) windowObject
                windowLength (1,1) double
            end
            arguments (Output)
                window (:,1) double
            end
            window = ones(windowLength, 1);
        end
    end
end


