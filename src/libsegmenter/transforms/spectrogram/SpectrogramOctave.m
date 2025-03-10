classdef SpectrogramOctave < handle
    %SpectrogramOctave
    % A class for computing the forward and inverse spectrogram.
    %
    % Currently, the normalization for the fourier transform cannot be controlled and is thus `backward` by default.

    methods
 	    function obj = SpectrogramOctave()
	       % Initialize spectrogram object.
	    end

      function spectrogram = forward(obj, input)
	    % Converts segments into a spectrogram.
	    %
	    % Args:
	    % input (2D / 3D array): Segments as generated by a Segmenter object.
	    if length(size(input)) == 2
		tmp = fft(input, [], 2);
		fftSize = size(input,2);
		if mod(fftSize,2) == 0
		    spectrogram = tmp(:,1:fftSize/2+1);
		else
		    error(['Only even length segments are supported. Received segmentSize = ' int2str(fftSize) ' and size(input) = [' int2str(size(input)) '].']);
		end
	    elseif length(size(input)) == 3
		tmp = fft(input, [], 3);
		fftSize = size(input,3);
		if mod(fftSize,2) == 0
		    spectrogram = tmp(:,:,1:fftSize/2+1);
	        else
		    error(['Only even length segments are supported. Received segmentSize = ' int2str(fftSize) ' and size(input) = [' int2str(size(input)) '].']);
		end
	    else
		error(['The input must be a 2 or 3 dimensional array. Received length(size(input)) = ' int2str(length(size(input))) ]);
	    end
	end

	function sequence = inverse(obj, input)
	    % Converts spectrogram into segments.
	    %
	    % Args:
	    %    input (2D / 3D complex array): Spectrogram resulting from a `forward` pass.
	    if length(size(input)) == 2
		rfftSize = size(input,2);
		% Even length time sequence
		tmp = [real(input(:,1)), input(:,2:rfftSize-1), real(input(:,rfftSize)), conj(fliplr(input(:,2:rfftSize-1)))];
		sequence = ifft(tmp, [], 2);
	    elseif length(size(input)) == 3
		batchSize = size(input,1);
		numSegments = size(input,2);
		rfftSize = size(input,3);
		sequence = zeros(batchSize, numSegments, 2*(rfftSize-1));
		for bIdx = 1:batchSize
		    tmp = squeeze(input(bIdx, :,:));
		    tmpSpectrum = [real(tmp(:,1)), tmp(:,2:rfftSize-1), real(tmp(:,rfftSize)), conj(fliplr(tmp(:,2:rfftSize-1)))];
		    sequence(bIdx,:,:) = ifft(tmpSpectrum,[],2);
		end
	    else
		error(['The input must be a 2 or 3 dimensional array. Received length(size(input)) = ' int2str(length(size(input))) ]);
	    end
	end
    end
end
