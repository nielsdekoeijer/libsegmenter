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
		spectrogram = fft(input, [], 2);
	    elseif length(size(input)) == 3
		spectrogram = fft(input, [], 3);
	    else
		error(['The input must be a 2 or 3 dimensional array. Received length(size(input)) = ' int2str(length(size(input))) ]);
	    end
	end

	function sequence = inverse(obj, input)
	    % Converts spectrogram into segments.
	    %
	    % Args:
	    %    input (2D / 3D complex array): Spectrogram resulting from a `forward` pass.
	    if length(size(input)) == 2 || length(size(input)) == 3
		sequence = real(ifft(input, [], length(size(input))));
	    else
		error(['The input must be a 2 or 3 dimensional array. Received length(size(input)) = ' int2str(length(size(input))) ]);
	    end
	end
    end
end
