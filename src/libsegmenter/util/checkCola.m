function [colaConditionSatisfied, normalization] = checkCola(window, hopSize, varargin)
%CHECKCOLA Script for validating COLA condition is satisfied
% This check seeks to validate whether the window function is zero at the frequency 2*pi / hopSize and all its harmonics.
% The implementation follows the example by Julius O. Smith: https://ccrma.stanford.edu/~jos/sasp/Periodic_Hamming_OLA_Poisson_Summation.html
if nargin == 2
    plotFlag = false;
    colaLimitDb = -40;
elseif nargin == 3
    colaLimitDb = varargin{1};
elseif nargin == 4
    colaLimitDb = varargin{1};
    plotFlag = varargin{2};
    plotName = '';
elseif nargin == 5
    colaLimitDb = varargin{1};
    plotFlag = varargin{2};
    plotName = varargin{3};
else
    error('Unsupported number of input argements');
end

windowLength = length(window);
% Frequency of the fundamental frequency (in normalized frequency, i.e., fs=1)
fundamentalFrequency = 1/hopSize; % frame rate (fs=1)
numSamples = windowLength;
upperBound = sum(window)/hopSize;  % try easy-to-compute upper bound
lowerBound = upperBound; % and lower bound
timeIdx = (0:numSamples-1)';

if plotFlag
    plotLength = 12*windowLength;
    windowSpectrum = fft(window, plotLength);
    windowSpectrum = windowSpectrum/abs(windowSpectrum(1));
    normFrequency = (0:plotLength-1)'/plotLength;
    minMag = -80;
    maxMag = 10;
    figure
    plot(normFrequency, 20*log10(abs(windowSpectrum)));
    hold on; grid on
    xlim([0 0.5]); ylim([minMag maxMag])
    xlabel('Normalized frequency'); ylabel('Magnitude [dB]')
    plot(1/(2*hopSize)*[1,1], [minMag, maxMag],'--');

    % Find idx corresponding to the folding frequency
    idx = find(normFrequency>1/(2*hopSize),1);
    stopBandRejectionRatio = 20*log10(max(abs(windowSpectrum(idx:round(plotLength/2)))));
    signalToAliasRatio = 20*log10(sum(abs(windowSpectrum(1:idx-1)))) - 20*log10(sum(abs(windowSpectrum(idx:round(plotLength/2)))));
end
for k = 1:hopSize-1 % traverse frame-rate harmonics
  harmonicFrequency = fundamentalFrequency*k;
  modulationFactor = exp(1j*2*pi*harmonicFrequency*timeIdx); % frame-rate harmonic

  % find exact window transform at frequency f
  dftCoefficient = window' * conj(modulationFactor(1:windowLength));
  % Update lower and upper bounds:
  upperBound = upperBound + abs(dftCoefficient)/hopSize; % build upper bound
  lowerBound = lowerBound - abs(dftCoefficient)/hopSize; % build lower bound
  if plotFlag
      plot([1,1]*k/hopSize, [minMag, maxMag], '--k')
  end
end
if plotFlag
    legend('Window spectrum','Folding frequency','Sampling rate harmonics')
    title([plotName '  Ripple = ' num2str(20*log10(ubound-lbound), '%.0f') ' dB | SAR = ' num2str(signalToAliasRatio, '%.0f') ' dB | SBR = ' num2str(stopBandRejectionRatio, '%.0f') ' dB'])
end
if 20*log10(abs((upperBound-lowerBound))) < colaLimitDb
    colaConditionSatisfied = true;
else
    colaConditionSatisfied = false;
    warning(['COLA check failed with 20*log10(ubound-lbound) = ' num2str(20*log10(abs((ubound-lbound))), '%.2f') ' dB']);
end
normalization = mean([lowerBound, upperBound]);

end
