function [colaConditionSatisfied, normalization] = checkCola(window, hopSize, varargin)
%COLA Summary of this function goes here
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

%   Detailed explanation goes here
windowLength = length(window);
ff = 1/hopSize; % frame rate (fs=1)
N = 6*windowLength;  % no. samples to look at OLA
sp = ones(N,1)*sum(window)/hopSize; % dc term (COLA term)
ubound = sp(1);  % try easy-to-compute upper bound
lbound = ubound; % and lower bound
n = (0:N-1)';

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

    % Find idx corresponding to the folding frequencyM
    idx = find(normFrequency>1/(2*hopSize),1);
    stopBandRejectionRatio = 20*log10(max(abs(windowSpectrum(idx:round(plotLength/2)))));
    signalToAliasRatio = 20*log10(sum(abs(windowSpectrum(1:idx-1)))) - 20*log10(sum(abs(windowSpectrum(idx:round(plotLength/2)))));
end
for k = 1:hopSize-1 % traverse frame-rate harmonics
  f=ff*k;
  csin = exp(1j*2*pi*f*n); % frame-rate harmonic
  % find exact window transform at frequency f
  Wf = window' * conj(csin(1:windowLength));
  hum = Wf*csin;   % contribution to OLA "hum"
  sp = sp + hum/hopSize; % "Poisson summation" into OLA
  % Update lower and upper bounds:
  Wfb = abs(Wf);
  ubound = ubound + Wfb/hopSize; % build upper bound
  lbound = lbound - Wfb/hopSize; % build lower bound
  if plotFlag
      plot([1,1]*k/hopSize, [minMag, maxMag], '--k')
  end
end
if plotFlag
    legend('Window spectrum','Folding frequency','Sampling rate harmonics')
    title([plotName '  Ripple = ' num2str(20*log10(ubound-lbound), '%.0f') ' dB | SAR = ' num2str(signalToAliasRatio, '%.0f') ' dB | SBR = ' num2str(stopBandRejectionRatio, '%.0f') ' dB'])
end
if 20*log10(abs((ubound-lbound))) < colaLimitDb
    colaConditionSatisfied = true;
else
    colaConditionSatisfied = false;
    warning(['COLA check failed with 20*log10(ubound-lbound) = ' num2str(20*log10(abs((ubound-lbound))), '%.2f') ' dB']);
end
normalization = mean([lbound, ubound]);

end
