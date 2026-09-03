function [FC, COVtau, Sigma] = hopf_linear_moments(C, G, a, omega, TR, tau_tr, beta, band)
%HOPF_LINEAR_MOMENTS  FC and lagged correlation of the linear (OU) Hopf whole-brain model.
%
%   [FC, COVtau] = hopf_linear_moments(C, G, a, omega, TR, tau_tr, beta, band)
%
%   C      N x N coupling (C(i,j): j -> i), G global coupling, a bifurcation parameter (scalar or N x 1),
%   omega  N x 1 angular node frequencies (2*pi*f), TR seconds, tau_tr lag in TRs, beta noise (default 0.02),
%   band   [flp fhi] Hz of the 2nd-order Butterworth band-pass applied to the data (filtfilt); [] = none.
%
%   Linearisation around z = 0 (a < 0):  A = [A11 -W; W A11], A11 = G*C + diag(a - G*sum(C,2)),
%   Sigma solves A*Sigma + Sigma*A' + beta^2 I = 0, Cov(X(t+tau), X(t)) = expm(A tau) Sigma.
%   With a band given, the TR-sampled covariance sequence is convolved with the filter's
%   autocorrelation (impulse response of filtfilt), exactly as in hopfec/models/hopf_linear.py.
%   COVtau(i,j) = corr(x_i(t+tau), x_j(t))  (positive: j leads i).

if nargin < 7 || isempty(beta), beta = 0.02; end
if nargin < 8, band = []; end
N = size(C, 1);
C = double(C); C(1:N+1:end) = 0;
a = a(:) .* ones(N, 1);
omega = omega(:);
s = G * sum(C, 2);
A11 = G * C + diag(a - s);
W = diag(omega);
A = [A11, -W; W, A11];
Sigma = lyap(A, beta^2 * eye(2 * N));           % A*Sigma + Sigma*A' + Q = 0
if isempty(band)
    S0 = Sigma;
    St = expm(A * tau_tr * TR) * Sigma;
else
    h2 = filter_autocorr(TR, band);              % h2(1) = lag 0
    M = numel(h2) - 1;
    P = expm(A * TR);
    K = tau_tr + M;
    c = cell(K + 1, 1); c{1} = Sigma;
    for k = 1:K, c{k+1} = P * c{k}; end
    S0 = h2(1) * c{1};
    for k = 1:M, S0 = S0 + h2(k+1) * (c{k+1} + c{k+1}'); end
    St = zeros(2 * N);
    for k = 0:K, St = St + h2(abs(tau_tr - k) + 1) * c{k+1}; end
    for kp = 1:(M - tau_tr), St = St + h2(kp + tau_tr + 1) * c{kp+1}'; end
end
Sxx = S0(1:N, 1:N); Txx = St(1:N, 1:N);
sd = sqrt(diag(Sxx));
FC = Sxx ./ (sd * sd'); FC(1:N+1:end) = 1;
COVtau = Txx ./ (sd * sd');
end

function h2 = filter_autocorr(TR, band)
fs = 1 / TR; nyq = fs / 2;
[b, a] = butter(2, [band(1) band(2)] / nyq, 'bandpass');
L = 20000;
x = zeros(2 * L + 1, 1); x(L + 1) = 1;
y = filtfilt(b, a, x);
h2 = y(L+1:end);
keep = find(abs(h2) > 1e-6 * max(abs(h2)), 1, 'last');
h2 = h2(1:keep);
end
