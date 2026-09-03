function [Ceff, hist] = hopf_gec_linear(FCemp, COVtauemp, SC, G, a, omega, TR, tau_tr, opts)
%HOPF_GEC_LINEAR  Heuristic GEC iteration with the linear Hopf model (fast: no simulation).
%
%   [Ceff, hist] = hopf_gec_linear(FCemp, COVtauemp, SC, G, a, omega, TR, tau_tr, opts)
%
%   opts fields (defaults): epsFC 1e-3, epsCOV 1e-3, maxIter 2000, patience 50, maxC 0.2,
%   band [0.008 0.08], beta 0.02, mask (logical N x N; default SC > 0 plus mirror-index homotopic pairs).
%
%   C_ij <- C_ij + epsFC (FCemp - FCsim)_ij + epsCOV (COVtauemp - COVtausim)_ij on allowed links,
%   clipped at 0, rescaled to max maxC; the best iterate (lowest RMSE) is returned.
%   NOTE: the Python package additionally provides an exact-gradient estimator (recommended); this
%   MATLAB function implements the published heuristic only.

if nargin < 9, opts = struct(); end
d = @(f, v) getfield_default(opts, f, v);
epsFC = d('epsFC', 1e-3); epsCOV = d('epsCOV', 1e-3); maxIter = d('maxIter', 2000);
patience = d('patience', 50); maxC = d('maxC', 0.2); band = d('band', [0.008 0.08]); beta = d('beta', 0.02);
N = size(SC, 1);
if isfield(opts, 'mask'), mask = logical(opts.mask);
else
    mask = SC > 0;
    for i = 1:N, mask(i, N - i + 1) = true; mask(N - i + 1, i) = true; end
end
mask(1:N+1:end) = false;
C = SC; C(~mask) = 0; if max(C(:)) > 0, C = C / max(C(:)) * maxC; end
best = inf; Ceff = C; stall = 0; hist = zeros(maxIter, 3);
od = ~eye(N);
for it = 1:maxIter
    [FCsim, COVsim] = hopf_linear_moments(C, G, a, omega, TR, tau_tr, beta, band);
    eFC = sqrt(mean((FCemp(od) - FCsim(od)).^2));
    eCOV = sqrt(mean((COVtauemp(:) - COVsim(:)).^2));
    err = sqrt(0.5 * (eFC^2 + eCOV^2));
    hist(it, :) = [err, corr(FCemp(triu(od)), FCsim(triu(od))), eCOV];
    if err < best, best = err; Ceff = C; stall = 0; else, stall = stall + 1; if stall >= patience, break; end, end
    C = C + epsFC * (FCemp - FCsim) + epsCOV * (COVtauemp - COVsim);
    C(~mask) = 0; C(C < 0) = 0;
    if max(C(:)) > 0, C = C / max(C(:)) * maxC; end
end
hist = hist(1:it, :);
end

function v = getfield_default(s, f, v)
if isfield(s, f), v = s.(f); end
end
