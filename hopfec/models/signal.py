"""Empirical statistics of parcellated BOLD: filtering, FC, lagged covariance, spectra, FCD."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp


def butter_coeffs(tr: float, band, order: int = 2):
    """Butterworth coefficients for band = [low, high] (Hz); either bound may be None."""
    fs = 1.0 / tr
    nyq = fs / 2.0
    lo, hi = (band if band is not None else (None, None))
    lo = None if lo in (None, 0, 0.0) else float(lo)
    hi = None if hi is None else float(hi)
    if hi is not None and hi >= nyq:
        hi = None
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        return sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    if lo is not None:
        return sps.butter(order, lo / nyq, btype="highpass")
    return sps.butter(order, hi / nyq, btype="lowpass")


def bandpass(ts: np.ndarray, tr: float, band, order: int = 2, axis: int = 0) -> np.ndarray:
    """Zero-phase Butterworth band-pass (filtfilt) of demeaned time series (time along *axis*)."""
    ts = np.asarray(ts, float)
    ts = ts - ts.mean(axis=axis, keepdims=True)
    ba = butter_coeffs(tr, band, order)
    if ba is None:
        return ts
    b, a = ba
    n = ts.shape[axis]
    padlen = min(3 * max(len(a), len(b)), n - 1)
    return sps.filtfilt(b, a, ts, axis=axis, padlen=padlen)


def filter_autocorrelation(tr: float, band, order: int = 2, n_lags: int | None = None, tol: float = 1e-8) -> np.ndarray:
    """Impulse response h2(m), m = 0..M, of the zero-phase (filtfilt) filter, i.e. IDFT(|B|^2).

    h2 is symmetric (h2(-m) = h2(m)); only the non-negative lags are returned, truncated where
    |h2| falls below tol * max(|h2|).  Returns [1.0] when no filtering is requested.
    """
    ba = butter_coeffs(tr, band, order)
    if ba is None:
        return np.array([1.0])
    b, a = ba
    # decay time-scale of the slowest pole -> generous length
    poles = np.roots(a)
    r = np.max(np.abs(poles)) if len(poles) else 0.0
    L = 2000 if r >= 1 else int(min(200000, max(500, np.ceil(np.log(tol) / np.log(max(r, 1e-6))) * 3)))
    if n_lags is not None:
        L = max(L, 2 * n_lags + 10)
    x = np.zeros(2 * L + 1)
    x[L] = 1.0
    h2 = sps.filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), 2 * L))
    h2 = h2[L:]  # non-negative lags
    keep = np.where(np.abs(h2) > tol * np.abs(h2).max())[0]
    M = int(keep[-1]) + 1 if len(keep) else 1
    if n_lags is not None:
        M = min(M, n_lags + 1)
    return h2[:M].copy()


def functional_connectivity(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, float)
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 1.0)
    return fc


def lagged_correlation(ts: np.ndarray, lag: int) -> np.ndarray:
    """COVtau[i, j] = < x_i(t+lag) x_j(t) > / (sd_i sd_j)  (x demeaned, sd over the full series).

    Positive entry (i, j) means j "leads" i.  lag is in samples (TRs)."""
    x = np.asarray(ts, float)
    x = x - x.mean(axis=0, keepdims=True)
    T = x.shape[0]
    lag = int(lag)
    if lag < 0:
        return lagged_correlation(ts, -lag).T
    if lag >= T - 2:
        raise ValueError("lag too large for the number of volumes")
    sd = x.std(axis=0, ddof=0)
    sd[sd == 0] = np.inf
    cov = x[lag:].T @ x[: T - lag] / (T - lag)
    return cov / np.outer(sd, sd)


def power_spectra(ts: np.ndarray, tr: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram (freqs, P[freq, node]) of demeaned time series."""
    x = np.asarray(ts, float)
    x = x - x.mean(axis=0, keepdims=True)
    T = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    P = (np.abs(X) ** 2) / (T / tr)
    freqs = np.fft.rfftfreq(T, d=tr)
    return freqs, P


def peak_frequencies(ts: np.ndarray | None, tr: float, band, smooth_hz: float = 0.01,
                     spectrum: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Per-node peak frequency (Hz) of the (Gaussian-smoothed) power spectrum within *band*."""
    if spectrum is None:
        freqs, P = power_spectra(ts, tr)
    else:
        freqs, P = spectrum
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    Ps = P
    if smooth_hz and smooth_hz > 0 and len(freqs) > 3:
        sigma = max(smooth_hz / df, 1e-6)
        Ps = gaussian_filter1d(P, sigma=sigma, axis=0, mode="nearest")
    lo, hi = (band if band is not None else (None, None))
    lo = 0.0 if lo is None else float(lo)
    hi = freqs[-1] if hi is None else float(hi)
    sel = (freqs >= lo) & (freqs <= hi)
    if sel.sum() == 0:
        sel = np.ones_like(freqs, bool)
    idx = np.argmax(Ps[sel], axis=0)
    f_peak = freqs[sel][idx]
    return f_peak, (freqs, P)


def average_spectra(spectra: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectra with possibly different frequency grids (interpolated onto the finest)."""
    grids = [s[0] for s in spectra]
    ref = min(grids, key=lambda g: (g[1] - g[0]) if len(g) > 1 else np.inf)
    acc = np.zeros((len(ref), spectra[0][1].shape[1]))
    for freqs, P in spectra:
        for j in range(P.shape[1]):
            acc[:, j] += np.interp(ref, freqs, P[:, j])
    return ref, acc / len(spectra)


def hilbert_phases(ts: np.ndarray) -> np.ndarray:
    x = np.asarray(ts, float)
    x = x - x.mean(axis=0, keepdims=True)
    return np.angle(sps.hilbert(x, axis=0))


def kuramoto_order(ts: np.ndarray) -> np.ndarray:
    ph = hilbert_phases(ts)
    return np.abs(np.mean(np.exp(1j * ph), axis=1))


def metastability(ts: np.ndarray) -> float:
    return float(np.std(kuramoto_order(ts)))


def synchrony(ts: np.ndarray) -> float:
    return float(np.mean(kuramoto_order(ts)))


def fcd_matrix(ts: np.ndarray, window: int = 30, step: int = 3) -> np.ndarray:
    """Sliding-window FCD: correlation between the FC vectors of every pair of windows."""
    x = np.asarray(ts, float)
    T, N = x.shape
    window = int(min(window, T))
    starts = list(range(0, T - window + 1, max(1, int(step))))
    iu = np.triu_indices(N, 1)
    vecs = []
    for s in starts:
        fc = np.corrcoef(x[s : s + window].T)
        vecs.append(np.nan_to_num(fc[iu]))
    V = np.array(vecs)
    if len(V) < 2:
        return np.ones((1, 1))
    return np.nan_to_num(np.corrcoef(V))


def fcd_distribution(ts: np.ndarray, window: int = 30, step: int = 3) -> np.ndarray:
    F = fcd_matrix(ts, window, step)
    iu = np.triu_indices(F.shape[0], 1)
    return F[iu]


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def ssim_matrix(A: np.ndarray, B: np.ndarray, sigma: float = 1.5) -> float:
    """Structural similarity between two matrices (Gaussian-window SSIM, data range from both)."""
    from scipy.ndimage import gaussian_filter

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    L = max(A.max(), B.max()) - min(A.min(), B.min())
    if L <= 0:
        return 1.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a, mu_b = gaussian_filter(A, sigma), gaussian_filter(B, sigma)
    saa = gaussian_filter(A * A, sigma) - mu_a ** 2
    sbb = gaussian_filter(B * B, sigma) - mu_b ** 2
    sab = gaussian_filter(A * B, sigma) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(np.mean(s))


def empirical_moments(ts: np.ndarray, tr: float, band, tau_tr: int, freq_band=None, smooth_hz: float = 0.01,
                      order: int = 2) -> dict:
    """Filter and compute everything the models are fitted to for one run."""
    xf = bandpass(ts, tr, band, order=order)
    fc = functional_connectivity(xf)
    covtau = lagged_correlation(xf, tau_tr)
    f_peak, spec = peak_frequencies(xf, tr, freq_band or band, smooth_hz)
    return {
        "FC": fc,
        "COVtau": covtau,
        "f_peak": f_peak,
        "spectrum": spec,
        "n_volumes": int(xf.shape[0]),
        "tr": float(tr),
        "metastability": metastability(xf),
        "filtered": xf,
    }
