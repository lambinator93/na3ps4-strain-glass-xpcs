"""
Sutton et al. (PRL 107, 015702, 2011) avalanche analysis
applied to two-time correlation / intensity matrices from XPCS.

The method detects heterogeneous, discontinuous dynamics (avalanches)
in two-time data by computing row-difference amplitudes near the diagonal.

Equations implemented:
    D_n(q; t1, t2) = <|I_n(q,t1) - I_n(q,t2)|>_pixels       (Eq. 1)
    A_n-(q; t1)   = (1/m) * sum_{ti=t1-m}^{t1-1} [D_n(t1, ti) - D_n(t1-1, ti)]  (Eq. 2)
    A_n+(q; t1)   = (1/m) * sum_{ti=t1+1}^{t1+m} [D_n(t1-1, ti) - D_n(t1, ti)]  (Eq. 3)
    A(q; t1)      = (A_n- + A_n+) / 2                         (Eq. 4)

Usage
-----
If you already have a precomputed two-time difference matrix D (shape [T, T]),
use `avalanche_amplitude_from_D`.

If you have raw pixel-level intensity traces (shape [T, N_pixels]),
use `compute_D_matrix` first, then feed the result into the amplitude function.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_D_matrix(I: np.ndarray) -> np.ndarray:
    """
    Compute the two-time difference matrix D(t1, t2) from raw intensity traces.

    D(t1, t2) = <|I(t1) - I(t2)|>  averaged over pixels in a bin.

    Parameters
    ----------
    I : ndarray, shape (T, N_pixels)
        Intensity as a function of frame index (rows) and pixel (columns).
        This corresponds to a single spatial bin / ROI.

    Returns
    -------
    D : ndarray, shape (T, T)
        Symmetric two-time difference matrix.
        D[t1, t2] = mean(|I[t1, :] - I[t2, :]|).
    """
    T = I.shape[0]
    # Vectorised: outer mean-absolute-difference
    # Memory note: for large T this is T x T x N_pixels — consider chunking.
    D = np.mean(np.abs(I[:, None, :] - I[None, :, :]), axis=-1)  # (T, T)
    return D


def avalanche_amplitude_from_D(
    D: np.ndarray,
    m: int = 15,
) -> np.ndarray:
    """
    Compute the avalanche amplitude A(t1) from a two-time difference matrix.

    Implements Eqs. (2)–(4) of Sutton et al. PRL 107, 015702 (2011).

    Parameters
    ----------
    D : ndarray, shape (T, T)
        Two-time difference matrix (see `compute_D_matrix`).
        Can also be a standard two-time *correlation* matrix C2 if you
        convert it to differences first: D = 1 - C2 works as a proxy.
    m : int, optional
        Number of time steps used in the row-difference averages.
        Default is 15 (as used in the paper).  Larger m gives smoother
        amplitude but reduces time resolution near the edges.

    Returns
    -------
    A : ndarray, shape (T,)
        Avalanche amplitude at each time step t1.
        Entries within m steps of the boundary are set to NaN.
    """
    T = D.shape[0]
    A = np.full(T, np.nan)

    for t1 in range(m + 1, T - m):
        # Eq. (2): look *before* t1 — fixed column index ti, vary row
        #   A_n-(t1) = (1/m) * sum_{ti=t1-m}^{t1-1} [D(t1, ti) - D(t1-1, ti)]
        ti_before = np.arange(t1 - m, t1)          # m indices before t1
        An_minus = np.mean(D[t1, ti_before] - D[t1 - 1, ti_before])

        # Eq. (3): look *after* t1 — fixed row index, vary column ti
        #   A_n+(t1) = (1/m) * sum_{ti=t1+1}^{t1+m} [D(t1-1, ti) - D(t1, ti)]
        ti_after = np.arange(t1 + 1, t1 + m + 1)   # m indices after t1
        An_plus = np.mean(D[t1 - 1, ti_after] - D[t1, ti_after])

        # Eq. (4)
        A[t1] = (An_minus + An_plus) / 2.0

    return A


# ──────────────────────────────────────────────────────────────────────────────
# Multi-bin / full-image pipeline
# ──────────────────────────────────────────────────────────────────────────────

def sutton_avalanche_pipeline(
    D: np.ndarray,
    m: int = 15,
    noise_D: np.ndarray | None = None,
    sigma_threshold: float = 3.0,
    min_separation: int | None = None,
    min_prominence: float = 0.0,
) -> dict:
    """
    Full Sutton pipeline:
      1. Compute per-time avalanche amplitude A(t1)  [Eqs. 2–4].
      2. Optionally normalise by static-noise amplitude.
      3. Detect avalanche events as *local maxima* (peaks) of A above threshold.
      4. Compute total avalanche amplitude A(σ) by summing peak heights.

    Detection uses scipy.signal.find_peaks rather than simple thresholding.
    This avoids the failure mode where shoulders of peaks, plateaus, and
    inter-peak troughs that happen to lie above threshold get counted as
    additional events.

    Parameters
    ----------
    D : ndarray, shape (T, T)
        Two-time difference matrix for *one* bin.
    m : int
        Averaging window (default 15).
    noise_D : ndarray, shape (T_noise, T_noise) or None
        Two-time difference matrix measured on a *static* reference sample.
        If provided, A(t1) is normalised by the noise amplitude so the
        result is in units of noise standard deviations (σ).
    sigma_threshold : float
        Minimum peak height (in units of σ) for an event to be detected.
    min_separation : int or None
        Minimum number of frames between adjacent detected peaks.
        Defaults to `m` (one event per averaging window — the natural choice
        given that m sets the temporal resolution of A(t1)).
        Set explicitly to override.
    min_prominence : float
        Minimum peak prominence (in units of σ) above surrounding baseline.
        Helps filter small wiggles that happen to be local maxima but are
        not distinct events. Default 0 (no prominence filter).

    Returns
    -------
    dict with keys:
        'A'                : ndarray (T,)  — raw amplitude
        'A_norm'           : ndarray (T,)  — amplitude normalised by noise σ
        'noise_level'      : float         — estimated noise amplitude (1σ)
        'avalanche_times'  : ndarray       — t1 indices of detected peaks
        'avalanche_amplitudes' : ndarray   — peak heights A_norm at events
        'total_amplitude'  : float         — sum of peak heights above threshold
        'peak_properties'  : dict          — full output of scipy.find_peaks
    """
    A = avalanche_amplitude_from_D(D, m=m)

    # ── noise normalisation ──────────────────────────────────────────────────
    if noise_D is not None:
        A_noise = avalanche_amplitude_from_D(noise_D, m=m)
        noise_level = np.nanstd(A_noise)
    else:
        # Estimate noise from the distribution of A itself.
        # Use the absolute deviations of the lower half (assumes noise is
        # roughly symmetric and not dominated by avalanches at small |A|).
        valid = A[~np.isnan(A)]
        noise_level = np.std(valid[valid < np.percentile(valid, 50)])
        if noise_level == 0 or np.isnan(noise_level):
            noise_level = 1.0  # fallback: no normalisation

    A_norm = A / noise_level

    # ── detect avalanche peaks (local maxima above threshold) ────────────────
    # Default: one event per averaging window (Sutton-consistent)
    if min_separation is None:
        min_separation = m

    # find_peaks rejects NaN, so feed it a NaN-cleaned copy where NaN -> -inf
    # so boundary regions never produce false peaks
    A_for_peaks = np.where(np.isnan(A_norm), -np.inf, A_norm)

    peaks, properties = find_peaks(
        A_for_peaks,
        height=sigma_threshold,
        distance=max(1, int(min_separation)),
        prominence=min_prominence if min_prominence > 0 else None,
    )

    avalanche_times = peaks
    avalanche_amplitudes = A_norm[peaks] if len(peaks) > 0 else np.array([])
    total_amplitude = float(np.sum(avalanche_amplitudes))

    return {
        "A": A,
        "A_norm": A_norm,
        "noise_level": noise_level,
        "avalanche_times": avalanche_times,
        "avalanche_amplitudes": avalanche_amplitudes,
        "total_amplitude": total_amplitude,
        "peak_properties": properties,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: if your two-time data is a *correlation* matrix C2
# ──────────────────────────────────────────────────────────────────────────────

def avalanche_from_C2(
    C2: np.ndarray,
    m: int = 15,
    noise_C2: np.ndarray | None = None,
    sigma_threshold: float = 3.0,
    min_separation: int | None = None,
    min_prominence: float = 0.0,
) -> dict:
    """
    Wrapper for systems where you have a two-time *correlation* matrix C2
    rather than raw intensities.

    Converts C2 → proxy difference matrix via D_proxy = 1 - C2, then
    runs the full Sutton pipeline.

    Parameters
    ----------
    C2 : ndarray, shape (T, T)
        Two-time intensity autocorrelation (g2 - 1) or normalised correlation,
        with values in [0, 1] where 1 = fully correlated.
    m, noise_C2, sigma_threshold, min_separation, min_prominence :
        see `sutton_avalanche_pipeline`.
    """
    D = 1.0 - C2
    noise_D = (1.0 - noise_C2) if noise_C2 is not None else None
    return sutton_avalanche_pipeline(
        D, m=m, noise_D=noise_D,
        sigma_threshold=sigma_threshold,
        min_separation=min_separation,
        min_prominence=min_prominence,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_sutton_analysis(
    D: np.ndarray,
    result: dict,
    times: np.ndarray | None = None,
    title: str = "",
    sigma_threshold: float = 3.0,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    ttc: bool = False,
    line_width: float = 1.0,
    marker_size: float = 40,
    marker_width: float = 0.6,
    legend_ax_font: float = 8,
):
    """
    Two-panel figure:
      (a) Two-time difference matrix D(t1, t2)
      (b) Avalanche amplitude A(t1) with threshold line and flagged peaks

    Parameters
    ----------
    D : ndarray (T, T)
    result : dict from `sutton_avalanche_pipeline`
    times : 1D array of physical times (s, frames, …); uses indices if None.
    title : overall figure title.
    sigma_threshold : drawn as a horizontal dashed line on panel (b).
    fig : existing Figure to draw into (creates new one if None).
    """
    T = D.shape[0]
    t = times if times is not None else np.arange(T)

    if ttc==True:
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        else:
            axes = fig.subplots(1, 2)

                # Panel (a): D matrix
        ax = axes[0]
        im = ax.imshow(
            D, origin="lower", aspect="auto",
            extent=[t[0], t[-1], t[0], t[-1]],
            norm=Normalize(vmin=np.nanpercentile(D, 0), vmax=np.nanpercentile(D, 100)),
            cmap="nipy_spectral",
        )
        plt.colorbar(im, ax=ax, label=r"$D_n(t_1, t_2)$")
        ax.set_xlabel("$t_2$")
        ax.set_ylabel("$t_1$")
        ax.set_title("Two-time difference matrix")

        # Panel (b): avalanche amplitude
        ax = axes[1]
        A_norm = result["A_norm"]
        ax.plot(t, A_norm, color='#373079', lw=3)#, label=r"$A(t_1)/\sigma_\mathrm{noise}$")
        ax.axhline(sigma_threshold, color='#d54848', ls="--", lw=line_width,
                label=f"{sigma_threshold}σ threshold")

        av_t = result["avalanche_times"]
        if len(av_t) > 0:
            ax.scatter(t[av_t], A_norm[av_t], color='#d54848', zorder=5, s=marker_size,
                    label="Avalanche events")

        ax.set_xlabel("$t_1$")
        ax.set_ylabel(r"$A(t_1) \;[\sigma]$")
        #ax.set_title("Avalanche amplitude")
        ax.legend(fontsize=8)
        ax.set_xlim(t[0], t[-1])

    if ttc==False:
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = fig
            ax = ax

        A_norm = result["A_norm"]
        ax.plot(t, A_norm, color='#373079', lw=2)#, label=r"$A($t$1)/\sigma_\mathrm{noise}$")
        ax.axhline(sigma_threshold, color='#d54848', ls="--", lw=line_width,
                label=f"{sigma_threshold}σ threshold")

        av_t = result["avalanche_times"]
        if len(av_t) > 0:
            ax.scatter(t[av_t], A_norm[av_t], color='#d54848', zorder=5, s=marker_size,
                    label="Avalanche events")
    


        ax.set_xlabel("$t_1$")
        ax.set_ylabel(r"$A(t_1) \;[\sigma]$")
        #ax.set_title("Avalanche amplitude")
        ax.legend(fontsize=legend_ax_font)
        ax.set_xlim(t[0], t[-1])

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Quick demonstration with synthetic data
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    T = 200
    N_pix = 100

    # Synthetic intensities: slow baseline + two sudden jumps (avalanches)
    I = rng.poisson(lam=10, size=(T, N_pix)).astype(float)

    jump_frames = [70, 140]
    for jf in jump_frames:
        I[jf:, :] += rng.choice([-1, 1]) * rng.poisson(15, size=(T - jf, N_pix))

    # ── compute D matrix ────────────────────────────────────────────────────
    print("Computing D matrix …")
    D = compute_D_matrix(I)

    # ── run Sutton pipeline ─────────────────────────────────────────────────
    result = sutton_avalanche_pipeline(D, m=15, sigma_threshold=3.0)

    print(f"Noise level (1σ): {result['noise_level']:.4f}")
    print(f"Total avalanche amplitude: {result['total_amplitude']:.2f} σ (sum of peak heights)")
    print(f"Detected avalanche frames: {result['avalanche_times']}")
    print(f"Detected avalanche amplitudes: {np.round(result['avalanche_amplitudes'], 2)}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig = plot_sutton_analysis(
        D, result,
        times=np.arange(T, dtype=float),
        title="Sutton avalanche analysis — synthetic demo",
        sigma_threshold=3.0,
    )
    plt.savefig("/mnt/user-data/outputs/sutton_demo.png", dpi=150, bbox_inches="tight")
    print("Saved demo figure → sutton_demo.png")
    plt.show()