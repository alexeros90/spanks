import numpy as np
from scipy.stats import norm
from scipy.signal import convolve
from scipy.special import expit, logit
from astropy.convolution import Gaussian1DKernel
from .utils import sigma_clipping
import warnings

def suppress_noise1d(profile, method ="simple", kernel = None, logit_alpha = 1.0, square_profile = False,
                     neighbor_booster = None, neighbors_weight = 0.5, neighbors_strength = 0.5, heavy_baseline = None,
                     heavy_average_weights = False):
    """
    Suppress background-dominated samples in a 1D profile using a noise-based
    probability model.

    The function assumes that the profile noise follows an approximately
    normal distribution. A noise probability profile is first computed from
    the profile intensity distribution and then refined using one of several
    methods. The resulting probability profile is applied as a weighting
    function, enhancing statistically significant features while attenuating
    samples consistent with the background noise.

    Parameters
    ----------
    profile : ndarray
        Input one-dimensional profile.

    method : {'simple', 'logit', 'layers'}, optional
        Method used to generate the probability profile.

        * ``'simple'``: local smoothing of the noise probability profile.
        * ``'logit'``: smoothing in logit space followed by a logistic
          transformation.
        * ``'layers'``: iterative probability refinement using an
          additional intensity-weighted layer.

        Default is ``'simple'``.

    kernel : ndarray, optional
        Kernel used for local probability aggregation. If ``None``, a
        Gaussian kernel with a width equivalent to
        ``Gaussian1DKernel(2.0)`` is used.

    logit_alpha : float, optional
        Scaling factor applied to the logistic transform when
        ``method='logit'``. Larger values increase the contrast between
        low- and high-probability regions.

    square_profile : bool, optional
        If True will use squared values to make the probability profile,
        else it will use absolute values. This increases the influence
        of large deviations from the estimated background level.
        Default is False.

    neighbor_booster : {None, "mean", "product", "heavy"}, optional
        Additional neighbor-based probability enhancement method.

        - ``None``: no additional enhancement.
        - ``"mean"``: weighted average with a neighbor probability profile.
        - ``"product"``: multiplicative combination with a neighbor profile.
        - ``"heavy"``: evidence propagation in log-odds space.

    neighbors_weight : float, optional
        Weight assigned to the neighbor probability profile when
        ``neighbor_booster="mean"``. Must be between 0 and 1.

    neighbors_strength : float, optional
        Strength of neighbor evidence propagation when
        ``neighbor_booster="heavy"``.

    heavy_baseline : float, optional
        Reference probability used by the heavy neighbor booster.
        Values above the baseline contribute positive evidence while
        values below contribute negative evidence. If `None`, the
        mean probability is used.

    heavy_average_weights : bool, optional
        If `True`, normalize the accumulated neighbor evidence by the
        available kernel weight when using the heavy neighbor booster.


    Returns
    -------
    ndarray
        Background-suppressed profile rescaled to approximately match the
        original profile intensity range.

    """
    if kernel is None:
        kernel = Gaussian1DKernel(2.0).array.copy() # Larger kernels yield higher noise level, but more symmetric noise distribution

    if neighbor_booster == "mean" and ((neighbors_weight > 1) or (neighbors_weight < 0)):
        warnings.warn("neighbors_weight should be between 0 and 1. Clipping value.", UserWarning, stacklevel=2)
        neighbors_weight = np.clip(neighbors_weight, 0,1)

    if neighbor_booster == "heavy" and heavy_baseline is not None:
        if (heavy_baseline > 1) or (heavy_baseline < 0):
            warnings.warn("heavy_baseline should be between 0 and 1. Clipping value.", UserWarning, stacklevel=2)
            heavy_baseline = np.clip(heavy_baseline, 0, 1)

    profile_noise_lvl, profile_noise_std = sigma_clipping(profile, robust=False)

    profile_centered = profile - profile_noise_lvl

    # Probability of obtaining x or lower:
    if square_profile:
        noise_p_profile = norm.cdf(profile_centered ** 2, loc=0, scale=profile_noise_std)
    else:
        noise_p_profile = norm.cdf(np.abs(profile_centered), loc=0, scale=profile_noise_std)


    if method == "logit":
        p_prof = _get_p_logit1d(noise_p_profile, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_prof = _get_p_layers1d(profile, noise_p_profile, kernel)
    else: #simple
        p_prof = _get_p_simple1d(noise_p_profile, kernel)

    if neighbor_booster == "mean":
        p_prof = mean_neighbors1d(p_prof, profile, noise_p_profile, kernel, method, neighbors_weight=neighbors_weight)
    elif neighbor_booster == "product":
        p_prof = prod_neighbors1d(p_prof, profile, noise_p_profile, kernel, method)
    elif neighbor_booster == "heavy":
        p_prof = heavy_neighbors1d(p_prof, kernel, neighbors_strength=neighbors_strength, baseline=heavy_baseline,
                                   average_weights = heavy_average_weights)

    noise_suppressed_profile = _apply_p_profile(profile_centered, p_prof) + profile_noise_lvl

    return noise_suppressed_profile


def heavy_neighbors1d(p_prof, kernel, eps=1e-6, neighbors_strength=0.5, baseline=None, average_weights = True):
    """
    Enhance probabilities using neighbor evidence in log-odds space.

    Each probability is converted to log-odds, and neighboring samples
    contribute evidence relative to a baseline probability. Positive
    evidence increases the probability of a sample belonging to the signal,
    while negative evidence decreases it.

    Parameters
    ----------
    p_prof : array-like
        Probability profile.

    kernel : array-like
        Neighbor weighting kernel. The central element is ignored.

    eps : float, optional
        Small value used to clip probabilities away from exactly 0 and 1,
        avoiding numerical instabilities in the logit transform.

    neighbors_strength : float, optional
        Scaling factor controlling how strongly neighboring evidence
        affects the final probabilities.

    baseline : float, optional
        Reference probability separating positive and negative evidence.
        Should be between 0 and 1. If `None`, the mean value of `p_prof` is used.

    average_weights : bool, optional
        If `True`, normalize accumulated neighbor evidence by the
        available kernel weight at each position to correct boundary
        effects.

    Returns
    -------
    ndarray
        Probability profile enhanced by neighbor evidence.

    Notes
    -----
    Neighbor evidence is computed as:

    ``logit(p) - logit(baseline)``

    and convolved with the supplied kernel. The resulting evidence is added
    to the original log-odds before transforming back to probability space.
    """

    if baseline is None:
        baseline = p_prof.mean()

    kernel = kernel.copy()
    kernel = kernel/kernel.max()
    center = len(kernel) // 2
    kernel[center] = 0.0

    p_prof = np.clip(p_prof, eps, 1 - eps)
    log_odds = logit(p_prof)

    evidence = log_odds - logit(baseline)
    neighbor_evidence = convolve(evidence, kernel, mode="same")
    # Correct for boundaries: normalize by available kernel weight.
    available_weight = convolve(np.ones_like(p_prof), kernel, mode="same")
    if average_weights:
        neighbor_evidence = np.divide(neighbor_evidence, available_weight, out=np.zeros_like(p_prof), where=available_weight > 0) # Uncomment for averaging
    final_log_odds = log_odds + neighbors_strength * neighbor_evidence
    return expit(final_log_odds)

def prod_neighbors1d(p_prof, profile, noise_p_profile, kernel, method = "simple", logit_alpha = 1.0):
    """
    Combine a probability profile with a neighbor-derived probability profile
    through multiplication.

    Parameters
    ----------
    p_prof : array-like
        Base probability profile.

    profile : array-like
        Original input profile.

    noise_p_profile : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Kernel used to construct the neighbor probability profile.

    method : {"simple", "logit", "layers"}, optional
        Method used when generating the neighbor probability profile.

    logit_alpha : float, optional
        Scaling parameter used by the ``"logit"`` method.

    Returns
    -------
    ndarray
        Probability profile obtained by multiplying the original and
        neighbor-derived probabilities.

    Notes
    -----
    This approach is conservative: probabilities are reduced unless both
    the local sample and its neighborhood indicate the presence of signal.
    """

    p_prof_neighbors = _make_neighbors_profile(profile, noise_p_profile, kernel.copy(), method, logit_alpha)
    return p_prof * p_prof_neighbors

def mean_neighbors1d(p_prof, profile, noise_p_profile, kernel, method = "simple", logit_alpha = 1.0, neighbors_weight = 0.5):
    """
    Combine a probability profile with a neighbor-derived probability profile
    through weighted averaging.

    Parameters
    ----------
    p_prof : array-like
        Base probability profile.

    profile : array-like
        Original input profile.

    noise_p_profile : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Kernel used to construct the neighbor probability profile.

    method : {"simple", "logit", "layers"}, optional
        Method used when generating the neighbor probability profile.

    logit_alpha : float, optional
        Scaling parameter used by the ``"logit"`` method.

    neighbors_weight : float, optional
        Relative contribution of the neighbor probability profile.
        A value of 0 returns the original profile, while a value of 1
        returns the neighbor-derived profile.

    Returns
    -------
    ndarray
        Weighted-average probability profile.

    Notes
    -----
    The combined probability is computed as

    ``p = w * p_neighbors + (1 - w) * p_prof``

    where ``w = neighbors_weight``.
    """
    p_prof_neighbors = _make_neighbors_profile(profile, noise_p_profile, kernel.copy(), method, logit_alpha)
    average_p_prof = neighbors_weight*p_prof_neighbors + (1-neighbors_weight)*p_prof
    return average_p_prof

def _make_neighbors_profile(profile, noise_p_profile, kernel, method = "simple", logit_alpha = 1.0):
    """
    Construct a probability profile using only neighboring samples.

    This helper generates a signal-probability profile in the same way as
    the main probability estimation methods, but excludes the contribution
    of the current sample by setting the central kernel weight to zero.
    It is primarily used by neighbor-based probability boosting methods.

    Parameters
    ----------
    profile : array-like
        Original input profile.

    noise_p_profile : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Convolution kernel defining the neighborhood. The central element
        is removed and the remaining weights are renormalized so that only
        neighboring samples contribute.

    method : {"simple", "logit", "layers"}, optional
        Method used to compute the neighbor probability profile.

        - ``"simple"``: kernel-weighted averaging of probabilities.
        - ``"logit"``: combines probabilities in log-odds space.
        - ``"layers"``: layered probability estimation.

    logit_alpha : float, optional
        Scaling factor used by the ``"logit"`` method.

    Returns
    -------
    ndarray
        Probability profile computed exclusively from neighboring samples.

    Notes
    -----
    Unlike the main probability estimation functions, this helper removes
    the central kernel weight before computing probabilities. The resulting
    profile therefore represents the neighborhood's assessment of whether
    a sample belongs to signal or noise, independent of the sample itself.
    """

    kernel[len(kernel) // 2] = 0.0
    kernel = kernel / kernel.sum()
    if method == "logit":
        p_prof_neighbors = _get_p_logit1d(noise_p_profile, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_prof_neighbors = _get_p_layers1d(profile, noise_p_profile, kernel)
    else: #simple
        p_prof_neighbors = _get_p_simple1d(noise_p_profile, kernel)
    return p_prof_neighbors

def _get_p_simple1d(p_prof, kernel):
    """
    Convolve probability profile (p_prof) with the input kernel.
    """
    p_out = convolve(p_prof, kernel, mode='same')
    return p_out

def _get_p_logit1d(p_prof, kernel, alpha = 1.0):
    """
    Transform probability profile (p_prof) into logits, convolve with the supplied
    kernel, and transform back into a probability profile.

    alpha is the scaling factor applied to the aggregated logits before converting
    back to probabilities. Values greater than one amplify local contrast,
    while values less than one reduce it.

    Notes
    -----
    Pixels with probability equal to one are clipped slightly below one
    before the logit transformation to avoid numerical singularities.
    """
    p_prof[p_prof >= 1] = 1-1e-9
    logit = np.log(p_prof / (1 - p_prof))
    L = convolve(logit, kernel, mode='same')  # weighted sum of logit
    p_out = 1.0 / (1.0 + np.exp(-alpha * L))
    return p_out

def _get_p_layers1d(data, p_prof, kernel, square_profile = False):
    """
    Generate a probability profile using a two-layer probability refinement
    scheme.

    An initial probability profile (p_prof) is first convolved with the provided kernel.
    The data is then weighted by this smoothed profile, and a second probability profile is
    estimated from the resulting intensity distribution. The final output
    is obtained by combining the two probability layers.
    """
    p_prof_cnv = convolve(p_prof, kernel, mode='same')
    data_p_prof_cnv = data * p_prof_cnv
    data_p_prof_cnv_noise_lvl, data_p_prof_cnv_noise_lvl_std = sigma_clipping(data_p_prof_cnv, robust=False)
    data_p_prof_cnv_center = data_p_prof_cnv - data_p_prof_cnv_noise_lvl
    if square_profile:
        p_prof_new_layer = norm.cdf(data_p_prof_cnv_center**2, loc=0, scale=data_p_prof_cnv_noise_lvl_std)
    else:
        p_prof_new_layer = norm.cdf(np.abs(data_p_prof_cnv_center), loc=0, scale=data_p_prof_cnv_noise_lvl_std)
    p_prof_new_layer_cnv = convolve(p_prof_new_layer, kernel, mode='same')
    p_out = p_prof_cnv * p_prof_new_layer_cnv
    return p_out

def _apply_p_profile(data, p_prof):
    """
    Apply a probability map (p_prof) to an image and rescale according to their maxima.
    """

    data_p_prof = data * p_prof
    data_p_prof_leveled = data_p_prof * data.max() / data_p_prof.max()
    return data_p_prof_leveled

if __name__ == "__main__":
    quit()


