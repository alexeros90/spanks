import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from scipy.special import expit, logit
from astropy.convolution import Gaussian2DKernel
from .utils import sigma_clipping

def suppress_noise2d(image, method ="simple", kernel = None, logit_alpha = 1.0, symmetric = False, square_map = False,
                     neighbor_booster = None, neighbors_weight = 0.5, neighbors_strength = 0.5, heavy_baseline = None,
                     heavy_average_weights = False):
    """
    Suppress background-dominated pixels in an image using a noise-based
    probability model.

    The function assumes that image noise follows an approximately normal
    distribution. A noise probability map is first computed from the image
    intensity distribution and then refined using one of several methods.
    The resulting probability map is applied as a weighting function,
    enhancing statistically significant structures while attenuating pixels
    consistent with the background noise.

    Parameters
    ----------
    image : ndarray
        Input image.

    method : {'simple', 'logit', 'layers'}, optional
        Method used to generate the probability map.

        * ``'simple'``: spatial smoothing of the noise probability map.
        * ``'logit'``: smoothing in logit space followed by a logistic
          transformation.
        * ``'layers'``: iterative probability refinement using an
          additional intensity-weighted layer.

        Default is ``'simple'``.

    kernel : ndarray, optional
        Convolution kernel used for local probability aggregation. If
        ``None``, a ``Gaussian2DKernel(2.0)`` is used.

    logit_alpha : float, optional
        Scaling factor applied to the logistic transform when
        ``method='logit'``. Larger values increase the contrast between
        low- and high-probability regions.

    symmetric : bool
        If True will suppress noise symmetrically towards its mean value.
        Default is False.
    square_map : bool
        If True will use squared values to make the probability map,
        else it will use absolute values. Default is False.

    Returns
    -------
    ndarray
        Background-suppressed image rescaled to approximately match the
        original image intensity range.

    See Also
    --------
    _get_p_simple2d
    _get_p_logit2d
    _get_p_layers2d
    _apply_p_map
    """

    if kernel is None:
        kernel = Gaussian2DKernel(2.0).array.copy() # Larger kernels yield higher noise level, but more symmetric noise distribution

    image_noise_lvl, image_noise_std = sigma_clipping(image, robust=False)

    # Probability of obtaining x or lower:
    if symmetric:
        if square_map:
            noise_p_map = norm.cdf(image**2, loc=image_noise_lvl, scale=image_noise_std)
        else:
            noise_p_map = norm.cdf(np.abs(image), loc=image_noise_lvl, scale=image_noise_std)
    else:
        noise_p_map = norm.cdf(image, loc=image_noise_lvl, scale=image_noise_std)

    if method == "logit":
        p_map = _get_p_logit2d(noise_p_map, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_map = _get_p_layers2d(image, noise_p_map, kernel, symmetric, square_map)
    else: #simple
        p_map = _get_p_simple2d(noise_p_map, kernel)

    if neighbor_booster == "mean":
        p_map = mean_neighbors2d(p_map, image, noise_p_map, kernel, method, neighbors_weight=neighbors_weight)
    elif neighbor_booster == "product":
        p_map = prod_neighbors2d(p_map, image, noise_p_map, kernel, method)
    elif neighbor_booster == "heavy":
        p_map = heavy_neighbors2d(p_map, kernel, neighbors_strength=neighbors_strength, baseline=heavy_baseline,
                                   average_weights = heavy_average_weights)

    bkg_reduced_image_leveled = _apply_p_map(image, p_map)

    return bkg_reduced_image_leveled


def heavy_neighbors2d(p_map, kernel, eps=1e-6, neighbors_strength=0.5, baseline=None, average_weights = True):
    """
    Enhance probabilities using neighbor evidence in log-odds space.

    Each probability is converted to log-odds, and neighboring samples
    contribute evidence relative to a baseline probability. Positive
    evidence increases the probability of a sample belonging to the signal,
    while negative evidence decreases it.

    Parameters
    ----------
    p_map : ndarray
        Probability map.

    kernel : ndarray, optional
        Neighbor weighting kernel. The central element is ignored.

    eps : float, optional
        Small value used to clip probabilities away from exactly 0 and 1,
        avoiding numerical instabilities in the logit transform.

    neighbors_strength : float, optional
        Scaling factor controlling how strongly neighboring evidence
        affects the final probabilities.

    baseline : float, optional
        Reference probability separating positive and negative evidence.
        Should be between 0 and 1. If `None`, the mean value of `p_map` is used.

    average_weights : bool, optional
        If `True`, normalize accumulated neighbor evidence by the
        available kernel weight at each position to correct boundary
        effects.

    Returns
    -------
    ndarray
        Probability map enhanced by neighbor evidence.

    Notes
    -----
    Neighbor evidence is computed as:

    ``logit(p) - logit(baseline)``

    and convolved with the supplied kernel. The resulting evidence is added
    to the original log-odds before transforming back to probability space.
    """

    if baseline is None:
        baseline = p_map.mean()

    kernel = kernel.copy()
    center_row = kernel.shape[0] // 2
    center_col = kernel.shape[1] // 2
    kernel[center_row][center_col] = 0.0
    kernel = kernel/kernel.sum()
    p_map = np.clip(p_map, eps, 1 - eps)
    log_odds = logit(p_map)

    evidence = log_odds - logit(baseline)
    neighbor_evidence = convolve2d(evidence, kernel, mode="same")
    # Correct for boundaries: normalize by available kernel weight.
    available_weight = convolve2d(np.ones_like(p_map), kernel, mode="same")
    if average_weights:
        neighbor_evidence = np.divide(neighbor_evidence, available_weight, out=np.zeros_like(p_map), where=available_weight > 0) # Uncomment for averaging
    final_log_odds = log_odds + neighbors_strength * neighbor_evidence
    return expit(final_log_odds)

def prod_neighbors2d(p_map, image, noise_p_map, kernel, method = "simple", logit_alpha = 1.0):
    """
    Combine a probability map with a neighbor-derived probability map
    through multiplication.

    Parameters
    ----------
    p_map : ndarray
        Base probability map.

    image : kernel : ndarray, optional
        Original input image.

    noise_p_map : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Kernel used to construct the neighbor probability map.

    method : {"simple", "logit", "layers"}, optional
        Method used when generating the neighbor probability map.

    logit_alpha : float, optional
        Scaling parameter used by the ``"logit"`` method.

    Returns
    -------
    ndarray
        Probability map obtained by multiplying the original and
        neighbor-derived probabilities.

    Notes
    -----
    This approach is conservative: probabilities are reduced unless both
    the local sample and its neighborhood indicate the presence of signal.
    """
    p_map_neighbors = _make_neighbors_map(image, noise_p_map, kernel.copy(), method, logit_alpha)
    return p_map * p_map_neighbors

def mean_neighbors2d(p_map, image, noise_p_map, kernel, method = "simple", logit_alpha = 1.0, neighbors_weight = 0.5):
    """
    Combine a probability maps with a neighbor-derived probability map
    through weighted averaging.

    Parameters
    ----------
    p_map : ndarray, optional
        Base probability map.

    image : ndarray, optional
        Original input image.

    noise_p_map : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Kernel used to construct the neighbor probability map.

    method : {"simple", "logit", "layers"}, optional
        Method used when generating the neighbor probability map.

    logit_alpha : float, optional
        Scaling parameter used by the ``"logit"`` method.

    neighbors_weight : float, optional
        Relative contribution of the neighbor probability map.
        A value of 0 returns the original image, while a value of 1
        returns the neighbor-derived map.

    Returns
    -------
    ndarray
        Weighted-average probability map.

    Notes
    -----
    The combined probability is computed as

    ``p = w * p_neighbors + (1 - w) * p_map``

    where ``w = neighbors_weight``.
    """
    p_map_neighbors = _make_neighbors_map(image, noise_p_map, kernel.copy(), method, logit_alpha)
    average_p_map = neighbors_weight*p_map_neighbors + (1-neighbors_weight)*p_map
    return average_p_map

def _make_neighbors_map(image, noise_p_map, kernel, method = "simple", logit_alpha = 1.0):
    """
    Construct a probability map using only neighboring samples.

    This helper generates a signal-probability map in the same way as
    the main probability estimation methods, but excludes the contribution
    of the current sample by setting the central kernel weight to zero.
    It is primarily used by neighbor-based probability boosting methods.

    Parameters
    ----------
    image : ndarray, optional
        Original input image.

    noise_p_map : array-like
        Per-sample probabilities derived from the noise model.

    kernel : array-like
        Convolution kernel defining the neighborhood. The central element
        is removed and the remaining weights are renormalized so that only
        neighboring samples contribute.

    method : {"simple", "logit", "layers"}, optional
        Method used to compute the neighbor probability map.

        - ``"simple"``: kernel-weighted averaging of probabilities.
        - ``"logit"``: combines probabilities in log-odds space.
        - ``"layers"``: layered probability estimation.

    logit_alpha : float, optional
        Scaling factor used by the ``"logit"`` method.

    Returns
    -------
    ndarray
        Probability map computed exclusively from neighboring samples.

    Notes
    -----
    Unlike the main probability estimation functions, this helper removes
    the central kernel weight before computing probabilities. The resulting
    map therefore represents the neighborhood's assessment of whether
    a sample belongs to signal or noise, independent of the sample itself.
    """

    center_row = kernel.shape[0] // 2
    center_col = kernel.shape[1] // 2
    kernel[center_row][center_col] = 0.0
    kernel = kernel / kernel.sum()
    if method == "logit":
        p_map_neighbors = _get_p_logit2d(noise_p_map, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_map_neighbors = _get_p_layers2d(image, noise_p_map, kernel)
    else: #simple
        p_map_neighbors = _get_p_simple2d(noise_p_map, kernel)
    return p_map_neighbors


def _get_p_simple2d(p_map, kernel):
    """
    Convolve probability map (p_map) with the input kernel.
    """
    p_out = convolve2d(p_map, kernel, mode='same', boundary='symm')
    return p_out

def _get_p_logit2d(p_map, kernel, alpha = 1.0):
    """
    Transform probability map (p_map) into logits, convolve with the supplied
    kernel, and transform back into a probability map.

    alpha is the scaling factor applied to the aggregated logits before converting
    back to probabilities. Values greater than one amplify local contrast,
    while values less than one reduce it.

    Notes
    -----
    Pixels with probability equal to one are clipped slightly below one
    before the logit transformation to avoid numerical singularities.
    """

    p_map[p_map >= 1] = 1-1e-9
    logit = np.log(p_map / (1 - p_map))
    L = convolve2d(logit, kernel, mode='same', boundary='symm')  # weighted sum of logit
    p_out = 1.0 / (1.0 + np.exp(-alpha * L))
    return p_out

def _get_p_layers2d(image, p_map, kernel, symmetric = False, square_map = False):
    """
    Generate a probability map using a two-layer probability refinement
    scheme.

    An initial probability map (p_map) is first convolved with the provided kernel.
    The image is then weighted by this smoothed map, and a second probability map is
    estimated from the resulting intensity distribution. The final output
    is obtained by combining the two probability layers.
    """

    p_map_cnv = convolve2d(p_map, kernel, mode='same', boundary='symm')
    image_p_map_cnv = image * p_map_cnv
    image_p_map_cnv_noise_lvl, image_p_map_cnv_noise_std = sigma_clipping(image_p_map_cnv, robust=False)
    if symmetric:
        if square_map:
            p_map_new_layer = norm.cdf(image_p_map_cnv**2, loc=image_p_map_cnv_noise_lvl, scale=image_p_map_cnv_noise_std)
        else:
            p_map_new_layer = norm.cdf(np.abs(image_p_map_cnv), loc=image_p_map_cnv_noise_lvl, scale=image_p_map_cnv_noise_std)
    else:
        p_map_new_layer = norm.cdf(image_p_map_cnv, loc=image_p_map_cnv_noise_lvl, scale=image_p_map_cnv_noise_std)
    p_map_new_layer_cnv = convolve2d(p_map_new_layer, kernel, mode='same', boundary='symm')
    p_out = p_map_cnv * p_map_new_layer_cnv
    return p_out

def _apply_p_map(image, p_map):
    """
    Apply a probability map (p_map) to an image and rescale according to their maxima.
    """

    image_p_map = image * p_map
    image_p_map_leveled = image_p_map * image.max() / image_p_map.max()
    return image_p_map_leveled

if __name__ == "__main__":
    quit()