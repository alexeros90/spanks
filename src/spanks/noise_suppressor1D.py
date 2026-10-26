import numpy as np
from scipy.stats import norm
from scipy.signal import convolve
from astropy.convolution import Gaussian1DKernel
from .utils import sigma_clipping

def suppress_noise1d(profile, method ="simple", kernel = None, logit_alpha = 1.0, abs_profile = False):
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

    abs_profile : bool, optional
        If True will use absolute values to make to probability profile,
        else it will use squared values. Default is False.

    Returns
    -------
    ndarray
        Background-suppressed profile rescaled to approximately match the
        original profile intensity range.

    See Also
    --------
    _get_p_simple_1d
    _get_p_logit_1d
    _get_p_layers_1d
    _apply_p_profile
    """

    if kernel is None:
        kernel = Gaussian1DKernel(2.0) # Larger kernels yield higher noise level, but more symmetric noise distribution

    profile_noise_lvl, profile_noise_std = sigma_clipping(profile, robust=False)

    profile_centered = profile - profile_noise_lvl

    # Probability of obtaining x or lower:
    if abs_profile:
        noise_p_profile = norm.cdf(np.abs(profile_centered), loc=0, scale=profile_noise_std)
    else:
        noise_p_profile = norm.cdf(profile_centered**2, loc=0, scale=profile_noise_std)

    if method == "logit":
        p_prof = _get_p_logit_1d(noise_p_profile, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_prof = _get_p_layers_1d(profile, noise_p_profile, kernel)
    else: #simple
        p_prof = _get_p_simple_1d(noise_p_profile, kernel)

    bkg_reduced_image_leveled = _apply_p_profile(profile_centered, p_prof) + profile_noise_lvl

    return bkg_reduced_image_leveled

def _get_p_simple_1d(p_prof, kernel):
    p_out = convolve(p_prof, kernel, mode='same')
    return p_out

def _get_p_logit_1d(p_prof, kernel, alpha = 1.0):
    p_prof[p_prof >= 1] = 1-1e-9
    logit = np.log(p_prof / (1 - p_prof))
    L = convolve(logit, kernel, mode='same')  # weighted sum of logit
    p_out = 1.0 / (1.0 + np.exp(-alpha * L))
    return p_out

def _get_p_layers_1d(data, p_prof, kernel, abs_profile = False):
    p_prof_cnv = convolve(p_prof, kernel, mode='same')
    data_p_prof_cnv = data * p_prof_cnv
    data_p_prof_cnv_noise_lvl, data_p_prof_cnv_noise_lvl_std = sigma_clipping(data_p_prof_cnv, robust=False)
    data_p_prof_cnv_center = data_p_prof_cnv - data_p_prof_cnv_noise_lvl
    if abs_profile:
        p_prof_new_layer = norm.cdf(np.abs(data_p_prof_cnv_center), loc=0, scale=data_p_prof_cnv_noise_lvl_std)
    else:
        p_prof_new_layer = norm.cdf(data_p_prof_cnv_center**2, loc=0, scale=data_p_prof_cnv_noise_lvl_std)
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


