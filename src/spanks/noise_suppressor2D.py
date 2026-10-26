import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from astropy.convolution import Gaussian2DKernel
from .utils import sigma_clipping

def suppress_noise2d(image, method ="simple", kernel = None, logit_alpha = 1.0, symmetric = False, abs_profile = False):
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
        kernel = Gaussian2DKernel(2.0) # Larger kernels yield higher noise level, but more symmetric noise distribution

    image_noise_lvl, image_noise_std = sigma_clipping(image, robust=False)

    # Probability of obtaining x or lower:
    if symmetric:
        if abs_profile:
            noise_p_map = norm.cdf(np.abs(image), loc=image_noise_lvl, scale=image_noise_std)
        else:
            noise_p_map = norm.cdf(image**2, loc=image_noise_lvl, scale=image_noise_std)
    else:
        noise_p_map = norm.cdf(image, loc=image_noise_lvl, scale=image_noise_std)

    if method == "logit":
        p_map = _get_p_logit2d(noise_p_map, kernel, alpha = logit_alpha)
    elif method == "layers":
        p_map = _get_p_layers2d(image, noise_p_map, kernel, symmetric, abs_profile)
    else: #simple
        p_map = _get_p_simple2d(noise_p_map, kernel)

    bkg_reduced_image_leveled = _apply_p_map(image, p_map)

    return bkg_reduced_image_leveled

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

def _get_p_layers2d(image, p_map, kernel, symmetric = False, abs_profile = False):
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
        if abs_profile:
            p_map_new_layer = norm.cdf(np.abs(image_p_map_cnv), loc=image_p_map_cnv_noise_lvl, scale=image_p_map_cnv_noise_std)
        else:
            p_map_new_layer = norm.cdf(image_p_map_cnv**2, loc=image_p_map_cnv_noise_lvl, scale=image_p_map_cnv_noise_std)
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