import numpy as np
from astropy.modeling.models import Sersic2D

def _get_psf_sources(number_of_psf_sources, image_size, psf_fwhm_range, psf_dynamic_range, symmetric):
    """
    Generate an image containing randomly positioned PSF-like sources.

    Each source is modeled as a circular two-dimensional Gaussian profile
    with a randomly selected position, width, and peak intensity. The
    resulting source images are summed to produce a single image.

    Parameters
    ----------
    number_of_psf_sources : int
        Number of PSF sources to generate.

    image_size : tuple of int
        Output image size as ``(nx, ny)``.

    psf_fwhm_range : tuple of float
        Minimum and maximum full width at half maximum (FWHM) values for
        the generated Gaussian sources.

    psf_dynamic_range : tuple of float
        Minimum and maximum absolute source amplitudes.

    symmetric : bool
        If ``True``, sources may have either positive or negative peak
        amplitudes. If ``False``, all amplitudes are positive.

    Returns
    -------
    ndarray
        Image containing the sum of all generated PSF sources.

    Notes
    -----
    The Gaussian standard deviation is computed from the FWHM using

    ``sigma = FWHM / 2.3548``.
    """

    all_psf_image = np.zeros(image_size, dtype = float)
    y, x = np.mgrid[:image_size[1], :image_size[0]]
    x_positions = np.random.random(number_of_psf_sources) * image_size[0]
    y_positions = np.random.random(number_of_psf_sources) * image_size[1]
    sources_fwhms = np.random.random(number_of_psf_sources) * (psf_fwhm_range[1] - psf_fwhm_range[0]) + psf_fwhm_range[0]
    sources_stds = sources_fwhms/2.3548 # std = fwhm/2.3548

    if symmetric:
        peaks = (2 * np.random.random(number_of_psf_sources) - 1) * (psf_dynamic_range[1] - psf_dynamic_range[0]) #+ psf_dynamic_range[0]
        peaks = peaks + psf_dynamic_range[0] * peaks / np.abs(peaks)
    else:
        peaks = (np.random.random(number_of_psf_sources)) * (psf_dynamic_range[1] - psf_dynamic_range[0]) + psf_dynamic_range[0]
    sources = zip(x_positions, y_positions, sources_stds, peaks)
    for this_source in sources:
        sigma = this_source[2]
        source_g = np.exp(-((x-this_source[0])**2 + (y-this_source[1])**2) / (2*sigma**2))
        source_g = this_source[3] * source_g/source_g.max()
        all_psf_image += source_g
    return all_psf_image

def _get_sersic_sources(number_of_sersic_sources, image_size, r_eff_range, n_range, ellip_range, theta_range,
                       dynamic_range, symmetric = False):
    """
    Generate an image containing randomly positioned Sérsic profile sources.

    Each source is modeled using a two-dimensional Sérsic profile with
    randomly selected structural parameters and position. The generated
    source images are summed to produce a composite image.

    Parameters
    ----------
    number_of_sersic_sources : int
        Number of Sérsic sources to generate.

    image_size : tuple of int
        Output image size as ``(nx, ny)``.

    r_eff_range : tuple of float
        Minimum and maximum effective radius values.

    n_range : tuple of float
        Minimum and maximum Sérsic indices.

    ellip_range : tuple of float
        Minimum and maximum ellipticities.

    theta_range : tuple of float
        Minimum and maximum position angles in radians.

    dynamic_range : tuple of float
        Minimum and maximum source amplitudes.

    symmetric : bool, optional
        If ``True``, each source is randomly assigned either a positive
        or negative sign. Default is ``False``.

    Returns
    -------
    ndarray
        Image containing the sum of all generated Sérsic sources.

    See Also
    --------
    astropy.modeling.functional_models.Sersic2D
    """


    all_sersic_image = np.zeros(image_size, dtype = float)
    y, x = np.mgrid[:image_size[1], :image_size[0]]
    x_positions = np.random.random(number_of_sersic_sources) * image_size[0]
    y_positions = np.random.random(number_of_sersic_sources) * image_size[1]
    r_eff = np.random.random(number_of_sersic_sources) * (r_eff_range[1] - r_eff_range[0]) + r_eff_range[0]
    n = np.random.random(number_of_sersic_sources) * (n_range[1] - n_range[0]) + n_range[0]
    ellip = np.random.random(number_of_sersic_sources) * (ellip_range[1] - ellip_range[0]) + ellip_range[0]
    theta = np.random.random(number_of_sersic_sources) * (theta_range[1] - theta_range[0]) + theta_range[0]
    peaks = (np.random.random(number_of_sersic_sources)) * (dynamic_range[1] - dynamic_range[0]) + dynamic_range[0]
    sources = zip(x_positions, y_positions, r_eff, n, ellip, theta, peaks)
    for this_source in sources:
        sersic_model = Sersic2D(amplitude=this_source[6], r_eff=this_source[2], n=this_source[3], x_0=this_source[0], y_0=this_source[1], ellip=this_source[4], theta=this_source[5])
        galaxy = sersic_model(x, y)
        galaxy = this_source[6] * galaxy/galaxy.max()
        if symmetric:
            rd = np.random.rand() - 0.5
            galaxy = galaxy * (rd/np.abs(rd))
        all_sersic_image += galaxy
    return all_sersic_image

def get_synthetic_sky_image(image_size = None, noise_std = 0.125, number_of_psf_sources = 25, psf_fwhm_range = None,
                            psf_dynamic_range = None, number_of_sersic_sources = 8, r_eff_range = None,  n_range = None,
                            ellip_range = None, theta_range = None, sersic_dynamic_range = None,
                            gradient_level = 0.5, gradient_slope = 1.5, symmetric = False):
    """
    Generate a synthetic astronomical image containing noise,
    background structure, point sources, and extended galaxies.

    The generated image consists of four components:

    * Gaussian-distributed background noise.
    * A large-scale background gradient.
    * PSF-like point sources represented by Gaussian profiles.
    * Extended sources represented by Sérsic profiles.

    This function is intended for testing and benchmarking image
    processing, source detection, and background estimation algorithms.

    Parameters
    ----------
    image_size : tuple of int, optional
        Output image size as ``(nx, ny)``.
        Default is ``(1000, 1000)``.

    noise_std : float, optional
        Standard deviation of the Gaussian background noise.
        Default is ``0.125``.

    number_of_psf_sources : int, optional
        Number of PSF sources to generate.
        Default is ``25``.

    psf_fwhm_range : tuple of float, optional
        Range of PSF FWHM values. Default is ``(1000, 1000)``.

    psf_dynamic_range : tuple of float, optional
        Range of PSF amplitudes. Default is ``(1, 4)``.

    number_of_sersic_sources : int, optional
        Number of Sérsic sources to generate.
        Default is ``8``.

    r_eff_range : tuple of float, optional
        Range of effective radii for Sérsic profiles.
        Default is ``(100, 170)``.

    n_range : tuple of float, optional
        Range of Sérsic indices.
        Default is ``(1, 4)``.

    ellip_range : tuple of float, optional
        Range of Sérsic ellipticities.
        Default is ``(0, 0.8)``.

    theta_range : tuple of float, optional
        Range of Sérsic position angles in radians.
        Default is ``(0, pi)``.

    sersic_dynamic_range : tuple of float, optional
        Range of Sérsic source amplitudes.
        Default is ``(1, 4)``.

    gradient_level : float, optional
        Constant background level of the smooth gradient component.
        Default is ``0.5``.

    gradient_slope : float, optional
        Controls the strength of the background gradient.
        Default is ``1.5``.

    symmetric : bool, optional
        If ``True``, generated sources may have positive or negative
        amplitudes. Default is ``False``.

    Returns
    -------
    ndarray
        Synthetic image containing background noise, gradient,
        PSF sources, and Sérsic sources.

    See Also
    --------
    _get_psf_sources
    _get_sersic_sources
    _get_simple_gradient
    """


    if image_size is None:
        image_size = (1000, 1000)
    if psf_fwhm_range is None:
        psf_fwhm_range = (1000, 1000)
    if psf_dynamic_range is None:
        psf_dynamic_range = (1, 4)
    if r_eff_range is None:
        r_eff_range = (100, 170)
    if n_range is None:
        n_range = (1, 4)
    if ellip_range is None:
        ellip_range = (0, 0.8)
    if theta_range is None:
        theta_range = (0, np.pi)
    if sersic_dynamic_range is None:
        sersic_dynamic_range = (1, 4)

    psfs_image = _get_psf_sources(number_of_psf_sources, image_size, psf_fwhm_range, psf_dynamic_range, symmetric)
    sersic_image = _get_sersic_sources(number_of_sersic_sources, image_size, r_eff_range, n_range, ellip_range, theta_range, sersic_dynamic_range, symmetric)
    full_map = np.zeros(image_size, dtype = float)
    bkg_gradient = _get_simple_gradient(image_size, gradient_level, gradient_slope)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=(image_size[1], image_size[0]))
    full_map += noise + bkg_gradient + psfs_image + sersic_image
    
    return full_map

def _get_simple_gradient(image_size, level = 0.5, slope = 1.5):
    """
    Generate a simple two-dimensional background gradient.

    The gradient increases linearly along both image axes and is
    normalized such that its minimum value is approximately equal to
    ``level``.

    Parameters
    ----------
    image_size : tuple of int
        Output image size as ``(nx, ny)``.

    level : float, optional
        Baseline offset added to the gradient.
        Default is ``0.5``.

    slope : float, optional
        Gradient strength. A value of ``0`` produces a flat image.
        Default is ``1.5``.

    Returns
    -------
    ndarray
        Two-dimensional background gradient image.

    Notes
    -----
    When ``slope == 0``, a zero-valued image with shape
    ``image_size`` is returned.
    """

    y, x = np.mgrid[:image_size[1], :image_size[0]]
    if slope != 0:
        gradient = slope * (x / image_size[1] + y / image_size[0]) / 2
        gradient = level + gradient/gradient.max()
    else:
        return np.zeros(image_size, dtype = float)
    # gradient = 0.2 + 0.5 * x / image_size[1]
    return gradient

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_size = (1000, 1000)
    symmetric = False
    noise_std = 0.125

    number_of_psf_sources = 25
    psf_fwhm_range = (2, 14)
    psf_dynamic_range = (1, 4)

    number_of_sersic_sources = 8
    r_eff_range = (100, 170)
    n_range = (1, 4)
    ellip_range = (0, 0.8)
    theta_range = (0, np.pi)
    sersic_dynamic_range = (1, 4)

    gradient_level = 0.2
    gradient_slope = 2

    full_map = get_synthetic_sky_image(image_size, noise_std, number_of_psf_sources, psf_fwhm_range, psf_dynamic_range,
                                       number_of_sersic_sources, r_eff_range, n_range, ellip_range, theta_range,
                                       sersic_dynamic_range, gradient_level, gradient_slope, symmetric)
    plt.imshow(full_map)
    plt.show()
