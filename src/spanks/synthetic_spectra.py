import numpy as np
from scipy.stats import norm as norm_dist

def _get_lines(number_of_lines, which_lines, lambda_range, fwhm_range, dynamic_range):
    """
    Generate a collection of random spectral line parameters.

    The line centers, widths, and amplitudes are drawn from uniform
    distributions within the specified ranges. Depending on
    ``which_lines``, the generated amplitudes correspond to emission
    lines, absorption lines, or a mixture of both.

    Parameters
    ----------
    number_of_lines : int
        Number of spectral lines to generate.

    which_lines : {'emission', 'absorption', 'both'}
        Type of lines to generate.

        * ``'emission'``: all amplitudes are positive.
        * ``'absorption'``: all amplitudes are negative.
        * ``'both'``: amplitudes may be either positive or negative.

    lambda_range : tuple of float
        Minimum and maximum wavelength values for line centers.

    fwhm_range : tuple of float
        Minimum and maximum full width at half maximum (FWHM) values.

    dynamic_range : tuple of float
        Minimum and maximum absolute line amplitudes.

    Returns
    -------
    zip
        Iterator containing ``(center, width, amplitude)`` tuples for
        each generated spectral line.
    """
    centers = np.random.random(number_of_lines) * (lambda_range[1] - lambda_range[0]) + lambda_range[0]
    widths = np.random.random(number_of_lines) * (fwhm_range[1] - fwhm_range[0]) + fwhm_range[0]

    if which_lines == "emission":
        heights = (np.random.random(number_of_lines)) * (dynamic_range[1] - dynamic_range[0]) + dynamic_range[0]
    elif which_lines == "absorption":
        heights = (np.random.random(number_of_lines) - 1) * (dynamic_range[1] - dynamic_range[0]) - dynamic_range[0]
    else: # both
        heights = (2 * np.random.random(number_of_lines) - 1) * (dynamic_range[1] - dynamic_range[0])
        heights = heights + dynamic_range[0] * heights / np.abs(heights)

    return zip(centers, widths, heights)

def _add_lines(number_of_lines, which_lines, lambda_range, fwhm_range, dynamic_range, plain_x, full_spectrum, noise,
               noise_std, clean_peaks = False):
    """
    Add synthetic spectral lines to an existing spectrum.

    Random Gaussian emission and/or absorption features are generated
    and added to the input spectrum. Line parameters are obtained from
    ``_get_lines``.

    When ``clean_peaks=True``, the generated line profiles are blended
    with the provided noise realization such that the line cores remain
    largely unaffected by noise while the wings transition smoothly into
    the background.

    Parameters
    ----------
    number_of_lines : int
        Number of spectral lines to generate.

    which_lines : {'emission', 'absorption', 'both'}
        Type of lines to generate.

    lambda_range : tuple of float
        Allowed wavelength range for line centers.

    fwhm_range : tuple of float
        Allowed range of line FWHM values.

    dynamic_range : tuple of float
        Allowed range of line amplitudes.

    plain_x : ndarray
        Wavelength grid.

    full_spectrum : ndarray
        Spectrum to which the generated lines will be added.

    noise : ndarray
        Noise realization used when ``clean_peaks=True``.

    noise_std : float
        Standard deviation of the noise distribution.

    clean_peaks : bool, optional
        If ``True``, reduce the influence of noise near strong line
        peaks. Default is ``False``.

    Returns
    -------
    full_spectrum : ndarray
        Spectrum with the generated lines added.

    lines_loc : ndarray
        Central wavelengths of the generated spectral lines.

    See Also
    --------
    _get_lines
    """
    lines = _get_lines(number_of_lines, which_lines, lambda_range, fwhm_range, dynamic_range)
    lines_loc = []
    for line in lines:
        lines_loc.append(line[0])
        g_line = norm_dist.pdf(plain_x, loc=line[0], scale=line[1]/2.3548) # std = fwhm/2.3548
        g_line = line[2] * g_line/g_line.max()
        if clean_peaks:
            noise_weight = norm_dist.pdf(g_line, loc=0, scale=noise_std)
            noise_weight = noise_weight/noise_weight.max()
            comb = noise*noise_weight + (1-noise_weight)*g_line
            full_spectrum += comb - noise
        else:
            full_spectrum += g_line 
    return full_spectrum, np.array(lines_loc)
            
def get_synthetic_spectrum(lambda_range = None, delta_lambda = 5, number_of_lines = 10, which_lines = "both",
                           fwhm_range = None, dynamic_range = None, noise_std = 0.125, continuum_amplitude = 2.5,
                           continuum_alpha = -1.3, break_amplitude = 0, break_grow_rate = 0.03, break_lambda = 3646,
                           clean_peaks = False):
    """
    Generate a synthetic spectrum containing continuum, noise, and
    spectral lines.

    The spectrum is constructed from three components:

    * A power-law continuum.
    * An optional smooth spectral break represented by a logistic
      function.
    * Random Gaussian emission and/or absorption lines.

    Gaussian noise is added before injecting the spectral lines. The
    resulting spectrum can be used for testing, benchmarking, or
    validating spectral-analysis algorithms.

    Parameters
    ----------
    lambda_range : tuple of float, optional
        Wavelength range of the synthetic spectrum as
        ``(lambda_min, lambda_max)``. Default is ``(3000, 8000)``.

    delta_lambda : float, optional
        Approximate wavelength sampling interval. Default is ``5``.

    number_of_lines : int, optional
        Number of spectral lines to generate. Default is ``10``.

    which_lines : {'emission', 'absorption', 'both'}, optional
        Type of spectral features to include. Default is ``'both'``.

    fwhm_range : tuple of float, optional
        Range of line FWHM values. If ``None``,
        ``(10, 60)`` is used.

    dynamic_range : tuple of float, optional
        Range of line amplitudes. If ``None``,
        ``(2, 5)`` is used.

    noise_std : float, optional
        Standard deviation of the Gaussian noise. Default is ``0.125``.

    continuum_amplitude : float, optional
        Multiplicative normalization of the continuum component.
        Default is ``2.5``.

    continuum_alpha : float, optional
        Spectral index of the power-law continuum:

        ``continuum ~ lambda**continuum_alpha``

        Default is ``-1.3``.

    break_amplitude : float, optional
        Amplitude of the smooth spectral break. A value of ``0``
        disables the break. Default is ``0``.

    break_grow_rate : float, optional
        Controls the steepness of the spectral break transition.
        Default is ``0.03``.

    break_lambda : float, optional
        Central wavelength of the spectral break. Default is ``3646``.

    clean_peaks : bool, optional
        If ``True``, reduce the impact of noise near strong line peaks.
        Default is ``False``.

    Returns
    -------
    plain_x : ndarray
        Wavelength grid of the synthetic spectrum.

    full_spectrum : ndarray
        Simulated spectrum containing continuum, noise, and spectral
        features.

    lines_loc : ndarray
        Central wavelengths of the generated spectral lines.

    See Also
    --------
    _get_lines
    _add_lines
    """

    if lambda_range is None:
        lambda_range = (3000, 8000)
    if fwhm_range is None:
        fwhm_range = (10, 60)
    if dynamic_range is None:
        dynamic_range = (2, 5)

    resolution = (lambda_range[1] - lambda_range[0])//delta_lambda
    plain_x = np.linspace(lambda_range[0], lambda_range[1], resolution)
    full_spectrum = np.zeros_like(plain_x, dtype = float)

    spec_break = break_amplitude / (1 + np.exp(-break_grow_rate * (plain_x - break_lambda))) - break_amplitude
    continuum = plain_x**continuum_alpha
    continuum = continuum_amplitude*continuum/continuum.max()
    continuum += spec_break

    noise = np.random.normal(scale = noise_std, size=len(plain_x))
    full_spectrum += noise
    full_spectrum, lines_loc = _add_lines(number_of_lines, which_lines, lambda_range, fwhm_range, dynamic_range,
                                          plain_x, full_spectrum, noise, noise_std, clean_peaks)
    full_spectrum += continuum
    return plain_x, full_spectrum, lines_loc

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lambda_range = (3000, 8000)
    unit_size = 5
    number_of_lines = 13
    dynamic_range = (0.5, 3.5)
    fwhm_range = (10, 60)
    noise_std = 0.125
    continuum_amplitude = 4
    continuum_alpha = -1.3
    break_amplitude = 2
    break_grow_rate = 0.03
    break_lambda = 3700
    clean_peaks = False

    plain_x, full_spectrum, lines_loc = get_synthetic_spectrum()
    print("Spectral lines located at:", lines_loc)
    fig = plt.figure(figsize = (15,4))
    plt.plot(plain_x, full_spectrum, '-', color='black', lw = 0.5)
    plt.xlim(plain_x[0], plain_x[-1])
    plt.show()
    quit()

