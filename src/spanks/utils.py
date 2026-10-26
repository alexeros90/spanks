import numpy as np

def smoothen_data(level = 1, data = None):
    """
    Iteratively smooth a one-dimensional dataset by averaging adjacent
    samples.

    Parameters
    ----------
    level : int, optional
        Number of smoothing iterations to perform. Higher values produce
        progressively smoother curves at the expense of resolution.
        Default is ``1``.

    data : ndarray
        Coordinate values of the dataset.

    Returns
    -------
    smooth_data : ndarray
        Smoothed array.
    """
    if data is None:
        return None
    smooth_data = data
    for i in range(level):
        smooth_data = (smooth_data[0:-1] + smooth_data[1:])/2
    return smooth_data

def get_padding_1D(x, y, percentages = None, left_size = 100, right_size = 100):
    """
    Generate synthetic padding regions for a one-dimensional dataset.

    The function estimates representative statistics from both ends of
    the input data using sigma-clipped measurements over several sample
    sizes. For each side, the sample size yielding the smallest clipped
    standard deviation is selected. Random Gaussian padding values are
    then generated using the corresponding mean and standard deviation.

    This can be useful when extending a signal prior to convolution,
    filtering, wavelet decomposition, or Fourier-based processing.

    Parameters
    ----------
    x : ndarray
        Coordinate array.

    y : ndarray
        Data values.

    percentages : sequence of float, optional
        Candidate percentages of the dataset length used to estimate
        edge statistics. The percentage producing the smallest
        sigma-clipped standard deviation is adopted independently for
        the left and right sides.

        If ``None``, ``(1, 2, 5, 10)`` is used.

    left_size : int, optional
        Number of padding samples to generate on the left side.
        Default is ``100``.

    right_size : int, optional
        Number of padding samples to generate on the right side.
        Default is ``100``.

    Returns
    -------
    (left_pad_x, right_pad_x) : tuple of ndarray
        Coordinate arrays for the left and right padding regions.

    (left_pad_y, right_pad_y) : tuple of ndarray
        Random Gaussian padding values generated from the selected
        edge statistics.

    Notes
    -----
    The left and right edge statistics are determined independently.
    Sigma clipping is used to reduce the influence of outliers when
    estimating the mean and standard deviation of each edge region.

    See Also
    --------
    sigma_clipping
    """

    if percentages is None:
        percentages = (1, 2, 5, 10)

    results = []

    for perc in percentages:
        sample_size = max(1, round(len(y) * perc / 100))

        left_mean, left_std = sigma_clipping(y[:sample_size])
        right_mean, right_std = sigma_clipping(y[-sample_size:])

        results.append({
            "sample_size": sample_size,
            "left": (left_mean, left_std),
            "right": (right_mean, right_std),
        })

    best_left = min(results, key=lambda r: r["left"][1])
    best_right = min(results, key=lambda r: r["right"][1])

    left_mean, left_std = best_left["left"]
    right_mean, right_std = best_right["right"]

    left_pad_y = np.random.normal(left_mean, left_std, left_size)
    right_pad_y = np.random.normal(right_mean, right_std, right_size)

    left_pad_x = x[:left_size] - x[left_size] + x[0]
    right_pad_x = x[-right_size:] - x[-right_size - 1] + x[-1]

    return (left_pad_x, right_pad_x), (left_pad_y, right_pad_y)

def sigma_clipping(image, sigma_factor=3, convergence_tolerance=0.001, iteration_limit=50, use_median=False,
                   use_iqr = False, robust=False, direction = "symmetric"):
    """
    Perform sigma clipping on an image to estimate the central value and spread.

    Parameters
    ----------
    image : 2D array
        Input image for sigma clipping.
    sigma_factor : float, optional
        Factor to determine the clipping threshold. Default is 3.
    convergence_tolerance : float, optional
        Tolerance for convergence of the central measure. Default is 0.001.
    iteration_limit : int, optional
        Maximum number of iterations for the clipping process. Default is 50.
    use_median : bool, optional
        Whether to use median instead of mean for the central measure. Default is False.
    use_iqr : bool, optional
        Whether to use interquartile range (IQR) instead of standard deviation for the spread. Default is False.
    robust : bool, optional
        Whether to use a robust method for outlier rejection. Default is False.
    direction : str, optional
        Direction of clipping: "up", "down", or "symmetric". Default is "symmetric".

    Returns
    -------
    central_measure : float
        The estimated central value (mean or median) after sigma clipping.
    sigma : float
        The estimated spread (standard deviation or IQR) after sigma clipping.
    """


    # Will return mean and standard deviation or median and iqr depending on use_median
    # If convergence_tolerance, the clipping will go on until there is no change in the central measurement or the iteration_limit is exceeded

    def trim_data(data_in, direction, central_measure, sigma_factor, sigma):
        if direction == "up":
            threshold = central_measure - sigma_factor * sigma
            data_out = data_in > threshold
        elif direction == "down":
            threshold = central_measure + sigma_factor * sigma
            data_out = data_in < threshold
        else:
            threshold_hi = central_measure + sigma_factor * sigma
            threshold_lo = central_measure - sigma_factor * sigma
            data_out = (data_in < threshold_hi) & (data_in > threshold_lo)
        return data_out

    def get_measurements(data_in, use_median, use_iqr):
        if use_median:
            central_measure = np.median(data_in)
        else:
            central_measure = np.mean(data_in)

        if use_iqr:
            q25, q75 = np.percentile(data_in, [25, 75])
            sigma = q75 - q25
        else:
            sigma = np.std(data_in, ddof=0)

        return central_measure, sigma

    iter_count = 0
    mask = np.ones(image.shape, dtype=bool)

    # Initial stats
    masked_image = image[mask]
    central_measure, sigma = get_measurements(masked_image, use_median, use_iqr)

    # Initial threshold
    new_mask = trim_data(image, direction, central_measure, sigma_factor, sigma)

    while iter_count < iteration_limit:
        masked_image = image[new_mask]
        if robust:
            q_lo, q_hi = np.percentile(masked_image, [5, 95])
            robust_mask = (masked_image >= q_lo) & (masked_image <= q_hi)
            trimmed = masked_image[robust_mask]
        else:
            trimmed = masked_image

        new_central, new_sigma = get_measurements(trimmed, use_median, use_iqr)

        # Check convergence
        convergence_threshold = np.abs(sigma) * convergence_tolerance
        mean_convergence = np.abs(new_central - central_measure) <= convergence_threshold
        std_convergence = np.abs(new_sigma - sigma) <= convergence_threshold
        if mean_convergence and std_convergence:
            break

        if new_mask.sum() == mask.sum():
            break

        central_measure = new_central
        sigma = new_sigma
        mask = new_mask

        new_mask = trim_data(image, direction, central_measure, sigma_factor, sigma)

        iter_count += 1

    return central_measure, sigma

