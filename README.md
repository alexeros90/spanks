# SPANKS (SPectrAl Noise Killer Script)

SPANKS is a lightweight Python package for suppressing noise in one-dimensional spectra and two-dimensional images using a probability-based approach, facilitating the detection of faint lines (1D) or sources (2D). Rather than applying a simple smoothing filter, SPANKS estimates how likely each sample or pixel is to belong to the background noise distribution and uses this information to attenuate noise-dominated regions while preserving statistically significant structures.

The package was originally designed for astronomical applications, but it can be applied to any signal or image where the background noise is approximately Gaussian.

---

## Features

- Noise suppression for **1D spectra and profiles**
- Noise suppression for **2D images**
- Automatic estimation of background level and noise standard deviation
- Multiple probability-refinement strategies:
  - `simple`
  - `logit`
  - `layers`
- Support for both positive-only and symmetric signal distributions
- Preservation of significant peaks, emission lines, absorption lines, and image structures
- Fully NumPy-based workflow
- Includes functions for creating synthetic spectra and sky images for testing purposes.

---

## Concept

SPANKS assumes that the background noise follows an approximately normal distribution.

The workflow is:

1. Estimate the noise properties using sigma clipping.
2. Compute a noise-probability profile/map.
3. Refine the probability distribution through local aggregation.
4. Use the resulting probabilities as adaptive weights.
5. Suppress background-dominated regions while preserving significant structures.

Unlike conventional smoothing filters, SPANKS attempts to distinguish between signal and noise statistically rather than averaging everything indiscriminately.

---

# Getting started

## `suppress_noise1d`

Suppress background-dominated samples in a one-dimensional profile or spectrum.

### Example

```python
from spanks import suppress_noise1d

clean_spectrum = suppress_noise1d(
    spectrum,
    method="layers"
)
```

---

## `suppress_noise2d`

Suppress background-dominated pixels in a two-dimensional image.

### Example

```python
from spanks import suppress_noise2d

clean_image = suppress_noise2d(
    image,
    method="simple"
)
```

## License

See the [`LICENSE`](LICENSE) file for licensing information.

---
- **GitHub:** [View the source code](https://github.com/alexeros90/spanks)
- **PyPI:** [View the package on PyPI](https://pypi.org/project/spanks/)
- **Issues:** [Report a bug or request a feature](https://github.com/alexeros90/spanks/issues)
---

*"Because sometimes the best way to find a signal is to give the noise a good spanking."*