import numpy as np
import pytest
import math

from star_tracker.algorithms.psf_fitting import gauss2d_rot, ring_median, fit_psf_on_candidate


def synthetic_gaussian_2d(
    h=15, w=15,
    A=100.0, x0=7.2, y0=6.8,
    sx=1.8, sy=2.2, theta=0.3,
    offset=30.0, noise=1.0, seed=42
) -> tuple[np.ndarray, dict]:
    """
    Returns synthetic 2D Gaussian image with noise and true parameters.
    """
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx.astype(float), yy.astype(float)
    model = gauss2d_rot((X, Y), A, x0, y0, sx, sy, theta, offset).reshape(h, w)
    img = model + rng.normal(0, noise, size=model.shape)

    return img.astype(np.float64), dict(A=A, x0=x0, y0=y0, sx=sx, sy=sy, theta=theta, offset=offset)


def test_gauss2d_rot_symmetry():
    """
    Test symmetry of the 2D Gaussian function.
    """
    h, w = 9, 9
    yy, xx = np.mgrid[0:h, 0:w]
    I = gauss2d_rot((xx, yy), 100.0, 4.0, 4.0, 2.0, 2.0, 0.0, 10.0).reshape(h, w)

    iy, ix = np.unravel_index(np.argmax(I), I.shape)

    assert abs(ix - 4) <= 1
    assert abs(iy - 4) <= 1

    diff = np.abs(I - np.flipud(np.fliplr(I)))

    assert np.mean(diff) < 1e-6

def test_gauss2d_rot_rotation_effect():
    """
    Test that rotation parameter affects the Gaussian shape.
    """
    h, w = 15, 15
    yy, xx = np.mgrid[0:h, 0:w]

    I0 = gauss2d_rot((xx, yy), 100.0, 7.0, 7.0, 1.0, 3.0, 0.0, 0.0)
    I1 = gauss2d_rot((xx, yy), 100.0, 7.0, 7.0, 1.0, 3.0, np.pi/4, 0.0)

    assert np.mean(np.abs(I1 - I0)) > 1.0


def test_ring_median_behavior():
    """
    Test ring_median function on synthetic data.
    """
    img = np.full((9, 9), 100.0)
    img[4, 4] = 1000.0
    med = ring_median(img, ring=1)

    assert abs(med - 100.0) < 1e-6


def test_ring_median_small_image():
    """
    Test ring_median on a small image.
    """
    img = np.full((3, 3), 55.0)
    val = ring_median(img, ring=1)

    assert val == 55.0
