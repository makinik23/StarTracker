import math
import os
import numpy as np
import matplotlib
import pytest


from star_tracker.algorithms.moment_tracker import (
    robust_threshold,
    label_components,
    compute_blob_moments,
    BlobMoment,
)


def _gaussian_spot(h=128, w=128, cx=40.7, cy=70.3, sigma=2.0, peak=180.0, bg=30.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), bg, dtype=float)
    g = peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    img += g
    return np.clip(img, 0, 255).astype(np.uint8)


def _detect_by_threshold(img, k=4.0):
    _, _, T = robust_threshold(img, k=k)
    mask = (img.astype(float) > T).astype(np.uint8)
    labels, _ = label_components(mask)
    return labels


@pytest.mark.parametrize("k", [3.0, 4.0, 5.0])
def test_robust_threshold_noise_only(k):
    rng = np.random.default_rng(123)
    img = rng.normal(loc=30.0, scale=2.0, size=(128, 128)).astype(np.float32)
    bg, sigma, T = robust_threshold(img, k=k)

    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    sigma_expected = 1.4826 * mad if mad > 0 else 3.0
    T_expected = med + k * sigma_expected

    assert bg == pytest.approx(med, abs=0.01)
    assert sigma == pytest.approx(sigma_expected, abs=0.01)
    assert T == pytest.approx(T_expected, abs=0.01)


def test_robust_threshold_with_bright_spot():
    img = _gaussian_spot()
    bg, sigma, T = robust_threshold(img, k=5.0)

    assert bg == pytest.approx(30.0, abs=0.1)
    assert T > bg + 5.0


def test_label_components_two_islands():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[1:3, 1:3] = 1
    m[6:9, 7:9] = 1
    labels, n = label_components(m)

    assert n == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_label_components_empty_mask():
    m = np.zeros((8, 8), dtype=np.uint8)
    labels, n = label_components(m)

    assert n == 0
    assert labels.shape == m.shape
    assert (labels == 0).all()


def test_compute_blob_moments_single_gaussian_centroid_accuracy():
    cx_true, cy_true = 40.7, 70.3
    img = _gaussian_spot(cx=cx_true, cy=cy_true, sigma=2.0)

    labels = _detect_by_threshold(img, k=4.0)
    blobs = compute_blob_moments(img, labels)

    assert len(blobs) == 1

    bm = blobs[0]

    assert bm.cx == pytest.approx(cx_true, abs=0.1)
    assert bm.cy == pytest.approx(cy_true, abs=0.1)
    assert bm.m00 > 0.0


def test_compute_blob_moments_two_gaussians():
    img = _gaussian_spot(cx=30.2, cy=30.1, sigma=1.8, peak=150.0)
    img = img.astype(float)
    img += _gaussian_spot(cx=90.4, cy=95.6, sigma=2.2, peak=200.0).astype(float)
    img = np.clip(img, 0, 255).astype(np.uint8)

    labels = _detect_by_threshold(img, k=4.0)
    blobs = compute_blob_moments(img, labels)

    assert len(blobs) == 2

    blobs = sorted(blobs, key=lambda b: b.cx)
    (b1, b2) = blobs

    assert b1.cx == pytest.approx(30.2, abs=0.1) and b1.cy == pytest.approx(30.1, abs=0.1)
    assert b2.cx == pytest.approx(90.4, abs=0.1) and b2.cy == pytest.approx(95.6, abs=0.1)
