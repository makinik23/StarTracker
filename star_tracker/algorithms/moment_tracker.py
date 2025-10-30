import argparse
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


from PIL import Image


def make_synthetic_starfield(H: int = 512, W: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=30.0, scale=2.0, size=(H, W))

    def gaussian_spot(cx: float, cy: float, sigma: float, peak: float) -> np.ndarray:
        """
        Generate a 2D Gaussian spot centered at specified coordinates.
        
        Creates a 2D Gaussian distribution over a grid defined by global variables H and W,
        centered at (cx, cy) with specified standard deviation and peak intensity.
        
        Args:
            cx (float): X-coordinate of the Gaussian center
            cy (float): Y-coordinate of the Gaussian center  
            sigma (float): Standard deviation of the Gaussian distribution
            peak (float): Peak intensity value at the center of the Gaussian
            
        Returns:
            numpy.ndarray: 2D array of shape (H, W) containing the Gaussian spot values
            
        Note:
            Requires global variables H and W to be defined for grid dimensions.
        """
        yy, xx = np.mgrid[0:H, 0:W]
        return peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    for _ in range(40):
        cx = rng.uniform(20, W - 20)
        cy = rng.uniform(20, H - 20)
        sigma = rng.uniform(1.2, 2.5)
        peak = rng.uniform(80, 220)
        img += gaussian_spot(cx, cy, sigma, peak)

    for ang_deg, length in [(25, 6.0), (-40, 8.0)]:
        cx = rng.uniform(60, W - 60)
        cy = rng.uniform(60, H - 60)
        theta = np.deg2rad(ang_deg)
        yy, xx = np.mgrid[0:H, 0:W]
        x0 = xx - cx
        y0 = yy - cy
        xr = x0 * np.cos(theta) + y0 * np.sin(theta)
        yr = -x0 * np.sin(theta) + y0 * np.cos(theta)
        sigma_x = 1.2 * length
        sigma_y = 1.5
        peak = 180
        g = peak * np.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
        img += g

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

    
def robust_threshold(img: np.ndarray, k: float = 5.0) -> Tuple[float, float, float]:
    """
    Calculates robust background statistics and threshold.

    Args:
        img (np.ndarray): Input image array.
        k (float): Scaling factor for thresholding.
    
    Returns:
        Background level, noise sigma, and threshold value.
    """
    bg = np.median(img)
    mad = np.median(np.abs(img - bg))
    sigma_n = 1.4826 * mad if mad > 0 else 3.0
    thresh = bg + k * sigma_n

    return bg, sigma_n, thresh


def label_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Perform two pass CCL with Union Find algorithm using 8-connectivity.

    Args:
        binary_mask (np.ndarray): Binary image mask.
    
    Returns:
        Labeled image and number of components.
    """
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    next_label = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] == 0:
                continue

            neigh = []

            if y > 0 and x > 0 and labels[y-1, x-1] > 0: neigh.append(labels[y-1, x-1])
            if y > 0 and labels[y-1, x] > 0: neigh.append(labels[y-1, x])
            if y > 0 and x+1 < w and labels[y-1, x+1] > 0: neigh.append(labels[y-1, x+1])
            if x > 0 and labels[y, x-1] > 0: neigh.append(labels[y, x-1])

            if not neigh:
                labels[y, x] = next_label
                parent.append(next_label)
                parent[next_label] = next_label
                next_label += 1

            else:
                m = min(neigh)
                labels[y, x] = m
                for n in neigh:
                    union(m, n)

    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                labels[y, x] = find(labels[y, x])

    unique = np.unique(labels[labels > 0])
    label_map = {old: i + 1 for i, old in enumerate(unique)}

    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                labels[y, x] = label_map[labels[y, x]]

    return labels, len(unique)


@dataclass
class BlobMoment:
    label: int
    m00: float
    cx: float
    cy: float
    theta_rad: float
    major_std: float
    minor_std: float

def compute_blob_moments(image: np.ndarray, labels: np.ndarray) -> List[BlobMoment]:
    """
    Calculates image moments for each labeled blob.

    Args:
        image (np.ndarray): Grayscale image array.
        labels (np.ndarray): Labeled image array.
    
    Returns:
        List of BlobMoment dataclasses for each blob.
    """
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out: List[BlobMoment] = []

    for L in range(1, labels.max() + 1):
        mask = (labels == L)
        if not np.any(mask):
            continue

        I = image[mask].astype(float)
        X = xx[mask].astype(float)
        Y = yy[mask].astype(float)

        m00 = I.sum()
        m10 = (X * I).sum()
        m01 = (Y * I).sum()
        cx = m10 / m00
        cy = m01 / m00

        x_c = X - cx
        y_c = Y - cy
        mu20 = (I * x_c**2).sum()
        mu02 = (I * y_c**2).sum()
        mu11 = (I * x_c * y_c).sum()

        cov_xx = mu20 / m00
        cov_xy = mu11 / m00
        cov_yy = mu02 / m00

        theta = 0.5 * math.atan2(2 * cov_xy, (cov_xx - cov_yy))
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy**2
        disc = max(trace * trace / 4 - det, 0.0)
        lam1 = trace / 2 + math.sqrt(disc)
        lam2 = trace / 2 - math.sqrt(disc)
        major_std = math.sqrt(max(lam1, 0.0))
        minor_std = math.sqrt(max(lam2, 0.0))

        out.append(BlobMoment(L, m00, cx, cy, theta, major_std, minor_std))

    return out


def draw_detections(img: np.ndarray, moments: List[BlobMoment], out_path: str) -> None:
    """
    Draws detected blob moments on the image and saves to file.

    Args:
        img (np.ndarray): Grayscale image array.
        moments (List[BlobMoment]): List of detected blob moments.
        out_path (str): Output file path for the image.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="upper", cmap="gray")

    for m in moments:
        ax.scatter([m.cx], [m.cy], s=20)
        L = 6.0
        x0 = m.cx - L * math.cos(m.theta_rad)
        y0 = m.cy - L * math.sin(m.theta_rad)
        x1 = m.cx + L * math.cos(m.theta_rad)
        y1 = m.cy + L * math.sin(m.theta_rad)
        ax.plot([x0, x1], [y0, y1])

    ax.set_title("Moment-based centroids and orientations")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_csv(moments: List[BlobMoment], csv_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "cx", "cy", "theta_rad", "major_std", "minor_std", "m00"])
        for m in moments:
            w.writerow([m.label, m.cx, m.cy, m.theta_rad, m.major_std, m.minor_std, m.m00])


def main(image_path: Optional[str], k: float, min_mass: float) -> None:
    if image_path:
        img = np.array(Image.open(image_path).convert("L"))
    else:
        img = make_synthetic_starfield()
        Image.fromarray(img).save("synthetic_starfield.png")
        image_path = "synthetic_starfield.png"

    _, _, T = robust_threshold(img, k=k)
    mask = (img.astype(float) > T).astype(np.uint8)

    labels, _ = label_components(mask)

    moments = [m for m in compute_blob_moments(img, labels) if m.m00 >= min_mass]

    out_png = "moment_detections.png"
    out_csv = "detections_centroids.csv"
    draw_detections(img, moments, out_png)
    save_csv(moments, out_csv)

    print(f"Input image: {image_path}")
    print(f"Saved: {out_png}, {out_csv}")
    print(f"Detections: {len(moments)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Moment Analysis for star-like blobs.")
    parser.add_argument("--image", type=str, default=None, help="Ścieżka do obrazu (grayscale). Brak -> generuje syntetyczny.")
    parser.add_argument("--k", type=float, default=5.0, help="Ile sigma powyżej tła (threshold).")
    parser.add_argument("--min-mass", type=float, default=400.0, help="Filtracja: minimalna masa (m00).")
    args = parser.parse_args()
    main(args.image, k=args.k, min_mass=args.min_mass)
