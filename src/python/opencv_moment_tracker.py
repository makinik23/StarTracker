import argparse
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_synthetic_starfield(H: int = 512, W: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=30.0, scale=2.2, size=(H, W))

    def add_gauss(cx, cy, sigma_x, sigma_y, theta, peak):
        yy, xx = np.mgrid[0:H, 0:W]
        x0 = xx - cx
        y0 = yy - cy
        ct = np.cos(theta)
        st = np.sin(theta)
        xr =  x0*ct + y0*st
        yr = -x0*st + y0*ct
        g = peak * np.exp(-(xr**2/(2*sigma_x**2) + yr**2/(2*sigma_y**2)))
        return g

    for _ in range(40):
        cx = rng.uniform(20, W-20)
        cy = rng.uniform(20, H-20)
        sig = rng.uniform(1.2, 2.5)
        peak = rng.uniform(80, 220)
        img += add_gauss(cx, cy, sig, sig, 0.0, peak)

    for ang_deg, length in [(25, 6.0), (-40, 8.0)]:
        cx = rng.uniform(60, W-60)
        cy = rng.uniform(60, H-60)
        theta = np.deg2rad(ang_deg)
        img += add_gauss(cx, cy, 1.2*length, 1.5, theta, 180)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def preprocess_opencv(gray: np.ndarray,
                      use_clahe: bool = True,
                      blur_ksize: int = 3,
                      thresh_mode: str = "adaptive",
                      block_size: int = 51,
                      C: int = -3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (img_eq, mask_bin)
    - CLAHE poprawia kontrast,
    - lekki blur gasi hot-pixele,
    - threshold: "adaptive" (Gaussian) lub "otsu".
    """
    img = gray

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    if thresh_mode == "adaptive":
        mask = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
    elif thresh_mode == "otsu":
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("thresh_mode must be 'adaptive' or 'otsu'.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return img, mask


@dataclass
class Blob:
    label: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    centroid_geom: Tuple[float, float]  # z OpenCV (geometriczne)
    centroid_int: Tuple[float, float]   # z momentów intensywności
    theta_rad: float
    major_std: float
    minor_std: float
    m00: float

def analyze_components(gray: np.ndarray, mask: np.ndarray, min_area: int = 5, min_mass: float = 200.0) -> List[Blob]:
    """
    Dla każdej etykiety:
      - bierzemy ROI,
      - liczymy momenty na obrazie I = gray * (mask>0),
      - wyciągamy centroid intensywności, macierz kowariancji, oś główną.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = gray.shape
    out: List[Blob] = []

    for L in range(1, n_labels):
        x, y, w, h, area = stats[L]
        if area < min_area:
            continue

        cx_g, cy_g = centroids[L]

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        roi_gray = gray[y0:y1, x0:x1]
        roi_mask = (labels[y0:y1, x0:x1] == L).astype(np.uint8)

        roi_int = (roi_gray.astype(np.float32)) * (roi_mask.astype(np.float32))

        M = cv2.moments(roi_int, binaryImage=False)

        m00 = M["m00"]
        if m00 < min_mass:
            continue

        cx = M["m10"] / m00
        cy = M["m01"] / m00

        mu20 = M["mu20"]
        mu02 = M["mu02"]
        mu11 = M["mu11"]

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

        out.append(Blob(
            label=L,
            bbox=(x, y, w, h),
            area=int(area),
            centroid_geom=(float(cx_g), float(cy_g)),
            centroid_int=(float(x0 + cx), float(y0 + cy)),
            theta_rad=float(theta),
            major_std=float(major_std),
            minor_std=float(minor_std),
            m00=float(m00)
        ))

    return out

def draw_overlays(gray: np.ndarray, blobs: List[Blob], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray, cmap="gray", origin="upper")
    for b in blobs:
        cx, cy = b.centroid_int
        ax.scatter([cx], [cy], s=20)

        L = 6.0
        x0 = cx - L * math.cos(b.theta_rad)
        y0 = cy - L * math.sin(b.theta_rad)
        x1 = cx + L * math.cos(b.theta_rad)
        y1 = cy + L * math.sin(b.theta_rad)
        ax.plot([x0, x1], [y0, y1])
    ax.set_title("OpenCV pipeline: centroids (intensity-weighted) + orientations")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_csv(blobs: List[Blob], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "cx_int", "cy_int", "theta_rad", "major_std", "minor_std", "m00", "area", "cx_geom", "cy_geom"])
        for b in blobs:
            w.writerow([
                b.label, b.centroid_int[0], b.centroid_int[1],
                b.theta_rad, b.major_std, b.minor_std, b.m00,
                b.area, b.centroid_geom[0], b.centroid_geom[1]
            ])

def main(img_path: Optional[str],
         use_clahe: bool,
         thresh_mode: str,
         block_size: int,
         C: int,
         blur_ksize: int,
         min_area: int,
         min_mass: float):

    if img_path:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"Nie można wczytać obrazu: {img_path}")
    else:
        gray = make_synthetic_starfield()
        cv2.imwrite("synthetic_starfield.png", gray)
        img_path = "synthetic_starfield.png"

    img_eq, mask = preprocess_opencv(
        gray,
        use_clahe=use_clahe,
        blur_ksize=blur_ksize,
        thresh_mode=thresh_mode,
        block_size=block_size,
        C=C
    )

    blobs = analyze_components(gray, mask, min_area=min_area, min_mass=min_mass)

    out_png = "opencv_moment_detections.png"
    out_csv = "opencv_detections_centroids.csv"
    draw_overlays(gray, blobs, out_png)
    save_csv(blobs, out_csv)

    print(f"Input image: {img_path}")
    print(f"Detections: {len(blobs)}")
    print(f"Saved: {out_png}, {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Star tracker — OpenCV pipeline + intensity-moment centroids")
    ap.add_argument("--image", type=str, default=None, help="Ścieżka do obrazu (grayscale). Brak -> generuje syntetyczny.")
    ap.add_argument("--no-clahe", action="store_true", help="Wyłącz CLAHE.")
    ap.add_argument("--thresh", type=str, default="adaptive", choices=["adaptive", "otsu"], help="Tryb binaryzacji.")
    ap.add_argument("--block-size", type=int, default=51, help="Rozmiar okna dla adaptive threshold (nieparzysty).")
    ap.add_argument("--C", type=int, default=-3, help="Stała odejmowana w adaptive threshold.")
    ap.add_argument("--blur", type=int, default=3, help="Kernel Gaussa (nieparzysty). 1 aby wyłączyć.")
    ap.add_argument("--min-area", type=int, default=5, help="Minimum area dla labelu (piksele).")
    ap.add_argument("--min-mass", type=float, default=200.0, help="Minimum m00 (suma intensywności w labelu).")
    args = ap.parse_args()

    if args.block_size % 2 == 0:
        args.block_size += 1
    if args.blur % 2 == 0:
        args.blur += 1

    main(
        args.image,
        use_clahe=not args.no_clahe,
        thresh_mode=args.thresh,
        block_size=args.block_size,
        C=args.C,
        blur_ksize=args.blur,
        min_area=args.min_area,
        min_mass=args.min_mass
    )
