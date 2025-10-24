import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
        sig = rng.uniform(1.2, 2.3)
        peak = rng.uniform(80, 220)
        img += add_gauss(cx, cy, sig, sig, 0.0, peak)

    for ang_deg, length in [(25, 6.0), (-40, 8.0)]:
        cx = rng.uniform(60, W-60)
        cy = rng.uniform(60, H-60)
        theta = np.deg2rad(ang_deg)
        img += add_gauss(cx, cy, 1.2*length, 1.5, theta, 180)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def preprocess(gray: np.ndarray,
               use_clahe: bool = True,
               blur_ksize: int = 3,
               thresh_mode: str = "adaptive",
               block_size: int = 51,
               C: int = -3) -> Tuple[np.ndarray, np.ndarray]:
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
            blockSize=block_size if block_size % 2 else block_size+1,
            C=C
        )
    else:
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    return img, mask

@dataclass
class Candidate:
    label: int
    bbox: Tuple[int,int,int,int] 
    area: int
    centroid: Tuple[float,float]  
    m00: float                    
    roi_gray: np.ndarray
    roi_mask: np.ndarray
    x0: int
    y0: int

def find_candidates(gray: np.ndarray, mask: np.ndarray,
                    min_area: int = 10, min_mass: float = 600.0) -> List[Candidate]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = gray.shape
    out: List[Candidate] = []
    for L in range(1, n_labels):
        x, y, w, h, area = stats[L]
        if area < min_area: 
            continue
        x0, y0 = max(0,x), max(0,y)
        x1, y1 = min(W, x+w), min(H, y+h)
        roi_gray = gray[y0:y1, x0:x1]
        roi_mask = (labels[y0:y1, x0:x1] == L).astype(np.uint8)
        m00 = float((roi_gray * roi_mask).sum())
        if m00 < min_mass:
            continue
        cx_g, cy_g = map(float, centroids[L])
        out.append(Candidate(L, (x,y,w,h), int(area), (cx_g, cy_g), m00, roi_gray, roi_mask, x0, y0))
    return out

def gauss2d_rot(coords, A, x0, y0, sx, sy, theta, offset):
    """
    Eliptyczny Gauss 2D z rotacją + offset tła.
    coords = (X, Y) – siatki pikseli (float).
    """
    X, Y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr =  (X - x0)*ct + (Y - y0)*st
    yr = -(X - x0)*st + (Y - y0)*ct
    G = offset + A*np.exp(-(xr**2/(2*sx**2) + yr**2/(2*sy**2)))
    return G.ravel()

@dataclass
class FitResult:
    label: int
    cx: float
    cy: float
    sx: float
    sy: float
    theta: float
    A: float
    offset: float
    sig_cx: float
    sig_cy: float
    red_chi2: float
    area: int
    m00: float

def ring_median(arr: np.ndarray, ring: int = 1) -> float:
    """Mediana piks. z obwódki ROI (prosty estymator tła)."""
    h, w = arr.shape
    if min(h,w) < 2*ring+1:
        return float(np.median(arr))
    core = arr[ring:-ring, ring:-ring]
    ring_vals = arr.ravel()
    if core.size > 0:
        ring_mask = np.ones_like(arr, dtype=bool)
        ring_mask[ring:-ring, ring:-ring] = False
        ring_vals = arr[ring_mask]
    return float(np.median(ring_vals))

def fit_psf_on_candidate(c: Candidate, roi_half: int = 6) -> Optional[FitResult]:
    """
    Wytnij małe ROI wokół centroidu CC i dopasuj eliptycznego Gaussa.
    Zwraca FitResult lub None jeśli fit się nie powiedzie.
    """
    H, W = c.roi_gray.shape
    cx0 = np.clip(int(round(c.centroid[0] - c.x0)), 0, W-1)
    cy0 = np.clip(int(round(c.centroid[1] - c.y0)), 0, H-1)

    x0 = max(0, cx0 - roi_half)
    y0 = max(0, cy0 - roi_half)
    x1 = min(W, cx0 + roi_half + 1)
    y1 = min(H, cy0 + roi_half + 1)

    I = c.roi_gray[y0:y1, x0:x1].astype(np.float64)
    if I.size < 9:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    Xloc = (xx - x0).astype(np.float64)
    Yloc = (yy - y0).astype(np.float64)
    # inicjalizacja parametrów
    offset0 = ring_median(I, ring=1)
    A0 = max(5.0, float(I.max() - offset0))
    x00 = float(np.argmax(I) % I.shape[1])
    y00 = float(np.argmax(I) // I.shape[1])
    sx0, sy0 = 1.8, 1.8
    theta0 = 0.0

    p0 = [A0, x00, y00, sx0, sy0, theta0, offset0]
    bounds = (
        [0.0,       0.0,       0.0,  0.5,  0.5, -np.pi/2, 0.0],
        [1e5,  I.shape[1]-1, I.shape[0]-1, 10.0, 10.0,  np.pi/2, 1e4]
    )

    try:
        popt, pcov = curve_fit(
            gauss2d_rot,
            (Xloc, Yloc), I.ravel(),
            p0=p0, bounds=bounds, maxfev=10000
        )
    except Exception:
        return None

    A, x0_loc, y0_loc, sx, sy, theta, offset = popt
    if pcov is None or not np.isfinite(pcov).all():
        sig = [np.nan]*len(popt)
    else:
        sig = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    sig_x0, sig_y0 = float(sig[1]), float(sig[2])

    # redukowane chi^2 (ocena jakości dopasowania)
    resid = I.ravel() - gauss2d_rot((Xloc, Yloc), *popt)
    dof = I.size - len(popt)
    red_chi2 = float((resid @ resid) / max(dof, 1))

    # współrzędne globalne
    cx = c.x0 + x0 + x0_loc
    cy = c.y0 + y0 + y0_loc

    return FitResult(
        label=c.label, cx=float(cx), cy=float(cy),
        sx=float(sx), sy=float(sy), theta=float(theta),
        A=float(A), offset=float(offset),
        sig_cx=float(sig_x0), sig_cy=float(sig_y0),
        red_chi2=red_chi2, area=c.area, m00=c.m00
    )

def draw_results(gray: np.ndarray, fits: List[FitResult], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(gray, cmap="gray", origin="upper")
    for fr in fits:
        ax.scatter([fr.cx], [fr.cy], s=25)
        L = 6.0
        x0 = fr.cx - L * math.cos(fr.theta)
        y0 = fr.cy - L * math.sin(fr.theta)
        x1 = fr.cx + L * math.cos(fr.theta)
        y1 = fr.cy + L * math.sin(fr.theta)
        ax.plot([x0, x1], [y0, y1])
    ax.set_title("PSF fit: Gauss2D centroids (+ principal axis)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_csv(fits: List[FitResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","x0","y0","sx","sy","theta_rad","A","offset","sigma_x0","sigma_y0","red_chi2","area","m00"])
        for fr in fits:
            w.writerow([fr.label, fr.cx, fr.cy, fr.sx, fr.sy, fr.theta, fr.A, fr.offset, fr.sig_cx, fr.sig_cy, fr.red_chi2, fr.area, fr.m00])

# =============== 4) Main ===============
def main(image_path: Optional[str],
         use_clahe: bool,
         thresh_mode: str,
         block_size: int,
         C: int,
         blur: int,
         min_area: int,
         min_mass: float,
         roi_half: int):
    if image_path:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"Nie można wczytać obrazu: {image_path}")
    else:
        gray = make_synthetic_starfield()
        cv2.imwrite("synthetic_starfield.png", gray)
        image_path = "synthetic_starfield.png"

    _, mask = preprocess(gray, use_clahe=use_clahe, blur_ksize=blur,
                         thresh_mode=thresh_mode, block_size=block_size, C=C)
    cands = find_candidates(gray, mask, min_area=min_area, min_mass=min_mass)

    fits: List[FitResult] = []
    for c in cands:
        fr = fit_psf_on_candidate(c, roi_half=roi_half)
        if fr is not None:
            fits.append(fr)

    draw_results(gray, fits, "psf_fit_detections.png")
    save_csv(fits, "psf_fit_results.csv")

    print(f"Input image: {image_path}")
    print(f"Candidates after CC: {len(cands)} | Successful PSF fits: {len(fits)}")
    if fits:
        avg_sig = np.nanmean([0.5*(f.sig_cx+f.sig_cy) for f in fits])
        print(f"~Average centroid 1σ (px): {avg_sig:.3f}")
    print("Saved: psf_fit_detections.png, psf_fit_results.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Star tracker — PSF (2D Gaussian) fitting pipeline")
    ap.add_argument("--image", type=str, default=None, help="Ścieżka do obrazu. Brak → syntetyczny kadr.")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--thresh", type=str, default="adaptive", choices=["adaptive","otsu"])
    ap.add_argument("--block-size", type=int, default=51)
    ap.add_argument("--C", type=int, default=-3)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--min-mass", type=float, default=600.0)
    ap.add_argument("--roi-half", type=int, default=6, help="Połowa rozmiaru ROI (okno ma wymiar 2*h+1).")
    args = ap.parse_args()

    if args.block_size % 2 == 0:
        args.block_size += 1
    if args.blur % 2 == 0:
        args.blur += 1

    main(args.image, use_clahe=not args.no_clahe, thresh_mode=args.thresh,
         block_size=args.block_size, C=args.C, blur=args.blur,
         min_area=args.min_area, min_mass=args.min_mass, roi_half=args.roi_half)
