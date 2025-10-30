import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import csv
from pathlib import Path



from algorithms.moment_tracker import (
    make_synthetic_starfield,
    robust_threshold,
    label_components,
    compute_blob_moments,
)


from algorithms.psf_fitting import (
    fit_psf_on_candidate,
)


def extract_roi(image: np.ndarray, cx: float, cy: float, half: int = 6) -> tuple[np.ndarray, int, int]:
    """
    Extract square ROI around (cx, cy) with given half-size.

    Args:
        image (np.ndarray): Grayscale image array.
        cx (float): X-coordinate of the center.
        cy (float): Y-coordinate of the center.
        half (int): Half-size of the ROI.
    
    Returns:
        tuple[np.ndarray, int, int]: ROI image, top-left x, top-left y.
    """
    H, W = image.shape

    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(W, cx + half + 1))
    y1 = int(min(H, cy + half + 1))

    return image[y0:y1, x0:x1], x0, y0


def hybrid_pipeline(
    image_path: str | None = None,
    k: float = 5.0,
    min_mass: float = 400.0,
    roi_half: int = 6,
    out_dir: str = "out"
) -> None:
    """
    Moment detection + PSF refinement. Results saved in /out.

    Args:
        image_path (str | None): Path to input image. If None, a synthetic starfield is generated.
        k (float): Thresholding parameter for robust threshold.
        min_mass (float): Minimum mass for moment detections to be considered.
        roi_half (int): Half-size of ROI for PSF fitting.
        out_dir (str): Output directory to save results.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if image_path:
        img = np.array(Image.open(image_path).convert("L"))
        image_name = Path(image_path).stem
    else:
        img = make_synthetic_starfield()
        image_name = "synthetic_starfield"
        Image.fromarray(img).save(out_path / f"{image_name}.png")

    print(f"[INFO] Input image: {image_path or (out_path / (image_name + '.png'))}")

    # --- Phase 1: moment-based detection ---
    _, _, T = robust_threshold(img, k)
    mask = (img.astype(float) > T).astype(np.uint8)
    labels, _ = label_components(mask)
    moments = [m for m in compute_blob_moments(img, labels) if m.m00 >= min_mass]

    print(f"[INFO] Moment detections: {len(moments)}")

    # --- Phase 2: PSF refinement ---
    refined = []
    for m in moments:
        roi_gray, x0, y0 = extract_roi(img, m.cx, m.cy, half=roi_half)

        if roi_gray.size < 9:
            continue

        class Tmp:
            pass
    
        c = Tmp()
        c.label = m.label
        c.x0, c.y0 = x0, y0
        c.roi_gray = roi_gray
        c.roi_mask = np.ones_like(roi_gray, dtype=np.uint8)
        c.centroid = (m.cx, m.cy)
        c.area = int((2 * roi_half) ** 2)
        c.m00 = m.m00

        fr = fit_psf_on_candidate(c, roi_half=roi_half)
        if fr is not None:
            refined.append(fr)

    print(f"[INFO] PSF fits successful: {len(refined)}")

    csv_path = out_path / f"{image_name}_hybrid_results.csv"
    img_path = out_path / f"{image_name}_hybrid_pipeline_detections.png"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "cx",
                "cy",
                "sx",
                "sy",
                "theta",
                "A",
                "offset",
                "sig_cx",
                "sig_cy",
                "red_chi2",
            ]
        )
        for fr in refined:
            w.writerow(
                [
                    fr.label,
                    fr.cx,
                    fr.cy,
                    fr.sx,
                    fr.sy,
                    fr.theta,
                    fr.A,
                    fr.offset,
                    fr.sig_cx,
                    fr.sig_cy,
                    fr.red_chi2,
                ]
            )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="upper")

    for fr in refined:
        ax.scatter([fr.cx], [fr.cy], s=20, c="lime")
        L = 6.0

        x0 = fr.cx - L * math.cos(fr.theta)
        y0 = fr.cy - L * math.sin(fr.theta)
        x1 = fr.cx + L * math.cos(fr.theta)
        y1 = fr.cy + L * math.sin(fr.theta)

        ax.plot([x0, x1], [y0, y1], c="yellow")

    ax.set_title("Hybrid pipeline: Moment + PSF centroids")
    plt.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved results to: {out_path}")
    print(f"         CSV: {csv_path.name}")
    print(f"         Image: {img_path.name}")


if __name__ == "__main__":
    hybrid_pipeline()
