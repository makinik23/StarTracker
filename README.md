# StarTracker

> CubeSat‑grade star detection pipeline using **image moments** (centroid/shape) and **PSF fitting** with a roadmap toward catalog cross‑match and attitude solution.

![status](https://img.shields.io/badge/status-WIP-yellow) ![python](https://img.shields.io/badge/Python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-informational)

---

## ✨ Highlights

* **Star detection via image moments**: centroid, total flux, second moments, ellipticity.
* **Optional PSF fitting**: refines sub‑pixel centroids and FWHM; configurable.
* **Robust preprocessing**: background estimation, denoising, thresholding, blob labeling.
* **Modular pipeline**: interchangeable steps; easy to extend to pattern matching (e.g., Liebe/triangle/quad).
* **Catalog‑ready I/O**: outputs clean star lists that can be fed to cross‑match modules (HYG v4.1 planned).

> **Note:** As of now, **catalog cross‑matching is not implemented** and **no integration tests** exist yet. Both are planned soon.

---

## 🗺️ Project goals

1. Detect stars reliably from raw frames (simulated or real sensor).
2. Produce precise centroids (moment → PSF refinement) with uncertainty estimates.
3. Enable catalog cross‑match (HYG v4.1) and then attitude determination.
4. Provide a clean, testable codebase for onboard and ground pipelines.

---

## 📦 Installation

```bash
# Recommended: create a fresh environment
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# or:  .venv\Scripts\activate                      # (Windows)

pip install -U pip
pip install -r requirements.txt  # Provided in this repo
```

**Minimum Python:** 3.10

**Core deps (typical):** `numpy`, `scipy`, `scikit-image`, `opencv-python`, `pandas`, `matplotlib`, `tqdm`, `pydantic`.

---

## 🚀 Quick start
