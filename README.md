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

### 🐍 Requirements

- **Python ≥ 3.10**
- **Hatch ≥ 1.10**  
  *(Install globally once: `pip install hatch`)*

### 🚀 Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-user>/star-tracker.git
cd star-tracker

# 2. Create the environment via Hatch
hatch env create

# 3. Run tests
hatch run unit-test
```

---

## 🧪 Development utilities

The following commands are predefined in the Hatch environment  
(`pyproject.toml → [tool.hatch.envs.default.scripts]`):

| Command | Description |
|----------|-------------|
| `hatch run unit-test` | Run all unit tests (pytest). |
| `hatch run test-cov`  | Run tests with coverage report. |
| `hatch run lint`      | Run Ruff linter. |
| `hatch run format`    | Auto-format code via Ruff formatter. |
| `hatch run type-check`| Run MyPy static type checking. |
| `hatch run all`       | Run lint, type check, and tests sequentially. |

---

## 🧭 Roadmap

- [x] Image moment analysis  
- [x] PSF fitting  
- [ ] Catalog cross-match (HYG)  
- [ ] Attitude determination (star vector matching)  
- [ ] Integration & performance testing  

---

## ⚖️ License

Released under the **MIT License** — see [LICENSE](LICENSE) for details.
