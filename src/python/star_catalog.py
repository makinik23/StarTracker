"""
Star catalog loader and utilities for the HYG Database (v3).

Provides:
- Loading and preprocessing of star catalog (RA, Dec, magnitude)
- Conversion to unit vectors in ICRF
- Filtering by magnitude
- Angular distance search (cone search)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Star:
    id: int
    name: str
    ra_deg: float
    dec_deg: float
    mag: float
    vector: np.ndarray


class StarCatalog:
    def __init__(self, path: str):
        """Load the HYG catalog from CSV."""
        self.df = pd.read_csv(path)

        self.df = self.df.dropna(subset=["ra", "dec", "mag"])
        self._vectors = None

    @property
    def vectors(self) -> np.ndarray:
        """Return Nx3 array of unit vectors (ICRF frame)."""
        if self._vectors is None:
            ra = np.deg2rad(self.df["ra"].to_numpy())
            dec = np.deg2rad(self.df["dec"].to_numpy())
            x = np.cos(dec) * np.cos(ra)
            y = np.cos(dec) * np.sin(ra)
            z = np.sin(dec)
            self._vectors = np.column_stack((x, y, z))
        return self._vectors

    def filtered(self, max_mag: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return unit vectors and magnitudes for stars brighter than given magnitude."""
        mask = self.df["mag"] <= max_mag
        return self.vectors[mask], self.df.loc[mask, "mag"].to_numpy()

    def cone_search(
        self, direction: np.ndarray, half_angle_deg: float
    ) -> pd.DataFrame:
        """
        Return stars within a cone of given half-angle [deg]
        around the direction vector (3D unit vector).
        """
        dir_unit = direction / np.linalg.norm(direction)
        cos_ang = np.cos(np.deg2rad(half_angle_deg))
        dots = self.vectors @ dir_unit
        mask = dots >= cos_ang
        res = self.df.loc[mask, ["id", "proper", "ra", "dec", "mag"]].copy()
        res["angle_deg"] = np.rad2deg(np.arccos(np.clip(dots[mask], -1, 1)))
        return res.sort_values("angle_deg")

    def as_stars(self, max_mag: float = 6.0) -> list[Star]:
        """Return list of Star dataclasses (for structured access)."""
        mask = self.df["mag"] <= max_mag
        ra = self.df.loc[mask, "ra"].to_numpy()
        dec = self.df.loc[mask, "dec"].to_numpy()
        mag = self.df.loc[mask, "mag"].to_numpy()
        ids = self.df.loc[mask, "id"].to_numpy(dtype=int)
        names = self.df.loc[mask, "proper"].fillna("").to_numpy()
        vecs = self.vectors[mask]
        return [
            Star(int(i), str(n), float(r), float(d), float(m), v)
            for i, n, r, d, m, v in zip(ids, names, ra, dec, mag, vecs)
        ]
