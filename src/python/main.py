from star_catalog import StarCatalog
import numpy as np

cat = StarCatalog("data/hygdata_v41.csv")

vecs, mags = cat.filtered(max_mag=6.0)
print(f"{len(vecs)} bright stars loaded")

res = cat.cone_search(np.array([0, 0, 1]), half_angle_deg=10)
print(res.head())