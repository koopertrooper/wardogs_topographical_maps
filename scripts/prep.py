"""Prepare the working folder inputs from the wardogs-data release (assets/maps/<world>/...).

usage: python prep.py <working-folder> <release-assets-maps-dir> <world> [<world> ...]

Creates, inside <working-folder>:
  <world>.npy            heightmap.png as a uint16 array (8192x8192, ~2 m/px)
  mosaic_<world>.png     zoom-6 satellite tiles stitched at 1 m/px, covering the area given in data/mosaic_meta.json
                         (tile pixel = world coordinate - map.json bounds.min; ox/oy in the meta are the world coordinates of mosaic pixel 0,0)
"""
import sys, os, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
work, maps = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(work, "mosaic_meta.json")))
for w in sys.argv[3:]:
    hm = Image.open(os.path.join(maps, w, "heightmap.png"))
    np.save(os.path.join(work, w + ".npy"), np.array(hm))
    t = json.load(open(os.path.join(work, "repo/wardogs-data/data/maps", w, "map.json")))
    bx, by = t["bounds"]["min"]
    m = meta[w]; Wp, Hp = m["size"]; T = m["tile"]
    tx0 = int(round((m["ox"] - bx) / T)); ty0 = int(round((m["oy"] - by) / T))
    out = Image.new("RGB", (Wp, Hp))
    for ix in range(Wp // T):
        for iy in range(Hp // T):
            p = os.path.join(maps, w, "tiles", "6", str(tx0 + ix), f"{ty0 + iy}.webp")
            if os.path.exists(p): out.paste(Image.open(p).convert("RGB"), (ix * T, iy * T))
    out.save(os.path.join(work, f"mosaic_{w}.png"))
    print(w, "heightmap", hm.size, "mosaic", out.size)
