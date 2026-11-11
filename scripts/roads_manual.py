"""Waypoint-guided road tracing: waypoints read off the 1 m imagery, snapped to the road surface by least-cost routing."""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.graph import route_through_array
from skimage.measure import approximate_polygon
SCR, w, wpfile = sys.argv[1], sys.argv[2], sys.argv[3]
OFF = {"kavkazi": (32, 35), "europe": (38, 39)}[w]
m = json.load(open(f"{SCR}/mosaic_meta.json"))[w]
a = np.asarray(Image.open(f"{SCR}/mosaic_{w}.png")).astype(np.float32); H, Wd, _ = a.shape
mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1); val = a.mean(2); br = a[..., 2] / (a[..., 0] + 1)
if w == "kavkazi":
    s = np.clip((0.58 - sat) / 0.22, 0, 1) * np.clip((br - 0.38) / 0.22, 0, 1) * ((val > 95) & (val < 200))
else:
    s = np.clip((0.55 - sat) / 0.22, 0, 1) * np.clip((br - 0.42) / 0.2, 0, 1) * ((val > 55) & (val < 125))
s = gaussian_filter(s.astype(np.float32), 1.5)
cost = 1 + 40 * (1 - s)
roads = json.load(open(wpfile)); out = []
pv = Image.fromarray(a.astype(np.uint8)); d = ImageDraw.Draw(pv)
for name, wps in roads.items():
    line = []
    for (x0, y0), (x1, y1) in zip(wps[:-1], wps[1:]):
        x0, x1 = [min(max(int(v), 0), Wd - 1) for v in (x0, x1)]; y0, y1 = [min(max(int(v), 0), H - 1) for v in (y0, y1)]
        # corridor: bounding box of the leg padded by 30 px
        xa, xb = max(min(x0, x1) - 30, 0), min(max(x0, x1) + 31, Wd); ya, yb = max(min(y0, y1) - 30, 0), min(max(y0, y1) + 31, H)
        sub = cost[ya:yb, xa:xb].copy()
        # penalise distance from the straight leg so the route cannot wander
        yy, xx = np.mgrid[ya:yb, xa:xb]; L = np.hypot(x1 - x0, y1 - y0) + 1e-6
        dline = np.abs((x1 - x0) * (y0 - yy) - (x0 - xx) * (y1 - y0)) / L
        sub = sub * (1 + np.clip(dline - 12, 0, None) / 6)
        path, _ = route_through_array(sub, (y0 - ya, x0 - xa), (y1 - ya, x1 - xa), fully_connected=True, geometric=True)
        line += [(c + xa, r + ya) for r, c in path]
    arr = np.array(line, float)
    k = 9; pad = np.pad(arr, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], 1); sm[0], sm[-1] = arr[0], arr[-1]
    sm = approximate_polygon(sm, tolerance=1.2)
    on = float(np.mean([s[int(y), int(x)] for x, y in arr]))
    out.append({"name": name, "pts": [[float(m["ox"] + x + .5 - OFF[0]), float(m["oy"] + y + .5 - OFF[1])] for x, y in sm], "roadness": round(on, 2)})
    d.line([tuple(p) for p in sm], fill=(255, 0, 255), width=3)
    for x, y in wps: d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 255, 255), width=2)
    print(f"{name:14s} roadness {on:.2f}  {len(sm)} pts")
json.dump(out, open(f"{SCR}/roads_manual_{w}.json", "w"))
pv.save(f"{SCR}/roads_manual_{w}.png"); t = pv.copy(); t.thumbnail((1100, 1100)); t.save(f"{SCR}/roads_manual_{w}_s.png")
