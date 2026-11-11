"""Trace review landmarks: water + sunflower fields (Bakurani map image), river main stem + summits (terrain)."""
import sys, json, heapq
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_opening, binary_closing, binary_fill_holes, label, gaussian_filter,
                           maximum_filter, generate_binary_structure, iterate_structure)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCR = sys.argv[1]; D = f"{SCR}/repo/wardogs-data/data/maps"
Image.MAX_IMAGE_PIXELS = None
FR = {"kavkazi": (-8873, -11752, 1282), "europe": (-7291, -7020, 1816)}

def mask_polys(mask, ox, oy, mpp, min_area):
    """Mask (rows=y) -> list of world-coord polygons via a 0.5 iso-contour of a lightly smoothed mask."""
    sm = gaussian_filter(mask.astype(float), 0.8)
    fig = plt.figure(); cs = plt.contour(sm, levels=[0.5]); plt.close(fig)
    out = []
    for seg in cs.allsegs[0]:
        if len(seg) < 4: continue
        xs, ys = seg[:, 0], seg[:, 1]
        area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) * mpp * mpp
        if area < min_area: continue
        out.append([[float(ox + (x + .5) * mpp), float(oy + (y + .5) * mpp)] for x, y in seg[::1]])
    return out

def heights(w):
    t = json.load(open(f"{D}/{w}/terrain.json"))
    (tx, ty), (ex, ey) = t["bounds"]["min"], t["bounds"]["max"]; mpp = (ex - tx) / 8192
    full = np.load(f"{SCR}/{w}.npy", mmap_mode="r")
    zr = t["zRange"]; base = zr["min"] + float(full.min()) / 65535 * (zr["max"] - zr["min"])
    x0, y0, W = FR[w]
    c0 = int((x0 - tx) / mpp); r0 = int((y0 - ty) / mpp); n = int(W / mpp)
    Z = zr["min"] + full[r0:r0 + n, c0:c0 + n].astype(np.float32) / 65535 * (zr["max"] - zr["min"]) - base
    return Z, tx + c0 * mpp, ty + r0 * mpp, mpp

def peaks(w, k, mind_m):
    Z, ox, oy, mpp = heights(w); zz = gaussian_filter(Z, 8)
    mx = zz == maximum_filter(zz, size=int(250 / mpp)); rr, cc = np.nonzero(mx)
    o = np.argsort(-zz[rr, cc]); out = []
    for i in o:
        r, c = rr[i], cc[i]
        if not (30 < r < Z.shape[0] - 30 and 30 < c < Z.shape[1] - 30): continue
        X, Y = ox + c * mpp, oy + r * mpp
        if all(np.hypot(X - a, Y - b) > mind_m for a, b, _ in out): out.append((X, Y, float(Z[r, c])))
        if len(out) >= k: break
    return out

def main_stem(w):
    Z, ox, oy, mpp = heights(w)
    z = gaussian_filter(Z[::2, ::2], 2).astype(np.float64); n = z.shape[0]; cm = mpp * 2
    done = np.zeros(z.shape, bool); h = []
    for r in range(n):
        for c in range(n):
            if r in (0, n - 1) or c in (0, n - 1): heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while h:
        zz, r, c = heapq.heappop(h)
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not done[rr, cc]:
                done[rr, cc] = True
                if z[rr, cc] <= zz: z[rr, cc] = zz + 1e-4
                heapq.heappush(h, (z[rr, cc], rr, cc))
    pad = np.pad(z, 1, constant_values=-1e9); best = np.zeros(z.shape); rec = np.arange(n * n).reshape(n, n)
    R, C = np.mgrid[0:n, 0:n]
    for dr, dc in nb:
        drop = (z - pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]) / np.hypot(dr, dc); b = drop > best
        best = np.where(b, drop, best); rec = np.where(b, np.clip(R + dr, 0, n - 1) * n + np.clip(C + dc, 0, n - 1), rec)
    rec = rec.ravel(); acc = np.ones(n * n); order = np.argsort(-z.ravel(), kind="stable")
    for i in order:
        j = rec[i]
        if j != i: acc[j] += acc[i]
    thr = 0.35 * acc.max(); on = np.nonzero(acc >= thr)[0]
    start = on[np.argmax(z.ravel()[on])]; path = [start]
    while True:
        j = rec[path[-1]]
        if j == path[-1] or acc[j] < thr: break
        path.append(j)
    pts = [[float(ox + (p % n + .5) * cm), float(oy + (p // n + .5) * cm)] for p in path]
    return pts, float(acc[path[-1]] * cm * cm / 1e6)

out = {}
# ---------------- Bakurani (map image traced)
w = "kavkazi"; m = json.load(open(f"{D}/{w}/map.json")); bx, by = m["bounds"]["min"]
x0, y0, W = FR[w]; pad = 60
c0 = int((x0 - pad - bx) / 4); r0 = int((y0 - pad - by) / 4); n = int((W + 2 * pad) / 4)
a = np.asarray(Image.open(f"assets/maps/{w}/overview.webp").convert("RGB"))[r0:r0 + n, c0:c0 + n].astype(int)
r, g, b = a[..., 0], a[..., 1], a[..., 2]; ox, oy = bx + c0 * 4, by + r0 * 4
# pond: very dark pixels in a small box around it
pb = (slice(int((-11260 - oy) / 4), int((-11110 - oy) / 4)), slice(int((-8110 - ox) / 4), int((-7990 - ox) / 4)))
pond_mask = np.zeros(r.shape, bool); pond_mask[pb] = ((r + g + b) < 90)[pb]
pond_mask = binary_fill_holes(binary_closing(pond_mask, iterations=1))
ys_, xs_ = np.nonzero(pond_mask)
bridges = [(-8560, -11400, "BRIDGE"), (-8615, -11290, "BRIDGE (2nd?)")]
sun = (r > 205) & (g > 165) & (b < 125) & ((r - b) > 105)
sun = binary_opening(sun, iterations=1)
out[w] = {
    "water": mask_polys(pond_mask, ox, oy, 4, 200),
    "sunflower": mask_polys(sun, ox, oy, 4, 1200),
    "pond": [[float(ox + (xs_.mean() + .5) * 4), float(oy + (ys_.mean() + .5) * 4)]],
    "bridges": [[x, y, t] for x, y, t in bridges],
    "peaks": peaks(w, 3, 350),
}
ys, xs = np.nonzero(sun & (np.arange(n)[None, :] * 4 + ox > x0) & (np.arange(n)[:, None] * 4 + oy > y0)
                    & (np.arange(n)[None, :] * 4 + ox < x0 + W) & (np.arange(n)[:, None] * 4 + oy < y0 + W))
out[w]["sun_label"] = [float(ox + np.median(xs) * 4), float(oy + np.median(ys) * 4)] if len(xs) else None
stem_k, area_k = main_stem(w); out[w]["river"] = stem_k; print("kavkazi stem", len(stem_k), "catchment km2", round(area_k, 1))
print(w, "water polys", len(out[w]["water"]), "sun polys", len(out[w]["sunflower"]), "peaks", [(round(x), round(y), round(z)) for x, y, z in out[w]["peaks"]])
# preview overlay on the image crop
pv = Image.fromarray(a.astype(np.uint8)).resize((n * 2, n * 2)); d = ImageDraw.Draw(pv)
for poly in out[w]["water"]: d.line([((x - ox) / 2, (y - oy) / 2) for x, y in poly + poly[:1]], fill=(0, 200, 255), width=2)
for poly in out[w]["sunflower"]: d.line([((x - ox) / 2, (y - oy) / 2) for x, y in poly + poly[:1]], fill=(255, 0, 255), width=2)
d.line([((x - ox) / 2, (y - oy) / 2) for x, y in stem_k], fill=(0, 120, 255), width=3)
for x, y, _t in out[w]["bridges"]: d.ellipse(((x - ox) / 2 - 8, (y - oy) / 2 - 8, (x - ox) / 2 + 8, (y - oy) / 2 + 8), outline=(255, 0, 0), width=3)
pv.save(f"{SCR}/feat_kavkazi.png")

# ---------------- Ozeti (terrain + named places)
w = "europe"
stem, area = main_stem(w); print("europe main stem", len(stem), "pts, catchment km2", round(area, 1))
out[w] = {"river": stem, "peaks": peaks(w, 3, 350)}
print("europe peaks", [(round(x), round(y), round(z)) for x, y, z in out[w]["peaks"]])
m2 = json.load(open(f"{D}/europe/map.json")); ex0, ey0 = m2["bounds"]["min"]
x0e, y0e, We = FR["europe"]; ce = int((x0e - ex0) / 4); re_ = int((y0e - ey0) / 4); ne = int(We / 4)
ae = Image.open("assets/maps/europe/overview.webp").convert("RGB").crop((ce, re_, ce + ne, re_ + ne)).resize((ne * 2, ne * 2))
de = ImageDraw.Draw(ae); de.line([((x - (ex0 + ce * 4)) / 2, (y - (ey0 + re_ * 4)) / 2) for x, y in stem], fill=(0, 120, 255), width=3)
for x, y, z in out["europe"]["peaks"]: de.ellipse(((x - (ex0 + ce * 4)) / 2 - 6, (y - (ey0 + re_ * 4)) / 2 - 6, (x - (ex0 + ce * 4)) / 2 + 6, (y - (ey0 + re_ * 4)) / 2 + 6), outline=(255, 0, 0), width=3)
ae.save(f"{SCR}/feat_europe.png")
json.dump(out, open(f"{SCR}/landmarks_review.json", "w"))
