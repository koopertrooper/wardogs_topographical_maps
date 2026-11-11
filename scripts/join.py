import sys, json, math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.graph import route_through_array
SCR = sys.argv[1]
OFF = {"kavkazi": (32, 35), "europe": (38, 39)}
RAD = {"kavkazi": 110, "europe": 70}
META = json.load(open(f"{SCR}/mosaic_meta.json"))
roads = json.load(open(f"{SCR}/roads_final.json"))
def L(l): return sum(math.dist(a, b) for a, b in zip(l, l[1:]))
def roadness(w):
    a = np.asarray(Image.open(f"{SCR}/mosaic_{w}.png")).astype(np.float32)
    mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1); val = a.mean(2); br = a[..., 2] / (a[..., 0] + 1)
    if w == "kavkazi": s = np.clip((0.58 - sat) / 0.22, 0, 1) * np.clip((br - 0.38) / 0.22, 0, 1) * ((val > 95) & (val < 200))
    else: s = np.clip((0.55 - sat) / 0.22, 0, 1) * np.clip((br - 0.42) / 0.2, 0, 1) * ((val > 55) & (val < 125))
    return gaussian_filter(s.astype(np.float32), 1.5)
def nearest(p, R, skip):
    best = (1e9, None)
    for j, l in enumerate(R):
        if j == skip: continue
        for a, b in zip(l, l[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]; n = dx * dx + dy * dy
            t = 0 if n == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
            q = (a[0] + t * dx, a[1] + t * dy); d = math.dist(p, q)
            if d < best[0]: best = (d, q)
    return best
out = {}
for w in ("kavkazi", "europe"):
    R = [l for l in roads[w] if L(l) >= 25]
    s = roadness(w); m = META[w]; ox, oy = m["ox"], m["oy"]; H, W = s.shape
    cost = 1 + 40 * (1 - s)
    topx = lambda p: (p[0] - ox - .5 + OFF[w][0], p[1] - oy - .5 + OFF[w][1])
    tog = lambda x, y: [ox + x + .5 - OFF[w][0], oy + y + .5 - OFF[w][1]]
    new = []; joined = 0
    for i, l in enumerate(R):
        for end in (0, -1):
            p = l[end]
            d, q = nearest(p, R, i)
            # already touching?
            if d < 4: continue
            # also consider own other end ignored; skip if this end is touched
            if d > RAD[w]: continue
            if d <= 12:
                new.append([p, list(q)]); joined += 1; continue
            x0, y0 = [int(round(v)) for v in topx(p)]; x1, y1 = [int(round(v)) for v in topx(q)]
            if not (0 <= x0 < W and 0 <= x1 < W and 0 <= y0 < H and 0 <= y1 < H): continue
            xa, xb = max(min(x0, x1) - 20, 0), min(max(x0, x1) + 21, W); ya, yb = max(min(y0, y1) - 20, 0), min(max(y0, y1) + 21, H)
            sub = cost[ya:yb, xa:xb]
            path, _ = route_through_array(sub, (y0 - ya, x0 - xa), (y1 - ya, x1 - xa), fully_connected=True, geometric=True)
            pts = [(c + xa, r + ya) for r, c in path]
            ons = float(np.mean([s[r + ya, c + xa] for r, c in path])); plen = len(pts)
            if ons >= 0.3 and plen <= 1.7 * d + 6:
                sm = pts[::6] + [pts[-1]]
                new.append([p] + [tog(x, y) for x, y in sm[1:-1]] + [list(q)]); joined += 1
    print(w, "dead-end joins:", joined)
    out[w] = R + new
json.dump(out, open(f"{SCR}/roads_joined.json", "w"))
