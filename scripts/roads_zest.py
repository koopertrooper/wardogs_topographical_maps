import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter, binary_closing, binary_opening, binary_dilation, label, distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects, disk
from skimage.measure import approximate_polygon
SCR = sys.argv[1]; D = f"{SCR}/repo/wardogs-data/data/maps"; w = "north-america"
ox_, oy_ = 0, 0                      # tile image is not offset from game coords on this map
MX, MY = -2304, -3328                # mosaic origin (game m)
a = np.asarray(Image.open(f"{SCR}/mosaic_{w}.png").convert("RGB")).astype(np.float32); H, Wd, _ = a.shape
mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1); val = a.mean(2)
road = (sat < 0.40) & (val > 105) & (val < 235) & (a[..., 2] > 0.55 * a[..., 0])
bm = np.zeros((H, Wd), bool)
for bb in json.load(open(f"{D}/{w}/buildings.json"))["buildings"]:
    x0 = int(bb["min"][0] - MX) - 2; x1 = int(bb["max"][0] - MX) + 3; y0 = int(bb["min"][1] - MY) - 2; y1 = int(bb["max"][1] - MY) + 3
    if x1 > 0 and y1 > 0 and x0 < Wd and y0 < H: bm[max(y0, 0):y1, max(x0, 0):x1] = True
road &= ~bm
road = binary_opening(road, structure=disk(1)); road = binary_closing(road, structure=disk(2))
wide = binary_opening(road, structure=disk(int(sys.argv[2]) if len(sys.argv) > 2 else 13))
road &= ~wide
road = remove_small_objects(road, max_size=250)
Image.fromarray((road * 255).astype(np.uint8)).save(f"{SCR}/roadmask_{w}.png")
sk = skeletonize(road); pts = set(zip(*np.nonzero(sk)))
nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def chains_of(pts):
    deg = {q: sum((q[0] + dr, q[1] + dc) in pts for dr, dc in nbr) for q in pts}
    nodes = {q for q, d in deg.items() if d != 2}; seen = set(); chains = []
    for s in nodes:
        for dr, dc in nbr:
            q = (s[0] + dr, s[1] + dc)
            if q not in pts or (s, q) in seen: continue
            ch = [s, q]; seen.add((s, q)); seen.add((q, s)); prev, cur = s, q
            while cur not in nodes:
                nx = [(cur[0] + a_, cur[1] + b_) for a_, b_ in nbr if (cur[0] + a_, cur[1] + b_) in pts and (cur[0] + a_, cur[1] + b_) != prev and (cur, (cur[0] + a_, cur[1] + b_)) not in seen]
                if not nx: break
                q2 = nx[0]; seen.add((cur, q2)); seen.add((q2, cur)); ch.append(q2); prev, cur = cur, q2
            chains.append(ch)
    return chains, deg
for _ in range(3):
    chains, deg = chains_of(pts)
    for ch in chains:
        if len(ch) < 30 and (deg.get(ch[0], 0) == 1 or deg.get(ch[-1], 0) == 1):
            for q in ch[1:-1]: pts.discard(q)
            if deg.get(ch[0], 0) == 1: pts.discard(ch[0])
            if deg.get(ch[-1], 0) == 1: pts.discard(ch[-1])
chains, deg = chains_of(pts)
skm = np.zeros((H, Wd), bool)
for q in pts: skm[q] = True
slab, _ = label(skm, structure=np.ones((3, 3))); clen = np.bincount(slab.ravel())
keep = []
dens = uniform_filter(skm.astype(np.float32), 41) * 41 * 41
for ch in chains:
    if clen[slab[ch[len(ch) // 2]]] < 200: continue
    L = len(ch); dd = float(np.hypot(ch[0][0] - ch[-1][0], ch[0][1] - ch[-1][1])); cl = dens[ch[L // 2]]
    if not (L >= 110 or (L >= 40 and dd / L > 0.85 and cl < 160)): continue
    if L < 110 and cl > 200: continue
    arr = approximate_polygon(np.array([[c, r] for r, c in ch], float), tolerance=1.5)
    keep.append([[float(MX + x + .5), float(MY + y + .5)] for x, y in arr])
json.dump(keep, open(f"{SCR}/roads_zest.json", "w")); print("segments", len(keep))
pv = Image.fromarray(a.astype(np.uint8)); d = ImageDraw.Draw(pv)
for k in keep: d.line([(x - MX, y - MY) for x, y in k], fill=(255, 0, 255), width=3)
pv.crop((494, 458, 1739, 1998)).save("/tmp/claude-0/zr_over.png"); Image.open(f"{SCR}/roadmask_{w}.png").crop((494, 458, 1739, 1998)).save("/tmp/claude-0/zr_mask.png")
