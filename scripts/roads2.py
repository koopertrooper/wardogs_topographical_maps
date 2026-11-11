"""Road tracing from 1 m/px map tiles: smooth low-saturation ribbons -> skeleton graph -> pruned polylines (game coords)."""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter, binary_closing, binary_opening, binary_dilation, label, distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects, disk
from skimage.measure import approximate_polygon
SCR = sys.argv[1]; D = f"{SCR}/repo/wardogs-data/data/maps"
OFF = {"kavkazi": (32, 35), "europe": (38, 39)}
P = json.load(open(f"{SCR}/road_params.json"))
meta = json.load(open(f"{SCR}/mosaic_meta.json")); out = {}
for w in ["europe"]:
    m = meta[w]; ox_, oy_ = OFF[w]; p = P[w]
    a = np.asarray(Image.open(f"{SCR}/mosaic_{w}.png")).astype(np.float32); H, Wd, _ = a.shape
    mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1); val = a.mean(2)
    mu = uniform_filter(val, 5); sd = np.sqrt(np.maximum(uniform_filter(val * val, 5) - mu * mu, 0))
    road = (sat < p["sat"]) & (val > p["vlo"]) & (val < p["vhi"]) & (sd < p["sd"]) & (a[..., 2] > p["br"] * a[..., 0])
    bm = np.zeros((H, Wd), bool)
    for bb in json.load(open(f"{D}/{w}/buildings.json"))["buildings"]:
        x0 = int(bb["min"][0] + ox_ - m["ox"]) - 2; x1 = int(bb["max"][0] + ox_ - m["ox"]) + 3
        y0 = int(bb["min"][1] + oy_ - m["oy"]) - 2; y1 = int(bb["max"][1] + oy_ - m["oy"]) + 3
        if x1 > 0 and y1 > 0 and x0 < Wd and y0 < H: bm[max(y0, 0):y1, max(x0, 0):x1] = True
    road &= ~bm
    rv = json.load(open(f"{SCR}/landmarks_review.json"))[w]["river"]
    rimg = Image.new("L", (Wd, H), 0); ImageDraw.Draw(rimg).line([(x + ox_ - m["ox"], y + oy_ - m["oy"]) for x, y in rv], fill=255, width=p["riverw"])
    road &= ~(np.asarray(rimg) > 0)
    road = binary_opening(road, structure=disk(1)); road = binary_closing(road, structure=disk(3))
    wide = binary_opening(road, structure=disk(p["wide"]))
    road &= ~binary_dilation(wide, structure=disk(3))
    road = remove_small_objects(road, max_size=p["minarea"])
    # drop blobs (yards/car parks): pixels far from the ribbon edge on a wide shape are kept, but whole comps that are compact are dropped
    lab, k = label(road); dist = distance_transform_edt(road)
    for i in range(1, k + 1):
        ys, xs = np.nonzero(lab == i); area = len(xs); ext = max(np.ptp(xs), np.ptp(ys)) + 1
        if area / (ext * ext) > 0.25 or ext < p["minext"]: road[lab == i] = False
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
    for _ in range(3):   # prune short dead-end spurs
        chains, deg = chains_of(pts)
        for ch in chains:
            if len(ch) < p["spur"] and (deg.get(ch[0], 0) == 1 or deg.get(ch[-1], 0) == 1):
                for q in ch[1:-1]: pts.discard(q)
                if deg.get(ch[0], 0) == 1: pts.discard(ch[0])
                if deg.get(ch[-1], 0) == 1: pts.discard(ch[-1])
    chains, deg = chains_of(pts)
    skm = np.zeros((H, Wd), bool)
    for q in pts: skm[q] = True
    slab, _ = label(skm, structure=np.ones((3, 3))); clen = np.bincount(slab.ravel())
    keep = []
    for ch in chains:
        if clen[slab[ch[len(ch) // 2]]] < p["mincomp"]: continue
        wid = float(np.median([dist[q] for q in ch]) * 2)
        arr = approximate_polygon(np.array([[c, r] for r, c in ch], float), tolerance=1.5)
        keep.append({"pts": [[float(m["ox"] + x + .5 - ox_), float(m["oy"] + y + .5 - oy_)] for x, y in arr], "len": len(ch), "width": wid})
    out[w] = keep; print(w, "segments", len(keep), "km", round(sum(k_["len"] for k_ in keep) / 1000, 1))
    pv = Image.fromarray(a.astype(np.uint8)); d = ImageDraw.Draw(pv)
    for k_ in keep: d.line([(x + ox_ - m["ox"], y + oy_ - m["oy"]) for x, y in k_["pts"]], fill=(255, 0, 255), width=3)
    pv.save(f"{SCR}/roads2_{w}.png")
json.dump(out, open(f"{SCR}/roads2.json", "w"))
