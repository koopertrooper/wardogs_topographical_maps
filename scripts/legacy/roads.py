"""Trace major roads + stadium outline from the game map images; output game-coord vectors."""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_opening, binary_closing, label, distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import approximate_polygon
SCR = sys.argv[1]; D = f"{SCR}/repo/wardogs-data/data/maps"; Image.MAX_IMAGE_PIXELS = None
OFF = {"kavkazi": (32, 36), "europe": (20, 60)}           # image = game + OFF
FR = {"kavkazi": (-8873, -11752, 1282), "europe": (-7291, -7020, 1816)}
out = {}
for w in ["kavkazi", "europe"]:
    bx, by = json.load(open(f"{D}/{w}/map.json"))["bounds"]["min"]; ox_, oy_ = OFF[w]
    x0, y0, W = FR[w]; pad = 150
    c0 = int((x0 - pad + ox_ - bx) / 4); r0 = int((y0 - pad + oy_ - by) / 4); n = int((W + 2 * pad) / 4)
    a = np.asarray(Image.open(f"/mnt/user-data/uploads/maps/{w}/overview.webp").convert("RGB"))[r0:r0 + n, c0:c0 + n].astype(float)
    mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1); val = a.mean(2)
    if w == "kavkazi":
        road = (sat < 0.37) & (val > 118) & (a[..., 2] > 80)
    else:
        road = (sat < 0.34) & (val > 48) & (val < 118)
    bm = np.zeros(road.shape, bool)
    for bb in json.load(open(f"{D}/{w}/buildings.json"))["buildings"]:
        xa = int((bb["min"][0] + ox_ - bx) / 4) - c0 - 1; xb = int((bb["max"][0] + ox_ - bx) / 4) - c0 + 2
        ya = int((bb["min"][1] + oy_ - by) / 4) - r0 - 1; yb = int((bb["max"][1] + oy_ - by) / 4) - r0 + 2
        if 0 <= xa < n and 0 <= ya < n: bm[max(ya, 0):yb, max(xa, 0):xb] = True
    road &= ~bm
    rv = json.load(open(f"{SCR}/landmarks_review.json"))[w]["river"]
    rimg = Image.new("L", (n, n), 0); ImageDraw.Draw(rimg).line([((x + ox_ - bx) / 4 - c0, (y + oy_ - by) / 4 - r0) for x, y in rv], fill=255, width=11)
    road &= ~(np.asarray(rimg) > 0)
    road = binary_closing(road, iterations=1)
    road = remove_small_objects(road, max_size=40)
    dist = distance_transform_edt(road)
    sk = skeletonize(road)
    # chain the skeleton
    H, Wd = sk.shape; nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pts = set(zip(*np.nonzero(sk)))
    deg = {p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in nbr) for p in pts}
    nodes = {p for p, d in deg.items() if d != 2}
    seen = set(); chains = []
    for s in nodes:
        for dr, dc in nbr:
            q = (s[0] + dr, s[1] + dc)
            if q not in pts or (s, q) in seen: continue
            ch = [s, q]; seen.add((s, q)); seen.add((q, s)); prev, cur = s, q
            while cur not in nodes:
                nxt = [(cur[0] + a_, cur[1] + b_) for a_, b_ in nbr if (cur[0] + a_, cur[1] + b_) in pts and (cur[0] + a_, cur[1] + b_) != prev]
                nxt = [p for p in nxt if (cur, p) not in seen]
                if not nxt: break
                p = nxt[0]; seen.add((cur, p)); seen.add((p, cur)); ch.append(p); prev, cur = cur, p
            chains.append(ch)
    sklab, _ = label(sk, structure=np.ones((3, 3)))
    comp_len = np.bincount(sklab.ravel()) * 4
    keep = []
    for ch in chains:
        L = len(ch) * 4
        wid = np.median([dist[r, c] for r, c in ch]) * 2 * 4
        comp = sklab[ch[len(ch) // 2]]
        spur = (ch[0] not in nodes or deg.get(ch[0], 0) == 1 or deg.get(ch[-1], 0) == 1) and L < 45
        if comp_len[comp] >= 500 and wid >= 7 and not spur and L >= 50 and np.hypot(ch[0][0] - ch[-1][0], ch[0][1] - ch[-1][1]) * 4 > 25:
            arr = np.array([[c, r] for r, c in ch], float)
            arr = approximate_polygon(arr, tolerance=1.2)
            g = [[float(bx + (c0 + x + .5) * 4 - ox_), float(by + (r0 + y + .5) * 4 - oy_)] for x, y in arr]
            keep.append({"pts": g, "len": L, "width": float(wid)})
    print(w, "chains", len(chains), "kept", len(keep), "total km", round(sum(k["len"] for k in keep) / 1000, 1))
    out[w] = {"roads": keep}
    # preview
    pv = Image.fromarray(a.astype(np.uint8)).resize((n * 2, n * 2)); d = ImageDraw.Draw(pv)
    for k in keep:
        d.line([((x + ox_ - bx) / 4 * 2 - c0 * 2, (y + oy_ - by) / 4 * 2 - r0 * 2) for x, y in k["pts"]], fill=(255, 0, 255), width=3)
    pv.thumbnail((900, 900)); pv.save(f"{SCR}/roads_{w}.png")
json.dump(out, open(f"{SCR}/roads.json", "w"))
