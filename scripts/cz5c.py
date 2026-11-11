"""Zoomed Control Zone sheets at native 2 m heightmap resolution."""
import sys, os, json, heapq, re, math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter1d, zoom
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = sys.argv[1]
OUT = os.path.join(SCR, "cz"); os.makedirs(OUT, exist_ok=True)
D = os.path.join(SCR, "repo/wardogs-data/data/maps")
PX, M = 1000, 44
S = PX + 2 * M
FIT = True
FIT_MARGIN = 12
LANDMARKS = {
    "europe": [
        {"name": "CHURCH", "x": -6052, "y": -6097},
        {"name": "POOL", "kind": "pool"},
        {"name": "STADIUM", "x": -6400, "y": -6328},
        {"name": "APARTMENTS", "kind": "apartment"},
        {"name": "LUMBERYARD", "x": -6038, "y": -5630},
    ],
    "kavkazi": [
        {"name": "APARTMENTS", "kind": "apartment"},
        {"name": "MARKET", "kind": "market"},
        {"name": "WAREHOUSES", "kind": "warehouse"},
    ],
}
import matplotlib.cm as _cm
def RAMP(v):
    t = 0.04 + 0.92 * np.clip(v, 0, 1)
    c = matplotlib.colormaps["turbo"](t)[..., :3]
    return 0.92 * c + 0.08
ORI = json.load(open(os.path.join(sys.argv[1], "orient.json"))) if os.path.exists(os.path.join(sys.argv[1], "orient.json")) else {}
def bpoly(bb):
    """world-space corner list for a building: rotated footprint recovered from imagery, else the AABB"""
    x0, y0 = bb["min"][:2]; x1, y1 = bb["max"][:2]; o = ORI.get(bb["id"]) if WORLD_ROT else None
    if o and abs(o[0]) > 4 and o[1] > 0.12:
        th = math.radians(o[0]); c, s = abs(math.cos(th)), abs(math.sin(th)); det = c * c - s * s; Wd, Hd = x1 - x0, y1 - y0
        if abs(det) > 0.15:
            w_ = (Wd * c - Hd * s) / det; h_ = (Hd * c - Wd * s) / det
            if w_ > 1 and h_ > 1:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                return [(cx + sx * w_ / 2 * math.cos(th) - sy * h_ / 2 * math.sin(th), cy + sx * w_ / 2 * math.sin(th) + sy * h_ / 2 * math.cos(th)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
WORLD_ROT = True
REV = json.load(open(os.path.join(sys.argv[1], "review_cfg.json"))); REV.setdefault("north-america", {"boxed": [], "generic": [], "roads": [], "river": [], "river_name": "", "water": [], "sunflower": [], "outlines": []})
TEAM = {"Alpha": ("#1f5fa8", "A"), "Bravo": ("#c9671a", "B"), "Charlie": ("#6b3fa0", "C")}

def halo_text(x, y, s, size=11, weight="normal", fill="#1f2a33", italic=False, anchor="middle"):
    st = ' font-style="italic"' if italic else ""
    c = f'x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{st}'
    return (f'<text {c} fill="none" stroke="#fbf8ef" stroke-width="3.2" stroke-linejoin="round">{s}</text>'
            f'<text {c} fill="{fill}">{s}</text>')

def d_of(p):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)

def flow(z):
    n = z.shape[0]; z = z.astype(np.float64).copy()
    done = np.zeros(z.shape, bool); h = []
    for r in range(n):
        for c in range(n):
            if r in (0, n - 1) or c in (0, n - 1):
                heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while h:
        zz, r, c = heapq.heappop(h)
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not done[rr, cc]:
                done[rr, cc] = True
                if z[rr, cc] <= zz: z[rr, cc] = zz + 1e-4
                heapq.heappush(h, (z[rr, cc], rr, cc))
    pad = np.pad(z, 1, constant_values=-1e9)
    best = np.zeros(z.shape); rec = np.arange(n * n).reshape(n, n)
    R, C = np.mgrid[0:n, 0:n]
    for dr, dc in nb:
        drop = (z - pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]) / np.hypot(dr, dc)
        b = drop > best; best = np.where(b, drop, best)
        rec = np.where(b, np.clip(R + dr, 0, n - 1) * n + np.clip(C + dc, 0, n - 1), rec)
    rec = rec.ravel(); acc = np.ones(n * n)
    for i in np.argsort(-z.ravel(), kind="stable"):
        j = rec[i]
        if j != i: acc[j] += acc[i]
    return rec, acc, n

def streams(rec, acc, n, T):
    on = acc >= T; indeg = np.zeros(n * n, int)
    for i in np.nonzero(on)[0]:
        j = rec[i]
        if j != i and on[j]: indeg[j] += 1
    lines = []; seen = set()
    for s in [i for i in np.nonzero(on)[0] if indeg[i] != 1]:
        cur = s; pts = [cur]
        while True:
            seen.add(cur); j = rec[cur]
            if j == cur or not on[j]: break
            pts.append(j)
            if indeg[j] != 1 or j in seen: break
            cur = j
        if len(pts) > 2:
            lines.append((np.array([(p % n, p // n) for p in pts], float), acc[pts[-1]]))
    return lines

def sheet(w, W, interval, idx_every, tag="", split=False):
    t = json.load(open(f"{D}/{w}/terrain.json"))
    (x0, y0), (x1, y1) = t["bounds"]["min"], t["bounds"]["max"]
    mpp = (x1 - x0) / 8192
    std = [c for c in json.load(open(f"{D}/{w}/objectives.json"))["configs"] if c["mode"] == "Standard"][0]
    alts = std["alternatives"]
    cx = np.mean([min(a["zone"]["center"][0] for a in alts), max(a["zone"]["center"][0] for a in alts)])
    cy = np.mean([min(a["zone"]["center"][1] for a in alts), max(a["zone"]["center"][1] for a in alts)])
    if FIT:
        rr_ = alts[0]["zone"]["extents"][0]
        lx_ = min(a["zone"]["center"][0] for a in alts) - rr_; hx_ = max(a["zone"]["center"][0] for a in alts) + rr_
        ly_ = min(a["zone"]["center"][1] for a in alts) - rr_; hy_ = max(a["zone"]["center"][1] for a in alts) + rr_
        cx, cy = (lx_ + hx_) / 2, (ly_ + hy_) / 2; W = max(hx_ - lx_, hy_ - ly_) + 2 * FIT_MARGIN
    wx0, wy0 = cx - W / 2, cy - W / 2
    c0 = int(round((wx0 - x0) / mpp)); r0 = int(round((wy0 - y0) / mpp)); n = int(round(W / mpp))
    wx0, wy0 = x0 + c0 * mpp, y0 + r0 * mpp; W = n * mpp
    a = np.load(f"{SCR}/{w}.npy")[r0:r0 + n, c0:c0 + n].astype(np.float32)
    from scipy import ndimage as _ndi
    _m = a == 0
    if _m.any() and not _m.all():
        _ix = _ndi.distance_transform_edt(_m, return_distances=False, return_indices=True); a = a[tuple(_ix)]
    Zfull = t["zRange"]["min"] + a / 65535 * (t["zRange"]["max"] - t["zRange"]["min"])
    full = np.load(f"{SCR}/{w}.npy", mmap_mode="r")
    base = t["zRange"]["min"] + float(full[full > 0].min()) / 65535 * (t["zRange"]["max"] - t["zRange"]["min"])
    Z = Zfull - base  # same datum as the full sheet
    sc = PX / n
    w2p = lambda x, y: (M + (x - wx0) / W * PX, M + (y - wy0) / W * PX)

    # relief image (2x)
    z2 = gaussian_filter(Z, 1.0)
    gy, gx = np.gradient(z2, mpp)
    az, alt = np.radians(315), np.radians(40)
    slope = np.arctan(np.hypot(gx, gy) * 1.3); aspect = np.arctan2(-gx, gy)
    sh = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    _lo, _hi = np.percentile(Z, 1), np.percentile(Z, 99)
    rel = np.clip((Z - _lo) / (_hi - _lo), 0, 1)
    rgb = RAMP(rel)
    img = np.clip(rgb * (0.70 + 0.30 * sh[..., None] / sh.max()), 0, 1)
    json.dump({"min": float(_lo), "max": float(_hi), "ramp": ["#%02x%02x%02x" % tuple(int(v * 255) for v in RAMP(np.array([q]))[0]) for q in np.linspace(0, 1, 12)]}, open(f"{OUT}/{w}-colramp.json", "w"))
    Image.fromarray((img * 255).astype(np.uint8)).resize((2000, 2000), Image.LANCZOS).save(f"{OUT}/{w}-cz{tag}-relief.webp", quality=86)

    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">',
         f'',
         f'<clipPath id="c"><rect x="{M}" y="{M}" width="{PX}" height="{PX}"/></clipPath><g clip-path="url(#c)">']
    # 100 m grid (aligned to the main sheet's grid)
    P.append('<g stroke="#1f2a33" stroke-width="0.4" stroke-opacity="0.22">')
    g0x = np.ceil((wx0 - x0) / 100) * 100 + x0; g0y = np.ceil((wy0 - y0) / 100) * 100 + y0
    for gxv in np.arange(g0x, wx0 + W, 100):
        if round((gxv - x0) / 100) % 10:
            px = w2p(gxv, 0)[0]; P.append(f'<line x1="{px:.1f}" y1="{M}" x2="{px:.1f}" y2="{M+PX}"/>')
    for gyv in np.arange(g0y, wy0 + W, 100):
        if round((gyv - y0) / 100) % 10:
            py = w2p(0, gyv)[1]; P.append(f'<line x1="{M}" y1="{py:.1f}" x2="{M+PX}" y2="{py:.1f}"/>')
    P.append("</g>")
    # contours
    zc = gaussian_filter(Z, 2.5)
    lv = np.arange(np.ceil(zc.min() / interval) * interval, zc.max(), interval)
    fig = plt.figure(); cs = plt.contour(np.arange(n), np.arange(n), zc, levels=lv); plt.close(fig)
    P.append('<g fill="none" stroke-linejoin="round" stroke-linecap="round">')
    idx_lines = []; Pm = []
    for level, segs in zip(cs.levels, cs.allsegs):
        idx = int(round(level / interval)) % idx_every == 0
        for s in segs:
            if len(s) < 5: continue
            p = np.c_[M + (s[:, 0] + .5) * sc, M + (s[:, 1] + .5) * sc]; p = np.vstack([p[::3], p[-1:]])
            if idx:
                P.append(f'<path d="{d_of(p)}" stroke="#14142b" stroke-opacity="0.62" stroke-width="1.0"/>'); idx_lines.append((level, p))
            else:
                (Pm if split else P).append(f'<path d="{d_of(p)}" stroke="#14142b" stroke-width="0.3" stroke-opacity="0.26"/>')
    P.append("</g>")
    # drainage at 4 m
    zf = gaussian_filter(Z[::2, ::2], 1.5)
    rec, acc, nn = flow(zf)
    T = 4000  # cells of 16 m² ≈ 0.064 km²
    P.append('<g fill="none" stroke="#3f7fb3" stroke-linecap="round" stroke-linejoin="round">')
    for pts, ac in []:
        sm = pts.copy()
        if len(sm) > 7:
            sm[1:-1] = np.c_[uniform_filter1d(pts[:, 0], 7)[1:-1], uniform_filter1d(pts[:, 1], 7)[1:-1]]
        p = np.c_[M + (sm[:, 0] + .5) * 2 * sc, M + (sm[:, 1] + .5) * 2 * sc][::2]
        P.append(f'<path d="{d_of(p)}" stroke-width="{0.9 + 1.3*np.log10(ac/T):.2f}"/>')
    P.append("</g>")
    # contour labels
    k = 0
    for level, p in idx_lines:
        if len(p) < 40: continue
        k += 1
        if k % 3: continue
        i = len(p) // 2; x, y = p[i]
        if not (M + 25 < x < M + PX - 25 and M + 25 < y < M + PX - 25): continue
        dx, dy = p[min(i + 2, len(p) - 1)] - p[max(i - 2, 0)]
        ang = np.degrees(np.arctan2(dy, dx)); ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        P.append(f'<g transform="translate({x:.1f},{y:.1f}) rotate({ang:.0f})"><rect x="-12" y="-5.5" width="24" height="11" fill="#f6f1e4" fill-opacity="0.92"/>'
                 f'<text y="3.4" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="8.5" fill="#7a4317">{int(level)}</text></g>')
    # review features: fields, water, river
    def poly_d(poly):
        return "M" + " L".join(f"{w2p(x, y)[0]:.1f},{w2p(x, y)[1]:.1f}" for x, y in poly) + "Z"
    for poly in REV[w].get("sunflower", []):
        P.append(f'<path d="{poly_d(poly)}" fill="#ffffff" fill-opacity="0.28" stroke="#14142b" stroke-width="1.6" stroke-dasharray="6 4"/>')
    if REV[w].get("river"):
        rpx = [w2p(x, y) for x, y in REV[w]["river"]]
        P.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in rpx) + '" fill="none" stroke="#0b2a6b" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
        P.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in rpx) + '" fill="none" stroke="#35c8ff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.85"/>')
    for road in REV[w].get("roads", []):
        rd = "M" + " L".join(f"{w2p(x, y)[0]:.1f},{w2p(x, y)[1]:.1f}" for x, y in road)
        P.append(f'<path d="{rd}" fill="none" stroke="#14142b" stroke-width="5.4" stroke-linecap="round" stroke-linejoin="round"/>')
    for road in REV[w].get("roads", []):
        rd = "M" + " L".join(f"{w2p(x, y)[0]:.1f},{w2p(x, y)[1]:.1f}" for x, y in road)
        P.append(f'<path d="{rd}" fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    for poly in REV[w].get("outlines", []):
        P.append(f'<path d="{poly_d(poly)}" fill="#e9e3cf" fill-opacity="0.55" stroke="#1f2a33" stroke-width="2"/>')
    for poly in REV[w].get("water", []):
        P.append(f'<path d="{poly_d(poly)}" fill="#2f6fb0" fill-opacity="0.85" stroke="#1f4f80" stroke-width="1"/>')
    # buildings
    P.append('<g fill="#2b2620" stroke="#fbf8ef" stroke-width="0.5">')
    for bb in json.load(open(f"{D}/{w}/buildings.json"))["buildings"]:
        ax, ay = w2p(bb["min"][0], bb["min"][1]); bx, by = w2p(bb["max"][0], bb["max"][1])
        if bx < M or ax > M + PX or by < M or ay > M + PX: continue
        fill = ' fill="#5a5047"' if bb["kind"] in ("warehouse", "garage") else ""
        P.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in (w2p(*q) for q in bpoly(bb))) + f'"{fill}/>')
    P.append("</g>")
    # zones
    P.append('<g fill="none" stroke="#14142b" stroke-width="2" stroke-dasharray="8 5">')
    zl = []
    for alt_ in alts:
        x, y = w2p(*alt_["zone"]["center"][:2]); r = alt_["zone"]["extents"][0] / W * PX
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>'); zl.append((x, y, r, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", alt_["id"].split(".")[2])))
    P.append("</g>")
    for x, y, r, nm in zl:
        P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#b3261e"/>')
    # objects
    objs = json.load(open(f"{D}/{w}/objects.json"))["objects"]
    for o in objs:
        x, y = w2p(*o["position"][:2])
        if not (M < x < M + PX and M < y < M + PX): continue
        if o["kind"] == "fuel-refill":
            P.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="#fbf8ef" stroke="#2b2620" stroke-width="1.4" transform="rotate(45 {x:.1f} {y:.1f})"/>')
        elif o["kind"] == "safe-zone" and o.get("faction") in TEAM:
            col, L = TEAM[o["faction"]]
            P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{col}" stroke="#fbf8ef" stroke-width="2"/><text x="{x:.1f}" y="{y+4.5:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#fff">{L}</text>')
    # capture towers (positions from WardogsHub tactical layer, aligned to game coords)
    for tw in json.load(open(os.path.join(SCR, "towers.json"))).get(w, []):
        tx, ty = w2p(tw["x"], tw["y"]); num = tw["name"].split()[-1]
        if not (M < tx < M + PX and M < ty < M + PX): continue
        P.append(f'<g><circle cx="{tx:.1f}" cy="{ty:.1f}" r="11" fill="#1f2a33" stroke="#f2b33d" stroke-width="2.4"/>'
                 f'<text x="{tx:.1f}" y="{ty+11*0.42:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{11*1.15:.0f}" font-weight="bold" fill="#ffffff">{num}</text></g>')
        if True:
            P.append(halo_text(tx + 16, ty - 9, f"TOWER {num}", size=11, weight="bold", fill="#1f2a33", anchor="start") if tx < M + PX - 90 else halo_text(tx - 16, ty - 9, f"TOWER {num}", size=11, weight="bold", fill="#1f2a33", anchor="end"))
    # spot heights
    zz = gaussian_filter(Z, 6); mx = zz == maximum_filter(zz, size=151)
    rr, cc = np.nonzero(mx); o = np.argsort(-zz[rr, cc]); pk = []; spots = []
    for i in o:
        r, c = rr[i], cc[i]
        if 40 < r < n - 40 and 40 < c < n - 40 and all(np.hypot(r - a, c - b) > 180 for a, b in pk):
            pk.append((r, c)); x, y = M + c * sc, M + r * sc
            P.append(f'<path d="M{x-4:.1f},{y+3:.1f} L{x:.1f},{y-4:.1f} L{x+4:.1f},{y+3:.1f}Z" fill="#2b2620"/>' + halo_text(x + 6, y + 4, f"{int(Z[r, c])}", size=10, anchor="start", fill="#2b2620")); spots.append((x, y, f"{int(Z[r, c])}"))
        if len(pk) >= 8: break
    # landmarks
    lms = list(REV[w]["boxed"])
    rp = [p for p in REV[w]["river"] if wx0 + 60 < p[0] < wx0 + W - 60 and wy0 + 60 < p[1] < wy0 + W - 60]
    if rp and REV[w].get("river_name"): lms.append({"name": REV[w]["river_name"], "x": rp[len(rp) // 3][0], "y": rp[len(rp) // 3][1]})
    bl = json.load(open(f"{D}/{w}/buildings.json"))["buildings"]
    placed = []; pts = []
    for L in lms:
        if L.get("kind"):
            sel = [bb for bb in bl if bb["kind"] == L["kind"]]
            sel = [bb for bb in sel if wx0 < (bb["min"][0]+bb["max"][0])/2 < wx0 + W and wy0 < (bb["min"][1]+bb["max"][1])/2 < wy0 + W]
            if not sel: continue
            if L.get("largest"): sel = [max(sel, key=lambda bb: (bb["max"][0]-bb["min"][0])*(bb["max"][1]-bb["min"][1]))]
            for bb in sel:
                ax, ay = w2p(bb["min"][0], bb["min"][1]); bx, by = w2p(bb["max"][0], bb["max"][1])
                P.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bx-ax:.1f}" height="{by-ay:.1f}" fill="#f2b33d" stroke="#1f2a33" stroke-width="1.2"/>')
            cxs = np.mean([(bb["min"][0]+bb["max"][0])/2 for bb in sel]); cys = np.mean([(bb["min"][1]+bb["max"][1])/2 for bb in sel])
            cc_ = np.array([[(bb["min"][0]+bb["max"][0])/2, (bb["min"][1]+bb["max"][1])/2] for bb in sel])
            k_ = np.argmin(np.hypot(cc_[:, 0] - cxs, cc_[:, 1] - cys)); cxs, cys = cc_[k_]
        else:
            cxs, cys = L["x"], L["y"]
        px_, py_ = w2p(cxs, cys); pts.append((px_, py_, L["name"]))
    others = [w2p(t_["x"], t_["y"]) for t_ in json.load(open(os.path.join(SCR, "towers.json"))).get(w, [])]
    inside_names = {L_["name"] for L_ in lms if L_.get("inside")}
    for px_, py_, name in pts:
        if name not in inside_names:
            P.append(f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="4.5" fill="#1f2a33" stroke="#fbf8ef" stroke-width="1.5"/>')
    occ_img = Image.new("L", (S, S), 0); od = ImageDraw.Draw(occ_img)
    for bb in bl:
        ax, ay = w2p(bb["min"][0], bb["min"][1]); bx2, by2 = w2p(bb["max"][0], bb["max"][1])
        od.polygon([w2p(*q) for q in bpoly(bb)], fill=255); od.rectangle((ax + 2, ay + 2, bx2 - 2, by2 - 2), fill=255) if False else None
    for road in REV[w].get("roads", []): od.line([w2p(x, y) for x, y in road], fill=255, width=7)
    if REV[w].get("river"): od.line([w2p(x, y) for x, y in REV[w]["river"]], fill=255, width=8)
    for poly in REV[w].get("outlines", []) + REV[w].get("water", []): od.polygon([w2p(x, y) for x, y in poly], fill=255)
    for a_, b_ in others: od.ellipse((a_ - 16, b_ - 16, a_ + 90, b_ + 12), fill=255)   # tower disc + its label
    for a_, b_, nm_ in pts:
        if nm_ not in inside_names: od.ellipse((a_ - 7, b_ - 7, a_ + 7, b_ + 7), fill=255)
    for a_, b_, t_ in spots: od.rectangle((a_ - 7, b_ - 8, a_ + 10 + 6.5 * len(t_), b_ + 8), fill=255)
    for G in REV[w].get("generic", []):
        gx_, gy_ = w2p(G["x"], G["y"])
        if M + 10 < gx_ < M + PX - 80 and M + 10 < gy_ < M + PX - 10: od.rectangle((gx_ - 4, gy_ - 7, gx_ + 8 + 5.8 * len(G["name"]), gy_ + 7), fill=255)
    for ci in range(int((wx0 - x0) // 1000), int((wx0 + W - x0) // 1000) + 1):
        for ri in range(int((wy0 - y0) // 1000), int((wy0 + W - y0) // 1000) + 1):
            gx_, gy_ = w2p(max(x0 + ci * 1000, wx0), max(y0 + ri * 1000, wy0))
            od.rectangle((gx_, gy_, gx_ + 52, gy_ + 24), fill=255)
    occ = (np.asarray(occ_img) > 0).astype(np.int32); I = occ.cumsum(0).cumsum(1)
    def boxsum0(x0_, y0_, x1_, y1_):
        x0_, y0_, x1_, y1_ = int(max(x0_, 1)), int(max(y0_, 1)), int(min(x1_, S - 1)), int(min(y1_, S - 1))
        return I[y1_, x1_] - I[y0_ - 1, x1_] - I[y1_, x0_ - 1] + I[y0_ - 1, x0_ - 1]
    zplaced = []
    for i_, (zx, zy, zr, nm) in enumerate(zl):
        txt = f"{nm.upper()} ZONE"; tw0 = 8.6 * len(txt) + 10; th0 = 18
        oth = [(a, b) for j, (a, b, _, _) in enumerate(zl) if j != i_]
        best = None
        for ang in np.radians(np.arange(0, 360, 6)):
            for off in (-16, -30, 16):
                lx = zx + (zr + off) * np.cos(ang); ly = zy + (zr + off) * np.sin(ang)
                if lx - tw0 / 2 < M + 6 or lx + tw0 / 2 > M + PX - 6 or ly - th0 / 2 < M + 6 or ly + th0 / 2 > M + PX - 6: continue
                frac = boxsum0(lx - tw0 / 2, ly - th0 / 2, lx + tw0 / 2, ly + th0 / 2) / (tw0 * th0)
                clash = any(abs(lx - qx) < (tw0 + qw) / 2 + 4 and abs(ly - qy) < (th0 + qh) / 2 + 2 for qx, qy, qw, qh in zplaced)
                dmin = min([np.hypot(lx - a, ly - b) for a, b in oth] + [400])
                score = -2000 * clash - 500 * frac + 0.15 * dmin - 0.5 * abs(off)
                if best is None or score > best[0]: best = (score, lx, ly)
        if best is None: continue
        _, lx, ly = best; zplaced.append((lx, ly, tw0, th0))
        P.append(halo_text(lx, ly + 4.5, txt, size=12, weight="bold", fill="#14142b", anchor="middle"))
        od.rectangle((lx - tw0 / 2 - 2, ly - th0 / 2 - 2, lx + tw0 / 2 + 2, ly + th0 / 2 + 2), fill=255)
    occ = (np.asarray(occ_img) > 0).astype(np.int32); I = occ.cumsum(0).cumsum(1)
    def boxsum(x0_, y0_, x1_, y1_):
        x0_, y0_, x1_, y1_ = int(max(x0_, 0)), int(max(y0_, 0)), int(min(x1_, S - 1)), int(min(y1_, S - 1))
        return I[y1_, x1_] - I[y0_ - 1, x1_] - I[y1_, x0_ - 1] + I[y0_ - 1, x0_ - 1] if x0_ > 0 and y0_ > 0 else occ[y0_:y1_ + 1, x0_:x1_ + 1].sum()
    order = sorted(range(len(pts)), key=lambda k: -boxsum(pts[k][0] - 40, pts[k][1] - 40, pts[k][0] + 40, pts[k][1] + 40))
    for k in order:
        px_, py_, name = pts[k]
        tw_ = 9.4 * len(name) + 24; th_ = 24
        if name in inside_names:
            placed.append((px_, py_, tw_, th_))
            P.append(f'<rect x="{px_-tw_/2:.1f}" y="{py_-th_/2:.1f}" width="{tw_:.1f}" height="{th_}" rx="3" fill="#fbf8ef" stroke="#1f2a33" stroke-width="1.6"/>')
            P.append(f'<text x="{px_:.1f}" y="{py_+4.6:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" letter-spacing="0.6" fill="#1f2a33">{name}</text>')
            od.rectangle((px_ - tw_ / 2 - 3, py_ - th_ / 2 - 3, px_ + tw_ / 2 + 3, py_ + th_ / 2 + 3), fill=255)
            occ = (np.asarray(occ_img) > 0).astype(np.int32); I = occ.cumsum(0).cumsum(1)
            continue
        best = None
        for rad in (16, 24, 34, 46, 62, 82):
            for ang in np.radians(np.arange(0, 360, 15)):
                lx = px_ + (rad + tw_ / 2 * abs(np.cos(ang))) * np.cos(ang); ly = py_ + (rad + th_ / 2 * abs(np.sin(ang))) * np.sin(ang)
                bx0, by0 = lx - tw_ / 2, ly - th_ / 2
                if bx0 < M + 6 or bx0 + tw_ > M + PX - 6 or by0 < M + 6 or by0 + th_ > M + PX - 6: continue
                frac = boxsum(bx0, by0, bx0 + tw_, by0 + th_) / (tw_ * th_)
                clash = any(abs(lx - qx) < (tw_ + qw) / 2 + 4 and abs(ly - qy) < (th_ + qh) / 2 + 3 for qx, qy, qw, qh in placed)
                score = -2000 * clash - 260 * frac - 2.2 * rad
                if best is None or score > best[0]: best = (score, lx, ly)
        _, lx, ly = best; placed.append((lx, ly, tw_, th_))
        od.rectangle((lx - tw_ / 2 - 3, ly - th_ / 2 - 3, lx + tw_ / 2 + 3, ly + th_ / 2 + 3), fill=255)
        occ = (np.asarray(occ_img) > 0).astype(np.int32); I = occ.cumsum(0).cumsum(1)
        # leader from the point to the nearest edge of the label box
        ex = min(max(px_, lx - tw_ / 2), lx + tw_ / 2); ey = min(max(py_, ly - th_ / 2), ly + th_ / 2)
        P.append(f'<line x1="{px_:.1f}" y1="{py_:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#1f2a33" stroke-width="1.4"/>')
        P.append(f'<rect x="{lx-tw_/2:.1f}" y="{ly-th_/2:.1f}" width="{tw_:.1f}" height="{th_}" rx="3" fill="#fbf8ef" stroke="#1f2a33" stroke-width="1.6"/>')
        P.append(f'<text x="{lx:.1f}" y="{ly+4.6:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" letter-spacing="0.6" fill="#1f2a33">{name}</text>')
    for G in REV[w].get("generic", []):
        gx, gy = w2p(G["x"], G["y"])
        if M + 10 < gx < M + PX - 80 and M + 10 < gy < M + PX - 10:
            P.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="2.6" fill="#5a5047"/>' + halo_text(gx + 5, gy + 4, G["name"], size=10, italic=True, fill="#5a5047", anchor="start"))
    # 1 km grid + refs
    P.append('<g stroke="#1f2a33" stroke-width="1" stroke-opacity="0.55">')
    kx = [x0 + k * 1000 for k in range(17) if wx0 < x0 + k * 1000 < wx0 + W]
    ky = [y0 + k * 1000 for k in range(17) if wy0 < y0 + k * 1000 < wy0 + W]
    for v in kx: px = w2p(v, 0)[0]; P.append(f'<line x1="{px:.1f}" y1="{M}" x2="{px:.1f}" y2="{M+PX}"/>')
    for v in ky: py = w2p(0, v)[1]; P.append(f'<line x1="{M}" y1="{py:.1f}" x2="{M+PX}" y2="{py:.1f}"/>')
    P.append("</g>")
    L = "ABCDEFGHIJKLMNOP"
    for ci in range(int((wx0 - x0) // 1000), int((wx0 + W - x0) // 1000) + 1):
        for ri in range(int((wy0 - y0) // 1000), int((wy0 + W - y0) // 1000) + 1):
            lx = max(x0 + ci * 1000, wx0); ly = max(y0 + ri * 1000, wy0)
            if min(x0 + ci * 1000 + 1000, wx0 + W) - lx < 250 or min(y0 + ri * 1000 + 1000, wy0 + W) - ly < 250: continue
            px, py = w2p(lx, ly)
            if px < M + PX - 30 and py < M + PX - 20 and 0 <= ci < 16 and 0 <= ri < 16:
                P.append(halo_text(px + 6, py + 16, f"{L[ci]}{ri+1}", size=14, weight="bold", anchor="start", fill="#1f2a33"))
    P.append("</g>")
    P.append(f'<rect x="{M}" y="{M}" width="{PX}" height="{PX}" fill="none" stroke="#2b2620" stroke-width="2"/>')
    # edge ticks every 100 m with 1 km labels
    P.append('<g stroke="#2b2620" stroke-width="1">')
    for gxv in np.arange(g0x, wx0 + W, 100):
        px = w2p(gxv, 0)[0]; ln = 10 if round((gxv - x0) / 100) % 10 == 0 else 5
        P.append(f'<line x1="{px:.1f}" y1="{M-ln}" x2="{px:.1f}" y2="{M}"/><line x1="{px:.1f}" y1="{M+PX}" x2="{px:.1f}" y2="{M+PX+ln}"/>')
    for gyv in np.arange(g0y, wy0 + W, 100):
        py = w2p(0, gyv)[1]; ln = 10 if round((gyv - y0) / 100) % 10 == 0 else 5
        P.append(f'<line x1="{M-ln}" y1="{py:.1f}" x2="{M}" y2="{py:.1f}"/><line x1="{M+PX}" y1="{py:.1f}" x2="{M+PX+ln}" y2="{py:.1f}"/>')
    P.append("</g></svg>")
    path = f"{OUT}/{w}-cz{tag}-overlay.svg"; open(path, "w").write("\n".join(P))
    if split:
        mp = f"{OUT}/{w}-cz{tag}-minor.svg"
        open(mp, "w").write("\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">', f'<clipPath id="c"><rect x="{M}" y="{M}" width="{PX}" height="{PX}"/></clipPath><g clip-path="url(#c)" fill="none" stroke-linejoin="round" stroke-linecap="round">'] + Pm + ["</g></svg>"]))
        print("  minor layer", os.path.getsize(mp) // 1024, "KB")
    print(w, f"window {W:.0f} m, relief {Z.max()-Z.min():.0f} m ({Z.min():.0f}-{Z.max():.0f}), {len(cs.levels)} levels, svg {os.path.getsize(path)//1024} KB")

sheet("north-america", 0, 0.5, 10, "zstc", True)
