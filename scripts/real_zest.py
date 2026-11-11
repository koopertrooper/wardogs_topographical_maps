"""Topographic sheets from WARDOGS game heightmaps (warpuppies/wardogs-data)."""
import sys, os, json, heapq
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, zoom, uniform_filter1d
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = sys.argv[1]
OUT = os.path.join(SCR, "real")
D = os.path.join(SCR, "repo/wardogs-data/data/maps")
os.makedirs(OUT, exist_ok=True)

PX, M = 1000, 44
S = PX + 2 * M
G = 1024  # analysis grid

def load(w):
    t = json.load(open(f"{D}/{w}/terrain.json"))
    a = np.load(f"{SCR}/{w}.npy").astype(np.float32)
    from scipy import ndimage as _ndi
    _m = a == 0
    if _m.any() and not _m.all():
        _ix = _ndi.distance_transform_edt(_m, return_distances=False, return_indices=True); a = a[tuple(_ix)]
    zmin, zmax = t["zRange"]["min"], t["zRange"]["max"]
    Z = zmin + a / 65535 * (zmax - zmin)
    return t, Z

def setup(t):
    (x0, y0), (x1, y1) = t["bounds"]["min"], t["bounds"]["max"]
    span = max(x1 - x0, y1 - y0)
    def w2p(x, y):  # world metres -> svg px
        return M + (x - x0) / span * PX, M + (y - y0) / span * PX
    def g2p(c, r):  # analysis grid -> svg px
        return M + c / G * PX * ((x1 - x0) / span), M + r / G * PX * ((y1 - y0) / span)
    return x0, y0, x1, y1, span, w2p, g2p

# ---------------------------------------------------------------- terrain products
def relief(Z, path, SPANM=16320):
    n = 2048
    z = Z.reshape(n, 4, n, 4).mean(axis=(1, 3))
    z = gaussian_filter(z, 0.8)
    cell = (SPANM) / n
    gy, gx = np.gradient(z, cell)
    az, alt = np.radians(315), np.radians(42)
    slope = np.arctan(np.hypot(gx, gy) * 1.4)
    aspect = np.arctan2(-gx, gy)
    sh = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    rel = (z - z.min()) / (z.max() - z.min())
    stops = np.array([0.0, 0.18, 0.45, 0.75, 1.0])
    cols = np.array([[201, 219, 178], [222, 228, 190], [236, 226, 192], [222, 202, 166], [244, 239, 228]]) / 255
    rgb = np.stack([np.interp(rel, stops, cols[:, k]) for k in range(3)], -1)
    img = rgb * (0.6 + 0.4 * sh[..., None] / sh.max())
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=86)

def contours(zg, interval, idx_every):
    lv = np.arange(np.ceil(zg.min() / interval) * interval, zg.max(), interval)
    fig = plt.figure()
    cs = plt.contour(np.arange(G), np.arange(G), zg, levels=lv)
    plt.close(fig)
    out = []
    for level, segs in zip(cs.levels, cs.allsegs):
        idx = int(round(level)) % (interval * idx_every) == 0
        for s in segs:
            if len(s) >= 5:
                out.append((level, idx, s))
    return out

def flow(zg):
    """Priority-flood fill then D8 flow accumulation. Returns receivers, accumulation."""
    n = zg.shape[0]
    z = zg.astype(np.float64).copy()
    done = np.zeros(z.shape, bool)
    h = []
    for r in range(n):
        for c in (0, n - 1):
            heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
    for c in range(1, n - 1):
        for r in (0, n - 1):
            heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while h:
        zz, r, c = heapq.heappop(h)
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not done[rr, cc]:
                done[rr, cc] = True
                if z[rr, cc] <= zz:
                    z[rr, cc] = zz + 1e-4
                heapq.heappush(h, (z[rr, cc], rr, cc))
    # D8 receivers
    pad = np.pad(z, 1, constant_values=-1e9)
    best = np.full(z.shape, 0.0); rec = np.arange(n * n).reshape(n, n)
    R, C = np.mgrid[0:n, 0:n]
    for dr, dc in nb:
        dist = np.hypot(dr, dc)
        drop = (z - pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]) / dist
        rr, cc = np.clip(R + dr, 0, n - 1), np.clip(C + dc, 0, n - 1)
        better = drop > best
        best = np.where(better, drop, best)
        rec = np.where(better, rr * n + cc, rec)
    rec = rec.ravel()
    order = np.argsort(-z.ravel(), kind="stable")
    acc = np.ones(n * n)
    for i in order:
        j = rec[i]
        if j != i:
            acc[j] += acc[i]
    return rec, acc

def streams(rec, acc, thresh):
    n = G
    on = acc >= thresh
    indeg = np.zeros(n * n, int)
    ids = np.nonzero(on)[0]
    for i in ids:
        j = rec[i]
        if j != i and on[j]:
            indeg[j] += 1
    starts = [i for i in ids if indeg[i] != 1]
    seen = set(); lines = []
    for s in starts:
        if s in seen and indeg[s] == 0:
            continue
        cur = s; pts = [cur]
        while True:
            seen.add(cur)
            j = rec[cur]
            if j == cur or not on[j]:
                break
            pts.append(j)
            if indeg[j] != 1 or j in seen:
                break
            cur = j
        if len(pts) > 2:
            lines.append((np.array([(p % n, p // n) for p in pts], float), acc[pts[-1]]))
    return lines

def peaks(zg, k, mind):
    zz = gaussian_filter(zg, 2)
    mx = zz == maximum_filter(zz, size=61)
    rr, cc = np.nonzero(mx)
    o = np.argsort(-zz[rr, cc]); out = []
    for i in o:
        r, c = rr[i], cc[i]
        if not (25 < r < G - 25 and 25 < c < G - 25):
            continue
        if all(np.hypot(r - a, c - b) > mind for a, b, _ in out):
            out.append((r, c, zg[r, c]))
        if len(out) >= k:
            break
    return out

# ---------------------------------------------------------------- svg helpers
def d_of(pts, step=1):
    p = pts[::step]
    if len(pts) and (len(pts) - 1) % step:
        p = np.vstack([p, pts[-1:]])
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)

def halo_text(x, y, s, size=11, weight="normal", fill="#1f2a33", italic=False, anchor="middle", spacing=0):
    st = ' font-style="italic"' if italic else ""
    c = f'x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" letter-spacing="{spacing}"{st}'
    return (f'<text {c} fill="none" stroke="#fbf8ef" stroke-width="3" stroke-linejoin="round" stroke-opacity="0.9">{s}</text>'
            f'<text {c} fill="{fill}">{s}</text>')

TEAM = {"Alpha": ("#1f5fa8", "A"), "Bravo": ("#c9671a", "B"), "Charlie": ("#6b3fa0", "C")}

def build(w, title_zone_prefix, interval, idx_every, stream_thresh):
    t, Z = load(w)
    x0, y0, x1, y1, span, w2p, g2p = setup(t)
    base = float(Z.min())
    relief(Z, os.path.join(OUT, f"{w}-relief.webp"), span)
    zg = Z.reshape(G, 8, G, 8).mean(axis=(1, 3)) - base
    zc = gaussian_filter(zg, 1.3)
    scale = PX / G * ((x1 - x0) / span)
    cs = contours(zc, interval, idx_every)
    rec, acc = flow(gaussian_filter(zg, 1.0))
    st = streams(rec, acc, stream_thresh)

    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">',
         f'<rect width="{S}" height="{S}" fill="#fbf8ef"/>',
         f'<rect x="{M}" y="{M}" width="{PX}" height="{PX}" fill="#ffffff"/>',
         f'<clipPath id="c"><rect x="{M}" y="{M}" width="{PX}" height="{PX}"/></clipPath>',
         '<g clip-path="url(#c)">']
    # contours
    P.append('<g fill="none" stroke-linejoin="round" stroke-linecap="round">')
    for level, idx, s in cs:
        pts = np.c_[M + s[:, 0] * scale, M + s[:, 1] * scale]
        step = 3 if len(pts) > 30 else 1
        if idx:
            P.append(f'<path d="{d_of(pts, step)}" stroke="#86491a" stroke-width="1.05"/>')
        else:
            P.append(f'<path d="{d_of(pts, step)}" stroke="#a8693a" stroke-width="0.45" stroke-opacity="0.75"/>')
    P.append("</g>")
    # streams
    P.append('<g fill="none" stroke="#3f7fb3" stroke-linecap="round" stroke-linejoin="round">')
    for pts, a in []:
        sm = pts.copy()
        if len(sm) > 5:
            sm[1:-1] = np.c_[uniform_filter1d(pts[:, 0], 5)[1:-1], uniform_filter1d(pts[:, 1], 5)[1:-1]]
        sp = np.c_[M + (sm[:, 0] + 0.5) * scale, M + (sm[:, 1] + 0.5) * scale]
        wdt = 0.7 + 1.1 * np.log10(max(a / stream_thresh, 1))
        P.append(f'<path d="{d_of(sp, 2)}" stroke-width="{wdt:.2f}"/>')
    P.append("</g>")
    # contour labels on index lines
    k = 0
    for level, idx, s in cs:
        if not idx or len(s) < 70:
            continue
        k += 1
        if k % 5:
            continue
        i = len(s) // 2
        x, y = M + s[i, 0] * scale, M + s[i, 1] * scale
        if not (M + 30 < x < M + PX - 30 and M + 30 < y < M + PX - 30):
            continue
        dx, dy = s[min(i + 2, len(s) - 1)] - s[max(i - 2, 0)]
        ang = np.degrees(np.arctan2(dy, dx))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        P.append(f'<g transform="translate({x:.1f},{y:.1f}) rotate({ang:.0f})"><rect x="-12" y="-5.5" width="24" height="11" fill="#f6f1e4" fill-opacity="0.92"/>'
                 f'<text y="3.4" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="8.5" fill="#7a4317">{int(level)}</text></g>')
    # buildings (axis-aligned boxes from game files)
    b = json.load(open(f"{D}/{w}/buildings.json"))["buildings"]
    P.append('<g fill="#2b2620">')
    for bb in b:
        ax, ay = w2p(bb["min"][0], bb["min"][1]); bx, by = w2p(bb["max"][0], bb["max"][1])
        P.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{max(bx-ax,1.2):.1f}" height="{max(by-ay,1.2):.1f}"/>')
    P.append("</g>")
    # control zone options
    ob = json.load(open(f"{D}/{w}/objectives.json"))["configs"]
    std = [c for c in ob if c["mode"] == "Standard"][0]
    P.append('<g fill="none" stroke="#b3261e" stroke-width="1.6" stroke-dasharray="6 4">')
    zl = []
    for alt in std["alternatives"]:
        cx, cy = w2p(*alt["zone"]["center"][:2])
        r = alt["zone"]["extents"][0] / span * PX
        P.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"/>')
        zl.append((cx, cy, r, alt["id"].split(".")[2]))
    P.append("</g>")
    for i, (cx, cy, r, name) in enumerate(zl):
        ang = np.radians(-90 + i * 360 / max(len(zl), 1) + 25)
        lx, ly = cx + (r + 12) * np.cos(ang), cy + (r + 12) * np.sin(ang)
        anchor = "start" if np.cos(ang) > 0.3 else "end" if np.cos(ang) < -0.3 else "middle"
        P.append(halo_text(lx, ly + 4, name, size=10, weight="bold", fill="#b3261e", anchor=anchor))
    # objects: safe zones (team), fuel
    objs = json.load(open(f"{D}/{w}/objects.json"))["objects"]
    for o in objs:
        if o["kind"] == "fuel-refill":
            x, y = w2p(*o["position"][:2])
            P.append(f'<rect x="{x-3.5:.1f}" y="{y-3.5:.1f}" width="7" height="7" fill="#fbf8ef" stroke="#2b2620" stroke-width="1.3" transform="rotate(45 {x:.1f} {y:.1f})"/>')
    for o in objs:
        if o["kind"] == "safe-zone" and o.get("faction") in TEAM:
            x, y = w2p(*o["position"][:2]); col, L = TEAM[o["faction"]]
            P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}" stroke="#fbf8ef" stroke-width="2"/>'
                     f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="bold" fill="#ffffff">{L}</text>')
    # capture towers (positions from WardogsHub tactical layer, aligned to game coords)
    for tw in json.load(open(os.path.join(SCR, "towers.json"))).get(w, []):
        tx, ty = w2p(tw["x"], tw["y"]); num = tw["name"].split()[-1]
        if not (M < tx < M + PX and M < ty < M + PX): continue
        P.append(f'<g><circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="#1f2a33" stroke="#f2b33d" stroke-width="2.4"/>'
                 f'<text x="{tx:.1f}" y="{ty+7*0.42:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{7*1.15:.0f}" font-weight="bold" fill="#ffffff">{num}</text></g>')
        if False:
            P.append(halo_text(tx + 7 + 5, ty - 7 + 2, f"TOWER {num}", size=11, weight="bold", fill="#1f2a33", anchor="start"))
    # spot heights
    for r, c, h in peaks(zg, 10, 120):
        x, y = M + (c + 0.5) * scale, M + (r + 0.5) * scale
        P.append(f'<path d="M{x-4:.1f},{y+3:.1f} L{x:.1f},{y-4:.1f} L{x+4:.1f},{y+3:.1f}Z" fill="#2b2620"/>')
        P.append(halo_text(x + 6, y + 4, f"{int(h)}", size=10, anchor="start", fill="#2b2620"))
    P.append("</g>")
    # grid: 1 km from bounds.min
    P.append(f'<g stroke="#1f2a33" stroke-opacity="0.4" stroke-width="0.7">')
    km = 1000 / span * PX
    for k in range(1, 18):
        v = M + k * km
        if v < M + PX - 1:
            P.append(f'<line x1="{v:.1f}" y1="{M}" x2="{v:.1f}" y2="{M+PX}"/><line x1="{M}" y1="{v:.1f}" x2="{M+PX}" y2="{v:.1f}"/>')
    P.append("</g>")
    P.append(f'<rect x="{M}" y="{M}" width="{PX}" height="{PX}" fill="none" stroke="#2b2620" stroke-width="2"/>')
    P.append('<g font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" fill="#2b2620" text-anchor="middle">')
    for k in range(int(span/1000)):
        c = M + (k + 0.5) * km
        L = "ABCDEFGHIJKLMNOPQ"[k]
        P.append(f'<text x="{c:.1f}" y="{M-15}">{L}</text><text x="{c:.1f}" y="{M+PX+28}">{L}</text>'
                 f'<text x="{M-22}" y="{c+4:.1f}">{k+1}</text><text x="{M+PX+22}" y="{c+4:.1f}">{k+1}</text>')
    P.append("</g></svg>")
    path = os.path.join(OUT, f"{w}-overlay.svg")
    open(path, "w").write("\n".join(P))
    print(w, "relief range", round(float(zg.max())), "m;", len(cs), "contours;", len(st), "streams;",
          os.path.getsize(path) // 1024, "KB svg;", os.path.getsize(os.path.join(OUT, f"{w}-relief.webp")) // 1024, "KB relief")

build("north-america", "Zestafona", 20, 5, 3000)
