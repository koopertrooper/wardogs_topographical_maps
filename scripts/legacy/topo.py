"""Generate illustrative topographic map layers (PNG relief + SVG contours/overlay)."""
import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
N = 500            # grid cells across 16 km  (32 m)
KM = 16.0
PX = 1000          # map px
M = 44             # margin for grid labels
S = PX + 2 * M

def fbm(seed, n=N, octaves=6, base=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, tot = 1.0, 0
    for o in range(octaves):
        k = base * 2 ** o
        g = rng.standard_normal((k + 1, k + 1))
        up = zoom(g, n / (k + 1), order=3)[:n, :n]
        out += amp * up
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.mean()) / out.std()

yy, xx = np.mgrid[0:N, 0:N] * (KM / N)   # km, y down (north at top)

def km2px(v):
    return M + v / KM * PX

def path_d(seg, close=False):
    pts = seg
    if len(pts) > 6:
        pts = pts[::2] if len(pts) < 400 else pts[::3]
        pts = np.vstack([pts, seg[-1:]])
    d = "M" + " L".join(f"{km2px(x):.1f},{km2px(y):.1f}" for x, y in pts)
    return d + ("Z" if close else "")

def smooth_line(pts, n=300):
    pts = np.asarray(pts, float)
    t = np.linspace(0, 1, len(pts))
    tt = np.linspace(0, 1, n)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t, pts, axis=0)
    return cs(tt)

def descend(z, x, y, steps=900, h=0.03):
    """Steepest-descent trace (km coords) for creeks."""
    gy, gx = np.gradient(gaussian_filter(z, 3))
    pts = []
    for _ in range(steps):
        i, j = y / KM * N, x / KM * N
        if not (1 < i < N - 2 and 1 < j < N - 2):
            break
        dx = map_coordinates(gx, [[i], [j]], order=1)[0]
        dy = map_coordinates(gy, [[i], [j]], order=1)[0]
        g = np.hypot(dx, dy)
        if g < 1e-4:
            break
        x -= h * dx / g
        y -= h * dy / g
        pts.append((x, y))
    return np.array(pts)

def relief_png(z, veg, fields, path, paper=(0.965, 0.945, 0.89)):
    gy, gx = np.gradient(z, KM * 1000 / N)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy) * 1.6)
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    base = np.ones(z.shape + (3,)) * np.array(paper)
    green = np.array([0.80, 0.88, 0.72])
    field = np.array([0.95, 0.91, 0.76])
    base = base * (1 - veg[..., None]) + green * veg[..., None]
    base = base * (1 - fields[..., None]) + field * fields[..., None]
    img = base * (0.72 + 0.28 * shade[..., None] / shade.max())
    img = np.clip(img, 0, 1)
    img = zoom(img, (PX / N, PX / N, 1), order=1)
    plt.imsave(path, img)

def contours(z, interval, index_every):
    lv = np.arange(np.floor(z.min() / interval) * interval, z.max() + interval, interval)
    fig = plt.figure()
    cs = plt.contour(xx[0], yy[:, 0], gaussian_filter(z, 1.2), levels=lv)
    plt.close(fig)
    out = []
    for level, segs in zip(cs.levels, cs.allsegs):
        idx = int(round(level)) % (interval * index_every) == 0
        for s in segs:
            if len(s) < 6:
                continue
            out.append((level, idx, s))
    return out

def label_index(cs, z, every=7):
    """Place elevation labels on index contours."""
    labels = []
    k = 0
    for level, idx, s in cs:
        if not idx or len(s) < 80:
            continue
        k += 1
        if k % every:
            continue
        i = len(s) // 2
        x, y = s[i]
        if not (0.6 < x < KM - 0.6 and 0.6 < y < KM - 0.6):
            continue
        dx, dy = s[min(i + 3, len(s) - 1)] - s[max(i - 3, 0)]
        ang = np.degrees(np.arctan2(dy, dx))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        labels.append((km2px(x), km2px(y), ang, int(level)))
    return labels

def svg_doc(name, cs, labels, layers, interval):
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">')
    # frame
    p.append(f'<rect x="0" y="0" width="{S}" height="{S}" fill="#fbf8ef"/>')
    p.append(f'<rect x="{M}" y="{M}" width="{PX}" height="{PX}" fill="none" stroke="#2b2620" stroke-width="2"/>')
    # contours
    p.append('<g fill="none" stroke-linejoin="round" stroke-linecap="round">')
    for level, idx, s in cs:
        if idx:
            p.append(f'<path d="{path_d(s)}" stroke="#8a4f1c" stroke-width="1.15" stroke-opacity="0.95"/>')
        else:
            p.append(f'<path d="{path_d(s)}" stroke="#a8693a" stroke-width="0.5" stroke-opacity="0.7"/>')
    p.append('</g>')
    p.extend(layers)
    # contour labels
    for x, y, a, v in labels:
        p.append(f'<g transform="translate({x:.1f},{y:.1f}) rotate({a:.0f})">'
                 f'<rect x="-13" y="-6" width="26" height="12" fill="#f3eedf" fill-opacity="0.9"/>'
                 f'<text x="0" y="3.5" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#7a4317">{v}</text></g>')
    # grid
    p.append('<g stroke="#1f2a33" stroke-opacity="0.38" stroke-width="0.7">')
    for k in range(1, 16):
        v = M + k * PX / 16
        p.append(f'<line x1="{v:.1f}" y1="{M}" x2="{v:.1f}" y2="{M+PX}"/>')
        p.append(f'<line x1="{M}" y1="{v:.1f}" x2="{M+PX}" y2="{v:.1f}"/>')
    p.append('</g>')
    letters = "ABCDEFGHIJKLMNOP"
    p.append('<g font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" fill="#2b2620" text-anchor="middle">')
    for k in range(16):
        c = M + (k + 0.5) * PX / 16
        p.append(f'<text x="{c:.1f}" y="{M-15}">{letters[k]}</text>')
        p.append(f'<text x="{c:.1f}" y="{M+PX+28}">{letters[k]}</text>')
        p.append(f'<text x="{M-22}" y="{c+4:.1f}">{k+1}</text>')
        p.append(f'<text x="{M+PX+22}" y="{c+4:.1f}">{k+1}</text>')
    p.append('</g>')
    p.append('</svg>')
    with open(os.path.join(OUT, f"{name}-overlay.svg"), "w") as f:
        f.write("\n".join(p))

def poly(pts, **attrs):
    d = "M" + " L".join(f"{km2px(x):.1f},{km2px(y):.1f}" for x, y in pts)
    a = " ".join(f'{k.replace("_","-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" {a}/>'

def text(x, y, s, size=12, weight="normal", fill="#1f2a33", italic=False, anchor="middle", spacing=0, halo=True):
    st = ' font-style="italic"' if italic else ""
    common = f'x="{km2px(x):.1f}" y="{km2px(y):.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" letter-spacing="{spacing}"{st}'
    out = ""
    if halo:
        out += f'<text {common} fill="none" stroke="#fbf8ef" stroke-width="3.2" stroke-linejoin="round">{s}</text>'
    return out + f'<text {common} fill="{fill}">{s}</text>'

def buildings(rng, cx, cy, n, spread, along=None, size=(0.05, 0.11)):
    out = []
    for _ in range(n):
        if along is not None:
            t = rng.uniform(0, 1)
            px, py = along(t)
            px += rng.normal(0, spread); py += rng.normal(0, spread)
        else:
            px, py = cx + rng.normal(0, spread), cy + rng.normal(0, spread)
        w = rng.uniform(*size); h = w * rng.uniform(0.6, 1.4)
        a = rng.uniform(-20, 20)
        out.append(f'<rect x="{km2px(px):.1f}" y="{km2px(py):.1f}" width="{w/KM*PX:.1f}" height="{h/KM*PX:.1f}" transform="rotate({a:.0f} {km2px(px):.1f} {km2px(py):.1f})" fill="#2b2620"/>')
    return out

def spot(x, y, z, label=True):
    s = f'<path d="M{km2px(x)-4:.1f},{km2px(y)+3:.1f} L{km2px(x):.1f},{km2px(y)-4:.1f} L{km2px(x)+4:.1f},{km2px(y)+3:.1f}Z" fill="#2b2620"/>'
    if label:
        s += text(x + 0.12, y + 0.08, f"{int(z)}", size=10, anchor="start", fill="#2b2620")
    return s

def zat(z, x, y):
    return map_coordinates(z, [[y / KM * N], [x / KM * N]], order=1)[0]

def peaks(z, k, mind=1.8):
    zz = gaussian_filter(z, 2)
    from scipy.ndimage import maximum_filter
    mx = (zz == maximum_filter(zz, size=25))
    ii, jj = np.where(mx)
    order = np.argsort(-zz[ii, jj])
    pts = []
    for o in order:
        x, y = jj[o] * KM / N, ii[o] * KM / N
        if not (0.5 < x < KM - 0.5 and 0.5 < y < KM - 0.5):
            continue
        if all(np.hypot(x - a, y - b) > mind for a, b, _ in pts):
            pts.append((x, y, z[ii[o], jj[o]]))
        if len(pts) >= k:
            break
    return pts

# ---------------------------------------------------------------- BAKURANI
def bakurani():
    rng = np.random.default_rng(11)
    # valley centreline: enters SW, meanders to NE
    cl = smooth_line([(2.2, 16.2), (3.8, 13.2), (6.2, 11.4), (7.4, 9.0), (9.6, 7.6), (11.0, 5.0), (13.4, 3.4), (16.2, 1.6)], 800)
    from scipy.spatial import cKDTree
    tree = cKDTree(cl)
    d, idx = tree.query(np.c_[xx.ravel(), yy.ravel()])
    d = d.reshape(N, N); t = (idx / len(cl)).reshape(N, N)
    n1, n2, n3 = fbm(3), fbm(8, base=6), fbm(21, base=2)
    floor = 430 + 140 * t                      # river falls to SW
    ridge = 1150 * (1 - np.exp(-(d / 2.4) ** 2.2))
    z = floor + ridge * (1 + 0.28 * n3) + 120 * n1 * np.clip(d / 1.5, 0, 1) + 45 * n2 * np.clip(d / 0.8, 0, 1)
    z = gaussian_filter(z, 1.0)
    # vegetation: forest above valley floor
    veg = np.clip((d - 1.0 + 0.5 * n2) / 0.6, 0, 1) * np.clip((1650 - z) / 150, 0.35, 1)
    fields = np.zeros_like(z)
    # field parcels on valley floor (blocky)
    blocks = (np.floor(xx * 2.2 + 0.4 * n3) + np.floor(yy * 2.6)) % 3
    fields = np.where((d < 1.0) & (blocks == 0), 1.0, 0.0) * 0.9
    fields = gaussian_filter(fields, 0.6)
    relief_png(z, veg, fields, os.path.join(OUT, "bakurani-relief.png"))

    cs = contours(z, 20, 5)
    labels = label_index(cs, z, every=9)
    L = []
    # river
    riv = cl + np.c_[0.12 * np.sin(np.linspace(0, 70, len(cl))), 0.10 * np.cos(np.linspace(0, 55, len(cl)))]
    L.append(f'<path d="{path_d(riv)}" fill="none" stroke="#3f7fb3" stroke-width="3.2" stroke-linecap="round"/>')
    # creeks from high ground
    for (sx, sy) in [(4.5, 7.5), (8.2, 3.5), (12.5, 8.5), (5.0, 15.0), (14.5, 6.5), (10.6, 12.3), (2.2, 9.5), (11.8, 1.2)]:
        tr = descend(z, sx, sy)
        if len(tr) > 20:
            L.append(f'<path d="{path_d(tr)}" fill="none" stroke="#4b8cc0" stroke-width="1.1" stroke-linecap="round"/>')
    # valley road (offset from river)
    road = cl + np.c_[np.full(len(cl), -0.32), np.full(len(cl), -0.22)]
    L.append(f'<path d="{path_d(road)}" fill="none" stroke="#2b2620" stroke-width="4.2" stroke-linecap="round"/>')
    L.append(f'<path d="{path_d(road)}" fill="none" stroke="#e8a33a" stroke-width="2.4" stroke-linecap="round"/>')
    # switchback track up east ridge
    trk = smooth_line([(9.3, 7.3), (10.3, 8.4), (9.9, 9.2), (11.0, 9.9), (10.7, 10.8), (12.0, 11.6), (13.5, 11.9)], 200)
    L.append(f'<path d="{path_d(trk)}" fill="none" stroke="#2b2620" stroke-width="1.3" stroke-dasharray="6 3"/>')
    trk2 = smooth_line([(6.0, 11.0), (5.0, 10.3), (5.4, 9.4), (4.2, 8.6), (3.4, 7.2)], 200)
    L.append(f'<path d="{path_d(trk2)}" fill="none" stroke="#2b2620" stroke-width="1.3" stroke-dasharray="6 3"/>')
    # bridge
    bx, by = 7.6, 8.75
    L.append(f'<rect x="{km2px(bx)-9:.1f}" y="{km2px(by)-3:.1f}" width="18" height="6" fill="#fbf8ef" stroke="#2b2620" stroke-width="1.2" transform="rotate(-50 {km2px(bx):.1f} {km2px(by):.1f})"/>')
    # town on valley floor
    town = lambda tt: road[int(300 + tt * 140)]
    L += buildings(rng, 0, 0, 170, 0.18, along=town)
    L += buildings(rng, 3.9, 13.0, 22, 0.15)
    L += buildings(rng, 13.2, 3.6, 18, 0.14)
    # labels
    L.append(text(8.6, 9.3, "BAKURANI", size=17, weight="bold", spacing=2))
    L.append(text(5.2, 12.9, "Bakurani River", size=11, italic=True, fill="#2f6a9a"))
    L.append(text(12.7, 2.7, "Valley Road", size=10, italic=True))
    L.append(text(4.2, 3.0, "WEST RIDGE", size=13, weight="bold", spacing=3, fill="#5b3a1a"))
    L.append(text(13.0, 13.2, "EAST RIDGE", size=13, weight="bold", spacing=3, fill="#5b3a1a"))
    for x, y, h in peaks(z, 9):
        L.append(spot(x, y, h))
    svg_doc("bakurani", cs, labels, L, 20)
    print("bakurani", z.min(), z.max(), len(cs))

# ---------------------------------------------------------------- OZETI
def ozeti():
    rng = np.random.default_rng(5)
    from scipy.spatial import cKDTree
    # valley runs W -> E, gently curving; town strung along road in the middle
    cl = smooth_line([(-0.2, 9.4), (3.0, 9.0), (6.0, 9.6), (8.5, 8.8), (11.0, 9.3), (13.5, 8.4), (16.2, 8.9)], 800)
    tree = cKDTree(cl)
    d, idx = tree.query(np.c_[xx.ravel(), yy.ravel()])
    d = d.reshape(N, N); t = (idx / len(cl)).reshape(N, N)
    n1, n2, n3 = fbm(41, base=3), fbm(17, base=7), fbm(9, base=2)
    floor = 610 + 60 * (1 - t)
    hills = 520 * (1 - np.exp(-(d / 3.2) ** 1.8))
    z = floor + hills * (1 + 0.35 * n3) + 90 * n1 * np.clip(d / 1.2, 0, 1) + 22 * n2
    # church knoll north of town
    z += 95 * np.exp(-(((xx - 8.9) / 0.55) ** 2 + ((yy - 7.6) / 0.45) ** 2))
    # terracing: quantise slopes a little
    terr = np.round(z / 10) * 10
    z = np.where((d > 0.5) & (d < 3.5), 0.55 * z + 0.45 * terr, z)
    z = gaussian_filter(z, 0.7)
    # vegetation: scattered pine (sparse), orchards near town, dry fields
    veg = np.clip((n2 - 0.9) / 0.5, 0, 1) * 0.8
    blocks = (np.floor(xx * 3.1 + 0.6 * n3) * 7 + np.floor(yy * 3.4)) % 4
    fields = np.where((d < 2.6) & (blocks == 1), 1.0, 0.0)
    relief_png(z, veg, gaussian_filter(fields, 0.5), os.path.join(OUT, "ozeti-relief.png"), paper=(0.975, 0.95, 0.88))

    cs = contours(z, 10, 5)
    labels = label_index(cs, z, every=11)
    L = []
    # dry creek
    riv = cl + np.c_[np.zeros(len(cl)), 0.35 + 0.1 * np.sin(np.linspace(0, 40, len(cl)))]
    L.append(f'<path d="{path_d(riv)}" fill="none" stroke="#4b8cc0" stroke-width="1.6" stroke-dasharray="7 3"/>')
    for (sx, sy) in [(4.0, 4.5), (11.5, 3.6), (6.5, 14.0), (13.5, 13.5), (2.0, 13.8)]:
        tr = descend(z, sx, sy)
        if len(tr) > 20:
            L.append(f'<path d="{path_d(tr)}" fill="none" stroke="#4b8cc0" stroke-width="0.9" stroke-dasharray="4 2"/>')
    # orchards: dot grids
    dots = []
    for (ox, oy, w, h) in [(5.4, 10.4, 1.4, 0.9), (10.4, 10.0, 1.6, 1.0), (7.0, 7.0, 1.0, 0.8), (12.0, 7.0, 1.2, 0.9), (3.0, 10.2, 1.1, 0.8)]:
        for gx in np.arange(ox, ox + w, 0.09):
            for gy in np.arange(oy, oy + h, 0.09):
                dots.append(f'<circle cx="{km2px(gx):.1f}" cy="{km2px(gy):.1f}" r="1.3"/>')
    L.append('<g fill="#4f7a3a" fill-opacity="0.8">' + "".join(dots) + "</g>")
    # scattered pine
    pines = []
    for _ in range(420):
        px, py = rng.uniform(0.3, 15.7, 2)
        if zat(z, px, py) > 820 and rng.random() < 0.55:
            pines.append(f'<circle cx="{km2px(px):.1f}" cy="{km2px(py):.1f}" r="2"/>')
    L.append('<g fill="#3f6b33" fill-opacity="0.75">' + "".join(pines) + "</g>")
    # road through town
    L.append(f'<path d="{path_d(cl)}" fill="none" stroke="#2b2620" stroke-width="4.2" stroke-linecap="round"/>')
    L.append(f'<path d="{path_d(cl)}" fill="none" stroke="#e8a33a" stroke-width="2.4" stroke-linecap="round"/>')
    side = smooth_line([(8.7, 9.0), (9.0, 8.3), (8.9, 7.6)], 60)
    L.append(f'<path d="{path_d(side)}" fill="none" stroke="#2b2620" stroke-width="1.4"/>')
    trk = smooth_line([(11.0, 9.3), (11.6, 11.0), (12.6, 12.4), (14.6, 13.8)], 120)
    L.append(f'<path d="{path_d(trk)}" fill="none" stroke="#2b2620" stroke-width="1.3" stroke-dasharray="6 3"/>')
    trk = smooth_line([(4.5, 9.1), (4.2, 7.4), (3.0, 5.8), (2.4, 3.2)], 120)
    L.append(f'<path d="{path_d(trk)}" fill="none" stroke="#2b2620" stroke-width="1.3" stroke-dasharray="6 3"/>')
    # buildings strung along the road
    town = lambda tt: cl[int(330 + tt * 230)]
    L += buildings(rng, 0, 0, 150, 0.09, along=town, size=(0.04, 0.09))
    # walled stadium at west end
    sx, sy = 6.25, 9.95
    L.append(f'<rect x="{km2px(sx)-17:.1f}" y="{km2px(sy)-11:.1f}" width="34" height="22" rx="10" fill="#e9e3cf" stroke="#2b2620" stroke-width="2.2"/>')
    L.append(f'<rect x="{km2px(sx)-11:.1f}" y="{km2px(sy)-6:.1f}" width="22" height="12" rx="5" fill="#cfe0b8" stroke="#2b2620" stroke-width="0.8"/>')
    # church on the knoll
    cx, cy = 8.9, 7.6
    L.append(f'<rect x="{km2px(cx)-5:.1f}" y="{km2px(cy)-3.5:.1f}" width="10" height="7" fill="#2b2620"/>')
    L.append(f'<path d="M{km2px(cx):.1f},{km2px(cy)-12:.1f} V{km2px(cy)-4:.1f} M{km2px(cx)-3:.1f},{km2px(cy)-9:.1f} H{km2px(cx)+3:.1f}" stroke="#2b2620" stroke-width="1.6"/>')
    L.append(text(sx, sy + 0.55, "Stadium", size=10, italic=True))
    L.append(text(cx, cy - 0.45, "Church", size=10, italic=True))
    L.append(text(10.3, 9.9, "OZETI", size=17, weight="bold", spacing=2))
    L.append(text(2.0, 8.6, "Valley Road", size=10, italic=True))
    L.append(text(14.0, 9.85, "dry creek", size=10, italic=True, fill="#2f6a9a"))
    for x, y, h in peaks(z, 10, mind=2.0):
        L.append(spot(x, y, h))
    svg_doc("ozeti", cs, labels, L, 10)
    print("ozeti", z.min(), z.max(), len(cs))

os.makedirs(OUT, exist_ok=True)
bakurani()
ozeti()
