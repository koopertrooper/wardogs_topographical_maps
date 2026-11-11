import sys, json, heapq
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
SCR = sys.argv[1]; D = f"{SCR}/repo/wardogs-data/data/maps"
FR = {"kavkazi": (-8873, -11752, 1282), "europe": (-7291, -7020, 1816)}
out = json.load(open(f"{SCR}/landmarks_review.json"))
for w in ["kavkazi", "europe"]:
    t = json.load(open(f"{D}/{w}/terrain.json")); (tx, ty), (ex, ey) = t["bounds"]["min"], t["bounds"]["max"]
    zr = t["zRange"]; a = np.load(f"{SCR}/{w}.npy").astype(np.float32)
    n = 2048; z = a.reshape(n, 4, n, 4).mean(axis=(1, 3)); z = zr["min"] + z / 65535 * (zr["max"] - zr["min"])
    z = gaussian_filter(z, 1.5).astype(np.float64); cm = (ex - tx) / n
    done = np.zeros(z.shape, bool); h = []
    for r in range(n):
        for c in (0, n - 1): heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
    for c in range(1, n - 1):
        for r in (0, n - 1): heapq.heappush(h, (z[r, c], r, c)); done[r, c] = True
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
    rec = rec.ravel(); acc = np.ones(n * n)
    for i in np.argsort(-z.ravel(), kind="stable"):
        j = rec[i]
        if j != i: acc[j] += acc[i]
    x0, y0, W = FR[w]
    cs, ce = int((x0 - tx) / cm), int((x0 + W - tx) / cm); rs, re_ = int((y0 - ty) / cm), int((y0 + W - ty) / cm)
    A = acc.reshape(n, n); sub = A[rs:re_, cs:ce]
    rr, cc = np.unravel_index(np.argmax(sub), sub.shape); outlet_acc = sub.max()
    # the main river: from the outlet, walk upstream along the donor with the largest accumulation
    donors = {}
    on = np.nonzero(acc > outlet_acc * 0.02)[0]
    for i in on:
        j = rec[i]
        if j != i: donors.setdefault(j, []).append(i)
    cur = (rs + rr) * n + (cs + cc); path = [cur]
    while cur in donors:
        nxt = max(donors[cur], key=lambda k: acc[k])
        if acc[nxt] < outlet_acc * 0.25: break
        r_, c_ = divmod(nxt, n)
        if not (rs - 40 <= r_ < re_ + 40 and cs - 40 <= c_ < ce + 40): path.append(nxt); break
        path.append(nxt); cur = nxt
    # also extend downstream past the outlet to the frame edge
    cur = path[0]; down = []
    for _ in range(400):
        j = rec[cur]
        if j == cur: break
        down.append(j); cur = j
        r_, c_ = divmod(j, n)
        if not (rs - 40 <= r_ < re_ + 40 and cs - 40 <= c_ < ce + 40): break
    full = list(reversed(path)) + down
    pts = np.array([[tx + (p % n + .5) * cm, ty + (p // n + .5) * cm] for p in full])
    if len(pts) > 7:
        pts[1:-1] = np.c_[uniform_filter1d(pts[:, 0], 5)[1:-1], uniform_filter1d(pts[:, 1], 5)[1:-1]]
    out[w]["river"] = pts.tolist()
    print(w, "river pts", len(pts), "catchment at outlet km2", round(outlet_acc * cm * cm / 1e6, 1))
json.dump(out, open(f"{SCR}/landmarks_review.json", "w"))
