s=open('gen/cz3.py').read()
def rep(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b)
rep('<rect x="{M}" y="{M}" width="{PX}" height="{PX}" fill="#fff"/>','')
rep('<rect width="{S}" height="{S}" fill="#fbf8ef"/>','')
rep("""    cols = np.array([[201, 219, 178], [222, 228, 190], [236, 226, 192], [222, 202, 166]]) / 255
    stops = np.array([0, 0.3, 0.65, 1])
    rgb = np.stack([np.interp(rel, stops, cols[:, k]) for k in range(3)], -1)
    img = np.clip(rgb * (0.62 + 0.38 * sh[..., None] / sh.max()), 0, 1)""","""    rgb = RAMP(rel)
    img = np.clip(rgb * (0.70 + 0.30 * sh[..., None] / sh.max()), 0, 1)
    json.dump({"min": float(Z.min()), "max": float(Z.max()), "ramp": ["#%02x%02x%02x" % tuple(int(v * 255) for v in RAMP(np.array([q]))[0]) for q in np.linspace(0, 1, 12)]}, open(f"{OUT}/{w}-colramp.json", "w"))""")
rep('REV = json.load','''import matplotlib.cm as _cm
def RAMP(v):
    t = 0.04 + 0.92 * np.clip(v, 0, 1)
    c = matplotlib.colormaps["turbo"](t)[..., :3]
    return 0.92 * c + 0.08
REV = json.load''')
rep('stroke="#86491a" stroke-width="1.1"','stroke="#14142b" stroke-opacity="0.62" stroke-width="1.0"')
rep('stroke="#a8693a" stroke-width="{0.32 if interval == 1 else 0.45}" stroke-opacity="0.8"','stroke="#14142b" stroke-width="0.3" stroke-opacity="0.26"')
rep('fill="#f5cf3a" fill-opacity="0.55" stroke="#a87800" stroke-width="1.2"','fill="#ffffff" fill-opacity="0.28" stroke="#14142b" stroke-width="1.6" stroke-dasharray="6 4"')
rep("""        P.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in rpx) + '" fill="none" stroke="#2f6fb0\"""","""        P.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in rpx) + '" fill="none" stroke="#ffffff" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
        P.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in rpx) + '" fill="none" stroke="#0b3d91\"""")
rep('stroke="#3a332c" stroke-width="5.2"','stroke="#14142b" stroke-width="5.4"')
rep('stroke="#f3e3b8" stroke-width="3"','stroke="#ffffff" stroke-width="3"')
rep('stroke="#b3261e" stroke-width="1.8" stroke-dasharray="8 5"','stroke="#14142b" stroke-width="2" stroke-dasharray="8 5"')
rep('fill="#b3261e", anchor="middle"','fill="#14142b", anchor="middle"')
rep('"rev", True)','"col", True)')
open('gen/cz4.py','w').write(s)
