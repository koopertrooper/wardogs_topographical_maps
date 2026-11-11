"""Render the sheet boards (sheets/*.dc.html) to PNG with Playwright, using the generated images in output/.
usage: python export_png.py <repo-root> <fonts-dir> [scale]   (fonts-dir: node_modules/@fontsource with barlow + barlow-condensed)"""
import sys, os, re, glob
from playwright.sync_api import sync_playwright
R, F = sys.argv[1], sys.argv[2]; SC = float(sys.argv[3]) if len(sys.argv) > 3 else 2
O = lambda p: os.path.join(R, "output", p)
def files(w, kind, zest=None):
    if w == "north-america":
        if kind == "full": return [O("real/north-america-relief.webp"), O("real/north-america-overlay.svg")]
        t = {"1m": "czzst", "rev": "czzst", "col": "czzstc"}[kind]
        ov = "czzstplain" if kind == "1m" else t
        return [O(f"cz/north-america-{t}-relief.webp"), O(f"cz/north-america-{t}-minor.svg"), O(f"cz/north-america-{ov}-overlay.svg")]
    if kind == "full": return [O(f"real/{w}-relief.webp"), O(f"real/{w}-overlay.svg")]
    if kind == "cz": return [O(f"cz/{w}-cz-relief.webp"), O(f"cz/{w}-cz-overlay.svg")]
    if kind == "1m": return [O(f"cz/{w}-cz-relief.webp"), O(f"cz/{w}-cz1m-minor.svg"), O(f"cz/{w}-cz1m-overlay.svg")]
    t = {"rev": "czrev", "col": "czcol"}[kind]
    return [O(f"cz/{w}-{t}-relief.webp"), O(f"cz/{w}-{t}-minor.svg"), O(f"cz/{w}-{t}-overlay.svg")]
BOARDS = [  # board file, world, kind, output name
 ("Main", "kavkazi", "full", "Bakurani_full-map"), ("Ozeti", "europe", "full", "Ozeti_full-map"), ("Zestafona", "north-america", "full", "Zestafona_full-map"),
 ("Bakurani-CZ", "kavkazi", "cz", "Bakurani_1A_control-zone-2m"), ("Ozeti-CZ", "europe", "cz", "Ozeti_2A_control-zone-2m"),
 ("Bakurani-CZ-1m", "kavkazi", "1m", "Bakurani_1B_contours-1m"), ("Ozeti-CZ-1m", "europe", "1m", "Ozeti_2B_contours-1m"), ("Zestafona-CZ-1m", "north-america", "1m", "Zestafona_3B_contours-0.5m"),
 ("Bakurani-CZ-Landmarks", "kavkazi", "rev", "Bakurani_1C_landmarks"), ("Ozeti-CZ-Landmarks", "europe", "rev", "Ozeti_2C_landmarks"), ("Zestafona-CZ-Landmarks", "north-america", "rev", "Zestafona_3C_landmarks"),
 ("Bakurani-CZ-Color", "kavkazi", "col", "Bakurani_1D_altitude-colour"), ("Ozeti-CZ-Color", "europe", "col", "Ozeti_2D_altitude-colour"), ("Zestafona-CZ-Color", "north-america", "col", "Zestafona_3D_altitude-colour"),
]
ff = lambda fam, wt, f: f"@font-face{{font-family:'{fam}';font-weight:{wt};src:url('file://{F}/{f}/files/{f}-latin-{wt}-normal.woff2')}}"
FONTS = ff("Barlow", 400, "barlow") + ff("Barlow", 600, "barlow") + ff("Barlow Condensed", 500, "barlow-condensed") + ff("Barlow Condensed", 700, "barlow-condensed")
TMP = os.path.join(R, ".exp_tmp"); os.makedirs(TMP, exist_ok=True)
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 1168, "height": 1400}, device_scale_factor=SC)
    for board, w, kind, name in BOARDS:
        h = open(os.path.join(R, "sheets", board + ".dc.html")).read()
        h = re.sub(r"<script.*?</script>", "", h, flags=re.S); h = re.sub(r'<link href="https://fonts[^>]*>|<link rel="preconnect"[^>]*>', "", h)
        h = h.replace("</style>", FONTS + "</style>", 1)
        fl = files(w, kind); imgs = re.findall(r'<img src="/_blob/[0-9a-f]+"', h); n = len(imgs)
        seq = [fl[0]] + fl[1:-1][:1] + [None] * (n - len(fl)) + [fl[-1]] if n >= len(fl) else fl
        pad = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        it = iter(seq)
        h = re.sub(r'<img src="/_blob/[0-9a-f]+"', lambda m: (lambda f: '<img src="' + (("file://" + f) if f else pad) + '"')(next(it)), h)
        p = os.path.join(TMP, board + ".html"); open(p, "w").write(h)
        pg.goto("file://" + p); pg.wait_for_timeout(800); pg.evaluate("document.fonts.ready")
        out = os.path.join(R, "exports", name + ".png")
        pg.locator("body > x-dc > div, x-dc > div").first.screenshot(path=out)
        print(name, os.path.getsize(out) // 1024, "KB", "imgs", n, "files", len(fl))
    b.close()
