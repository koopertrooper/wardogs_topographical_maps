# Method

## 1. Inputs
From https://github.com/warpuppies/wardogs-data (game build 25937608):
- `data/maps/<world>/terrain.json`, `map.json` (bounds, z range), `buildings.json` (213-432 axis-aligned boxes per map with a `kind`), `objects.json`, `objectives.json` (Control Zone options, safe zones, towers)
- release assets: `heightmap.png` (16-bit, 8192 x 8192, about 2 m per pixel; `z = zMin + v / 65535 * (zMax - zMin)`) and the satellite tile pyramid `tiles/<z>/<x>/<y>.webp`
- WardogsHub (wardogshub.gg) place markers (names with normalised 0..1 coordinates), used only for landmark names/positions and, on Zestafona, tower positions

`scripts/prep.py` turns the heightmap into a numpy array and stitches the zoom-6 tiles into a 1 m/px mosaic (256 px tiles; tile pixel = world coordinate minus `map.json` bounds.min). The mosaic windows are in `data/mosaic_meta.json`.

## 2. Relief and contours
- Heightmap is lightly smoothed (Gaussian) and contoured with matplotlib at a per-sheet interval (20 / 10 m full maps, 2 / 1 / 0.5 m close-ups). Every Nth line is an index contour with a height label; spot heights are the local peaks.
- The relief image (`*-relief.webp`) is a hillshade (light from 315 degrees, 40 degrees altitude) over a pale green-to-sand height tint.
- Fine contours go to a separate `*-minor.svg` layer because of file size.
- Altitude-colour sheets map height through the matplotlib `turbo` ramp, stretched to the 1st-99th percentile of the window (Zestafona) or its min/max (others).
- **Zestafona nodata:** its heightmap stores 0 for nodata (a large border region plus isolated pixels). These are filled from the nearest valid pixel and the lowest *valid* height is used as the base (about 25 m), otherwise the colour scale is wrong.
- Window selection: the close-ups are centred on the Control Zone options (`objectives.json`) and fitted to the zone circles plus a margin. Grid squares are lettered from `map.json` bounds.min.

## 3. Image offset
The satellite imagery is shifted from game coordinates: about (32, 35) m on Bakurani, (38, 39) m on Ozeti and (0, 0) on Zestafona (`data/image_offset.json`). It was found by matching mosaic features to terrain/building features. Anything traced from the imagery (roads, river, pond, building angles) is converted back to game coordinates with this offset.

## 4. Roads
- Bakurani and Ozeti: a road-likeness map from the mosaic (low saturation, blue/red ratio, brightness window) is skeletonised and traced with least-cost paths between hand-placed waypoints (`roads_manual.py`, `data/wp_*.json`); `roads2.py` is the automatic tracer. The result was cleaned by hand (junk yards and loops removed), then `join.py` joins dead ends to the nearest road when the least-cost path between them runs over road surface.
- Zestafona (a regular grid of wide grey roads): `roads_zest.py` thresholds the mosaic, removes buildings and wide blobs, skeletonises, prunes spurs and short or tangled chains (it only keeps arterials and long smooth chains; the dense centre and rail yard are dropped as noise). `join_zest.py` then closes gaps: each dead end is extended toward the nearest road in its direction (within 55 degrees, up to 170 m) along the road surface, accepted only if the path is nearly straight.
- Final polylines are stored in game coordinates in `data/review_cfg.json` (`roads`) and drawn as a dark casing plus a cream line.

## 5. Landmarks, river, towers
- `data/review_cfg.json` per world: `boxed` landmarks (a name plus either a position or a building `kind`, optional `largest`/`inside`), traced `river`, `water`, `sunflower` fields and `outlines` (e.g. the stadium), and `roads`.
- Capture towers come from `objectives.json` on Bakurani and Ozeti (`data/towers.json`). On Zestafona the objectives are not available, so Tower 1-3 are the hub's named place markers: **approximate (about 50 m), not confirmed capture towers.**
- Zestafona landmark dots also come from hub markers (about 50 m accuracy); there is no building highlighting on that map yet.

## 6. Building rotation
`buildings.json` only holds axis-aligned boxes, so every building looks upright. `orient_all.py` / `orient.py` recover each building's real angle from the mosaic:
1. Sobel gradients of the greyscale mosaic inside the box (Gaussian-weighted towards the centre).
2. Dominant edge direction modulo 90 degrees using the 4-theta trick (`theta = atan2(sum m^2 sin 4phi, sum m^2 cos 4phi) / 4`), with a confidence value.
3. Smoothing over neighbours within 40 m (so garage rows and blocks share their orientation).
4. The rotated rectangle's width and height are solved so that its bounding box equals the original box; angles below 4 degrees or near 45 degrees (ill-conditioned) fall back to the upright box.
Result: `data/orient.json` (id -> [angle, confidence]); only buildings inside the mosaic windows are measured (the rest stay upright). `cz*.py`, `real.py`, `cz5*.py` draw them via `bpoly()`.

## 7. Assembling the sheets
Each sheet is a design board: relief image (1000 x 1000 inside a 44 px margin), the minor-contour SVG (multiply blend), then the overlay SVG, plus HTML legend, scale bar and title. See `sheets/*.dc.html`. The boards were published as a Design canvas artifact; `sheets/canvas.json` has their layout.

## Known limits
- Roads on Zestafona are auto-traced and incomplete in the dense centre; Zestafona landmarks and towers are positioned from hub markers.
- Building angles are estimated from roofs and shadows; a few large or odd roofs may be off by several degrees.
- Zestafona terrain is very flat (68 m of relief in the Control Zone window), hence 0.5 m contours.
- `cz2.py`, `topo.py`, `roads.py` in `scripts/legacy` are superseded earlier versions kept for reference.
- Scripts expect the working-folder layout from `scripts/prep.py` and `README.md`; some paths and the Zestafona mosaic origin are hard-coded.
