"""FillFinder v2 (proof of concept)

Give it ANY embroidery file (DST/PES/EXP/JEF/...):
  * lines-only design  -> uses all lines as outlines
  * finished design    -> auto-detects which parts are FILLS and which are OUTLINES,
                          throws the fills away, keeps the outlines, then re-fills
                          every enclosed area automatically

Every run is saved in its own folder (nothing is overwritten):
  runs/<design>_<YYYYmmdd_HHMMSS>/
    input.<ext>              copy of the original file
    01_original.svg          render of the original design
    02_outlines_used.svg     only the stitches treated as outlines
    03_regions.svg           numbered detected regions (preview)
    04_autofilled.svg        render of the auto-filled design
    regions_for_wilcom.svg   clean filled shapes, same mm coordinates, for import
    autofilled.dst / .pes    the new stitch file (PES keeps colours)
    regions.csv              id, area, centre, fill stitches per region
    summary.json             settings and numbers for this run
    report.html              one-page before/after report for the client

Only needs pyembroidery, shapely, numpy (no matplotlib/Pillow, so it runs on PCs
where Windows Application Control blocks those DLLs). Python 3.8+.

Usage:
  python fillfinder.py design.dst
  python fillfinder.py design.dst --exclude 3,8 --angle 30 --spacing 0.45
"""
import argparse
import csv
import json
import math
import os
import shutil
import time
import colorsys
from datetime import datetime

import numpy as np
import pyembroidery as pe
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

UNIT = 10.0  # pyembroidery: 0.1 mm per unit


# ====================================================================== reading
def read_design(path):
    """Return pattern and a list of colour blocks; each block is a list of
    stitch paths, each path a list of (x_mm, y_mm). Paths split at jumps/trims."""
    pattern = pe.read(path)
    if pattern is None:
        raise SystemExit("Could not read " + path)
    blocks, paths, cur = [], [], []

    def flush():
        if len(cur) >= 2:
            paths.append(list(cur))

    for x, y, cmd in pattern.stitches:
        c = cmd & pe.COMMAND_MASK
        pt = (x / UNIT, y / UNIT)
        if c == pe.STITCH:
            if not cur or cur[-1] != pt:
                cur.append(pt)
            continue
        flush()
        cur = [pt] if c == pe.JUMP else []
        if c == pe.COLOR_CHANGE:
            blocks.append(paths)
            paths = []
        if c == pe.END:
            break
    flush()
    blocks.append(paths)
    colors = []
    for i in range(len(blocks)):
        try:
            t = pattern.threadlist[i]
            colors.append("#%06x" % (t.color & 0xFFFFFF))
        except (IndexError, AttributeError):
            colors.append(None)
    return pattern, blocks, colors


# ============================================================ raster helpers (numpy only)
def _filter1d(a, k, axis, fn):
    if k <= 1:
        return a
    r = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="constant", constant_values=(fn is np.minimum) * 1)
    out = p.take(range(0, a.shape[axis]), axis=axis).copy()
    for s in range(1, 2 * r + 1):
        out = fn(out, p.take(range(s, s + a.shape[axis]), axis=axis))
    return out


def dilate(a, k):
    return _filter1d(_filter1d(a, k, 0, np.maximum), k, 1, np.maximum)


def erode(a, k):
    return _filter1d(_filter1d(a, k, 0, np.minimum), k, 1, np.minimum)


class Raster:
    def __init__(self, blocks, res):
        pts = [p for b in blocks for path in b for p in path]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.res = res
        self.x0, self.y0 = min(xs) - 5, min(ys) - 5
        self.w = int((max(xs) + 5 - self.x0) / res) + 1
        self.h = int((max(ys) + 5 - self.y0) / res) + 1

    def draw(self, paths):
        m = np.zeros((self.h, self.w), dtype=np.uint8)
        for path in paths:
            a = np.asarray(path)
            p0, p1 = a[:-1], a[1:]
            n = np.maximum(1, np.ceil(np.hypot(*(p1 - p0).T) / (self.res * 0.5))).astype(int)
            t = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
            s = np.repeat(p0, n + 1, axis=0) + np.repeat(p1 - p0, n + 1, axis=0) * t[:, None]
            ix = ((s[:, 0] - self.x0) / self.res).astype(int)
            iy = ((s[:, 1] - self.y0) / self.res).astype(int)
            m[iy, ix] = 1
        return m

    def lookup(self, mask, x, y):
        ix = int((x - self.x0) / self.res)
        iy = int((y - self.y0) / self.res)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            return mask[iy, ix]
        return 0


# ============================================================ outline / fill detection
def _split_by_mask(paths, R, mask):
    """Keep only path segments whose midpoint is NOT on the mask."""
    kept = []
    for path in paths:
        seg = []
        for a, b in zip(path, path[1:]):
            if R.lookup(mask, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2):
                if len(seg) >= 2:
                    kept.append(seg)
                seg = []
            else:
                if not seg:
                    seg = [a]
                seg.append(b)
        if len(seg) >= 2:
            kept.append(seg)
    return kept


def classify_blocks(blocks, thick_mm=3.5, res=None):
    """Pass 1, per colour block: areas wider than `thick_mm` are fills. A block that
    is mostly thick is a fill block; otherwise its thin segments are outline candidates.
    Pass 2: candidate segments lying DEEP inside any single fill (travel runs, texture/hatch lines on top of fills) are dropped, while
    segments on the fill EDGE (real borders) are kept. For a lines-only design there
    are no fills, so every line is kept."""
    if sum(len(p) for b in blocks for p in b) == 0:
        raise SystemExit("No stitches found.")
    tmp = Raster(blocks, 1.0)
    if res is None:  # keep raster around 2000 px on the long side
        res = max(0.2, max(tmp.w, tmp.h) / 2000.0)
    R = Raster(blocks, res)
    close_k = max(3, int(round(1.0 / res)) | 1)  # bridges tatami rows ~1 mm apart
    er_k = int(round(thick_mm / res)) | 1        # erosion window = thickness
    edge_k = int(round(2.4 / res)) | 1
    deep = np.zeros((R.h, R.w), dtype=np.uint8)
    fill_any = np.zeros((R.h, R.w), dtype=np.uint8)
    cands, info = [], []
    for bi, paths in enumerate(blocks):
        if not paths:
            info.append({"block": bi, "stitches": 0, "fill_share": 0, "role": "empty"})
            continue
        m = R.draw(paths)
        closed = erode(dilate(m, close_k), close_k)
        thick = dilate(erode(closed, er_k), er_k + 2)
        share = min(1.0, float((thick & closed).sum()) / max(1, closed.sum()))
        n = sum(len(p) for p in paths)
        if share > 0.45:
            role = "fill"
            deep |= erode(closed, edge_k)
            fill_any |= closed
        else:
            fill_any |= thick & closed
            deep |= erode(thick & closed, edge_k)
            cands.append((bi, _split_by_mask(paths, R, thick)))
            role = "outline" if share < 0.05 else "mixed"
        info.append({"block": bi, "stitches": n, "fill_share": round(share, 3), "role": role})
    # pass 2: drop candidates buried deep inside ONE fill (each fill checked on its
    # own, so a border between two different fills is kept)
    outlines = []
    for bi, paths in cands:
        kept = _split_by_mask(paths, R, deep)
        before = sum(len(p) for p in paths)
        after = sum(len(p) for p in kept)
        if before and after < 0.5 * before and info[bi]["role"] == "outline":
            info[bi]["role"] = "texture"  # mostly sits on top of fills
        outlines.extend(kept)
    outlines = [p for p in outlines if len(p) >= 2]
    return outlines, info, (R, fill_any)


# ====================================================================== regions
def as_polys(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def find_regions(paths, gap=0.6, overlap=0.3, min_area=4.0, max_area_frac=0.15, simplify=0.05):
    """Thicken lines into walls (radius gap/2: closes small gaps, solidifies zigzag),
    take every enclosed hole as a region, grow it back under the outline."""
    r = gap / 2.0
    walls = unary_union([LineString(p).buffer(r, resolution=6) for p in paths if len(p) >= 2])
    hull_area = walls.convex_hull.area
    regions, skipped = [], 0
    for poly in as_polys(walls):
        for ring in poly.interiors:
            for piece in as_polys(Polygon(ring).difference(walls)):
                if piece.area < min_area:
                    continue
                if piece.area > max_area_frac * hull_area:
                    skipped += 1  # background-like (e.g. neckline opening)
                    continue
                g = piece.buffer(r + overlap, join_style=1).simplify(simplify)
                regions.extend(as_polys(g))
    regions.sort(key=lambda g: (round(g.centroid.y / 8), g.centroid.x))
    return regions, skipped


def fill_coverage(region, R, fill_any, step=1.0):
    """Share of the region that was stitched with fill in the ORIGINAL design."""
    from shapely import contains_xy
    minx, miny, maxx, maxy = region.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    xs, ys = xs.ravel(), ys.ravel()
    inside = contains_xy(region, xs, ys)
    if not inside.any():
        return 0.0
    xs, ys = xs[inside], ys[inside]
    ix = np.clip(((xs - R.x0) / R.res).astype(int), 0, R.w - 1)
    iy = np.clip(((ys - R.y0) / R.res).astype(int), 0, R.h - 1)
    return float(fill_any[iy, ix].mean())


# ====================================================================== stitching
def tatami_rows(region, angle=45.0, spacing=0.4, stitch_len=3.5):
    """Basic tatami: parallel rows clipped to the region, alternating direction,
    staggered needle points. No underlay or pull compensation (demo quality)."""
    c = region.centroid
    rot = affinity.rotate(region, -angle, origin=c)
    minx, miny, maxx, maxy = rot.bounds
    rows, flip, y, offset = [], False, miny + spacing / 2, 0.0
    while y < maxy:
        seg = rot.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
        parts = [seg] if seg.geom_type == "LineString" else list(getattr(seg, "geoms", []))
        parts = [p for p in parts if p.geom_type == "LineString" and p.length > 0.3]
        parts.sort(key=lambda p: min(p.coords[0][0], p.coords[-1][0]), reverse=flip)
        for p in parts:
            xa, xb = sorted([p.coords[0][0], p.coords[-1][0]])
            x0, x1 = (xb, xa) if flip else (xa, xb)
            n = max(1, int(abs(x1 - x0) / stitch_len))
            pts = [(x0 + (x1 - x0) * min(1.0, (k + (offset if 0 < k < n else 0)) / n), y) for k in range(n + 1)]
            rows.append(list(affinity.rotate(LineString(pts), angle, origin=c).coords))
        offset = (offset + 0.33) % 1.0
        flip = not flip
        y += spacing
    return rows


def build_filled_pattern(outline_paths, regions, colors, angle, spacing):
    """Fills first (one colour per region), outlines last on top."""
    p = pe.EmbPattern()
    fill_counts = []
    for i, g in enumerate(regions):
        p.add_thread({"color": colors[i]})
        if i > 0:
            p.add_command(pe.COLOR_CHANGE)
        last, count = None, 0
        for row in tatami_rows(g, angle=angle, spacing=spacing):
            for j, (x, y) in enumerate(row):
                if j == 0 and (last is None or math.hypot(x - last[0], y - last[1]) > 2.0):
                    p.add_command(pe.TRIM)
                    p.add_stitch_absolute(pe.JUMP, x * UNIT, y * UNIT)
                p.add_stitch_absolute(pe.STITCH, x * UNIT, y * UNIT)
                last = (x, y)
                count += 1
        fill_counts.append(count)
    p.add_thread({"color": "#1a1a8c"})
    if regions:
        p.add_command(pe.COLOR_CHANGE)
    for path in outline_paths:
        p.add_command(pe.TRIM)
        p.add_stitch_absolute(pe.JUMP, path[0][0] * UNIT, path[0][1] * UNIT)
        for x, y in path:
            p.add_stitch_absolute(pe.STITCH, x * UNIT, y * UNIT)
    p.add_command(pe.END)
    return p, fill_counts


# ====================================================================== SVG output
def palette(n):
    return ["#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.5, 0.92))
            for i in range(n)]


def ring_d(coords):
    return "M " + " L ".join("%.2f,%.2f" % (x, y) for x, y in coords) + " Z"


def poly_d(g):
    return ring_d(g.exterior.coords) + "".join(" " + ring_d(r.coords) for r in g.interiors)


def svg_open(bounds, dark=False):
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    bg = '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>' % (
        minx, miny, w, h, "#15151c" if dark else "#ffffff")
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="%.2fmm" height="%.2fmm" '
            'viewBox="%.2f %.2f %.2f %.2f">' % (w, h, minx, miny, w, h), bg]


def polylines(paths, color, width):
    o = ['<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round">' % (color, width)]
    for p in paths:
        o.append('<polyline points="%s"/>' % " ".join("%.2f,%.2f" % q for q in p))
    o.append("</g>")
    return o


def render_blocks_svg(out, blocks, colors, bounds):
    o = svg_open(bounds, dark=True)
    pal = palette(len(blocks))
    for i, paths in enumerate(blocks):
        o += polylines(paths, colors[i] or pal[i], 0.18)
    o.append("</svg>")
    _write(out, o)


def render_outlines_svg(out, outlines, bounds):
    o = svg_open(bounds) + polylines(outlines, "#1a1a8c", 0.35) + ["</svg>"]
    _write(out, o)


def regions_svg(out, outlines, regions, colors, bounds, labels=True, lines=True, marks=True, background=True):
    o = svg_open(bounds) if background else svg_open(bounds)[:1]
    o.append('<g id="regions">')
    for i, g in enumerate(regions, 1):
        o.append('<path id="region_%d" d="%s" fill="%s" fill-rule="evenodd"/>' % (i, poly_d(g), colors[i - 1]))
    o.append("</g>")
    if lines:
        o += polylines(outlines, "#1a1a8c", 0.3)
    if labels:
        fs = max(1.5, min(4.0, (bounds[2] - bounds[0]) / 90))
        o.append('<g font-family="Arial" font-size="%.2f" text-anchor="middle" fill="#000">' % fs)
        for i, g in enumerate(regions, 1):
            pt = g.representative_point()
            o.append('<text x="%.2f" y="%.2f" dominant-baseline="middle">%d</text>' % (pt.x, pt.y, i))
        o.append("</g>")
    if marks:
        o.append('<g stroke="#e00000" stroke-width="0.2"><line x1="-3" y1="0" x2="3" y2="0"/>'
                 '<line x1="0" y1="-3" x2="0" y2="3"/></g>')
    o.append("</svg>")
    _write(out, o)


def render_pattern_svg(out, pattern, colors_by_block, bounds):
    """Render a stitch pattern (our generated one) as coloured thread lines."""
    blocks, paths, cur = [], [], []
    for x, y, cmd in pattern.stitches:
        c = cmd & pe.COMMAND_MASK
        if c == pe.STITCH:
            cur.append((x / UNIT, y / UNIT))
            continue
        if len(cur) >= 2:
            paths.append(cur)
        cur = [(x / UNIT, y / UNIT)] if c == pe.JUMP else []
        if c == pe.COLOR_CHANGE:
            blocks.append(paths)
            paths = []
    if len(cur) >= 2:
        paths.append(cur)
    blocks.append(paths)
    o = svg_open(bounds, dark=True)
    for i, ps in enumerate(blocks):
        col = colors_by_block[i] if i < len(colors_by_block) else "#4a6cff"
        o += polylines(ps, col, 0.2)
    o.append("</svg>")
    _write(out, o)


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ====================================================================== report
def write_report(out, s):
    rows = "".join(
        "<tr><td>%d</td><td>%s</td><td>%d</td><td>%.0f%%</td></tr>" % (
            b["block"] + 1, b["role"], b["stitches"], b["fill_share"] * 100)
        for b in s["blocks"])
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FillFinder report: {name}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1d1d24}}
header{{background:#1d1d24;color:#fff;padding:20px 28px}}
header h1{{margin:0;font-size:22px}} header p{{margin:4px 0 0;opacity:.75}}
main{{padding:24px 28px;max-width:1300px;margin:auto}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:24px}}
.stat{{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.stat b{{display:block;font-size:26px}} .stat span{{font-size:13px;opacity:.7}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}}
figure{{margin:0;background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
figure img{{width:100%;height:auto;display:block;border-radius:6px;background:#fff}}
figcaption{{font-weight:600;margin-top:8px}} figcaption small{{display:block;font-weight:400;opacity:.7}}
table{{border-collapse:collapse;background:#fff;margin-top:24px;width:100%;max-width:520px}}
td,th{{border-bottom:1px solid #e3e4e8;padding:6px 10px;text-align:left;font-size:14px}}
</style></head><body>
<header><h1>Automatic fill: {name}</h1>
<p>Generated {when} in {secs:.1f} s &middot; {mode} &middot; FillFinder proof of concept</p></header>
<main>
<div class="stats">
<div class="stat"><b>{regions}</b><span>areas found and filled automatically</span></div>
<div class="stat"><b>{outline_paths}</b><span>outline paths used</span></div>
<div class="stat"><b>{fill_stitches:,}</b><span>fill stitches generated</span></div>
<div class="stat"><b>{size}</b><span>design size (mm)</span></div>
</div>
<div class="grid">
<figure><img src="01_original.svg"><figcaption>1. Original design<small>{in_stitches:,} stitches, {nblocks} colour blocks</small></figcaption></figure>
<figure><img src="02_outlines_used.svg"><figcaption>2. Outlines only<small>fills removed automatically, only borders kept</small></figcaption></figure>
<figure><img src="03_regions.svg"><figcaption>3. Areas detected<small>every closed area, including ones closed only by crossing lines</small></figcaption></figure>
<figure><img src="04_autofilled.svg"><figcaption>4. Auto-filled result<small>tatami fill, angle {angle}&deg;, spacing {spacing} mm</small></figcaption></figure>
</div>
<table><tr><th>Colour block</th><th>Treated as</th><th>Stitches</th><th>Fill share</th></tr>{rows}</table>
<p style="font-size:13px;opacity:.7;margin-top:18px">Files in this folder: autofilled.dst / autofilled.pes (new stitch file),
regions_for_wilcom.svg (filled shapes for import into Wilcom at the same position), regions.csv.
Fill stitches are demo quality (no underlay or pull compensation); final stitching is meant to be done by Wilcom from the imported shapes.</p>
</main></body></html>""".format(rows=rows, **s)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description="Auto-detect and fill enclosed areas in any embroidery file.")
    ap.add_argument("design", help="DST/PES/EXP/JEF/... file")
    ap.add_argument("--blocks", default=None, help="force these colour blocks as outlines, e.g. 9,17 (1-based)")
    ap.add_argument("--exclude", default=None, help="region numbers NOT to fill, e.g. 3,8 (from 03_regions.svg)")
    ap.add_argument("--fill-all", action="store_true",
                    help="finished designs: also fill areas that were empty in the original")
    ap.add_argument("--thick", type=float, default=3.5, help="anything wider than this (mm) counts as a fill")
    ap.add_argument("--gap", type=float, default=0.6, help="close openings up to this size (mm)")
    ap.add_argument("--overlap", type=float, default=0.3, help="fill tucks under outline by this (mm)")
    ap.add_argument("--min-area", type=float, default=4.0, help="ignore areas smaller than this (mm2)")
    ap.add_argument("--max-area", type=float, default=0.15, help="skip areas bigger than this share of the design")
    ap.add_argument("--angle", type=float, default=45.0, help="fill angle (degrees)")
    ap.add_argument("--spacing", type=float, default=0.4, help="fill row spacing (mm)")
    ap.add_argument("--runs", default=None, help="folder for run outputs (default: runs next to this script)")
    a = ap.parse_args()

    t0 = time.time()
    name = os.path.splitext(os.path.basename(a.design))[0]
    runs = a.runs or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    outdir = os.path.join(runs, "%s_%s" % (name, datetime.now().strftime("%Y%m%d_%H%M%S")))
    base_dir, k = outdir, 2
    while os.path.exists(outdir):  # never overwrite an earlier run
        outdir = "%s_%d" % (base_dir, k)
        k += 1
    os.makedirs(outdir)
    ext = os.path.splitext(a.design)[1].lower() or ".dst"
    shutil.copy2(a.design, os.path.join(outdir, "input" + ext))
    J = lambda f: os.path.join(outdir, f)

    print("[1/5] Reading", a.design)
    pattern, blocks, colors = read_design(a.design)
    n_in = sum(len(p) for b in blocks for p in b)
    print("      %d stitches in %d colour blocks" % (n_in, len(blocks)))

    print("[2/5] Separating outlines from fills")
    if a.blocks:
        keep = set(int(b) - 1 for b in a.blocks.split(","))
        outlines = [p for i, b in enumerate(blocks) if i in keep for p in b]
        fillinfo = None
        info = [{"block": i, "stitches": sum(len(p) for p in b), "fill_share": 0,
                 "role": "outline (forced)" if i in keep else "ignored"} for i, b in enumerate(blocks)]
    else:
        outlines, info, fillinfo = classify_blocks(blocks, thick_mm=a.thick)
        if not any(b["role"] == "fill" for b in info):
            # lines-only design: use every line as-is, nothing to compare against
            fillinfo = None
            outlines = [p for b in blocks for p in b]
            for b in info:
                b["role"] = "outline" if b["stitches"] else "empty"
    for b in info:
        print("      block %2d: %-8s %6d stitches  fill share %3.0f%%" % (
            b["block"] + 1, b["role"], b["stitches"], b["fill_share"] * 100))
    if not outlines:
        raise SystemExit("No outline stitches found. Try --thick 5 or --blocks.")

    print("[3/5] Finding enclosed areas")
    regions, skipped = find_regions(outlines, a.gap, a.overlap, a.min_area, a.max_area)
    print("      %d closed areas found (%d background-size areas skipped)" % (len(regions), skipped))
    empty_skipped = 0
    if fillinfo is not None and not a.fill_all:
        # finished design: only fill areas that were filled in the original
        R, fill_any = fillinfo
        keep = [g for g in regions if fill_coverage(g, R, fill_any) >= 0.5]
        empty_skipped = len(regions) - len(keep)
        regions = keep
        print("      %d areas were empty in the original and are left empty (use --fill-all to fill them)"
              % empty_skipped)
    if a.exclude:
        drop = set(int(x) for x in a.exclude.split(","))
        regions = [g for i, g in enumerate(regions, 1) if i not in drop]
    print("      %d areas will be filled" % len(regions))

    pts = [q for b in blocks for p in b for q in p]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    bounds = (min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5)
    cols = palette(len(regions))

    print("[4/5] Generating fill stitches")
    filled, counts = build_filled_pattern(outlines, regions, cols, a.angle, a.spacing)
    pe.write_dst(filled, J("autofilled.dst"))
    pe.write_pes(filled, J("autofilled.pes"))

    print("[5/5] Writing previews and report")
    render_blocks_svg(J("01_original.svg"), blocks, colors, bounds)
    render_outlines_svg(J("02_outlines_used.svg"), outlines, bounds)
    regions_svg(J("03_regions.svg"), outlines, regions, cols, bounds)
    render_pattern_svg(J("04_autofilled.svg"), filled, cols + ["#39d0c8"], bounds)
    regions_svg(J("regions_for_wilcom.svg"), outlines, regions, cols, bounds,
                labels=False, lines=False, marks=True, background=False)
    with open(J("regions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "area_mm2", "centre_x_mm", "centre_y_mm", "fill_stitches"])
        for i, (g, n) in enumerate(zip(regions, counts), 1):
            c = g.centroid
            w.writerow([i, round(g.area, 1), round(c.x, 2), round(c.y, 2), n])

    summary = {
        "name": name, "when": datetime.now().strftime("%d %b %Y %H:%M"),
        "secs": time.time() - t0, "regions": len(regions), "outline_paths": len(outlines),
        "fill_stitches": sum(counts), "empty_skipped": empty_skipped,
        "mode": "finished design (fills removed and rebuilt)" if fillinfo is not None else "lines-only design", "in_stitches": n_in, "nblocks": len(blocks),
        "size": "%.0f x %.0f" % (max(xs) - min(xs), max(ys) - min(ys)),
        "angle": a.angle, "spacing": a.spacing, "blocks": info,
        "settings": {k: v for k, v in vars(a).items()},
    }
    with open(J("summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_report(J("report.html"), summary)
    print("\nDone in %.1f s. All outputs saved in:\n  %s\nOpen report.html to show the client." % (
        summary["secs"], outdir))


if __name__ == "__main__":
    main()
