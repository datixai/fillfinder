"""FillFinder core engine (web version).

Same idea as the desktop fillfinder.py, but it works on bytes in memory and
returns light SVG strings that are small enough to send over HTTP.

process(file_bytes, filename, options) -> dict with steps, stats and downloads.
"""
import base64
import colorsys
import csv
import io
import math
import os
import tempfile
import time

import numpy as np
import pyembroidery as pe
from shapely import contains_xy
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

UNIT = 10.0  # pyembroidery works in 0.1 mm


# ------------------------------------------------------------------ reading
def read_design(path):
    """Read a stitch file into colour blocks of paths, in millimetres."""
    pattern = pe.read(path)
    if pattern is None:
        raise ValueError("This file could not be read as an embroidery design.")
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
            colors.append("#%06x" % (pattern.threadlist[i].color & 0xFFFFFF))
        except (IndexError, AttributeError):
            colors.append(None)
    if not any(blocks):
        raise ValueError("No stitches found in this file.")
    return blocks, colors


# ------------------------------------------------------- raster helpers (numpy only)
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
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.res = res
        self.x0, self.y0 = min(xs) - 5, min(ys) - 5
        self.w = int((max(xs) + 5 - self.x0) / res) + 1
        self.h = int((max(ys) + 5 - self.y0) / res) + 1

    def draw(self, paths):
        m = np.zeros((self.h, self.w), dtype=np.uint8)
        for path in paths:
            a = np.asarray(path, dtype=float)
            p0, p1 = a[:-1], a[1:]
            n = np.maximum(1, np.ceil(np.hypot(*(p1 - p0).T) / (self.res * 0.5))).astype(int)
            t = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
            s = np.repeat(p0, n + 1, axis=0) + np.repeat(p1 - p0, n + 1, axis=0) * t[:, None]
            ix = np.clip(((s[:, 0] - self.x0) / self.res).astype(int), 0, self.w - 1)
            iy = np.clip(((s[:, 1] - self.y0) / self.res).astype(int), 0, self.h - 1)
            m[iy, ix] = 1
        return m

    def lookup(self, mask, x, y):
        ix = int((x - self.x0) / self.res)
        iy = int((y - self.y0) / self.res)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            return mask[iy, ix]
        return 0


# ------------------------------------------------------- outlines vs fills
def _split_by_mask(paths, R, mask):
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


def classify_blocks(blocks, thick_mm=3.5, max_px=1600):
    """Split the design into outlines and fills. Returns (outlines, info, fillinfo)."""
    tmp = Raster(blocks, 1.0)
    res = max(0.25, max(tmp.w, tmp.h) / float(max_px))
    R = Raster(blocks, res)
    close_k = max(3, int(round(1.0 / res)) | 1)
    er_k = int(round(thick_mm / res)) | 1
    edge_k = int(round(2.4 / res)) | 1
    deep = np.zeros((R.h, R.w), dtype=np.uint8)
    fill_any = np.zeros((R.h, R.w), dtype=np.uint8)
    cands, info = [], []
    for bi, paths in enumerate(blocks):
        n = sum(len(p) for p in paths)
        if not paths:
            info.append({"block": bi + 1, "stitches": 0, "fill_share": 0.0, "role": "empty"})
            continue
        m = R.draw(paths)
        closed = erode(dilate(m, close_k), close_k)
        thick = dilate(erode(closed, er_k), er_k + 2)
        share = min(1.0, float((thick & closed).sum()) / max(1, int(closed.sum())))
        if share > 0.45:
            role = "fill"
            deep |= erode(closed, edge_k)
            fill_any |= closed
        else:
            deep |= erode(thick & closed, edge_k)
            fill_any |= thick & closed
            cands.append(_split_by_mask(paths, R, thick))
            role = "outline" if share < 0.05 else "mixed"
        info.append({"block": bi + 1, "stitches": n, "fill_share": round(share, 3), "role": role})

    outlines = []
    for paths in cands:
        outlines.extend(_split_by_mask(paths, R, deep))
    outlines = [p for p in outlines if len(p) >= 2]

    has_fill = any(b["role"] == "fill" for b in info)
    if not has_fill:  # lines-only design: keep every line exactly as drawn
        outlines = [p for b in blocks for p in b]
        for b in info:
            if b["stitches"]:
                b["role"] = "outline"
        return outlines, info, None
    return outlines, info, (R, fill_any)


# ------------------------------------------------------------------ regions
def as_polys(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def find_regions(paths, gap=0.6, overlap=0.3, min_area=4.0, max_area_frac=0.15, simplify=0.08):
    r = gap / 2.0
    walls = unary_union([LineString(p).buffer(r, resolution=5) for p in paths if len(p) >= 2])
    hull_area = walls.convex_hull.area
    regions, skipped = [], 0
    for poly in as_polys(walls):
        for ring in poly.interiors:
            for piece in as_polys(Polygon(ring).difference(walls)):
                if piece.area < min_area:
                    continue
                if piece.area > max_area_frac * hull_area:
                    skipped += 1
                    continue
                regions.extend(as_polys(piece.buffer(r + overlap, join_style=1).simplify(simplify)))
    regions.sort(key=lambda g: (round(g.centroid.y / 8), g.centroid.x))
    return regions, skipped


def fill_coverage(region, R, fill_any, step=1.0):
    minx, miny, maxx, maxy = region.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    xs, ys = xs.ravel(), ys.ravel()
    if xs.size == 0:
        return 0.0
    inside = contains_xy(region, xs, ys)
    if not inside.any():
        return 0.0
    xs, ys = xs[inside], ys[inside]
    ix = np.clip(((xs - R.x0) / R.res).astype(int), 0, R.w - 1)
    iy = np.clip(((ys - R.y0) / R.res).astype(int), 0, R.h - 1)
    return float(fill_any[iy, ix].mean())


# ------------------------------------------------------------------ stitching
def tatami_rows(region, angle=45.0, spacing=0.4, stitch_len=3.5):
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
            pts = [(x0 + (x1 - x0) * min(1.0, (k + (offset if 0 < k < n else 0)) / n), y)
                   for k in range(n + 1)]
            rows.append(list(affinity.rotate(LineString(pts), angle, origin=c).coords))
        offset = (offset + 0.33) % 1.0
        flip = not flip
        y += spacing
    return rows


def build_pattern(outlines, regions, colors, angle, spacing):
    p = pe.EmbPattern()
    counts = []
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
                last, count = (x, y), count + 1
        counts.append(count)
    p.add_thread({"color": "#1a1a8c"})
    if regions:
        p.add_command(pe.COLOR_CHANGE)
    for path in outlines:
        p.add_command(pe.TRIM)
        p.add_stitch_absolute(pe.JUMP, path[0][0] * UNIT, path[0][1] * UNIT)
        for x, y in path:
            p.add_stitch_absolute(pe.STITCH, x * UNIT, y * UNIT)
    p.add_command(pe.END)
    return p, counts


# ------------------------------------------------------------------ SVG (kept small)
def palette(n):
    return ["#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.5, 0.9))
            for i in range(n)]


def _thin(path, step):
    if step <= 1 or len(path) <= 3:
        return path
    out = path[::step]
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def _pl(paths, color, width, step=1):
    o = ['<g fill="none" stroke="%s" stroke-width="%.2f" stroke-linejoin="round">' % (color, width)]
    for p in paths:
        o.append('<polyline points="%s"/>' % " ".join("%.1f,%.1f" % (q[0], q[1]) for q in _thin(p, step)))
    o.append("</g>")
    return o


def _ring(coords):
    return "M " + " L ".join("%.1f,%.1f" % (x, y) for x, y in coords) + " Z"


def _poly(g):
    return _ring(g.exterior.coords) + "".join(" " + _ring(r.coords) for r in g.interiors)


def _head(bounds, bg=None):
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%.1fmm" height="%.1fmm" '
         'viewBox="%.1f %.1f %.1f %.1f">' % (w, h, minx, miny, w, h)]
    if bg:
        o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>' % (minx, miny, w, h, bg))
    return o


def svg_original(blocks, colors, bounds, step):
    o = _head(bounds, "#ffffff")
    pal = palette(len(blocks))
    for i, paths in enumerate(blocks):
        o += _pl(paths, colors[i] or pal[i], 0.25, step)
    return "\n".join(o + ["</svg>"])


def svg_outlines(outlines, bounds, step):
    return "\n".join(_head(bounds, "#ffffff") + _pl(outlines, "#2440a8", 0.35, step) + ["</svg>"])


def svg_regions(outlines, regions, cols, bounds, step, labels=True, lines=True, bg="#ffffff", marks=False):
    o = _head(bounds, bg)
    for i, g in enumerate(regions):
        o.append('<path d="%s" fill="%s" fill-rule="evenodd"/>' % (_poly(g), cols[i]))
    if lines:
        o += _pl(outlines, "#2440a8", 0.3, step)
    if labels:
        fs = max(1.6, (bounds[2] - bounds[0]) / 80.0)
        o.append('<g font-family="Arial" font-size="%.1f" text-anchor="middle" fill="#111">' % fs)
        for i, g in enumerate(regions, 1):
            pt = g.representative_point()
            o.append('<text x="%.1f" y="%.1f" dominant-baseline="middle">%d</text>' % (pt.x, pt.y, i))
        o.append("</g>")
    if marks:
        o.append('<g stroke="#e00000" stroke-width="0.2"><line x1="-3" y1="0" x2="3" y2="0"/>'
                 '<line x1="0" y1="-3" x2="0" y2="3"/></g>')
    return "\n".join(o + ["</svg>"])


def svg_result(outlines, regions, cols, bounds, step):
    """Filled areas plus the outlines on top, on a dark background: the 'after' picture."""
    o = _head(bounds, "#15151c")
    for i, g in enumerate(regions):
        o.append('<path d="%s" fill="%s" fill-rule="evenodd" opacity="0.95"/>' % (_poly(g), cols[i]))
    o += _pl(outlines, "#4ce0d0", 0.3, step)
    return "\n".join(o + ["</svg>"])


# ------------------------------------------------------------------ main entry
def _b64(data):
    return base64.b64encode(data).decode("ascii")


def process(file_bytes, filename, opts=None):
    """Run the whole pipeline and return everything the web page needs."""
    o = dict(thick=3.5, gap=0.6, overlap=0.3, min_area=4.0, max_area=0.15,
             angle=45.0, spacing=0.4, fill_all=False, make_stitches=True)
    o.update(opts or {})
    t0 = time.time()

    ext = os.path.splitext(filename)[1].lower() or ".dst"
    if ext not in (".dst", ".pes", ".exp", ".jef", ".vp3", ".xxx", ".hus", ".pec", ".emb"):
        raise ValueError("Unsupported file type: %s" % ext)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(file_bytes)
        tmp.close()
        blocks, colors = read_design(tmp.name)
    finally:
        os.unlink(tmp.name)

    n_in = sum(len(p) for b in blocks for p in b)
    pts = [q for b in blocks for p in b for q in p]
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    bounds = (min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5)
    width_mm, height_mm = max(xs) - min(xs), max(ys) - min(ys)
    step = 1 if n_in < 20000 else (2 if n_in < 60000 else 3)  # keep the SVGs light

    outlines, info, fillinfo = classify_blocks(blocks, thick_mm=o["thick"])
    if not outlines:
        raise ValueError("No outline stitches were found. The design may be fills only.")
    n_outline = sum(len(p) for p in outlines)

    regions, skipped = find_regions(outlines, o["gap"], o["overlap"], o["min_area"], o["max_area"])
    found = len(regions)
    empty_skipped = 0
    if fillinfo is not None and not o["fill_all"]:
        R, fill_any = fillinfo
        keep = [g for g in regions if fill_coverage(g, R, fill_any) >= 0.5]
        empty_skipped = len(regions) - len(keep)
        regions = keep
    if not regions:
        raise ValueError("No closed areas were found. Try a larger gap value.")

    cols = palette(len(regions))
    counts = []
    downloads = {}
    if o["make_stitches"]:
        pattern, counts = build_pattern(outlines, regions, cols, o["angle"], o["spacing"])
        for fmt, writer in (("dst", pe.write_dst), ("pes", pe.write_pes)):
            buf = io.BytesIO()
            writer(pattern, buf)
            downloads["autofilled." + fmt] = _b64(buf.getvalue())
    else:
        counts = [0] * len(regions)

    wilcom = svg_regions(outlines, regions, cols, bounds, step, labels=False, lines=False,
                         bg=None, marks=True)
    downloads["regions_for_wilcom.svg"] = _b64(wilcom.encode("utf-8"))

    sio = io.StringIO()
    w = csv.writer(sio)
    w.writerow(["region", "area_mm2", "centre_x_mm", "centre_y_mm", "fill_stitches"])
    for i, (g, n) in enumerate(zip(regions, counts), 1):
        c = g.centroid
        w.writerow([i, round(g.area, 1), round(c.x, 2), round(c.y, 2), n])
    downloads["regions.csv"] = _b64(sio.getvalue().encode("utf-8"))

    mode = "finished" if fillinfo is not None else "lines"
    areas = sorted((g.area for g in regions), reverse=True)
    steps = [
        {"key": "input", "title": "1. Your design as it arrived",
         "text": ("This file has %s stitches in %s colour block(s) and measures %.0f x %.0f mm. "
                  "Nothing has been changed yet." % ("{:,}".format(n_in), len(blocks), width_mm, height_mm)),
         "svg": svg_original(blocks, colors, bounds, step)},
        {"key": "outlines", "title": "2. Outlines kept, fills removed",
         "text": ("The tool checks how wide each stitched part is. Anything wider than %.1f mm is a fill, so it "
                  "is removed. Thin borders, run lines and zigzag edges are kept, because those are the lines "
                  "that shape your design. %s outline stitches were kept."
                  % (o["thick"], "{:,}".format(n_outline))) if mode == "finished" else
                 ("Your design is made of lines only, so every line is used as an outline. "
                  "%s stitches in total." % "{:,}".format(n_outline)),
         "svg": svg_outlines(outlines, bounds, step)},
        {"key": "regions", "title": "3. Closed areas found automatically",
         "text": ("Every line is made slightly thicker so it works like a wall. Any space that is fully "
                  "surrounded by walls becomes an area you can fill. This also works when a shape is closed only "
                  "because two lines cross, and small openings up to %.1f mm are closed automatically. "
                  "%d areas were found%s." % (o["gap"], found,
                  ", and %d of them were empty in your original design so they were left empty" % empty_skipped
                  if empty_skipped else "")),
         "svg": svg_regions(outlines, regions, cols, bounds, step)},
        {"key": "result", "title": "4. Every area filled",
         "text": ("Each area now has its own fill, generated automatically. No tracing and no reshaping. "
                  "%s fill stitches were created at %d degrees with %.2f mm spacing."
                  % ("{:,}".format(sum(counts)), int(o["angle"]), o["spacing"])
                  if o["make_stitches"] else "Each area now has its own fill shape, generated automatically."),
         "svg": svg_result(outlines, regions, cols, bounds, step)},
    ]

    return {
        "ok": True,
        "filename": filename,
        "mode": mode,
        "seconds": round(time.time() - t0, 1),
        "stats": {
            "stitches_in": n_in,
            "colour_blocks": len(blocks),
            "size_mm": "%.0f x %.0f" % (width_mm, height_mm),
            "areas_found": found,
            "areas_filled": len(regions),
            "areas_left_empty": empty_skipped,
            "background_skipped": skipped,
            "fill_stitches": sum(counts),
            "largest_area_mm2": round(areas[0], 1) if areas else 0,
            "smallest_area_mm2": round(areas[-1], 1) if areas else 0,
        },
        "blocks": info,
        "regions": [{"n": i, "area": round(g.area, 1), "stitches": c}
                    for i, (g, c) in enumerate(zip(regions, counts), 1)],
        "steps": steps,
        "downloads": downloads,
    }
