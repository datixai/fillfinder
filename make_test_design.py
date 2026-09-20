"""Creates a test DST that mimics the client's workflow:
a flower drawn only with OPEN run lines that cross each other,
plus one zigzag (satin-like) border. No closed objects anywhere.
Usage: python make_test_design.py test_flower.dst
"""
import math
import sys
import pyembroidery as pe

MM = 10  # pyembroidery units are 0.1 mm


def run_line(pattern, pts, step_mm=2.5):
    """Add an open running-stitch line through pts (mm)."""
    pattern.add_command(pe.TRIM)
    first = True
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step_mm))
        for i in range(n + 1):
            if i == 0 and not first:
                continue
            t = i / n
            x, y = (x0 + (x1 - x0) * t) * MM, (y0 + (y1 - y0) * t) * MM
            pattern.add_stitch_absolute(pe.JUMP if first else pe.STITCH, x, y)
            first = False


def zigzag_line(pattern, pts, width_mm=2.0, step_mm=0.8):
    """Add a zigzag border following pts (mm)."""
    pattern.add_command(pe.TRIM)
    dense = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step_mm))
        for i in range(n):
            t = i / n
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, x1 - x0, y1 - y0, d))
    for i, (x, y, dx, dy, d) in enumerate(dense):
        nx, ny = -dy / d, dx / d
        s = width_mm / 2 * (1 if i % 2 else -1)
        cmd = pe.JUMP if i == 0 else pe.STITCH
        pattern.add_stitch_absolute(cmd, (x + nx * s) * MM, (y + ny * s) * MM)


def petal(cx, cy, length, width, angle_deg, n=60, overshoot=0.06, gap=0.0):
    """Leaf-shaped petal drawn as ONE OPEN line. With overshoot>0 its ends run
    past each other and cross; with gap>0 its ends stop short (tiny opening)."""
    a = math.radians(angle_deg)
    pts = []
    start, end = -overshoot + gap, 1 + overshoot - gap
    for i in range(n + 1):
        t = start + (end - start) * i / n
        u = 2 * math.pi * t
        lx = length * (1 - math.cos(u)) / 2           # 0 at base, length at tip
        ly = width * math.sin(u) * (0.6 + 0.4 * (1 - math.cos(u)) / 2)
        x = cx + lx * math.cos(a) - ly * math.sin(a)
        y = cy + lx * math.sin(a) + ly * math.cos(a)
        pts.append((x, y))
    return pts


def main(out):
    p = pe.EmbPattern()
    p.add_thread({"color": "#2b2bb0"})
    # outer ring of 6 big petals
    for k in range(6):
        run_line(p, petal(0, 0, 55, 16 + (k % 2) * 3, k * 60 + 15, gap=0.004 if k == 2 else 0.0))
    # inner ring of 5 small petals, rotated, overlapping the big ones
    for k in range(5):
        run_line(p, petal(0, 0, 30, 9, k * 72 + 40))
    # centre circle drawn as an open line that overlaps itself a bit
    circ = [(8 * math.cos(math.radians(d)), 8 * math.sin(math.radians(d))) for d in range(-20, 370, 10)]
    run_line(p, circ)
    # one zigzag border across the lower petals
    arc = [(40 * math.cos(math.radians(d)), 40 * math.sin(math.radians(d))) for d in range(20, 101, 5)]
    zigzag_line(p, arc)
    p.add_command(pe.END)
    pe.write_dst(p, out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_flower.dst")
