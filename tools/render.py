#!/usr/bin/env python3
"""Studio for cutest-picture-01 -- renders entry.png offline.

This file is a TOOL, not the entry: the shipped entry is main.py, which only copies
entry.png. Run it with an interpreter that has Pillow + numpy (the studio venv):

    python3 tools/render.py /root/makefun/art/work/a2.png a
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SZ = 1024
SS = 2
W = SZ * SS

_arr = (np.arange(W, dtype=np.float32) + 0.5) / SS
X, Y = np.meshgrid(_arr, _arr)

RNG = np.random.default_rng(20261113)


# ---------------------------------------------------------------- mask helpers
def bl(mask, r):
    if r <= 0:
        return mask.astype(np.float32)
    im = Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0


def ell(cx, cy, rx, ry, soft=1.3, rot=0.0):
    dx = X - cx
    dy = Y - cy
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = dx * c + dy * s, -dx * s + dy * c
    r = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return np.clip(0.5 - (r - 1.0) * min(rx, ry) / soft, 0.0, 1.0).astype(np.float32)


def poly_mask(pts, soft=1.4):
    im = Image.new("L", (W, W), 0)
    ImageDraw.Draw(im).polygon([(x * SS, y * SS) for x, y in pts], fill=255)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return bl(a, soft) if soft else a


def blob(cx, cy, r, n=8, irr=0.16, phase=0.0, aspect=0.72, soft=1.8, rot=0.0):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1.0 + irr * math.sin(3.0 * a + phase) + irr * 0.6 * math.cos(5.0 * a - phase))
        px, py = rr * math.cos(a), rr * math.sin(a) * aspect
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            px, py = px * c - py * s, px * s + py * c
        pts.append((cx + px, cy + py))
    return poly_mask(pts, soft=soft)


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return [0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * t
                   + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2
                   + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3) for i in (0, 1)]


def stroke(points, radii, soft=1.0, total=420):
    p = [points[0]] + list(points) + [points[-1]]
    r = [radii[0]] + list(radii) + [radii[-1]]
    segs = len(p) - 3
    per = max(3, total // segs)
    im = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(im)
    for s in range(segs):
        for k in range(per + 1):
            t = k / per
            x, y = _catmull(p[s], p[s + 1], p[s + 2], p[s + 3], t)
            rr = r[s + 1] * (1 - t) + r[s + 2] * t
            d.ellipse([(x - rr) * SS, (y - rr) * SS, (x + rr) * SS, (y + rr) * SS], fill=255)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return bl(a, soft) if soft else a


def stroke_pts(points, radii, soft=1.0):
    """Dense (point, radius) samples along a smoothed path, for ring stripes."""
    p = [points[0]] + list(points) + [points[-1]]
    r = [radii[0]] + list(radii) + [radii[-1]]
    out = []
    for s in range(len(p) - 3):
        for k in range(11):
            t = k / 10.0
            x, y = _catmull(p[s], p[s + 1], p[s + 2], p[s + 3], t)
            out.append(((x, y), r[s + 1] * (1 - t) + r[s + 2] * t))
    return out


def heart_shape(cx, cy, s, soft=3.0):
    """Two lobes over a point: an unmistakable heart."""
    m = np.clip(ell(cx - 0.42 * s, cy - 0.34 * s, 0.50 * s, 0.50 * s, soft=2.0)
                + ell(cx + 0.42 * s, cy - 0.34 * s, 0.50 * s, 0.50 * s, soft=2.0)
                + poly_mask([(cx - 0.90 * s, cy - 0.12 * s),
                             (cx + 0.90 * s, cy - 0.12 * s),
                             (cx, cy + 0.94 * s)], soft=soft), 0, 1)
    return bl(m, 2.0)


def heart(cx, cy, s, soft=1.2, rot=0.0):
    dx = (X - cx) / s
    dy = -(Y - cy) / s
    if rot:
        c, sn = math.cos(rot), math.sin(rot)
        dx, dy = dx * c + dy * sn, -dx * sn + dy * c
    v = dy * 0.86 + 0.30
    q = dx * dx + v * v - 1.0
    return np.clip(0.5 - (q * q * q - dx * dx * v * v * v) / 3.0 / soft, 0.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------- lighting
LIGHT = (-0.44, -0.58, 0.69)
_l = math.sqrt(sum(c * c for c in LIGHT))
LIGHT = tuple(c / _l for c in LIGHT)
_H = (LIGHT[0], LIGHT[1], LIGHT[2] + 1.0)
_h = math.sqrt(sum(c * c for c in _H))
HALF = tuple(c / _h for c in _H)


def _light(nx, ny, nz, base, amb, diff, spec, sp, bounce, bcol):
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0.0, 1.0)
    col = np.empty((W, W, 3), np.float32)
    base = np.asarray(base, np.float32)
    f = amb + diff * lam
    for i in range(3):
        col[:, :, i] = base[i] * f
    if spec > 0:
        s = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0.0, 1.0) ** sp
        col += (s * spec * 255.0)[:, :, None]
    if bounce > 0:
        bn = np.clip(-nx * 0.45 + ny * 0.80 + 0.10, 0.0, 1.0) * bounce
        for i in range(3):
            col[:, :, i] += bcol[i] * bn
    return col


def sphere(mask, cx, cy, rx, ry, base, amb=0.50, diff=0.50, spec=0.16, sp=16,
           bounce=0.16, bcol=(255, 206, 168), grain=0.0):
    nx = (X - cx) / rx
    ny = (Y - cy) / ry
    r2 = nx * nx + ny * ny
    col = _light(nx, ny, np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)), base, amb, diff, spec, sp,
                 bounce, bcol)
    if grain:
        n = bl(RNG.normal(0.0, 1.0, (W, W)).astype(np.float32), 3.0)
        n /= max(1e-6, float(np.abs(n).max()))
        col *= (1.0 + n * grain * np.clip(1.0 - r2, 0.0, 1.0))[:, :, None]
    return col


def pillow(mask, base, blur=16.0, k=0.85, amb=0.58, diff=0.42, spec=0.12, sp=12,
           bounce=0.10, bcol=(255, 208, 172)):
    gy, gx = np.gradient(bl(mask, blur))
    m = np.sqrt(gx * gx + gy * gy)
    mm = float(m.max()) or 1.0
    nx = -gx / mm * k
    ny = -gy / mm * k
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return _light(nx / ln, ny / ln, nz / ln, base, amb, diff, spec, sp, bounce, bcol)


def torus_shade(mask, cx, cy, rx, ry, base, amb=0.55, diff=0.45, spec=0.14, sp=8,
         bounce=0.10, bcol=(255, 206, 168), tilt=0.78):
    """Smooth torus-ish shading for a rim: normal points outward around the ellipse."""
    dx = (X - cx) / rx
    dy = (Y - cy) / ry
    ln = np.sqrt(dx * dx + dy * dy) + 1e-6
    nx = dx / ln * tilt
    ny = dy / ln * tilt
    nz = np.full_like(nx, 0.66)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return _light(nx / l, ny / l, nz / l, base, amb, diff, spec, sp, bounce, bcol)


def cyl(mask, cx, hw_of_y, base, amb=0.40, diff=0.58, spec=0.12, sp=10,
        bounce=0.14, bcol=(255, 200, 160)):
    hw = hw_of_y(Y)
    nx = np.clip((X - cx) / np.maximum(hw, 1.0), -1.0, 1.0)
    return _light(nx, np.zeros_like(nx), np.sqrt(np.clip(1.0 - nx * nx, 0.0, 1.0)),
                  base, amb, diff, spec, sp, bounce, bcol)


def over(img, col, mask, alpha=1.0):
    m = (mask * alpha)[:, :, None]
    img *= 1.0 - m
    img += col * m
    return img


def flat(img, mask, rgb, alpha=1.0):
    m = (mask * alpha)[:, :, None]
    img *= 1.0 - m
    img += np.asarray(rgb, np.float32) * m
    return img


def shade(img, mask, alpha=1.0, bcol=(70, 58, 52)):
    m = (mask * alpha)[:, :, None]
    img *= 1.0 - m * (1.0 - np.asarray(bcol, np.float32) / 255.0)
    return img


def save(img, path):
    a = np.clip(img, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(a, "RGB").resize((SZ, SZ), Image.LANCZOS).save(path, optimize=True)
    return path


# ---------------------------------------------------------------- palette
P = dict(
    bg_top=(176, 206, 148),
    bg_bot=(244, 234, 207),
    glow=(255, 250, 233),
    cup=(243, 225, 192),
    cup_dark=(224, 199, 160),
    cup_in=(206, 176, 141),
    rim=(191, 98, 68),
    saucer=(240, 226, 197),
    fur=(255, 250, 241),
    stripe=(230, 191, 150),
    inner_ear=(241, 175, 173),
    blush=(247, 164, 168),
    face=(93, 68, 57),
    nose=(232, 143, 155),
    paw=(255, 252, 246),
    steam=(255, 253, 247),
    biscuit=(220, 176, 111),
    mint=(133, 179, 108),
    whisker=(196, 176, 154),
)


def render_a():
    it = float(os.environ.get("ITER", "0"))
    img = np.zeros((W, W, 3), np.float32)

    # ---------------------------------------------- background
    f = np.clip(Y / SZ, 0.0, 1.0)[:, :, None]
    img[:] = (np.asarray(P["bg_top"], np.float32) * (1 - f)
              + np.asarray(P["bg_bot"], np.float32) * f)
    img *= (1.0 - (np.clip((Y - 726.0) / 70.0, 0, 1))[:, :, None] * 0.055)
    rv = np.sqrt((X - 512) ** 2 + (Y - 520) ** 2) / 820.0
    img *= (1.0 - (np.clip(rv - 0.60, 0, 1) ** 1.5)[:, :, None] * 0.25)

    # ---------------------------------------------- saucer
    scx, scy, srx, sry = 512, 806, 342, 108
    img = shade(img, bl(ell(512, 886, 340, 52, soft=8), 28), 0.38, (116, 100, 84))
    m = ell(scx, scy, srx, sry, soft=1.3)
    img = over(img, sphere(m, scx, scy, srx, sry, P["saucer"], amb=0.68, diff=0.32,
                           spec=0.10, sp=8, bounce=0.08), m)
    well = ell(scx, scy - 8, srx * 0.76, sry * 0.74, soft=1.6)
    img = shade(img, well, 0.18, (150, 128, 104))
    img = shade(img, ell(scx, scy + 24, srx * 0.55, sry * 0.42, soft=26), 0.16, (128, 106, 86))
    ring = np.clip(ell(scx, scy - 6, srx * 0.846, sry * 0.829, soft=0.9)
                   - ell(scx, scy - 6, srx * 0.832, sry * 0.814, soft=0.9), 0, 1)
    img = flat(img, bl(ring, 0.7), P["rim"], 0.85)

    # ---------------------------------------------- cup handle (behind the body)
    h_out = ell(250, 612, 72, 88, soft=1.3)
    h_in = ell(278, 612, 42, 56, soft=1.3)
    handle = np.clip(h_out - h_in, 0, 1)
    img = over(img, pillow(handle, (230, 204, 166), blur=13, k=0.85, amb=0.46, diff=0.54,
                           spec=0.20, sp=10), handle)
    img = flat(img, np.clip(bl(handle, 2.0) - handle, 0, 1), (150, 120, 94), 0.22)
    img = shade(img, np.clip((bl(h_out, 3.0) - h_out) + (h_in - bl(h_in, 3.0)), 0, 1) * handle,
                0.16, (156, 124, 96))
    # clean silhouette gap between handle and cup wall
    gap = ell(286, 612, 18, 92, soft=4.0)
    img = shade(img, np.clip(gap * handle, 0, 1), 0.48, (72, 54, 44))
    img = flat(img, np.clip(bl(handle, 2.6) - handle, 0, 1), (110, 86, 68), 0.42)

    # ---------------------------------------------- cup body
    rim_cy, body_y = 470, 778
    hw_top, hw_bot = 296, 170

    def hw_at(y):
        t = np.clip((y - rim_cy) / (body_y - rim_cy), 0.0, 1.0)
        return hw_bot + (hw_top - hw_bot) * (1.0 - t) ** 1.22

    ys = np.arange(rim_cy, body_y, 2.0)
    left = [(512 - float(hw_at(y)), float(y)) for y in ys]
    right = [(512 + float(hw_at(y)), float(y)) for y in ys][::-1]
    cap = [(512 + hw_bot * math.cos(a), body_y + 34 * math.sin(a))
           for a in np.linspace(0, -math.pi, 30)]
    body = np.clip(poly_mask(left + cap + right, soft=1.0)
                   + ell(512, body_y, hw_bot, 34, soft=1.2), 0, 1)
    img = over(img, cyl(body, 512, hw_at, P["cup"], amb=0.46, diff=0.54, spec=0.10, sp=9,
                        bounce=0.16), body)
    img = shade(img, bl(body, 70) * np.clip((Y - 540) / 300.0, 0, 1), 0.18, (140, 112, 86))
    img = shade(img, bl(ell(268, 622, 64, 138, soft=6), 36), 0.20, (150, 118, 92))
    img = shade(img, bl(ell(258, 542, 40, 34, soft=10), 12), 0.24, (146, 114, 88))
    img = shade(img, bl(ell(262, 690, 40, 32, soft=10), 12), 0.24, (146, 114, 88))
    # foot ring at the base
    foot = np.clip(ell(512, body_y - 6, hw_bot * 0.94, 30, soft=1.4)
                   - ell(512, body_y + 6, hw_bot * 0.80, 22, soft=1.4), 0, 1)
    img = shade(img, np.clip(foot * body, 0, 1), 0.22, (150, 120, 92))
    img = shade(img, bl(ell(512, body_y + 14, hw_bot * 0.84, 20, soft=8), 5), 0.30, (126, 102, 80))
    img = shade(img, bl(ell(512, body_y + 36, hw_bot * 0.96, 20, soft=10), 13), 0.18, (140, 112, 88))
    # sheen and a thin hand-painted line on the flank
    img = flat(img, bl(ell(404, 606, 52, 150, soft=30), 34), (255, 250, 236), 0.30)
    y_line = 552.0 + 0.00055 * (X - 512.0) ** 2
    line = (np.clip((2.9 - np.abs(Y - y_line)) / 1.5, 0, 1)
            * np.clip((hw_at(Y) - np.abs(X - 512.0) - 5.0) / 4.0, 0, 1))
    img = flat(img, bl(line, 1.0), P["rim"], 0.8)

    # ---------------------------------------------- cup mouth
    mouth = ell(512, rim_cy, 272, 66, soft=1.3)
    img = over(img, sphere(mouth, 512, rim_cy - 12, 272, 66, (216, 190, 158), amb=0.76,
                           diff=0.24, spec=0.04, sp=6, bounce=0.0), mouth)
    img = shade(img, mouth * np.clip((424.0 - Y) / 95.0, 0, 1), 0.40, (114, 88, 62))
    band = np.clip(ell(512, rim_cy, 296, 82, soft=1.2)
                   - ell(512, rim_cy, 266, 62, soft=1.2), 0, 1)
    w_far = np.clip((rim_cy + 8.0 - Y) / 16.0, 0, 1)
    far = band * w_far

    # ---------------------------------------------- kitten: shoulders in the cup
    img = shade(img, bl(np.clip(ell(512, 588, 240, 90, soft=20), 0, 1) * mouth, 26), 0.22,
                (122, 96, 70))
    img = shade(img, bl(np.clip(ell(512, 456, 248, 128, soft=34), 0, 1), 34), 0.15, (128, 102, 78))
    img = over(img, torus_shade(far, 512, rim_cy, 281, 72, P["rim"], amb=0.59, diff=0.41,
                         spec=0.13, sp=9, tilt=0.80), far)

    # ---------------------------------------------- ears (behind the head)
    earL = stroke([(414, 296), (402, 244), (386, 192), (378, 148)], [86, 78, 46, 17], soft=1.4)
    earR = stroke([(618, 296), (650, 240), (684, 198), (712, 168)], [86, 76, 46, 16], soft=1.4)
    tufts = np.zeros((W, W), np.float32)
    for ty, tl in ((452, 46), (486, 40), (518, 32)):
        tufts = np.clip(tufts + stroke([(360, ty), (336, ty - 6), (318, ty + tl * 0.0)],
                                       [18, 11, 2], soft=1.4), 0, 1)
        tufts = np.clip(tufts + stroke([(650, ty), (674, ty - 6), (692, ty)],
                                       [18, 11, 2], soft=1.4), 0, 1)
    ears = np.clip(earL + earR + tufts, 0, 1)
    img = over(img, pillow(ears, P["fur"], blur=24, k=0.55, amb=0.60, diff=0.40, spec=0.06), ears)

    # ---------------------------------------------- head
    hcx, hcy, hrx, hry = 505, 362, 198, 182
    head = ell(hcx, hcy, hrx, hry, soft=1.3)
    img = shade(img, np.clip(bl(head, 13.0) - bl(head, 2.0), 0, 1) * mouth, 0.26, (128, 100, 74))
    img = over(img, sphere(head, hcx, hcy, hrx, hry, P["fur"], amb=0.54, diff=0.46, spec=0.0,
                           sp=30, bounce=0.24, bcol=(255, 214, 184), grain=0.035), head)
    # matte directional fur: soft tufts + a gentle top-left light wash (no gloss)
    for path, rads in (([(352, 300), (322, 326), (302, 362)], [7, 5, 2]),
                       ([(660, 296), (688, 324), (708, 360)], [7, 5, 2]),
                       ([(432, 214), (472, 198), (520, 196)], [6, 5, 2])):
        tuft_h = stroke(path, rads, soft=2.2)
        img = shade(img, np.clip(tuft_h * head, 0, 1), 0.055, (152, 122, 96))
    img = flat(img, bl(head, 40) * np.clip((360.0 - Y) / 300.0, 0, 1) * head, (255, 253, 246),
               0.10)

    # inner ears, only on the visible part of each ear
    vis = bl(np.clip(bl(ears, 4.0) - bl(head, 4.0), 0, 1), 1.2)
    ieL = stroke([(408, 288), (398, 246), (386, 200), (380, 166)], [50, 42, 22, 7], soft=1.6)
    ieR = stroke([(624, 286), (652, 240), (690, 214), (724, 226), (744, 252)],
                 [50, 40, 22, 10, 5], soft=1.6)
    ieR = stroke([(622, 288), (650, 242), (680, 202), (702, 176)], [50, 42, 22, 7], soft=1.6)
    inner = bl(np.clip((ieL + ieR) * vis, 0, 1), 1.6)
    img = over(img, pillow(inner, P["inner_ear"], blur=8, k=0.5, amb=0.70, diff=0.30, spec=0.05),
               inner, 0.85)
    fold = stroke([(604, 292), (636, 268), (662, 258)], [9, 8, 6], soft=2.4)
    img = shade(img, np.clip(fold * bl(vis, 3), 0, 1), 0.15, (176, 146, 118))

    # ---------------------------------------------- warm cheek fur, no hard crown marks
    img = flat(img, bl(ell(506, 452, 190, 120, soft=44), 44), (253, 238, 226), 0.18)

    # blush
    for bx in (378, 634):
        rr = np.sqrt(((X - bx) / 54.0) ** 2 + ((Y - 442) / 40.0) ** 2)
        img = flat(img, np.clip((1.0 - rr) / 0.55, 0, 1), P["blush"], 0.40)

    # whiskers
    wh = np.zeros((W, W), np.float32)
    for path in ([(400, 448), (368, 440), (342, 430)],
                 ([(404, 466), (372, 466), (348, 468)]),
                 ([(612, 448), (644, 440), (670, 430)]),
                 ([(608, 466), (640, 466), (664, 468)])):
        wh = np.clip(wh + stroke(path, [3.4, 2.6, 1.2], soft=1.0), 0, 1)
    img = flat(img, bl(wh, 1.5), P["whisker"], 0.30)

    # ---------------------------------------------- face
    face = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(face)
    for ex in (430, 582):
        d.arc([(ex - 49) * SS, (368 - 38) * SS, (ex + 49) * SS, (368 + 38) * SS], 22, 158,
              fill=255, width=int(17 * SS))
    img = flat(img, bl(np.asarray(face, np.float32) / 255.0, 0.8), P["face"])

    nose = poly_mask([(506, 462), (487, 442), (506, 431), (525, 442)], soft=2.4)
    img = over(img, sphere(nose, 506, 446, 20, 17, P["nose"], amb=0.60, diff=0.40, spec=0.32,
                           sp=14, bounce=0.10), nose)
    img = flat(img, bl(ell(500, 440, 6, 4, soft=2.0), 1.5), (255, 250, 250), 0.75)
    mm_ = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(mm_)
    d.arc([(492 - 12) * SS, (458 - 12) * SS, (492 + 12) * SS, (458 + 12) * SS], 350, 120,
          fill=255, width=int(7 * SS))
    d.arc([(520 - 12) * SS, (458 - 12) * SS, (520 + 12) * SS, (458 + 12) * SS], 60, 190,
          fill=255, width=int(7 * SS))
    img = flat(img, bl(np.asarray(mm_, np.float32) / 255.0, 0.7), P["face"], 0.9)

    # ---------------------------------------------- near lip
    near = band * (1.0 - w_far)
    img = over(img, torus_shade(near, 512, rim_cy, 281, 72, P["rim"], amb=0.58, diff=0.42,
                         spec=0.14, sp=9, tilt=0.80), near)
    img = shade(img, bl(ell(512, 440, 210, 130, soft=40), 26), 0.20, (118, 94, 72))


    # ---------------------------------------------- two paws gripping the front rim
    for px in (406, 618):
        cx0, cy0, pw, ph = px, 520, 38, 27
        paw = ell(cx0, cy0, pw, ph, soft=1.6)
        for dx in (-17, 0, 17):
            paw = np.clip(paw + ell(cx0 + dx, cy0 - 26, 13, 14, soft=1.6), 0, 1)
        img = flat(img, np.clip(bl(paw, 1.5) - paw, 0, 1), (150, 120, 94), 0.60)
        img = over(img, sphere(paw, cx0 - 4, cy0 - 8, pw, ph * 1.2, P["paw"], amb=0.58,
                               diff=0.42, spec=0.0, sp=10, bounce=0.24,
                               bcol=(255, 210, 180)), paw)
        img = shade(img, np.clip(bl(paw, 5.0) - paw, 0, 1), 0.30, (128, 104, 82))
        for dx in (-9, 9):
            cr = stroke([(cx0 + dx, cy0 - 38), (cx0 + dx * 1.6, cy0 + 14)], [4.6, 3.0],
                        soft=0.7)
            img = shade(img, np.clip(cr * paw, 0, 1), 0.55, (134, 106, 82))
        tip = stroke([(cx0, cy0 - 40), (cx0, cy0 + 8)], [5.0, 3.2], soft=0.7)
        img = shade(img, np.clip(tip * paw, 0, 1), 0.22, (146, 118, 94))

    # ---------------------------------------------- tail curled around the rim
    if os.environ.get("FLICK", "0") == "1":
        tp = [(604, 552), (612, 526), (620, 502)]
        tr = [17, 15, 7]
        tail = stroke(tp, tr, soft=1.3)
        img = flat(img, np.clip(bl(tail, 1.7) - tail, 0, 1), (168, 138, 112), 0.40)
        img = over(img, pillow(tail, P["fur"], blur=14, k=0.7, amb=0.52, diff=0.48, spec=0.12),
                   tail)

    # ---------------------------------------------- biscuit crumbs
    img = shade(img, bl(ell(628, 862, 84, 30, soft=10), 12), 0.24, (140, 118, 96))
    for (cx0, cy0, ang, ln, wd) in ((630, 852, -0.26, 108, 36),):
        ux, uy = math.cos(ang), math.sin(ang)
        path = [(cx0 - ux * ln * 0.5, cy0 - uy * ln * 0.5),
                (cx0, cy0),
                (cx0 + ux * ln * 0.5, cy0 + uy * ln * 0.5)]
        lf = stroke(path, [wd * 0.42, wd * 0.52, 2.0], soft=1.8)
        img = over(img, pillow(lf, (152, 188, 130), blur=13, k=0.7, amb=0.56, diff=0.44,
                               spec=0.14), lf)
        img = flat(img, np.clip(stroke(path, [wd * 0.10, wd * 0.11, 1.0], soft=1.0) * lf, 0, 1),
                   (104, 140, 84), 0.72)

    # ---------------------------------------------- one bold mint leaf on the saucer
    img = shade(img, bl(ell(720, 860, 84, 26, soft=10), 12), 0.22, (140, 118, 96))
    stem_m = stroke([(694, 892), (686, 880), (678, 872)], [3.2, 2.8, 2.2], soft=0.9)
    img = flat(img, stem_m, (108, 148, 86), 0.95)
    leaf = poly_mask([(668, 872), (698, 879), (730, 877), (755, 864), (775, 844),
                      (785, 820), (786, 796), (764, 787), (738, 787), (711, 797),
                      (689, 814), (674, 842)], soft=1.5)
    img = shade(img, np.clip(bl(leaf, 1.7) - leaf, 0, 1), 0.30, (74, 104, 58))
    img = over(img, pillow(leaf, P["mint"], blur=8, k=0.68, amb=0.56, diff=0.44, spec=0.05),
               leaf)
    img = flat(img, np.clip(stroke([(672, 868), (722, 838), (782, 798)], [2.6, 2.0, 1.2],
                                   soft=0.8) * leaf, 0, 1), (84, 122, 64), 0.65)
    for a, b in (((709, 845), (730, 876)), ((709, 845), (689, 815)),
                 ((733, 830), (754, 862)), ((733, 830), (711, 798))):
        img = flat(img, np.clip(stroke([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), b],
                                       [1.6, 1.3, 0.9], soft=0.8) * leaf, 0, 1), (84, 122, 64), 0.45)

    # ---------------------------------------------- three separate curls of steam
    steam = np.zeros((W, W), np.float32)
    for path, rads in (([(146, 418), (176, 370), (138, 320), (168, 268), (140, 222)],
                        [14, 12, 10, 8, 6]),
                       ([(208, 432), (240, 380), (198, 326), (230, 274), (202, 226)],
                        [14, 12, 10, 8, 6]),
                       ([(272, 436), (302, 382), (260, 328), (292, 276), (266, 234)],
                        [12, 10, 9, 7, 5])):
        steam = np.clip(steam + stroke(path, rads, soft=1.0), 0, 1)
    img = flat(img, bl(steam, 10.0), (244, 249, 255), 0.26)
    img = flat(img, bl(steam, 3.2), (250, 252, 255), 0.48)
    img = flat(img, np.clip(steam, 0, 1), (253, 254, 255), 0.74 + 0.02 * it)

    # ---------------------------------------------- teabag tag on the near rim
    string = stroke([(548, 500), (566, 540), (586, 574)], [1.7, 1.5, 1.2], soft=1.0)
    img = flat(img, string, (168, 148, 118), 0.60)
    tag = poly_mask([(586, 574), (576, 606), (610, 614), (618, 582)], soft=1.6)
    img = over(img, pillow(tag, (250, 245, 233), blur=6, k=0.6, amb=0.64, diff=0.36,
                           spec=0.04), tag)
    img = flat(img, np.clip(bl(tag, 1.4) - tag, 0, 1), (150, 128, 102), 0.35)
    tln = stroke([(592, 586), (596, 600), (606, 602)], [1.9, 1.9, 1.6], soft=0.7)
    img = flat(img, np.clip(tln * tag, 0, 1), (160, 140, 112), 0.50)

    return img


# ---------------------------------------------------------------- concept B (comparison render)
def render_b():
    img = np.zeros((W, W, 3), np.float32)
    f = np.clip(Y / SZ, 0, 1)[:, :, None]
    img[:] = np.asarray((168, 194, 206), np.float32) * (1 - f) + \
        np.asarray((196, 214, 214), np.float32) * f
    r = np.sqrt((X - 500) ** 2 + (Y - 460) ** 2) / 780.0
    g = np.clip(1 - r, 0, 1) ** 2 * 0.4
    img = img * (1 - g[:, :, None]) + np.asarray((214, 228, 226), np.float32) * g[:, :, None]
    gr = np.clip((Y - 720) / 40.0, 0, 1)
    img = img * (1 - gr[:, :, None]) + np.asarray((176, 196, 152), np.float32) * gr[:, :, None]
    pud = ell(512, 900, 250, 46, soft=8)
    img = over(img, sphere(pud, 512, 900, 250, 46, (150, 176, 190), amb=0.7, diff=0.3, spec=0.3,
                           sp=8, bounce=0.0), pud, 0.85)
    stem_m = poly_mask([(470, 760), (486, 470), (546, 470), (560, 760)], soft=2.0)
    img = over(img, cyl(stem_m, 512, lambda y: np.full_like(y, 40.0), (238, 230, 208),
                        amb=0.5, diff=0.5, spec=0.1), stem_m)
    cap = np.clip(ell(512, 470, 250, 190, soft=2.0) * np.clip((560 - Y) / 90.0, 0, 1), 0, 1)
    img = over(img, sphere(cap, 512, 520, 250, 200, (206, 62, 58), amb=0.5, diff=0.5, spec=0.2,
                           sp=14, bounce=0.12, bcol=(255, 190, 150)), cap)
    for cx0, cy0, r0 in ((430, 420, 40), (556, 396, 48), (620, 470, 34), (470, 500, 30),
                         (360, 486, 30), (600, 372, 24)):
        m = ell(cx0, cy0, r0, r0 * 0.86, soft=1.6)
        img = over(img, sphere(m, cx0 - 5, cy0 - 6, r0, r0 * 0.86, (250, 246, 236), amb=0.6,
                               diff=0.4, spec=0.12), m)
    body = ell(470, 792, 150, 118, soft=2.0)
    img = over(img, sphere(body, 470, 780, 150, 118, (252, 214, 84), amb=0.5, diff=0.5,
                           spec=0.16, sp=14, bounce=0.14, bcol=(255, 230, 170)), body)
    head = ell(452, 626, 118, 106, soft=2.0)
    img = over(img, sphere(head, 452, 618, 118, 106, (253, 220, 96), amb=0.48, diff=0.52,
                           spec=0.18, sp=16, bounce=0.16), head)
    beak = poly_mask([(352, 620), (300, 630), (352, 646)], soft=2.4)
    img = over(img, sphere(beak, 340, 630, 40, 20, (238, 152, 56), amb=0.55, diff=0.45,
                           spec=0.2), beak)
    for ex in (404, 470):
        m = ell(ex, 610, 22, 24, soft=1.6)
        img = over(img, sphere(m, ex, 610, 22, 24, (52, 46, 52), amb=0.32, diff=0.68, spec=0.5,
                               sp=40), m)
        img = flat(img, ell(ex - 6, 601, 7, 7, soft=1.4), (255, 255, 255), 0.95)
    lf = ell(470, 470, 130, 70, soft=2.0, rot=-0.25)
    img = over(img, pillow(lf, (96, 158, 86), blur=14, k=0.7, amb=0.5, diff=0.5, spec=0.16), lf)
    st2 = stroke([(560, 520), (536, 620), (508, 700)], [9, 8, 7], soft=1.2)
    img = flat(img, st2, (120, 150, 96))
    boot = ell(452, 906, 62, 40, soft=1.8)
    img = over(img, sphere(boot, 452, 900, 62, 40, (242, 126, 66), amb=0.5, diff=0.5, spec=0.2),
               boot)
    rn = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(rn)
    for i in range(90):
        x0, y0 = (i * 137 % 1024), (i * 293 % 1024)
        d.line([(x0 * SS, y0 * SS), ((x0 + 26) * SS, (y0 + 70) * SS)], fill=170, width=int(3 * SS))
    img = flat(img, bl(np.asarray(rn, np.float32) / 255.0, 1.0), (236, 244, 248), 0.45)
    return img


if __name__ == "__main__":
    out = sys.argv[1]
    which = (sys.argv[2] if len(sys.argv) > 2 else "a").lower()
    save(render_a() if which == "a" else render_b(), out)
    print("wrote", out)
