#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v9.2  (Algebraic Least-Squares Shape Fitting)
========================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

Pipeline (algebraic shape fitting → true closed shapes → CAD):
  1.  Load image              (cv2.imread)
  2.  Median Blur             (cv2.medianBlur — salt-and-pepper noise reduction)
  3.  Adaptive Threshold      (cv2.adaptiveThreshold — binarisation, lines=WHITE)
  4.  Morph Clean             (cv2.morphologyEx MORPH_OPEN — speckle removal)
  5.  Canny Edge Detection    (cv2.Canny — thin precise edges)
  6.  Contour Extraction      (cv2.findContours → drop noise specks below a
                                minimum perimeter → cv2.approxPolyDP simplification)
  7.  Shape Classification    (algebraic least-squares: Kasa circle fit + rect fit;
                                rect candidates must have polygon-fill-ratio >= 0.5,
                                rejecting thin dimension-line/arrow artifacts)
  8.  Dedup & Filter          - group near-identical duplicates
                               - resolve same-type "offset" double-edge pairs
                                 (discard the larger of each pair)
                               - discard shapes outside the main part's boundary
                                 (off-board annotation/text artifacts)
                               - cross-type circle-vs-rect offset resolution
                                 (drop a circle that duplicates a rect's footprint)
                               - eliminate shapes smaller than the two largest
                                 circles (both area AND min-dimension checks)
                               - axis-cluster alignment: snap shapes that already
                                 share a centreline/row onto a common axis WITHOUT
                                 collapsing genuinely separate columns/rows
  9.  DXF Export — CLEAN      (one CIRCLE or closed LWPOLYLINE per true shape)
  10. PNG Preview             (side-by-side comparison: denoised Canny — pure
                                black background, no speckle dots — vs clean shapes)
  11. PDF Export              (reportlab — edge image rendered to PDF page)
"""

import sys, os, json, time, traceback, math
from pathlib import Path
from collections import defaultdict

# ── Graceful optional imports ────────────────────────────────────────────────
def _try(fn):
    try: return fn()
    except Exception: return None

cv2           = _try(lambda: __import__("cv2"))
np            = _try(lambda: __import__("numpy"))
ezdxf         = _try(lambda: __import__("ezdxf"))
Image         = _try(lambda: __import__("PIL.Image", fromlist=["Image"]))
reportlab_mod = _try(lambda: __import__("reportlab"))

HAS_CV  = cv2 is not None and np is not None
HAS_DXF = ezdxf is not None
HAS_PIL = Image is not None
HAS_RL  = reportlab_mod is not None

def now_ms(): return int(time.time() * 1000)
def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD IMAGE
# ════════════════════════════════════════════════════════════════════════════

def load_image(image_path):
    if not HAS_CV:
        raise RuntimeError("OpenCV (cv2) is not installed.")
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"cv2.imread returned None for: {image_path}")

    dpi = 96.0
    if HAS_PIL:
        try:
            pil  = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (96, 96))
            dpi  = float(xdpi[0]) if xdpi and xdpi[0] > 1 else 96.0
        except Exception:
            pass

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = bgr.shape[:2]
    return bgr, gray, dpi, img_w, img_h


# ════════════════════════════════════════════════════════════════════════════
# STEPS 2-6 — STANDARD CV PRE-PROCESSING (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def median_blur(gray, ksize=5):
    if ksize % 2 == 0: ksize += 1
    return cv2.medianBlur(gray, ksize)

def adaptive_threshold_binarize(blurred):
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )

def morph_clean(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

def canny_edges(cleaned, low_threshold=30, high_threshold=100):
    return cv2.Canny(cleaned, low_threshold, high_threshold)

def extract_simplified_contours(edges, epsilon_factor=0.5, min_perimeter=9.0):
    """
    Extract and simplify contours from a Canny edge map.

    min_perimeter: contours with arcLength below this value are discarded
    as speckle/JPEG-compression noise (isolated dots scattered across the
    background that do NOT belong to any real drawn shape).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    simplified = []
    for cnt in contours:
        if len(cnt) < 2: continue
        arc = cv2.arcLength(cnt, closed=False)
        if arc < min_perimeter:
            continue  # noise speck — discard entirely
        epsilon = epsilon_factor * arc / max(len(cnt), 1)
        epsilon = max(epsilon, 0.3)
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=False)
        if len(approx) >= 2:
            simplified.append(approx)
    return simplified


def build_clean_edge_mask(edges, min_perimeter=9.0):
    """
    Produce a denoised version of the Canny edge map with all small
    speckle/dot contours removed and the background fully black.
    Used for the comparison-PNG preview so it never shows stray white
    dots — only the real traced edges of the drawing.
    """
    if not HAS_CV or not np:
        return edges

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(edges)
    for cnt in contours:
        if len(cnt) < 2:
            continue
        if cv2.arcLength(cnt, closed=False) < min_perimeter:
            continue  # speck — leave as black background
        cv2.drawContours(mask, [cnt], -1, 255, 1)
    return mask


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — ALGEBRAIC LEAST-SQUARES SHAPE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def _fit_circle_algebraic(pts_xy):
    """
    Kasa algebraic circle fit.
    Solves: x² + y² = a·x + b·y + c  (linear system)
    Returns (cx, cy, r, rms_residual).
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    A    = np.column_stack([x, y, np.ones(len(x))])
    b_   = x**2 + y**2
    res, _, _, _ = np.linalg.lstsq(A, b_, rcond=None)
    cx = res[0] / 2.0
    cy = res[1] / 2.0
    r  = math.sqrt(abs(res[2] + cx**2 + cy**2))
    dists    = np.sqrt((x - cx)**2 + (y - cy)**2)
    rms      = float(np.sqrt(((dists - r)**2).mean()))
    return cx, cy, r, rms

def _fit_rect_algebraic(pts_xy):
    """
    Axis-aligned rectangle: min/max extents of point cloud.
    Returns (cx, cy, w, h).
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    cx   = (float(x.min()) + float(x.max())) / 2.0
    cy   = (float(y.min()) + float(y.max())) / 2.0
    w    = float(x.max() - x.min())
    h    = float(y.max() - y.min())
    return cx, cy, w, h

def _polygon_area(pts_xy):
    """Shoelace formula — signed area of a (possibly open) polygon path."""
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _classify_contour(pts_xy, min_pts_for_circle=12, circle_rms_tol=0.12,
                       min_rect_fill_ratio=0.5):
    """
    Classify a contour as 'circle' or 'rect' using algebraic LS.

    Circle test: aspect ratio > 0.6 and enough points, with the Kasa fit's
    RMS radial residual small relative to the fitted radius
    (rel_err < circle_rms_tol). NOTE: a polygon-area-vs-circle-area check
    is deliberately NOT used here — Canny often traces a thin ring as a
    single "annulus" loop whose shoelace area is tiny even for a perfectly
    correct circle, which would cause false rejections.

    Rectangle test: the contour's enclosed polygon area (shoelace) must
    cover at least `min_rect_fill_ratio` of its bounding-box area. A true
    rectangle outline (4 corners) has fill ≈ 1.0. Thin dimension-line /
    arrow / leader-line artifacts have fill « 0.5 and are rejected
    entirely (return None) — they are not real geometry.

    Returns a shape dict, or None if the contour is not a real shape.
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    w    = float(x.max() - x.min())
    h    = float(y.max() - y.min())
    if w < 1e-6 or h < 1e-6:
        return None

    aspect = min(w, h) / max(w, h)

    # Try circle fit when aspect ratio is reasonably round and enough points
    if aspect > 0.60 and len(pts_xy) >= min_pts_for_circle:
        try:
            cx, cy, r, rms = _fit_circle_algebraic(pts_xy)
            rel_err = rms / (r + 1e-9)
            if rel_err < circle_rms_tol and r > 1e-6:
                return {
                    'type': 'circle',
                    'cx': float(cx), 'cy': float(cy), 'r': float(r),
                    'err': float(rel_err),
                    'area': math.pi * r * r,
                    'w': w, 'h': h,
                }
        except Exception:
            pass

    # Fall back to rectangle — but reject thin/hollow artifacts (dimension
    # lines, arrows, leader lines) whose enclosed area is much smaller
    # than their bounding box.
    poly_area  = _polygon_area(pts_xy)
    bbox_area  = w * h
    fill_ratio = poly_area / bbox_area if bbox_area > 0 else 0.0
    if fill_ratio < min_rect_fill_ratio:
        return None

    cx_b = (float(x.min()) + float(x.max())) / 2.0
    cy_b = (float(y.min()) + float(y.max())) / 2.0
    return {
        'type': 'rect',
        'cx': cx_b, 'cy': cy_b, 'w': w, 'h': h,
        'area': w * h,
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — DEDUPLICATION + SIZE FILTERING
# ════════════════════════════════════════════════════════════════════════════

def _dedup_circles(circs, center_tol=15.0, radius_rel_tol=0.15):
    """
    Group circles by proximity of centre and similar radius.
    Keep the one with lowest fit error per group.
    """
    used   = [False] * len(circs)
    result = []
    for i, s in enumerate(circs):
        if used[i]: continue
        grp = [s]; used[i] = True
        for j, s2 in enumerate(circs):
            if used[j]: continue
            dc = math.hypot(s['cx'] - s2['cx'], s['cy'] - s2['cy'])
            dr = abs(s['r'] - s2['r']) / (max(s['r'], s2['r']) + 1e-9)
            if dc < center_tol and dr < radius_rel_tol:
                grp.append(s2); used[j] = True
        best = min(grp, key=lambda x: x.get('err', 1.0))
        result.append(best)
    return result

def _dedup_rects(rects, center_tol=20.0, dim_tol=20.0):
    """
    Group rectangles by similar centre and dimensions.
    Keep the largest (outer) one per group.
    """
    used   = [False] * len(rects)
    result = []
    for i, s in enumerate(rects):
        if used[i]: continue
        grp = [s]; used[i] = True
        for j, s2 in enumerate(rects):
            if used[j]: continue
            dc = math.hypot(s['cx'] - s2['cx'], s['cy'] - s2['cy'])
            dw = abs(s['w'] - s2['w'])
            dh = abs(s['h'] - s2['h'])
            if dc < center_tol and dw < dim_tol and dh < dim_tol:
                grp.append(s2); used[j] = True
        best = max(grp, key=lambda x: x['w'] * x['h'])
        result.append(best)
    return result

def _filter_outside_main_boundary(rects, circles, margin_frac=0.03):
    """
    Determine the "main" shape — the largest rectangle by area, or the
    largest circle if no rectangles were detected — and discard any OTHER
    detected shape whose centre falls outside that main shape's bounding
    box (expanded by margin_frac on each side).

    This removes dimension-line / arrow / annotation-text artifacts that
    sit outside the actual part outline (e.g. "900mm" labels and their
    dimension arrows drawn above/beside a sketch), while keeping every
    real feature that lies within the part's footprint.
    """
    if not rects and not circles:
        return rects, circles

    if rects:
        main = max(rects, key=lambda r: r['w'] * r['h'])
        x0, x1 = main['cx'] - main['w'] / 2.0, main['cx'] + main['w'] / 2.0
        y0, y1 = main['cy'] - main['h'] / 2.0, main['cy'] + main['h'] / 2.0
    else:
        main = max(circles, key=lambda c: c['r'])
        x0, x1 = main['cx'] - main['r'], main['cx'] + main['r']
        y0, y1 = main['cy'] - main['r'], main['cy'] + main['r']

    mx = margin_frac * (x1 - x0)
    my = margin_frac * (y1 - y0)
    x0 -= mx; x1 += mx
    y0 -= my; y1 += my

    def _inside(s):
        return (x0 <= s['cx'] <= x1) and (y0 <= s['cy'] <= y1)

    rects_f   = [r for r in rects   if r is main or _inside(r)]
    circles_f = [c for c in circles if c is main or _inside(c)]
    return rects_f, circles_f


def _resolve_offset_duplicates(shapes, kind):
    """
    Detect pairs of same-type shapes that share (almost) the same centre but
    differ slightly in size — concentric "offset" duplicates produced when
    Canny traces both the inner and outer edge of a single drawn line.

    For every such pair, the LARGER shape is discarded and the SMALLER
    (inner) shape is kept, leaving exactly one entity per true edge.

    kind: 'circle' or 'rect'
    """
    n    = len(shapes)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue

            a, b = shapes[i], shapes[j]
            dc = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])

            if kind == 'circle':
                ra, rb = a['r'], b['r']
                center_tol = max(10.0, 0.05 * max(ra, rb))
                size_rel   = abs(ra - rb) / max(ra, rb, 1e-9)
                larger_is_a = ra >= rb
            else:  # rect
                size_a = max(a['w'], a['h'])
                size_b = max(b['w'], b['h'])
                center_tol = max(15.0, 0.04 * max(size_a, size_b))
                w_rel = abs(a['w'] - b['w']) / max(a['w'], b['w'], 1e-9)
                h_rel = abs(a['h'] - b['h']) / max(a['h'], b['h'], 1e-9)
                size_rel = max(w_rel, h_rel)
                larger_is_a = (a['w'] * a['h']) >= (b['w'] * b['h'])

            # Same centre + similar size  ⇒  offset duplicate of one edge
            if dc <= center_tol and size_rel <= 0.20:
                if larger_is_a:
                    keep[i] = False
                else:
                    keep[j] = False
                    # 'a' (i) stays — re-check it against subsequent shapes
                if not keep[i]:
                    break  # i was removed; move to next i

    return [s for k, s in zip(keep, shapes) if k]


def _resolve_cross_type_offsets(circles, rects):
    """
    Eliminate spurious CIRCLE entities that are actually offset/duplicate
    traces of a RECT contour — i.e. a circle whose centre coincides with a
    rectangle's centre and whose diameter is comparable to (roughly the
    same order of magnitude as) that rectangle's own dimensions.

    This catches algebraic mis-fits where a near-square or large
    rectangular contour (often the outer-boundary double-edge, or a
    dimension-line artifact) happens to satisfy the circle RMS test but
    geometrically represents the SAME edge as an already-detected
    rectangle. The rectangle is kept; the offending circle is discarded.

    A genuinely small circular feature (a real hole/bore whose diameter is
    much smaller than the rectangle it sits inside) is left untouched.
    """
    if not circles or not rects:
        return circles

    keep = [True] * len(circles)
    for i, c in enumerate(circles):
        diameter = 2.0 * c['r']
        for r in rects:
            dc = math.hypot(c['cx'] - r['cx'], c['cy'] - r['cy'])
            center_tol = max(30.0, 0.06 * max(r['w'], r['h']))
            if dc > center_tol:
                continue
            # "Comparable size" ⇒ the circle roughly spans the same extent
            # as the rectangle (within 0.5x–1.5x of either dimension) —
            # this is the signature of a misclassified duplicate edge,
            # NOT a small bore hole drilled inside a larger panel.
            lo = 0.5 * min(r['w'], r['h'])
            hi = 1.5 * max(r['w'], r['h'])
            if lo <= diameter <= hi:
                keep[i] = False
                break

    return [c for k, c in zip(keep, circles) if k]


def _cluster_align_axis(shapes, key, tol):
    """
    Group shapes whose `key` coordinate (e.g. 'cx' or 'cy') is within `tol`
    of each other (chained proximity) and snap every member of a group to
    the group's mean value.

    This preserves multi-column / multi-row layouts (shapes in different
    columns keep their own distinct X position) while still aligning
    shapes that already share an approximate centreline — giving the
    "perpendicular & centred" look for single-column designs without
    collapsing genuinely separate shapes onto one axis.
    """
    n = len(shapes)
    if n <= 1:
        return shapes

    order = sorted(range(n), key=lambda i: shapes[i][key])
    groups = [[order[0]]]
    for idx in order[1:]:
        if abs(shapes[idx][key] - shapes[groups[-1][-1]][key]) <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    result = list(shapes)
    for grp in groups:
        mean_val = float(np.mean([shapes[i][key] for i in grp]))
        for i in grp:
            result[i] = {**result[i], key: mean_val}
    return result


def build_clean_shapes(simplified_contours, img_w, img_h):
    """
    Full algebraic LS pipeline:
      a. Classify every contour as circle or rect (with strict circle
         area-ratio check to reject rect-like contours).
      b. Deduplicate overlapping/offset copies (near-identical groups).
      c. Resolve remaining "offset" pairs — same centre, slightly different
         size — by discarding the larger of each pair (same-type AND
         cross-type circle-vs-rect).
      d. Filter shapes smaller than the two largest circles, AND require
         each rectangle's smallest dimension to meet that same threshold
         (eliminates thin dimension-line / annotation artifacts).
      e. Snap shapes that already share an approximate centreline/row onto
         a common axis — WITHOUT collapsing genuinely separate columns or
         rows (preserves true layout spacing).
    Returns list of shape dicts {'type', 'cx', 'cy', 'r'|'w'+'h'}.
    """
    raw_circles = []
    raw_rects   = []

    for cnt in simplified_contours:
        pts_xy = np.array([[int(p[0][0]), int(p[0][1])] for p in cnt], dtype=float)
        if len(pts_xy) < 2:
            continue
        shape = _classify_contour(pts_xy)
        if shape is None:
            continue
        if shape['type'] == 'circle':
            raw_circles.append(shape)
        else:
            raw_rects.append(shape)

    # Step 1: group near-identical duplicates (tight tolerance)
    circles_dedup = _dedup_circles(raw_circles)
    rects_dedup   = _dedup_rects(raw_rects)

    # Step 2: resolve remaining concentric "offset" pairs — for every pair
    # of same-type shapes sharing a centre but differing slightly in size
    # (Canny double-edge artifacts), discard the LARGER of the pair so only
    # one entity per true edge survives.
    circles_dedup = _resolve_offset_duplicates(circles_dedup, 'circle')
    rects_dedup   = _resolve_offset_duplicates(rects_dedup, 'rect')

    # Step 2a: discard any shape whose centre lies outside the main part's
    # boundary (e.g. dimension-line / annotation-text artifacts drawn
    # beside or above the actual outline).
    rects_dedup, circles_dedup = _filter_outside_main_boundary(rects_dedup, circles_dedup)

    # Step 2b: cross-type offset resolution — eliminate any circle that is
    # really an offset/duplicate trace of a rectangle's outline (same
    # centre, comparable overall size).
    circles_dedup = _resolve_cross_type_offsets(circles_dedup, rects_dedup)

    # Sort circles by radius descending
    circles_sorted = sorted(circles_dedup, key=lambda c: c['r'], reverse=True)

    # Minimum size = smaller of two largest circles (or single largest if only one)
    if len(circles_sorted) >= 2:
        min_r = circles_sorted[1]['r']
    elif len(circles_sorted) == 1:
        min_r = circles_sorted[0]['r']
    else:
        min_r = 0.0

    min_area_thresh = math.pi * min_r * min_r * 0.5  # generous lower bound
    min_dim_thresh  = min_r  # smallest rect dimension must reach this

    # Filter: keep circles whose radius >= 90% of min_r
    circles_final = [c for c in circles_sorted if c['r'] >= min_r * 0.90]

    # Filter rects:
    #  - area must clear the minimum-circle-derived area threshold, AND
    #  - BOTH dimensions must clear min_dim_thresh — this removes thin
    #    elongated artifacts (dimension lines / leader lines / annotation
    #    strokes) that can otherwise have large bounding-box area despite
    #    being only a few px wide.
    rects_final = [
        r for r in rects_dedup
        if r['w'] * r['h'] >= min_area_thresh
        and min(r['w'], r['h']) >= min_dim_thresh * 0.9
    ]

    # Final safety pass: resolve any remaining offset pairs across the
    # filtered sets (in case the size filter let a near-duplicate through
    # that wasn't caught above due to ordering), including cross-type.
    circles_final = _resolve_offset_duplicates(circles_final, 'circle')
    rects_final   = _resolve_offset_duplicates(rects_final, 'rect')
    circles_final = _resolve_cross_type_offsets(circles_final, rects_final)

    # ── Axis alignment: snap shapes that already share an approximate
    # centreline/row onto a common axis, preserving genuinely distinct
    # columns/rows (multi-shape layouts) instead of collapsing everything
    # onto one master centre line.
    all_shapes = rects_final + circles_final
    cx_tol = max(15.0, 0.02 * img_w)
    cy_tol = max(15.0, 0.02 * img_h)
    all_shapes = _cluster_align_axis(all_shapes, 'cx', cx_tol)
    all_shapes = _cluster_align_axis(all_shapes, 'cy', cy_tol)

    n_rects_final = len(rects_final)
    rects_final   = all_shapes[:n_rects_final]
    circles_final = all_shapes[n_rects_final:]
    final_shapes  = all_shapes

    # Reference centre-X (for reporting/analysis only — no longer forced
    # onto every shape).
    if rects_final:
        largest_rect = max(rects_final, key=lambda r: r['w'] * r['h'])
        master_cx    = largest_rect['cx']
    elif circles_final:
        master_cx = float(np.mean([c['cx'] for c in circles_final]))
    else:
        master_cx = img_w / 2.0

    return final_shapes, circles_final, rects_final, master_cx


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — CLEAN DXF EXPORT (one entity per shape)
# ════════════════════════════════════════════════════════════════════════════

def build_clean_dxf(final_shapes, img_w, img_h, out_path):
    """
    Write a DXF containing only true closed shapes:
      - CIRCLE entities for circles
      - Closed LWPOLYLINE (rectangle) entities for rects
    One entity per individual shape — no raw contour polylines.
    """
    if not HAS_DXF or not final_shapes:
        return None, 0, 0

    from ezdxf import units as ezunits

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"]   = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"]   = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"]   = (0.0, 0.0)
    doc.header["$LIMMAX"]   = (float(img_w), float(img_h))

    msp = doc.modelspace()
    doc.layers.new("SHAPES",  dxfattribs={"color": 7,  "linetype": "CONTINUOUS"})
    doc.layers.new("CIRCLES", dxfattribs={"color": 1,  "linetype": "CONTINUOUS"})
    doc.layers.new("RECTS",   dxfattribs={"color": 3,  "linetype": "CONTINUOUS"})

    entity_count = 0

    for s in final_shapes:
        if s['type'] == 'circle':
            # Native DXF CIRCLE entity — not a polyline approximation
            msp.add_circle(
                (s['cx'], s['cy'], 0.0),
                s['r'],
                dxfattribs={"layer": "CIRCLES", "color": 256}
            )
            entity_count += 1

        elif s['type'] == 'rect':
            # Closed LWPOLYLINE rectangle (4 corners + close flag)
            x0, y0 = s['cx'] - s['w'] / 2.0, s['cy'] - s['h'] / 2.0
            x1, y1 = s['cx'] + s['w'] / 2.0, s['cy'] + s['h'] / 2.0
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly = msp.add_lwpolyline(
                pts, format="xy",
                dxfattribs={"layer": "RECTS", "color": 256}
            )
            poly.close(True)
            entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════
# STEP 10 — PNG PREVIEW (side-by-side comparison)
# ════════════════════════════════════════════════════════════════════════════

def build_comparison_png(edges, final_shapes, img_w, img_h, out_path, min_perimeter=18.0):
    """
    Two-panel PNG:
      Left  — denoised Canny edge result (white edges on pure-black bg,
              all isolated speckle/dot noise removed)
      Right — Clean fitted shapes on dark bg (circles=red, rects=green)
    """
    if not HAS_CV or not np:
        return False

    try:
        # Left panel: denoised canny preview — background is fully black,
        # only contours above the noise-perimeter threshold are drawn.
        clean_edges = build_clean_edge_mask(edges, min_perimeter=min_perimeter)
        left = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        left[clean_edges > 0] = (255, 255, 255)

        # Right panel: clean shapes
        right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        right[:] = (15, 12, 10)

        for s in final_shapes:
            cx_px = int(round(s['cx']))
            cy_px = int(round(s['cy']))

            if s['type'] == 'circle':
                r_px = int(round(s['r']))
                cv2.circle(right, (cx_px, cy_px), r_px, (80, 80, 220), 2, cv2.LINE_AA)

            elif s['type'] == 'rect':
                x0 = int(round(s['cx'] - s['w'] / 2.0))
                y0 = int(round(s['cy'] - s['h'] / 2.0))
                x1 = int(round(s['cx'] + s['w'] / 2.0))
                y1 = int(round(s['cy'] + s['h'] / 2.0))
                cv2.rectangle(right, (x0, y0), (x1, y1), (80, 200, 80), 2, cv2.LINE_AA)

        # Add labels
        font    = cv2.FONT_HERSHEY_SIMPLEX
        n_circ  = sum(1 for s in final_shapes if s['type'] == 'circle')
        n_rect  = sum(1 for s in final_shapes if s['type'] == 'rect')
        cv2.putText(left,  "Canny Edge Detection",
                    (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(right,
                    f"LS Shapes: {n_rect} rect(s) + {n_circ} circle(s)",
                    (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Combine side-by-side with a thin separator
        sep   = np.full((img_h, 4, 3), 40, dtype=np.uint8)
        panel = np.concatenate([left, sep, right], axis=1)

        ok = cv2.imwrite(str(out_path), panel)
        return bool(ok and out_path.exists())
    except Exception as e:
        sys.stderr.write(f"PNG preview error: {e}\n")
        return False


# ════════════════════════════════════════════════════════════════════════════
# STEP 11 — PDF EXPORT (unchanged from v8)
# ════════════════════════════════════════════════════════════════════════════

def export_pdf(edges, out_path, orig_bgr=None):
    if not HAS_RL:
        return False

    import tempfile, os as _os
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    try:
        page_w, page_h = landscape(A4)
        c = rl_canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

        margin = 30
        col_w  = (page_w - margin * 3) / 2
        col_h  = page_h - margin * 2 - 40

        c.setFillColorRGB(0.04, 0.05, 0.06)
        c.rect(0, page_h - 36, page_w, 36, fill=1, stroke=0)
        c.setFillColorRGB(0.9, 0.91, 0.93)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, page_h - 24, "SheetForge — Algebraic LS Shape Fitting Preview")
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.5, 0.55, 0.6)
        from datetime import datetime
        c.drawRightString(page_w - margin, page_h - 24,
                          f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

        def _arr_to_reader(arr):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, arr)
            return ImageReader(tmp.name), tmp.name

        tmp_files = []

        left_x = margin
        if orig_bgr is not None:
            reader, tname = _arr_to_reader(orig_bgr)
            tmp_files.append(tname)
            _draw_panel(c, reader, left_x, margin, col_w, col_h, "Original Image")
        else:
            c.setFillColorRGB(0.08, 0.1, 0.13)
            c.roundRect(left_x, margin, col_w, col_h, 6, fill=1, stroke=0)

        right_x   = margin * 2 + col_w
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        reader, tname = _arr_to_reader(edges_bgr)
        tmp_files.append(tname)
        _draw_panel(c, reader, right_x, margin, col_w, col_h, "Canny Edge Detection")

        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 12,
                            "SheetForge v9.0  •  Algebraic Least-Squares Shape Fitting  •  Clean DXF")
        c.save()

        for f in tmp_files:
            try: _os.unlink(f)
            except Exception: pass

        return True
    except Exception as e:
        sys.stderr.write(f"PDF export error: {e}\n{traceback.format_exc()}\n")
        return False


def _draw_panel(c, img_reader, x, y, w, h, title):
    title_h = 22
    img_h   = h - title_h

    c.setFillColorRGB(0.08, 0.1, 0.13)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.12, 0.15, 0.2)
    c.roundRect(x, y + img_h, w, title_h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.55, 0.65, 0.85)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + w / 2, y + img_h + 7, title)

    pad = 8
    c.drawImage(img_reader, x + pad, y + title_h + pad,
                width=w - pad * 2, height=img_h - pad * 2,
                preserveAspectRatio=True, anchor="c", mask="auto")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    opts       = {}
    if len(sys.argv) > 2:
        try: opts = json.loads(sys.argv[2])
        except Exception: pass

    blur_ksize     = int(opts.get("blurKsize",    5))
    canny_low      = int(opts.get("cannyLow",    30))
    canny_high     = int(opts.get("cannyHigh",  100))
    epsilon_factor = float(opts.get("epsilonFactor", 0.5))
    # Contours with arcLength below this are speckle/JPEG-noise dots and are
    # discarded entirely — both from shape detection AND the preview image.
    min_perimeter  = float(opts.get("minPerimeter", 18.0))

    steps = []

    # ── STEP 1: Load ──────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record("CV-1: Load Image", f"{img_w}×{img_h}px  DPI={dpi:.0f}", t0))

    # ── STEP 2: Median Blur ───────────────────────────────────────────────
    t0 = now_ms()
    blurred = median_blur(gray, ksize=blur_ksize)
    steps.append(step_record(f"CV-2: Median Blur (ksize={blur_ksize})", "Noise reduced", t0))

    # ── STEP 3: Adaptive Threshold ───────────────────────────────────────
    t0 = now_ms()
    binary   = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record("CV-3: Adaptive Threshold", f"{white_px} white px", t0))

    # ── STEP 4: Morph Open ───────────────────────────────────────────────
    t0 = now_ms()
    cleaned    = morph_clean(binary)
    cleaned_px = int(np.count_nonzero(cleaned))
    steps.append(step_record("CV-4: MORPH_OPEN", f"{white_px - cleaned_px} speckles removed", t0))

    # ── STEP 5: Canny ────────────────────────────────────────────────────
    t0 = now_ms()
    edges   = canny_edges(cleaned, canny_low, canny_high)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(f"CV-5: Canny (lo={canny_low}, hi={canny_high})", f"{edge_px} edge px", t0))

    # ── STEP 6: Contour extraction ───────────────────────────────────────
    t0 = now_ms()
    simplified_contours = extract_simplified_contours(edges, epsilon_factor, min_perimeter)
    total_pts = sum(len(c) for c in simplified_contours)
    steps.append(step_record(
        f"CV-6: findContours + approxPolyDP (ε={epsilon_factor}, min-perimeter={min_perimeter}px)",
        f"{len(simplified_contours)} contours  |  {total_pts} vertices  (noise specks discarded)", t0))

    # ── STEP 7: Algebraic LS shape classification ─────────────────────────
    t0 = now_ms()
    final_shapes, circles_f, rects_f, master_cx = build_clean_shapes(
        simplified_contours, img_w, img_h)
    n_circles = len(circles_f)
    n_rects   = len(rects_f)
    steps.append(step_record(
        "LS-7: Algebraic Least-Squares Shape Fitting (Kasa circle + rect)",
        f"{len(simplified_contours)} raw contours → {n_circles} circle(s) + {n_rects} rect(s) detected",
        t0))

    # ── STEP 8: Deduplication + filter ────────────────────────────────────
    t0 = now_ms()
    n_final = len(final_shapes)
    n_circ_final = sum(1 for s in final_shapes if s['type'] == 'circle')
    n_rect_final = sum(1 for s in final_shapes if s['type'] == 'rect')
    steps.append(step_record(
        "LS-8: Dedup, offset-pair resolution (incl. cross-type), size filter & axis alignment",
        f"{n_final} final shapes  |  {n_rect_final} rect(s)  |  {n_circ_final} circle(s)  |  ref CX={master_cx:.1f}",
        t0))

    # ── Output dir ────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str       = int(time.time())
    dxf_name     = f"design_{ts_str}.dxf"
    pdf_name     = f"design_{ts_str}.pdf"
    png_name     = f"preview_{ts_str}.png"

    dxf_path     = server_out_dir / dxf_name
    pdf_path     = server_out_dir / pdf_name
    png_path     = server_out_dir / png_name

    # ── STEP 9: Clean DXF export ──────────────────────────────────────────
    t0 = now_ms()
    _, entity_count, dxf_size = build_clean_dxf(final_shapes, img_w, img_h, dxf_path)

    # Read DXF content for frontend blob download (up to 200 KB)
    dxf_content_str = ""
    if dxf_size and dxf_size > 0:
        try:
            with open(dxf_path, encoding="utf-8", errors="replace") as f:
                dxf_content_str = f.read(200_000)
        except Exception:
            pass

    steps.append(step_record(
        "DXF-9: Clean export (CIRCLE + closed LWPOLYLINE, one entity per shape)",
        f"{entity_count} entities  |  {dxf_size // 1024 if dxf_size else 0} KB", t0))

    # ── STEP 10: PNG comparison preview ───────────────────────────────────
    t0 = now_ms()
    png_ok   = build_comparison_png(edges, final_shapes, img_w, img_h, png_path, min_perimeter)
    png_size = png_path.stat().st_size if png_ok and png_path.exists() else 0
    steps.append(step_record(
        "PNG-10: Side-by-side comparison preview (Canny vs LS shapes)",
        f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # ── STEP 11: PDF export ───────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-11: Export edge preview", "OK" if pdf_ok else "FAILED", t0))

    # ── Analysis summary ──────────────────────────────────────────────────
    analysis = {
        "width"          : float(img_w),
        "height"         : float(img_h),
        "dpi"            : dpi,
        "edgePixels"     : edge_px,
        "edges"          : entity_count,      # final DXF entity count
        "contours"       : len(simplified_contours),
        "mergedContours" : n_final,
        "closedContours" : n_final,           # all output shapes are closed
        "totalVertices"  : total_pts,
        "blurKsize"      : blur_ksize,
        "cannyLow"       : canny_low,
        "cannyHigh"      : canny_high,
        "epsilonFactor"  : epsilon_factor,
        "imgW"           : img_w,
        "imgH"           : img_h,
        "scaleMmPerDu"   : round(25.4 / dpi, 4),
        "coordSystem"    : "origin=top-left px, Y-down (image convention)",
        # Shape-fitting specific
        "circlesDetected": n_circ_final,
        "rectsDetected"  : n_rect_final,
        "masterCx"       : round(master_cx, 2),
        "shapeSummary"   : (
            f"{n_rect_final} rectangle(s), {n_circ_final} circle(s)"
            f" — algebraic LS fitted, offset-pairs resolved (incl. cross-type),"
            f" off-board annotations removed, axis-aligned"
        ),
    }

    print(json.dumps({
        "steps"        : steps,
        "analysis"     : analysis,
        "dwg": {
            "entities"        : entity_count,
            "fileSize"        : dxf_size or 0,
            "filename"        : dxf_name if dxf_size else "",
            "dxfAbsPath"      : str(dxf_path) if dxf_size else "",
            "pdfFilename"     : pdf_name if pdf_ok else "",
            "edgePngFilename" : png_name if png_ok else "",
            "edgePngPath"     : str(png_path) if png_ok else "",
            "gcodeFiles"      : {},
            "gcodeFilePaths"  : {},
        },
        "dxfContent"   : dxf_content_str,    # full DXF text for blob download
        "dxfAvailable" : bool(dxf_size and dxf_size > 0),
        "pdfAvailable" : pdf_ok,
        "pngAvailable" : png_ok,
        "gcodeAvailable": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error"    : str(e),
            "traceback": traceback.format_exc(),
            "steps"    : [],
            "analysis" : {},
            "dwg"      : {"entities": 0, "fileSize": 0},
            "dxfContent": "",
        }))
        sys.exit(1)
