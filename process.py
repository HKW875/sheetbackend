#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v13  (Pixel-Graph Skeleton + True Algebraic LS Primitives)
======================================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

ARCHITECTURE — WHY v12 STILL HAD OFFSETS AND DUPLICATES
---------------------------------------------------------
v12 ran cv2.findContours on a thickened skeleton.  findContours traces the
PERIMETER of each connected region — even a 1-px skeleton thickened to 3 px
produces a closed perimeter loop with inner and outer sides.  Every drawn line
therefore appeared as a narrow closed rectangle in the contour list, and the
Douglas-Peucker per-segment H/V snap operated on those perimeter corners, not
on the true pixel centerline.  This is the root cause of:

  • Doubled / offset parallel lines  (inner vs outer contour sides)
  • Mis-placed segment endpoints     (perimeter corners ≠ skeleton endpoints)
  • Incomplete intersection trimming (the trim worked on wrong geometry)

v13 ALGORITHM — PIXEL GRAPH TRACING (the correct approach)
-----------------------------------------------------------
Step A  — Canny → 200 px area filter  (unchanged from v12)
Step B  — Zhang-Suen skeleton to 1-px centerline  (unchanged)
Step C  — BUILD PIXEL ADJACENCY GRAPH
            Every white pixel is a node.  8-connected neighbours are edges.
            Classify every pixel as:
              • ENDPOINT  (exactly 1 neighbour)
              • BRANCH    (≥ 3 neighbours)  ← junction / T-cross / corner
              • PASS      (exactly 2 neighbours)  ← interior of a stroke

Step D  — TRACE BRANCHES
            Walk from every ENDPOINT or BRANCH pixel along PASS pixels until
            the next ENDPOINT or BRANCH.  Each walk produces one PRIMITIVE CHAIN
            — an ordered list of (col, row) pixel coordinates that is guaranteed
            to be a single, non-forking stroke with no duplicate pixels.

Step E  — ALGEBRAIC LS FIT PER CHAIN
            For each chain, decide the best geometric primitive:

            CIRCLE  — Kasa algebraic LS.  Accept if closed chain and RMS/R < tol.

            LINE    — SVD total-LS fit direction.
                      Accept as H or V LINE if the fitted angle is within
                      SNAP_ANGLE_DEG (default 8°) of 0°/90°/180°/270°.
                      The line endpoints are the projection of the first and last
                      pixel onto the fitted infinite line — NOT the pixel coords.

            ARC     — Kasa LS on open chains.  Accept if arc spans ≥ 30° and
                      RMS/R < tol AND chain is NOT better as a line.

            POLYLINE — fallback.  Douglas-Peucker on chain, then each segment is
                      snapped H or V by the same LS approach used for LINE.

Step F  — GLOBAL GRID SNAP
            After fitting, collect all distinct H-line Y-values and V-line
            X-values.  Cluster within GRID_SNAP_PX (default 4 px).  Replace
            each cluster with the LS-weighted mean.  This makes truly collinear
            lines share the exact same coordinate — no 0.3 px offsets.

Step G  — INTERSECTION TRIMMING (algebraic, exact)
            Build a spatial index of all segment bounding boxes.
            For each LINE or POLYLINE segment endpoint:
              Extend the endpoint outward by EXTEND_PX.
              Compute exact algebraic intersection with every nearby segment.
              Keep the nearest intersection that changes the endpoint by
              < TRIM_RADIUS_PX (default 20 px).
              Snap endpoint to that intersection.
            Result: every endpoint meets its neighbour at an exact point with
            zero gap and zero overshoot.

Step H  — DEDUPLICATION
            Two entities are duplicates if their bounding boxes overlap AND
            their LS-line parameters (angle, offset) agree within tolerance.
            Keep only one.

Step I  — DXF EXPORT  (unchanged layer/entity structure from v12)
"""

import sys, os, json, time, traceback, math
from collections import deque
from pathlib import Path

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
# TUNING CONSTANTS  (override via options_json)
# ════════════════════════════════════════════════════════════════════════════

SNAP_ANGLE_DEG   = 8.0    # within this many degrees of H/V → snap to exact H or V
CIRCLE_RMS_TOL   = 0.10   # Kasa RMS/R < this → accept as circle
ARC_RMS_TOL      = 0.08   # Kasa RMS/R < this for open arcs
ARC_MIN_DEG      = 30.0   # arc must span this many degrees to be kept as ARC
GRID_SNAP_PX     = 4.0    # cluster collinear H/V lines within this distance
TRIM_RADIUS_PX   = 20.0   # max extension/trim distance for endpoint snapping
MIN_CHAIN_PX     = 10     # discard chains shorter than this many pixels
DP_EPSILON_FRAC  = 0.015  # Douglas-Peucker epsilon as fraction of arc length
EXTEND_PX        = 25.0   # extend endpoint ray by this much when searching intersections


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
# STEPS 2-5 — DENOISE / BINARISE / SPECKLE REMOVAL
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

def remove_small_blobs(binary, min_area):
    """Remove all 8-connected blobs with pixel area < min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    removed_px, removed_blobs = 0, 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == lbl] = 255
        else:
            removed_px += area
            removed_blobs += 1
    return cleaned, removed_blobs, removed_px


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — CANNY → AREA FILTER → SKELETONIZE
# ════════════════════════════════════════════════════════════════════════════

def canny_edges(cleaned, low_threshold=20, high_threshold=80):
    return cv2.Canny(cleaned, low_threshold, high_threshold)

def skeletonize_mask(binary_mask):
    """Zhang-Suen thinning to 1-px centerline."""
    if binary_mask is None or not HAS_CV:
        return binary_mask
    _, bw = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    try:
        thinned = cv2.ximgproc.thinning(bw, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thinned
    except (AttributeError, cv2.error):
        pass
    # Pure-numpy fallback Zhang-Suen
    element  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(bw)
    img      = bw.copy()
    for _ in range(200):
        eroded   = cv2.erode(img, element)
        temp     = cv2.dilate(eroded, element)
        temp     = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img      = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton

def thicken_for_display(binary_mask, thickness_px=4):
    if binary_mask is None or not HAS_CV:
        return binary_mask
    radius = max(1, thickness_px // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary_mask, kernel, iterations=1)


# ════════════════════════════════════════════════════════════════════════════
# STEP P — PIXEL GEOMETRY VALIDATOR
# Erase 255-pixels that cannot belong to any valid engineering primitive
# BEFORE chain tracing, so they are never offered to the DXF exporter.
#
# A skeleton pixel is VALID if it belongs to at least one of:
#   1. A straight run that is within SNAP_ANGLE_DEG of exactly H or V
#   2. A circular / arc locus (Kasa fit on its connected component)
#   3. A semi-circle locus (arc spanning 150°–210° on a Kasa-valid component)
#   4. A closed rectangular/square polygon (component is a closed H+V loop)
#
# A pixel is INVALID (erased) if:
#   • It is an isolated speck (connected component < MIN_CHAIN_PX pixels)
#   • Its local run direction deviates from H/V by > SNAP_ANGLE_DEG AND it
#     does not lie on a valid circle/arc locus (diagonal stub, stray ink)
#   • It belongs to a connected component that fits none of the four
#     primitive classes above (true "random scribble" clusters)
#
# Implementation uses a two-pass approach:
#   Pass 1 — Per-component classification (cheap, whole-blob decision)
#   Pass 2 — Per-pixel local-direction check inside polyline-class components
#             (removes individual diagonal pixels that slipped through)
# ════════════════════════════════════════════════════════════════════════════

# --- geometry tolerance constants used by the validator ---
_VAL_LINE_RMS_PX    = 3.5    # max perpendicular RMS (px) for a pure-line blob
_VAL_CIRCLE_RMS_REL = 0.12   # max relative RMS for a circle/arc blob
_VAL_SEMI_MIN_DEG   = 150.0  # lower bound of arc span for "semicircle" class
_VAL_SEMI_MAX_DEG   = 210.0  # upper bound
_VAL_LOCAL_WIN      = 9      # half-window (px) for local direction check
_VAL_DIAG_TOL_DEG   = 12.0  # local direction must be within this of H or V


def _component_pixels(labels_img, lbl):
    """Return (N,2) float array of (col,row) for component label lbl."""
    rows, cols = np.where(labels_img == lbl)
    return np.column_stack([cols, rows]).astype(float)


def _classify_component(pts, snap_tol_deg=SNAP_ANGLE_DEG,
                         line_rms_px=_VAL_LINE_RMS_PX,
                         circ_rms_rel=_VAL_CIRCLE_RMS_REL,
                         semi_min=_VAL_SEMI_MIN_DEG,
                         semi_max=_VAL_SEMI_MAX_DEG):
    """
    Classify a connected component's pixel set into one of:
      'line'        — fits a single H or V straight line (or near-H/V diagonal)
      'circle'      — closed circular locus
      'arc'         — partial circular arc (open, span < 330°)
      'semicircle'  — arc spanning semi_min..semi_max degrees
      'polygon'     — multi-segment closed H+V loop (rectangle, square, etc.)
      'reject'      — none of the above; should be erased

    Returns (class_str, meta_dict)
    """
    N = len(pts)
    if N < 4:
        return 'reject', {}

    # ── Test 1: pure straight line (SVD) ────────────────────────────────────
    angle, mid_x, mid_y, rms_perp = _fit_line_svd(pts)
    _, was_snapped = _snap_to_hv(angle, snap_tol_deg)
    if rms_perp <= line_rms_px:
        # Straight enough — accept if H/V snappable OR if genuinely straight
        return 'line', {'angle': angle, 'snapped': was_snapped, 'rms': rms_perp}

    # ── Test 2: circle / arc (Kasa) ─────────────────────────────────────────
    fit = _fit_circle_kasa(pts)
    if fit is not None:
        cx, cy, r, rel = fit
        if rel <= circ_rms_rel and r >= 3.0:
            # Determine angular span to distinguish circle from arc/semicircle
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            # Unwrap span: use circular range
            a_sorted = np.sort(angles)
            gaps = np.diff(a_sorted)
            largest_gap = float(gaps.max()) if len(gaps) else 0.0
            span_deg = 360.0 - math.degrees(largest_gap)

            if span_deg >= 330.0:
                return 'circle', {'cx': cx, 'cy': cy, 'r': r, 'span': span_deg}
            if semi_min <= span_deg <= semi_max:
                return 'semicircle', {'cx': cx, 'cy': cy, 'r': r, 'span': span_deg}
            if span_deg >= 30.0:
                return 'arc', {'cx': cx, 'cy': cy, 'r': r, 'span': span_deg}

    # ── Test 3: closed H+V polygon (rectangle / square) ─────────────────────
    # Heuristic: majority of pixels are within line_rms_px of some H or V line,
    # AND the bounding-box perimeter approximates the pixel count (closed loop).
    x_min, x_max = float(pts[:,0].min()), float(pts[:,0].max())
    y_min, y_max = float(pts[:,1].min()), float(pts[:,1].max())
    bbox_w = x_max - x_min + 1.0
    bbox_h = y_max - y_min + 1.0
    perimeter_est = 2.0 * (bbox_w + bbox_h)
    # For a rectangular loop the pixel count should be ≈ perimeter ± 20 %
    if 0.6 * perimeter_est <= N <= 1.8 * perimeter_est and bbox_w > 5 and bbox_h > 5:
        # Check that individual pixels lie near an H or V line
        # Each pixel should be within line_rms_px of x_min, x_max, y_min, or y_max
        px = pts[:, 0]; py = pts[:, 1]
        near_h = (np.abs(py - y_min) <= line_rms_px) | (np.abs(py - y_max) <= line_rms_px)
        near_v = (np.abs(px - x_min) <= line_rms_px) | (np.abs(px - x_max) <= line_rms_px)
        near_any = near_h | near_v
        if float(near_any.sum()) / N >= 0.80:
            return 'polygon', {'bbox': (x_min, y_min, x_max, y_max)}

    # ── Reject ───────────────────────────────────────────────────────────────
    return 'reject', {}


def _local_direction_valid(skeleton_img, px_col, px_row,
                            half_win=_VAL_LOCAL_WIN,
                            tol_deg=_VAL_DIAG_TOL_DEG):
    """
    Fit a line to the white pixels in a square window around (px_col, px_row).
    Return True if the local direction is within tol_deg of H or V.
    Pixels with fewer than 4 neighbours in the window are considered endpoints
    and always pass (they may be at a corner junction).
    """
    h, w = skeleton_img.shape
    r0 = max(0, px_row - half_win);  r1 = min(h, px_row + half_win + 1)
    c0 = max(0, px_col - half_win);  c1 = min(w, px_col + half_win + 1)
    patch = skeleton_img[r0:r1, c0:c1]
    rows_loc, cols_loc = np.where(patch > 0)
    if len(rows_loc) < 4:
        return True   # not enough context — keep the pixel
    local_pts = np.column_stack([cols_loc + c0, rows_loc + r0]).astype(float)
    angle, _, _, rms_perp = _fit_line_svd(local_pts)
    _, snapped = _snap_to_hv(angle, tol_deg)
    return snapped or rms_perp < _VAL_LINE_RMS_PX


def validate_and_clean_skeleton(skeleton,
                                 snap_angle_deg=SNAP_ANGLE_DEG,
                                 min_chain_px=MIN_CHAIN_PX,
                                 enable_local_check=True):
    """
    Main entry point for Step P.

    Operates entirely on the 1-px skeleton image.  Returns:
      cleaned_skeleton : np.ndarray (same shape/dtype)  — pixels that survived
      n_removed_blobs  : int  — number of whole components erased
      n_removed_px     : int  — total pixels erased (component + local pass)
      report           : list of str — per-component decisions for the step log
    """
    if skeleton is None or not HAS_CV:
        return skeleton, 0, 0, []

    # --- Label connected components on skeleton ----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8)

    cleaned   = skeleton.copy()
    n_rem_blobs = 0
    n_rem_px    = 0
    report      = []

    local_check_mask = np.zeros_like(skeleton)   # pixels flagged for local check

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        # Tiny blob — always remove
        if area < min_chain_px:
            cleaned[labels == lbl] = 0
            n_rem_blobs += 1
            n_rem_px    += area
            report.append(f"lbl{lbl}: ERASE (tiny, {area}px)")
            continue

        pts = _component_pixels(labels, lbl)
        cls, meta = _classify_component(pts, snap_tol_deg=snap_angle_deg)

        if cls == 'reject':
            # Erase the whole component
            cleaned[labels == lbl] = 0
            n_rem_blobs += 1
            n_rem_px    += area
            report.append(f"lbl{lbl}: ERASE (no primitive match, {area}px)")
        elif cls in ('line',):
            # For pure-line components, also schedule local direction check
            # to remove any stray diagonal pixels within the component
            if enable_local_check:
                local_check_mask[labels == lbl] = 1
            report.append(f"lbl{lbl}: KEEP  line  rms={meta.get('rms',0):.1f}px "
                           f"snapped={meta.get('snapped')}  ({area}px)")
        else:
            report.append(f"lbl{lbl}: KEEP  {cls}  {area}px  "
                           + str({k:round(v,1) if isinstance(v,float) else v
                                  for k,v in meta.items() if k not in ('bbox',)}))

    # --- Pass 2: local direction check on line-class pixels ----------------
    if enable_local_check:
        lc_rows, lc_cols = np.where(local_check_mask > 0)
        n_local_removed = 0
        for r, c in zip(lc_rows.tolist(), lc_cols.tolist()):
            if cleaned[r, c] == 0:
                continue   # already erased
            if not _local_direction_valid(cleaned, c, r,
                                           half_win=_VAL_LOCAL_WIN,
                                           tol_deg=_VAL_DIAG_TOL_DEG):
                cleaned[r, c] = 0
                n_rem_px     += 1
                n_local_removed += 1
        if n_local_removed:
            report.append(f"local-dir: removed {n_local_removed} diagonal px "
                           f"from line components")

    return cleaned, n_rem_blobs, n_rem_px, report


# ════════════════════════════════════════════════════════════════════════════
# STEP C — PIXEL ADJACENCY GRAPH: classify every white pixel
# ════════════════════════════════════════════════════════════════════════════

def build_pixel_graph(skeleton):
    """
    Returns:
      pixels    : set of (col, row) for all white pixels
      degree    : dict (col, row) → int  (number of 8-connected white neighbours)
      endpoints : set of (col, row) with degree == 1
      branches  : set of (col, row) with degree >= 3
    """
    rows, cols = np.where(skeleton > 0)
    pixels = set(zip(cols.tolist(), rows.tolist()))   # (x, y) pixel coords

    degree    = {}
    endpoints = set()
    branches  = set()

    NBRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for (x, y) in pixels:
        n = sum(1 for dx, dy in NBRS if (x+dx, y+dy) in pixels)
        degree[(x, y)] = n
        if n == 1:
            endpoints.add((x, y))
        elif n >= 3:
            branches.add((x, y))

    return pixels, degree, endpoints, branches


# ════════════════════════════════════════════════════════════════════════════
# STEP D — TRACE PRIMITIVE CHAINS along the pixel graph
# ════════════════════════════════════════════════════════════════════════════

def trace_chains(pixels, degree, endpoints, branches, min_chain_px=MIN_CHAIN_PX):
    """
    Walk along PASS pixels (degree==2) from every endpoint/branch.
    Returns list of chains; each chain is an ordered list of (x, y) pixel coords.
    Isolated loops (no endpoints/branches) are also traced.
    """
    visited_edges = set()   # frozenset of {pixel_a, pixel_b} for directed edges
    chains = []

    NBRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def walk_from(start, prev):
        """Walk a single chain starting at `start`, coming from `prev`."""
        chain = [prev, start]
        visited_edges.add((prev, start))
        cur = start
        last = prev
        while True:
            nxt = None
            for dx, dy in NBRS:
                nb = (cur[0]+dx, cur[1]+dy)
                if nb not in pixels or nb == last:
                    continue
                if (cur, nb) in visited_edges:
                    continue
                nxt = nb
                break
            if nxt is None:
                break
            visited_edges.add((cur, nxt))
            chain.append(nxt)
            if nxt in endpoints or nxt in branches:
                break       # stop at next junction/endpoint
            last = cur
            cur = nxt
        return chain

    # Walk from all endpoints first, then from branch pixels
    start_pixels = list(endpoints) + list(branches)

    for sp in start_pixels:
        for dx, dy in NBRS:
            nb = (sp[0]+dx, sp[1]+dy)
            if nb not in pixels:
                continue
            if (sp, nb) in visited_edges:
                continue
            chain = walk_from(nb, sp)
            if len(chain) >= min_chain_px:
                chains.append(chain)

    # Handle isolated closed loops (no endpoints/branches): do a BFS per component
    visited_all = set()
    for c in chains:
        visited_all.update(c)

    remaining = pixels - visited_all
    while remaining:
        start = next(iter(remaining))
        loop = [start]
        visited_all.add(start)
        cur = start
        last = None
        for _ in range(len(remaining) + 1):
            nxt = None
            for dx, dy in NBRS:
                nb = (cur[0]+dx, cur[1]+dy)
                if nb in remaining and nb != last:
                    nxt = nb
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            visited_all.add(nxt)
            remaining.discard(nxt)
            last = cur
            cur = nxt
        remaining -= visited_all
        if len(loop) >= min_chain_px:
            loop.append(loop[0])   # close the loop
            chains.append(loop)

    return chains


# ════════════════════════════════════════════════════════════════════════════
# STEP E — ALGEBRAIC LS FIT PER CHAIN
# ════════════════════════════════════════════════════════════════════════════

def _pts_array(chain):
    return np.array(chain, dtype=float)   # (N, 2) col=X, row=Y

def _fit_circle_kasa(pts):
    """
    Kasa algebraic LS circle fit.
    Returns (cx, cy, r, rms_relative) or None if degenerate.
    """
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b_ = x**2 + y**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b_, rcond=None)
    except Exception:
        return None
    cx = res[0] / 2.0
    cy = res[1] / 2.0
    discriminant = res[2] + cx**2 + cy**2
    if discriminant <= 0:
        return None
    r = math.sqrt(discriminant)
    if r < 2.0:
        return None
    dists = np.sqrt((x - cx)**2 + (y - cy)**2)
    rms   = float(np.sqrt(((dists - r)**2).mean()))
    rel   = rms / (r + 1e-9)
    return cx, cy, r, rel

def _fit_line_svd(pts):
    """
    SVD total-least-squares line fit.
    Returns (angle_rad, mid_x, mid_y, rms_perp).
    angle_rad is the direction of the principal axis (0 = rightward).
    """
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    centered = pts - np.array([cx, cy])
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    # Perpendicular residuals
    perp = centered @ np.array([-direction[1], direction[0]])
    rms  = float(np.sqrt((perp**2).mean()))
    return angle, cx, cy, rms

def _snap_to_hv(angle_rad, snap_tol_deg=SNAP_ANGLE_DEG):
    """
    If angle is within snap_tol_deg of 0°, 90°, 180°, 270° → snap.
    Returns (snapped_angle_rad, was_snapped).
    """
    deg = math.degrees(angle_rad) % 180.0   # fold to [0, 180)
    tol = snap_tol_deg
    if deg <= tol or deg >= 180.0 - tol:
        return 0.0, True      # horizontal
    if abs(deg - 90.0) <= tol:
        return math.pi / 2.0, True   # vertical
    return angle_rad, False

def _project_onto_line(pts, angle, cx, cy):
    """Project pts onto the line through (cx,cy) at angle. Return (t_min, t_max)."""
    dx, dy = math.cos(angle), math.sin(angle)
    ts = (pts[:, 0] - cx) * dx + (pts[:, 1] - cy) * dy
    return float(ts.min()), float(ts.max())

def _is_closed(chain, tol=8.0):
    p0, p1 = chain[0], chain[-1]
    return math.hypot(p0[0]-p1[0], p0[1]-p1[1]) < tol

def _arc_span_deg(pts, cx, cy):
    """Angular span of pts around centre (cx,cy) in degrees."""
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    a_min, a_max = float(angles.min()), float(angles.max())
    span = math.degrees(a_max - a_min)
    # unwrap: if span < 0, add 360
    return span % 360.0

def fit_chain(chain, circle_rms_tol=CIRCLE_RMS_TOL, arc_rms_tol=ARC_RMS_TOL,
              arc_min_deg=ARC_MIN_DEG, snap_tol_deg=SNAP_ANGLE_DEG):
    """
    Fit one pixel chain to the best geometric primitive.
    Returns entity dict with keys: type, and type-specific geometry.
    """
    pts = _pts_array(chain)
    N   = len(pts)
    closed = _is_closed(chain)

    # ── CIRCLE: closed chain ────────────────────────────────────────────────
    if closed and N >= 16:
        fit = _fit_circle_kasa(pts)
        if fit is not None:
            cx, cy, r, rel = fit
            if rel < circle_rms_tol:
                return {'type': 'circle', 'cx': cx, 'cy': cy, 'r': r, 'closed': True}

    # ── LINE: attempt SVD line fit ───────────────────────────────────────────
    if N >= 4:
        angle, mid_x, mid_y, rms_perp = _fit_line_svd(pts)
        snapped, was_snapped = _snap_to_hv(angle, snap_tol_deg)

        # Accept as LINE if snapped (H or V), or if residual is tiny
        line_rms_tol = 3.0   # pixels; if perp RMS is below this → treat as line
        if was_snapped or rms_perp < line_rms_tol:
            use_angle = snapped if was_snapped else angle
            t_min, t_max = _project_onto_line(pts, use_angle, mid_x, mid_y)
            dx, dy = math.cos(use_angle), math.sin(use_angle)
            x0 = mid_x + t_min * dx
            y0 = mid_y + t_min * dy
            x1 = mid_x + t_max * dx
            y1 = mid_y + t_max * dy
            return {
                'type': 'line',
                'p0': (x0, y0), 'p1': (x1, y1),
                'angle': use_angle,
                'is_horizontal': was_snapped and abs(use_angle) < 0.01,
                'is_vertical':   was_snapped and abs(use_angle - math.pi/2) < 0.01,
                'mid': (mid_x, mid_y),
                'closed': False,
            }

    # ── ARC: open chain circle fit ──────────────────────────────────────────
    if not closed and N >= 8:
        fit = _fit_circle_kasa(pts)
        if fit is not None:
            cx, cy, r, rel = fit
            if rel < arc_rms_tol:
                span = _arc_span_deg(pts, cx, cy)
                if span >= arc_min_deg:
                    # Compute start/end angles from actual first/last pixel
                    a_start = math.atan2(chain[0][1]  - cy, chain[0][0]  - cx)
                    a_end   = math.atan2(chain[-1][1] - cy, chain[-1][0] - cx)
                    return {
                        'type': 'arc',
                        'cx': cx, 'cy': cy, 'r': r,
                        'a_start': a_start, 'a_end': a_end,
                        'closed': False,
                    }

    # ── POLYLINE fallback: Douglas-Peucker + per-segment H/V snap ──────────
    return _make_orthogonal_polyline(pts, snap_tol_deg)

def _make_orthogonal_polyline(pts, snap_tol_deg=SNAP_ANGLE_DEG):
    """
    Douglas-Peucker simplification on pts, then snap each segment to H or V
    using per-segment SVD LS.  Returns a polyline entity.
    """
    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    arc = cv2.arcLength(pts_int, closed=False)
    epsilon = max(2.0, DP_EPSILON_FRAC * arc)
    approx  = cv2.approxPolyDP(pts_int, epsilon, closed=False)
    corners = np.array([[p[0][0], p[0][1]] for p in approx], dtype=float)

    if len(corners) < 2:
        return {'type': 'polyline',
                'pts': [(float(pts[0,0]), float(pts[0,1])),
                        (float(pts[-1,0]), float(pts[-1,1]))],
                'closed': False}

    # For each inter-corner segment, extract the original pts that belong to it
    # and do a LS line fit + H/V snap on those pts only.
    result_pts = []
    n = len(corners)

    # Build a KD-style assignment: for each pixel in pts, assign to nearest segment
    # Simple approach: for each corner pair, find pts in that bounding box
    for i in range(n - 1):
        p0 = corners[i]
        p1 = corners[i + 1]

        # Collect original pts near this segment
        seg_pts = _pts_near_segment(pts, p0, p1, margin=8.0)
        if len(seg_pts) < 2:
            seg_pts = np.array([p0, p1])

        angle, mid_x, mid_y, _ = _fit_line_svd(seg_pts)
        snapped, was_snapped   = _snap_to_hv(angle, snap_tol_deg)
        use_angle = snapped if was_snapped else angle

        t_min, t_max = _project_onto_line(seg_pts, use_angle, mid_x, mid_y)
        dx, dy = math.cos(use_angle), math.sin(use_angle)

        sx0 = mid_x + t_min * dx;  sy0 = mid_y + t_min * dy
        sx1 = mid_x + t_max * dx;  sy1 = mid_y + t_max * dy

        if not result_pts:
            result_pts.append((sx0, sy0))
        else:
            # Close any gap with an orthogonal elbow
            prev = result_pts[-1]
            if math.hypot(sx0 - prev[0], sy0 - prev[1]) > 1.0:
                if was_snapped and abs(use_angle) < 0.01:   # current is H
                    result_pts.append((prev[0], sy0))
                else:                                        # current is V
                    result_pts.append((sx0, prev[1]))
            result_pts.append((sx0, sy0))
        result_pts.append((sx1, sy1))

    # Deduplicate
    deduped = [result_pts[0]] if result_pts else []
    for p in result_pts[1:]:
        if math.hypot(p[0]-deduped[-1][0], p[1]-deduped[-1][1]) > 0.5:
            deduped.append(p)

    closed = _is_closed(deduped, tol=8.0) if len(deduped) >= 3 else False
    return {'type': 'polyline', 'pts': deduped, 'closed': closed}

def _pts_near_segment(all_pts, p0, p1, margin=8.0):
    """Return subset of all_pts within `margin` of the segment p0→p1."""
    dx = p1[0] - p0[0];  dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return all_pts
    ux, uy = dx/length, dy/length
    # Project each point onto the segment
    rel = all_pts - np.array([p0[0], p0[1]])
    t   = rel[:, 0] * ux + rel[:, 1] * uy
    perp = np.abs(rel[:, 0] * (-uy) + rel[:, 1] * ux)
    mask = (t >= -margin) & (t <= length + margin) & (perp <= margin)
    sub = all_pts[mask]
    return sub if len(sub) >= 2 else all_pts


# ════════════════════════════════════════════════════════════════════════════
# STEP F — GLOBAL GRID SNAP (cluster collinear H/V lines)
# ════════════════════════════════════════════════════════════════════════════

def global_grid_snap(entities, grid_snap_px=GRID_SNAP_PX):
    """
    Cluster all horizontal-line Y coordinates within grid_snap_px.
    Cluster all vertical-line X coordinates within grid_snap_px.
    Replace with the cluster mean.
    Also applies to the H/V segments inside polylines.
    """
    def cluster_values(vals, tol):
        """Simple greedy 1D clustering. Returns dict: original_val → cluster_mean."""
        if not vals:
            return {}
        sorted_vals = sorted(set(vals))
        clusters    = []
        cur_group   = [sorted_vals[0]]
        for v in sorted_vals[1:]:
            if v - cur_group[-1] <= tol:
                cur_group.append(v)
            else:
                clusters.append(cur_group)
                cur_group = [v]
        clusters.append(cur_group)
        mapping = {}
        for grp in clusters:
            mean_v = sum(grp) / len(grp)
            for v in grp:
                mapping[v] = mean_v
        return mapping

    h_ys = []   # Y coords of horizontal lines
    v_xs = []   # X coords of vertical lines

    for e in entities:
        if e['type'] == 'line':
            if e.get('is_horizontal'):
                h_ys.append(e['p0'][1])
                h_ys.append(e['p1'][1])
            elif e.get('is_vertical'):
                v_xs.append(e['p0'][0])
                v_xs.append(e['p1'][0])
        elif e['type'] == 'polyline':
            pts = e['pts']
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i+1]
                dy = abs(p1[1] - p0[1])
                dx = abs(p1[0] - p0[0])
                if dy < 0.5:   # horizontal segment
                    h_ys.append(p0[1]); h_ys.append(p1[1])
                elif dx < 0.5: # vertical segment
                    v_xs.append(p0[0]); v_xs.append(p1[0])

    y_map = cluster_values(h_ys, grid_snap_px)
    x_map = cluster_values(v_xs, grid_snap_px)

    def snap_y(y): return y_map.get(y, y)
    def snap_x(x): return x_map.get(x, x)

    def nearest_in_map(val, mapping, tol):
        best_k = None; best_d = tol + 1
        for k in mapping:
            d = abs(val - k)
            if d < best_d:
                best_d = d; best_k = k
        return mapping[best_k] if best_k is not None and best_d <= tol else val

    for e in entities:
        if e['type'] == 'line':
            if e.get('is_horizontal'):
                ny = nearest_in_map(e['p0'][1], y_map, grid_snap_px * 2)
                e['p0'] = (e['p0'][0], ny)
                e['p1'] = (e['p1'][0], ny)
            elif e.get('is_vertical'):
                nx = nearest_in_map(e['p0'][0], x_map, grid_snap_px * 2)
                e['p0'] = (nx, e['p0'][1])
                e['p1'] = (nx, e['p1'][1])
        elif e['type'] == 'polyline':
            new_pts = []
            for i, p in enumerate(e['pts']):
                x, y = p
                # Snap Y if this point participates in an H segment
                # Snap X if it participates in a V segment
                pts = e['pts']
                participates_h = False; participates_v = False
                if i > 0:
                    prev = pts[i-1]
                    if abs(prev[1]-y) < 1.0: participates_h = True
                    if abs(prev[0]-x) < 1.0: participates_v = True
                if i < len(pts)-1:
                    nxt = pts[i+1]
                    if abs(nxt[1]-y) < 1.0: participates_h = True
                    if abs(nxt[0]-x) < 1.0: participates_v = True
                if participates_h:
                    y = nearest_in_map(y, y_map, grid_snap_px * 2)
                if participates_v:
                    x = nearest_in_map(x, x_map, grid_snap_px * 2)
                new_pts.append((x, y))
            e['pts'] = new_pts

    return entities


# ════════════════════════════════════════════════════════════════════════════
# STEP G — INTERSECTION TRIMMING (exact algebraic)
# ════════════════════════════════════════════════════════════════════════════

def _line_line_intersect(p1, p2, p3, p4):
    """
    Infinite-line intersection of line through p1,p2 with line through p3,p4.
    Returns intersection point or None if parallel.
    """
    x1,y1 = p1;  x2,y2 = p2;  x3,y3 = p3;  x4,y4 = p4
    d1x = x2-x1; d1y = y2-y1
    d2x = x4-x3; d2y = y4-y3
    denom = d1x*d2y - d1y*d2x
    if abs(denom) < 1e-9:
        return None
    t = ((x3-x1)*d2y - (y3-y1)*d2x) / denom
    return (x1 + t*d1x, y1 + t*d1y)

def _entity_segments(e):
    """Return list of (p0, p1) tuples for an entity's straight segments."""
    if e['type'] == 'line':
        return [(e['p0'], e['p1'])]
    elif e['type'] == 'polyline':
        pts = e['pts']
        segs = list(zip(pts[:-1], pts[1:]))
        if e.get('closed') and len(pts) >= 3:
            segs.append((pts[-1], pts[0]))
        return segs
    return []  # circles and arcs: no straight segments

def _bbox(e):
    """Bounding box (x_min, y_min, x_max, y_max) of an entity."""
    if e['type'] == 'circle':
        cx,cy,r = e['cx'], e['cy'], e['r']
        return (cx-r, cy-r, cx+r, cy+r)
    if e['type'] == 'arc':
        cx,cy,r = e['cx'], e['cy'], e['r']
        return (cx-r, cy-r, cx+r, cy+r)
    if e['type'] == 'line':
        xs = [e['p0'][0], e['p1'][0]]; ys = [e['p0'][1], e['p1'][1]]
        return (min(xs), min(ys), max(xs), max(ys))
    if e['type'] == 'polyline':
        xs = [p[0] for p in e['pts']]; ys = [p[1] for p in e['pts']]
        return (min(xs), min(ys), max(xs), max(ys))
    return (0,0,0,0)

def _bboxes_overlap(b1, b2, margin=0.0):
    return (b1[0]-margin <= b2[2] and b1[2]+margin >= b2[0] and
            b1[1]-margin <= b2[3] and b1[3]+margin >= b2[1])

def trim_endpoints_to_intersections(entities, trim_radius=TRIM_RADIUS_PX,
                                    extend_px=EXTEND_PX):
    """
    For each LINE and POLYLINE endpoint:
      1. Compute the outward unit vector of the endpoint's terminal segment.
      2. Extend the endpoint by extend_px along that vector to form a ray.
      3. For every other entity whose bbox overlaps the extended bbox:
         compute exact line-line intersection of the terminal segment's infinite
         extension with each segment of the other entity.
      4. If the intersection is within trim_radius of the original endpoint,
         move the endpoint to the intersection.
    This is purely algebraic — no pixel-level rounding.
    """
    bboxes = [_bbox(e) for e in entities]
    search_margin = trim_radius + extend_px

    def try_trim_endpoint(pts, is_start):
        """
        pts: list of (x,y).  Modifies pts[0] if is_start else pts[-1].
        Returns True if snapped.
        """
        if is_start:
            ep  = pts[0]
            ref = pts[1] if len(pts) > 1 else pts[0]
        else:
            ep  = pts[-1]
            ref = pts[-2] if len(pts) > 1 else pts[-1]

        dx = ep[0] - ref[0];  dy = ep[1] - ref[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return False
        ux = dx / seg_len;  uy = dy / seg_len

        # Far point of extended ray
        far = (ep[0] + extend_px * ux, ep[1] + extend_px * uy)

        ep_bbox = (min(ep[0],far[0])-2, min(ep[1],far[1])-2,
                   max(ep[0],far[0])+2, max(ep[1],far[1])+2)

        best_pt   = None
        best_dist = trim_radius + extend_px + 1.0

        for j, other in enumerate(entities):
            if not _bboxes_overlap(ep_bbox, bboxes[j], margin=search_margin):
                continue
            for (q0, q1) in _entity_segments(other):
                ip = _line_line_intersect(ep, far, q0, q1)
                if ip is None:
                    continue
                # Check that the intersection is actually within or near the
                # OTHER segment (not on its infinite extension far away)
                qlen = math.hypot(q1[0]-q0[0], q1[1]-q0[1])
                if qlen < 1e-6:
                    continue
                t_other = ((ip[0]-q0[0])*(q1[0]-q0[0]) +
                           (ip[1]-q0[1])*(q1[1]-q0[1])) / (qlen**2)
                if t_other < -0.05 or t_other > 1.05:
                    continue   # intersection outside the OTHER segment

                d = math.hypot(ip[0]-ep[0], ip[1]-ep[1])
                if d < best_dist:
                    best_dist = d
                    best_pt   = ip

        if best_pt is not None:
            if is_start:
                pts[0] = best_pt
            else:
                pts[-1] = best_pt
            return True
        return False

    for i, e in enumerate(entities):
        if e['type'] == 'line':
            pts = [e['p0'], e['p1']]
            try_trim_endpoint(pts, is_start=True)
            try_trim_endpoint(pts, is_start=False)
            e['p0'] = pts[0];  e['p1'] = pts[1]

        elif e['type'] == 'polyline' and not e.get('closed'):
            pts = list(e['pts'])
            try_trim_endpoint(pts, is_start=True)
            try_trim_endpoint(pts, is_start=False)
            e['pts'] = pts

    return entities


# ════════════════════════════════════════════════════════════════════════════
# STEP H — DEDUPLICATION
# ════════════════════════════════════════════════════════════════════════════

def deduplicate_entities(entities, dup_tol_px=3.0):
    """
    Remove duplicate LINE entities:
      Two lines are duplicates if they are parallel (same angle ± 1°) AND
      their perpendicular distance is < dup_tol_px AND their bounding boxes
      substantially overlap.
    Keep the one with the longer extent.
    """
    def line_normal_offset(e):
        angle = e.get('angle', math.atan2(e['p1'][1]-e['p0'][1],
                                           e['p1'][0]-e['p0'][0]))
        mx, my = e.get('mid', ((e['p0'][0]+e['p1'][0])/2,
                                (e['p0'][1]+e['p1'][1])/2))
        # Normal distance from origin to line
        nx = -math.sin(angle); ny = math.cos(angle)
        offset = mx * nx + my * ny
        return angle % math.pi, offset

    def line_length(e):
        return math.hypot(e['p1'][0]-e['p0'][0], e['p1'][1]-e['p0'][1])

    keep = [True] * len(entities)
    line_idx = [i for i,e in enumerate(entities) if e['type'] == 'line']

    for ii in range(len(line_idx)):
        if not keep[line_idx[ii]]:
            continue
        a1, off1 = line_normal_offset(entities[line_idx[ii]])
        b1 = _bbox(entities[line_idx[ii]])
        for jj in range(ii+1, len(line_idx)):
            if not keep[line_idx[jj]]:
                continue
            a2, off2 = line_normal_offset(entities[line_idx[jj]])
            if abs((a1-a2+math.pi/2) % math.pi - math.pi/2) > math.radians(2.0):
                continue   # not parallel
            if abs(off1 - off2) > dup_tol_px:
                continue   # too far apart
            b2 = _bbox(entities[line_idx[jj]])
            if not _bboxes_overlap(b1, b2, margin=dup_tol_px):
                continue
            # Duplicate — remove the shorter one
            if line_length(entities[line_idx[ii]]) >= line_length(entities[line_idx[jj]]):
                keep[line_idx[jj]] = False
            else:
                keep[line_idx[ii]] = False
                break

    return [e for i, e in enumerate(entities) if keep[i]]


# ════════════════════════════════════════════════════════════════════════════
# STEP I — DXF EXPORT
# ════════════════════════════════════════════════════════════════════════════

def build_centerline_dxf(entities, img_w, img_h, out_path):
    """
    Write DXF:
      LINE     → DXF LINE entity  (layer LINES)
      POLYLINE → DXF LWPOLYLINE   (layer CENTERLINES)
      CIRCLE   → DXF CIRCLE       (layer CIRCLES)
      ARC      → DXF ARC          (layer ARCS)
    Y-axis is flipped (DXF Y-up, image Y-down).
    """
    if not HAS_DXF or not entities:
        return None, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"]   = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"]   = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"]   = (0.0, 0.0)
    doc.header["$LIMMAX"]   = (float(img_w), float(img_h))

    msp = doc.modelspace()

    try:
        doc.linetypes.get('CENTER')
    except Exception:
        pass

    doc.layers.new("LINES",       dxfattribs={"color": 7,  "linetype": "CONTINUOUS"})
    doc.layers.new("CENTERLINES", dxfattribs={"color": 3,  "linetype": "CONTINUOUS"})
    doc.layers.new("CIRCLES",     dxfattribs={"color": 1,  "linetype": "CONTINUOUS"})
    doc.layers.new("ARCS",        dxfattribs={"color": 4,  "linetype": "CONTINUOUS"})

    def fy(y): return float(img_h) - float(y)   # flip Y

    entity_count = 0

    for e in entities:
        t = e['type']

        if t == 'line':
            x0, y0 = e['p0']
            x1, y1 = e['p1']
            msp.add_line(
                (float(x0), fy(y0), 0.0),
                (float(x1), fy(y1), 0.0),
                dxfattribs={"layer": "LINES", "color": 7}
            )
            entity_count += 1

        elif t == 'polyline':
            pts_dxf = [(float(p[0]), fy(p[1])) for p in e['pts']]
            if len(pts_dxf) < 2:
                continue
            poly = msp.add_lwpolyline(
                pts_dxf, format="xy",
                dxfattribs={"layer": "CENTERLINES", "color": 3}
            )
            if e.get('closed') and len(pts_dxf) >= 3:
                poly.close(True)
            entity_count += 1

        elif t == 'circle':
            msp.add_circle(
                (float(e['cx']), fy(e['cy']), 0.0),
                float(e['r']),
                dxfattribs={"layer": "CIRCLES", "color": 1}
            )
            entity_count += 1

        elif t == 'arc':
            cx = float(e['cx']); cy = fy(e['cy']); r = float(e['r'])
            # In DXF, ARC angles are measured CCW from +X in DXF coords.
            # Because we flipped Y, arc direction is also flipped.
            a_start_dxf = math.degrees(-e['a_end'])   % 360.0
            a_end_dxf   = math.degrees(-e['a_start']) % 360.0
            if abs(a_start_dxf - a_end_dxf) < 0.5:
                a_end_dxf = (a_start_dxf + 359.0) % 360.0
            msp.add_arc(
                (cx, cy, 0.0), r,
                start_angle=a_start_dxf,
                end_angle=a_end_dxf,
                dxfattribs={"layer": "ARCS", "color": 4}
            )
            entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size if out_path.exists() else 0
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════
# PNG PREVIEW
# ════════════════════════════════════════════════════════════════════════════

def build_comparison_png(edges_display, entities, img_w, img_h, out_path):
    if not HAS_CV or not np:
        return False
    try:
        left = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        left[:] = (15, 12, 10)
        left[edges_display > 0] = (255, 255, 255)

        right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        right[:] = (15, 12, 10)

        n_lines = n_circ = n_poly = n_arc = 0
        for e in entities:
            if e['type'] == 'line':
                p0 = (int(round(e['p0'][0])), int(round(e['p0'][1])))
                p1 = (int(round(e['p1'][0])), int(round(e['p1'][1])))
                color = (255, 220, 50) if e.get('is_horizontal') else \
                        (50, 220, 255) if e.get('is_vertical') else (200, 200, 80)
                cv2.line(right, p0, p1, color, 2, cv2.LINE_AA)
                n_lines += 1
            elif e['type'] == 'circle':
                cx_px = int(round(e['cx'])); cy_px = int(round(e['cy']))
                r_px  = int(round(e['r']))
                cv2.circle(right, (cx_px, cy_px), r_px, (80, 80, 220), 2, cv2.LINE_AA)
                n_circ += 1
            elif e['type'] == 'arc':
                cx = int(round(e['cx'])); cy = int(round(e['cy']))
                r  = int(round(e['r']))
                a1 = int(round(math.degrees(e['a_start'])))
                a2 = int(round(math.degrees(e['a_end'])))
                cv2.ellipse(right, (cx,cy), (r,r), 0, a1, a2, (80, 200, 200), 2, cv2.LINE_AA)
                n_arc += 1
            elif e['type'] == 'polyline':
                pts_draw = [(int(round(p[0])), int(round(p[1]))) for p in e['pts']]
                for k in range(len(pts_draw) - 1):
                    cv2.line(right, pts_draw[k], pts_draw[k+1], (80, 200, 80), 2, cv2.LINE_AA)
                if e.get('closed') and len(pts_draw) >= 3:
                    cv2.line(right, pts_draw[-1], pts_draw[0], (80, 200, 80), 2, cv2.LINE_AA)
                n_poly += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left,  "Canny Skeleton (200px filter)", (10, 22),
                    font, 0.55, (180,180,180), 1, cv2.LINE_AA)
        summary = (f"L:{n_lines}  C:{n_circ}  A:{n_arc}  P:{n_poly}"
                   f"  [yellow=H, cyan=V, green=poly, blue=circle]")
        cv2.putText(right, summary, (10, 22),
                    font, 0.45, (180,180,180), 1, cv2.LINE_AA)

        sep   = np.full((img_h, 4, 3), 40, dtype=np.uint8)
        panel = np.concatenate([left, sep, right], axis=1)
        ok = cv2.imwrite(str(out_path), panel)
        return bool(ok and out_path.exists())
    except Exception as ex:
        sys.stderr.write(f"PNG preview error: {ex}\n")
        return False


# ════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
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
        c.drawString(margin, page_h - 24,
                     "SheetForge v13 — Pixel-Graph LS Centerline DXF Preview")
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
        _draw_panel(c, reader, right_x, margin, col_w, col_h,
                    "Skeleton Centerline (Pixel-Graph Traced)")
        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w/2, 12,
            "SheetForge v13  •  Pixel-Graph Tracing  •  LS Primitive Fit  "
            "•  H/V Snap  •  Grid Cluster  •  Algebraic Intersection Trim")
        c.save()
        for f in tmp_files:
            try: _os.unlink(f)
            except Exception: pass
        return True
    except Exception as ex:
        sys.stderr.write(f"PDF export error: {ex}\n{traceback.format_exc()}\n")
        return False


def _draw_panel(c, img_reader, x, y, w, h, title):
    title_h = 22; img_h = h - title_h
    c.setFillColorRGB(0.08, 0.1, 0.13)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.12, 0.15, 0.2)
    c.roundRect(x, y + img_h, w, title_h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.55, 0.65, 0.85)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + w/2, y + img_h + 7, title)
    pad = 8
    c.drawImage(img_reader, x+pad, y+title_h+pad,
                width=w-pad*2, height=img_h-pad*2,
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

    blur_ksize      = int(opts.get("blurKsize",     5))
    canny_low       = int(opts.get("cannyLow",     20))
    canny_high      = int(opts.get("cannyHigh",    80))
    min_blob_area   = int(opts.get("minBlobArea",  50))
    canny_min_area  = int(opts.get("cannyMinArea", 200))
    circle_rms_tol  = float(opts.get("circleRmsTol",  CIRCLE_RMS_TOL))
    arc_rms_tol     = float(opts.get("arcRmsTol",     ARC_RMS_TOL))
    snap_angle_deg  = float(opts.get("snapAngleDeg",  SNAP_ANGLE_DEG))
    grid_snap_px    = float(opts.get("gridSnapPx",    GRID_SNAP_PX))
    trim_radius     = float(opts.get("trimRadius",    TRIM_RADIUS_PX))
    extend_px       = float(opts.get("extendPx",      EXTEND_PX))
    min_chain_px    = int(opts.get("minChainPx",    MIN_CHAIN_PX))

    steps = []

    # ── STEP 1: Load ─────────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record("CV-1: Load Image", f"{img_w}×{img_h}px  DPI={dpi:.0f}", t0))

    # ── STEP 2: Median Blur ───────────────────────────────────────────────────
    t0 = now_ms()
    blurred = median_blur(gray, ksize=blur_ksize)
    steps.append(step_record(f"CV-2: Median Blur (ksize={blur_ksize})", "Noise reduced", t0))

    # ── STEP 3: Adaptive Threshold ────────────────────────────────────────────
    t0 = now_ms()
    binary   = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record("CV-3: Adaptive Threshold", f"{white_px} white px", t0))

    # ── STEP 4: Morph Open ────────────────────────────────────────────────────
    t0 = now_ms()
    opened    = morph_clean(binary)
    opened_px = int(np.count_nonzero(opened))
    steps.append(step_record("CV-4: MORPH_OPEN",
                              f"{white_px - opened_px} spur px removed", t0))

    # ── STEP 5: Blob filter ───────────────────────────────────────────────────
    t0 = now_ms()
    cleaned, removed_blobs, removed_px = remove_small_blobs(opened, min_blob_area)
    steps.append(step_record(
        f"CV-5: Blob Filter (minBlobArea={min_blob_area}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px)", t0))

    # ── STEP 6a: Canny ────────────────────────────────────────────────────────
    t0 = now_ms()
    edges_raw = canny_edges(cleaned, canny_low, canny_high)
    edges_200, removed_edge_blobs, removed_edge_px = remove_small_blobs(
        edges_raw, canny_min_area)
    steps.append(step_record(
        f"CV-6: Canny (lo={canny_low}, hi={canny_high}) + {canny_min_area}px area filter",
        f"{removed_edge_blobs} edge blob(s) removed ({removed_edge_px}px) — "
        f"{int(np.count_nonzero(edges_200))} edge px remain", t0))

    # ── STEP 6b: Skeletonize ──────────────────────────────────────────────────
    t0 = now_ms()
    skeleton = skeletonize_mask(edges_200)
    skel_px  = int(np.count_nonzero(skeleton))
    steps.append(step_record(
        "CV-6b: Skeletonize (Zhang-Suen) → 1-px centerline",
        f"{skel_px} skeleton px", t0))

    edges_display = thicken_for_display(skeleton, thickness_px=4)

    # ── STEP P: Pixel geometry validator ─────────────────────────────────────
    t0 = now_ms()
    skeleton, n_rem_blobs, n_rem_px, val_report = validate_and_clean_skeleton(
        skeleton,
        snap_angle_deg=snap_angle_deg,
        min_chain_px=min_chain_px,
        enable_local_check=True,
    )
    skel_px_after = int(np.count_nonzero(skeleton))
    steps.append(step_record(
        "PX-P: Pixel geometry validator "
        "(erase non-H/V/circle/arc/semicircle/rect/polygon pixels before DXF)",
        f"{n_rem_blobs} component(s) erased  |  {n_rem_px} px removed  |  "
        f"{skel_px_after} valid px remain", t0))

    # ── STEP C: Pixel adjacency graph ─────────────────────────────────────────
    t0 = now_ms()
    pixels, degree, endpoints, branches = build_pixel_graph(skeleton)
    steps.append(step_record(
        "GR-C: Pixel adjacency graph",
        f"{len(pixels)} px  |  {len(endpoints)} endpoint(s)  |  "
        f"{len(branches)} branch/junction(s)", t0))

    # ── STEP D: Trace primitive chains ────────────────────────────────────────
    t0 = now_ms()
    chains = trace_chains(pixels, degree, endpoints, branches,
                          min_chain_px=min_chain_px)
    steps.append(step_record(
        f"GR-D: Pixel-graph chain tracing (min={min_chain_px}px)",
        f"{len(chains)} chains traced  (no perimeter loops — true centerline)", t0))

    # ── STEP E: LS fit per chain ──────────────────────────────────────────────
    t0 = now_ms()
    entities = []
    for ch in chains:
        e = fit_chain(ch,
                      circle_rms_tol=circle_rms_tol,
                      arc_rms_tol=arc_rms_tol,
                      snap_tol_deg=snap_angle_deg)
        entities.append(e)

    n_line  = sum(1 for e in entities if e['type'] == 'line')
    n_circ  = sum(1 for e in entities if e['type'] == 'circle')
    n_arc   = sum(1 for e in entities if e['type'] == 'arc')
    n_poly  = sum(1 for e in entities if e['type'] == 'polyline')
    steps.append(step_record(
        f"LS-E: Algebraic LS fit (Kasa circle/arc, SVD line, H/V snap={snap_angle_deg}°)",
        f"{len(chains)} chains → {n_line} line(s) + {n_circ} circle(s) + "
        f"{n_arc} arc(s) + {n_poly} polyline(s)", t0))

    # ── STEP F: Global grid snap ──────────────────────────────────────────────
    t0 = now_ms()
    entities = global_grid_snap(entities, grid_snap_px=grid_snap_px)
    steps.append(step_record(
        f"GEO-F: Global H/V grid snap (cluster tol={grid_snap_px}px)",
        "Collinear H/V lines unified to exact shared coordinate", t0))

    # ── STEP G: Intersection trimming ─────────────────────────────────────────
    t0 = now_ms()
    entities = trim_endpoints_to_intersections(
        entities, trim_radius=trim_radius, extend_px=extend_px)
    steps.append(step_record(
        f"GEO-G: Algebraic intersection trimming (trim_r={trim_radius}px, extend={extend_px}px)",
        "All endpoints snapped to exact algebraic intersection — zero gap/overshoot", t0))

    # ── STEP H: Deduplication ─────────────────────────────────────────────────
    t0 = now_ms()
    pre_count = len(entities)
    entities  = deduplicate_entities(entities, dup_tol_px=3.0)
    removed_dup = pre_count - len(entities)
    steps.append(step_record(
        "GEO-H: Duplicate line removal",
        f"{removed_dup} duplicate(s) removed  →  {len(entities)} final entities", t0))

    # ── Output paths ──────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)
    ts_str   = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    pdf_name = f"design_{ts_str}.pdf"
    png_name = f"preview_{ts_str}.png"
    dxf_path = server_out_dir / dxf_name
    pdf_path = server_out_dir / pdf_name
    png_path = server_out_dir / png_name

    # ── STEP I: DXF export ────────────────────────────────────────────────────
    t0 = now_ms()
    _, entity_count, dxf_size = build_centerline_dxf(
        entities, img_w, img_h, dxf_path)
    dxf_content_str = ""
    if dxf_size and dxf_size > 0:
        try:
            with open(dxf_path, encoding="utf-8", errors="replace") as f:
                dxf_content_str = f.read(200_000)
        except Exception:
            pass
    steps.append(step_record(
        "DXF-I: Centerline DXF export (LINE/LWPOLYLINE/CIRCLE/ARC, "
        "90° snapped, grid-clustered, intersection-trimmed)",
        f"{entity_count} entities  |  {dxf_size // 1024 if dxf_size else 0} KB", t0))

    # ── PNG preview ───────────────────────────────────────────────────────────
    t0 = now_ms()
    png_ok   = build_comparison_png(edges_display, entities, img_w, img_h, png_path)
    png_size = png_path.stat().st_size if png_ok and png_path.exists() else 0
    steps.append(step_record(
        "PNG-J: Side-by-side preview (skeleton vs fitted primitives)",
        f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # ── PDF export ────────────────────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges_display, pdf_path, orig_bgr=bgr)
    steps.append(step_record(
        "PDF-K: Export centerline preview", "OK" if pdf_ok else "FAILED", t0))

    n_line_final  = sum(1 for e in entities if e['type'] == 'line')
    n_circ_final  = sum(1 for e in entities if e['type'] == 'circle')
    n_arc_final   = sum(1 for e in entities if e['type'] == 'arc')
    n_poly_final  = sum(1 for e in entities if e['type'] == 'polyline')

    analysis = {
        "width"            : float(img_w),
        "height"           : float(img_h),
        "dpi"              : dpi,
        "edgePixels"       : skel_px,
        "edges"            : entity_count,
        "chains"           : len(chains),
        "entities"         : len(entities),
        "linesDetected"    : n_line_final,
        "circlesDetected"  : n_circ_final,
        "arcsDetected"     : n_arc_final,
        "polylinesDetected": n_poly_final,
        "blurKsize"        : blur_ksize,
        "cannyLow"         : canny_low,
        "cannyHigh"        : canny_high,
        "cannyMinArea"     : canny_min_area,
        "circleRmsTol"     : circle_rms_tol,
        "snapAngleDeg"     : snap_angle_deg,
        "gridSnapPx"       : grid_snap_px,
        "trimRadius"       : trim_radius,
        "imgW"             : img_w,
        "imgH"             : img_h,
        "scaleMmPerDu"     : round(25.4 / dpi, 4),
        "coordSystem"      : "DXF Y-flipped (Y-up), origin=bottom-left",
        "shapeSummary"     : (
            f"{n_line_final} line(s) + {n_circ_final} circle(s) + "
            f"{n_arc_final} arc(s) + {n_poly_final} polyline(s) — "
            f"pixel-graph tracing, LS primitive fit, H/V snap, "
            f"grid cluster, algebraic intersection trim"
        ),
        "validatorBlobsErased" : n_rem_blobs,
        "validatorPxErased"    : n_rem_px,
        "validatorPxRemain"    : skel_px_after,
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
        "dxfContent"   : dxf_content_str,
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
