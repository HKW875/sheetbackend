#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v11  (Hierarchy-Aware Shape Fitting + Real-MM Export)
================================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, mmDxfContent, pdfAvailable }

WHAT'S NEW IN v11 (on top of v10)
-----------------------------------
v10 already produces a clean, true-position px-space DXF via hierarchy-aware
offset-pair averaging + algebraic LS shape fitting (rects/circles). v11 ADDS
a second, independent export path that many CAM workflows want directly:

  1. PROPER THINNING (Zhang-Suen, STEP 5b)
     The cleaned speckle-free mask is reduced to a true 1px topological
     skeleton. Unlike a quick erode/dilate loop, Zhang-Suen guarantees a
     single-pixel centreline without breaking connectivity or shortening
     strokes — needed so Hough circle radii reflect the stroke centreline,
     not its outer edge.

  2. MAIN EXTERNAL CONTOUR + VERY TIGHT approxPolyDP (STEP 11b)
     cv2.findContours(cleaned, RETR_EXTERNAL) finds the part's outer
     boundary; cv2.approxPolyDP with epsilon ≈ 0.15% of perimeter removes
     only staircase/anti-aliasing jitter while keeping every real corner.

  3. HOUGH-CIRCLE HOLES FROM THE SKELETON (STEP 11b)
     cv2.HoughCircles runs on the 1px skeleton so detected hole radii are
     true centreline radii, exported as real CIRCLE entities.

  4. PIXEL -> MILLIMETRE WITH Y-FLIP (STEP 11b)
     scale = 25.4 / dpi; every coordinate's Y is flipped
     (mm_y = (img_h - px_y) * scale) to convert from the image convention
     (origin top-left, Y-down) to the standard CAD convention (origin
     bottom-left, Y-up).

  5. SECOND DXF, REAL-WORLD UNITS (STEP 11b)
     A standalone DXF (`design_mm_<ts>.dxf`) with $INSUNITS=4 (millimeters),
     $MEASUREMENT=1, layer "OUTLINE" (closed LWPOLYLINE) and layer "HOLES"
     (CIRCLE per detected hole) — ready to open directly at true scale in
     any CAD/CAM package.

This is purely ADDITIVE: the original v10 px-space DXF (`design_<ts>.dxf`,
hierarchy-paired rect/circle shapes) is still produced unchanged; the new
mm-scale outline+holes DXF is exported alongside it as `mmDxfFilename` /
`mmDxfAbsPath` in the JSON `dwg` block.

================================================================================
v10 NOTES (unchanged) — Hierarchy-Aware Shape Fitting
================================================================================

WHAT CHANGED FROM v9.1 AND WHY
-------------------------------
v9.1 produced excellent results on clean scans (door_lock) but broke down
on noisier images (oven_top) for three root causes:

  1. SPECKLE / 1px DOTS
     A 3x3 MORPH_OPEN alone does not reliably remove every isolated
     1-2px speckle.  Any speckle that survives becomes its own tiny
     contour, OR — worse — gets 8-connected to a real shape's contour
     and silently drags that shape's bounding box (and therefore its
     centre/size) off in some direction.  This is the single biggest
     cause of "offset" rectangles/circles.
     FIX: cv2.connectedComponentsWithStats() removes EVERY connected
     blob below an area threshold, by construction — not "mostly",
     100% — regardless of where it sits on the page.

  2. DOUBLE-EDGE "OFFSET" DUPLICATES
     Running findContours on the *Canny* output means every drawn
     line produces TWO parallel contours (its inner edge and its
     outer edge).  v9.1 tried to fix this after the fact by comparing
     every shape to every other shape and discarding "the larger one"
     — fragile, and biased (it always keeps the inner edge, which is
     not the true centreline of the drawn stroke).
     FIX: findContours now runs on the *cleaned binary mask* (the
     stroke itself, not its Canny silhouette) with RETR_TREE.  A
     stroke drawn as a closed ring produces an outer contour and an
     inner ("hole") contour that are PARENT/CHILD in the hierarchy.
     We pair them explicitly using that hierarchy relationship and
     AVERAGE their geometry — giving the true stroke centreline,
     with no offset and exactly one entity per shape.

  3. FORCED "MASTER CX" RE-CENTERING
     v9.1 snapped every shape's cx to a single master_cx (taken from
     the largest rectangle). That is a door-lock-specific assumption
     (one tall part, everything aligned on its vertical axis). For a
     2-D layout like oven_top (4 corner cut-outs + 1 centre hole) this
     actively MOVES every shape away from its true detected position,
     producing exactly the "not well spaced" / offset symptom reported.
     FIX: master-axis re-centering is removed entirely. Every shape's
     (cx, cy) is the value measured directly from the image — DXF
     geometry now matches the Canny image pixel-for-pixel.

Pipeline:
  1.  Load image
  2.  Median Blur                (salt-and-pepper pre-clean)
  3.  Adaptive Threshold         (binarise, strokes = WHITE)
  4.  Morph Open                 (remove small attached spurs)
  5.  Connected-Component Filter (remove EVERY blob < minBlobArea — 100% dot removal)
  6.  Canny Edge Detection       (visualisation / PDF preview only)
  7.  Contour + Hierarchy        (cv2.findContours RETR_TREE on the CLEANED MASK)
  8.  Shape Classification       (algebraic LS: Kasa circle fit w/ outlier trim + rect bbox)
  9.  Hierarchy Pairing          (outer/inner stroke-edge pairs -> averaged centreline shape)
  10. Residual Dedup + Filter    (proximity-average safety net, absolute-area noise filter)
  11. DXF Export — CLEAN         (one CIRCLE or closed LWPOLYLINE per true shape, true cx/cy)
  12. PNG Preview                (side-by-side: Canny vs final shapes)
  13. PDF Export                 (reportlab)
"""

import sys, os, json, time, traceback, math
from pathlib import Path

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
    """
    Remove EVERY 8-connected white blob whose pixel area is below
    `min_area`. Unlike MORPH_OPEN (which only erodes-then-dilates and can
    miss speckles or merge them into nearby strokes), this is exact and
    deterministic: connectedComponentsWithStats labels every blob, and any
    blob below the threshold is dropped completely, wherever it sits.

    This is what guarantees the cleaned mask used for contour extraction
    is 100% free of the 1px "salt" dots seen in oven_top_grayscale.png —
    the same property door_lock_grayscale.png already had "for free"
    because its source scan simply had no speckle to begin with.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    removed_px = 0
    removed_blobs = 0
    for lbl in range(1, num_labels):  # 0 = background
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == lbl] = 255
        else:
            removed_px += area
            removed_blobs += 1
    return cleaned, removed_blobs, removed_px

def canny_edges(cleaned, low_threshold=20, high_threshold=80):
    return cv2.Canny(cleaned, low_threshold, high_threshold)

def clean_edge_preview(edges, min_blob_area=3):
    """
    Cosmetic pass on the Canny output used ONLY for the side-by-side
    preview / PDF: Canny's internal Gaussian smoothing + gradient steps
    can leave a handful of isolated 1-2px edge specks even when the input
    mask is itself 100% speckle-free (these specks never reach contour
    extraction, which runs on `cleaned`, not `edges` — so they cannot
    affect the DXF). Removing them here makes the preview match the
    "no dots" look of door_lock_grayscale.png for ANY input image.
    """
    edges_clean, removed_blobs, removed_px = remove_small_blobs(edges, min_blob_area)
    return edges_clean, removed_blobs, removed_px


# ════════════════════════════════════════════════════════════════════════════
# STEP 5b — ZHANG-SUEN THINNING (proper 1px skeletonisation)
# ════════════════════════════════════════════════════════════════════════════

def zhang_suen_thinning(binary, max_iter=60):
    """
    Proper Zhang-Suen thinning of a {0,255} binary mask down to a true 1px
    skeleton.

    WHY THIS REPLACES "ERODE THEN DILATE" THINNING
    -----------------------------------------------
    A naive `while True: erode -> dilate -> subtract -> OR-into-skeleton`
    loop (as in quick one-off scripts) does NOT reliably converge to a
    single-pixel-wide centreline: depending on the kernel and stroke
    geometry it can leave 2px-wide residue on diagonal strokes, create
    small gaps at junctions, or erode away short spurs/corners entirely.

    Zhang-Suen instead repeatedly removes only "boundary" pixels that are
    provably safe to delete without breaking 8-connectivity or shortening
    the skeleton (the B/A neighbour-count + zero-pattern conditions below),
    alternating two sub-iterations until no pixel qualifies for removal.
    The result is a topologically faithful 1px centreline for every stroke
    — exactly what HoughCircles needs to report a hole's TRUE radius
    (the centreline radius), not the outer-edge radius of a thick stroke.

    `max_iter` caps the number of full (sub-iteration A + B) passes as a
    safety net for pathological inputs; in practice convergence on
    sketch-sized strokes (a handful of px wide) takes well under 20 passes.
    """
    img = (binary > 0).astype(np.uint8)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2 = np.roll(img,  1, axis=0)
            p3 = np.roll(np.roll(img,  1, axis=0), -1, axis=1)
            p4 = np.roll(img, -1, axis=1)
            p5 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
            p6 = np.roll(img, -1, axis=0)
            p7 = np.roll(np.roll(img, -1, axis=0),  1, axis=1)
            p8 = np.roll(img,  1, axis=1)
            p9 = np.roll(np.roll(img,  1, axis=0),  1, axis=1)

            neigh = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(neigh)

            # A = number of 0->1 transitions walking p2..p9 then back to p2
            seq = neigh + [p2]
            A = np.zeros_like(img, dtype=np.uint8)
            for k in range(8):
                A += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)

            base = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond = base & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = base & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            if cond.any():
                img[cond] = 0
                changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# STEP 11b — MAIN-OUTLINE + HOUGH-CIRCLE HOLES, IN REAL MILLIMETRES
# ════════════════════════════════════════════════════════════════════════════

def extract_main_outline_and_holes(cleaned, skeleton, dpi, img_h,
                                    epsilon_factor_tight=0.0015,
                                    hough_min_radius_px=8,
                                    hough_max_radius_px=400,
                                    coverage_threshold=0.7):
    """
    A second, simpler export path that produces real-world millimetre
    geometry from just two OpenCV operations:

      1. MAIN EXTERNAL CONTOUR
         cv2.findContours(cleaned, RETR_EXTERNAL, CHAIN_APPROX_NONE) on the
         speckle-free stroke mask (NOT the 1px skeleton — a skeleton's own
         outer boundary is itself, which would collapse a drawn outline to
         a zero-width contour). The largest-area external contour is the
         part's outer boundary.

      2. VERY TIGHT approxPolyDP
         epsilon = epsilon_factor_tight * perimeter (default 0.15%). This
         removes only staircase/anti-aliasing jitter while preserving every
         genuine corner — far tighter than the v10 shape-classification
         epsilon, which is intentionally loose because it feeds an
         algebraic rect/circle fit rather than an exact polyline.

      3. HOUGH CIRCLES ON THE THINNED SKELETON, COVERAGE-VERIFIED
         cv2.HoughCircles runs on `skeleton` (the Zhang-Suen 1px
         centreline), not on `cleaned`. A thick stroke's HoughCircles
         radius is biased toward the OUTER edge of the ring; the skeleton's
         radius is the stroke centreline — the dimension a CAM operator
         actually wants for a bore/hole.

         HoughCircles on a sparse 1px skeleton readily proposes spurious
         circles that fit a rectangle's corner/edge segments (their centre
         and radius satisfy the Hough accumulator locally without the
         circle actually being drawn). Each candidate is therefore
         VERIFIED by sampling 72 points around its circumference and
         measuring what fraction land on a 2px-dilated skeleton pixel —
         a real drawn circle scores ~0.95+, a corner/edge false-positive
         typically scores well under 0.3. Only candidates scoring at or
         above `coverage_threshold` are kept as real holes.

      4. PIXEL -> MILLIMETRE, WITH Y-FLIP
         scale = 25.4 / dpi. Every point's Y is flipped via
         `mm_y = (img_h - px_y) * scale` so the output uses the standard
         CAD convention (origin bottom-left, Y increasing upward) instead
         of the image convention (origin top-left, Y increasing downward).

    Returns:
      outline_mm : [(x_mm, y_mm), ...] closed outer-boundary polygon
      holes_mm   : [(cx_mm, cy_mm, r_mm), ...] one tuple per detected hole
      scale      : the px -> mm factor that was applied
    """
    scale = 25.4 / float(dpi) if dpi and dpi > 1 else 25.4 / 96.0

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], [], scale

    main_contour = max(contours, key=cv2.contourArea)
    perim   = cv2.arcLength(main_contour, closed=True)
    epsilon = max(epsilon_factor_tight * perim, 0.5)
    approx  = cv2.approxPolyDP(main_contour, epsilon, closed=True)

    outline_mm = []
    for p in approx:
        px, py = float(p[0][0]), float(p[0][1])
        outline_mm.append((px * scale, (img_h - py) * scale))

    holes_mm = []
    circles = cv2.HoughCircles(
        skeleton, cv2.HOUGH_GRADIENT, dp=1,
        minDist=max(20, hough_min_radius_px * 2),
        param1=50, param2=25,
        minRadius=hough_min_radius_px, maxRadius=hough_max_radius_px,
    )
    if circles is not None:
        dil = cv2.dilate(skeleton, np.ones((5, 5), np.uint8), iterations=1)
        h_img, w_img = dil.shape[:2]
        n_samples = 72
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

        for x, y, r in np.round(circles[0, :]).astype(float):
            xs = np.round(x + r * np.cos(angles)).astype(int)
            ys = np.round(y + r * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)
            hits = 0
            if valid.any():
                hits = int(dil[ys[valid], xs[valid]].astype(bool).sum())
            coverage = hits / float(n_samples)
            if coverage >= coverage_threshold:
                holes_mm.append((x * scale, (img_h - y) * scale, r * scale))

    return outline_mm, holes_mm, scale


def build_mm_dxf(outline_mm, holes_mm, out_path):
    """
    Writes a standalone, real-world-scale DXF:
      - $INSUNITS = 4 (millimeters) + $MEASUREMENT = 1 (metric) in the
        header, so CAD/CAM software imports the part at true physical size.
      - Layer "OUTLINE": one closed LWPOLYLINE for the main external contour.
      - Layer "HOLES":   one CIRCLE entity per Hough-detected hole.
    All coordinates are already millimetres with a bottom-left, Y-up CAD
    origin (see extract_main_outline_and_holes).
    """
    if not HAS_DXF or not outline_mm:
        return None, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"]    = 4   # 4 = Millimeters
    doc.header["$MEASUREMENT"] = 1   # 1 = Metric

    xs = [p[0] for p in outline_mm]
    ys = [p[1] for p in outline_mm]
    doc.header["$EXTMIN"] = (min(xs), min(ys), 0.0)
    doc.header["$EXTMAX"] = (max(xs), max(ys), 0.0)
    doc.header["$LIMMIN"] = (min(xs), min(ys))
    doc.header["$LIMMAX"] = (max(xs), max(ys))

    msp = doc.modelspace()
    doc.layers.new("OUTLINE", dxfattribs={"color": 7, "linetype": "CONTINUOUS"})
    doc.layers.new("HOLES",   dxfattribs={"color": 1, "linetype": "CONTINUOUS"})

    poly = msp.add_lwpolyline(
        outline_mm, format="xy",
        dxfattribs={"layer": "OUTLINE", "color": 256},
    )
    poly.close(True)
    entity_count = 1

    for cx, cy, r in holes_mm:
        msp.add_circle((cx, cy, 0.0), r, dxfattribs={"layer": "HOLES", "color": 256})
        entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — CONTOUR + HIERARCHY EXTRACTION (on the CLEANED BINARY MASK)
# ════════════════════════════════════════════════════════════════════════════

def extract_contours_with_hierarchy(cleaned_mask, epsilon_factor=0.5):
    """
    Find contours of the cleaned stroke mask using RETR_TREE so that the
    inner/outer edge of every drawn ring (rectangle outline, circle outline)
    is captured as an explicit parent/child pair. Each contour is simplified
    with approxPolyDP (closed=True, since these are always closed blob
    outlines).

    Returns:
      simplified: list of approx-point arrays (Nx1x2)
      parents:    list (same length) — parent index per contour, or -1
      children:   list (same length) — first-child index per contour, or -1
    """
    contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    simplified, parents, children = [], [], []

    # cv2.findContours preserves contour order alongside hierarchy rows,
    # so index i in `contours` corresponds to hierarchy[0][i].
    for i, cnt in enumerate(contours):
        if len(cnt) < 3:
            continue
        arc     = cv2.arcLength(cnt, closed=True)
        epsilon = epsilon_factor * arc / max(len(cnt), 1)
        epsilon = max(epsilon, 0.3)
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 2:
            continue
        simplified.append(approx)
        # hierarchy[0][i] = [next, previous, first_child, parent]
        h = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]
        parents.append(int(h[3]))
        children.append(int(h[2]))

    return simplified, parents, children


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — ALGEBRAIC LEAST-SQUARES SHAPE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def _fit_circle_algebraic(pts_xy, outlier_trim_passes=1):
    """
    Kasa algebraic circle fit, with optional outlier-rejection passes:
    after the first fit, points whose residual distance from the fitted
    circle exceeds 2*rms are dropped and the circle is refit. This guards
    against any stray point (e.g. a small connecting spur where two
    contours nearly touch) dragging the fit off-centre.
    Returns (cx, cy, r, rms).
    """
    x, y = pts_xy[:, 0].copy(), pts_xy[:, 1].copy()
    cx = cy = r = rms = 0.0
    passes = max(0, outlier_trim_passes)
    for _ in range(passes + 1):
        A  = np.column_stack([x, y, np.ones(len(x))])
        b_ = x**2 + y**2
        res, _, _, _ = np.linalg.lstsq(A, b_, rcond=None)
        cx = res[0] / 2.0
        cy = res[1] / 2.0
        r  = math.sqrt(abs(res[2] + cx**2 + cy**2))
        dists = np.sqrt((x - cx)**2 + (y - cy)**2)
        rms   = float(np.sqrt(((dists - r)**2).mean()))
        if passes <= 0:
            break
        keep = np.abs(dists - r) <= max(2.0 * rms, 1.0)
        if keep.sum() < 8 or keep.all():
            break
        x, y = x[keep], y[keep]
        passes -= 1
    return cx, cy, r, rms

def _fit_rect_algebraic(pts_xy):
    """
    Axis-aligned rectangle: min/max extents of the contour's point cloud.
    Because pts_xy now comes from a single clean blob outline (speckle and
    spur-free), the extreme points ARE the true corners — no percentile
    trimming is applied here, since trimming would clip the very corner
    points that define w/h and reintroduce an offset.
    Returns (cx, cy, w, h).
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    cx   = (float(x.min()) + float(x.max())) / 2.0
    cy   = (float(y.min()) + float(y.max())) / 2.0
    w    = float(x.max() - x.min())
    h    = float(y.max() - y.min())
    return cx, cy, w, h

def _classify_contour(pts_xy, min_pts_for_circle=12, circle_rms_tol=0.12):
    """
    Classify a contour as 'circle' or 'rect' using algebraic LS.
    Returns a shape dict or None.
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    w    = float(x.max() - x.min())
    h    = float(y.max() - y.min())
    if w < 1e-6 or h < 1e-6:
        return None

    aspect = min(w, h) / max(w, h)

    if aspect > 0.60 and len(pts_xy) >= min_pts_for_circle:
        try:
            cx, cy, r, rms = _fit_circle_algebraic(pts_xy, outlier_trim_passes=1)
            rel_err = rms / (r + 1e-9)
            if rel_err < circle_rms_tol:
                return {
                    'type': 'circle',
                    'cx': float(cx), 'cy': float(cy), 'r': float(r),
                    'err': float(rel_err),
                    'area': math.pi * r * r,
                    'w': w, 'h': h,
                }
        except Exception:
            pass

    cx_b, cy_b, w_b, h_b = _fit_rect_algebraic(pts_xy)
    return {
        'type': 'rect',
        'cx': cx_b, 'cy': cy_b, 'w': w_b, 'h': h_b,
        'area': w_b * h_b,
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — HIERARCHY-BASED OFFSET-PAIR AVERAGING
# ════════════════════════════════════════════════════════════════════════════

def _shapes_similar_for_pairing(a, b):
    """
    True only if `a` and `b` look like the inner/outer edge of the SAME
    drawn stroke: same classified type, centres essentially coincident,
    and overall size differing only by roughly a stroke-width.

    This deliberately does NOT match e.g. a big rectangle's inner "hole"
    contour against a much smaller cut-out that happens to be its child in
    the hierarchy tree — those have very different centres/sizes and will
    correctly be treated as independent shapes.
    """
    if a['type'] != b['type']:
        return False
    dc = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
    if a['type'] == 'circle':
        size_a, size_b = a['r'], b['r']
    else:
        size_a, size_b = max(a['w'], a['h']), max(b['w'], b['h'])
    if size_a <= 0 or size_b <= 0:
        return False
    size_rel   = abs(size_a - size_b) / max(size_a, size_b)
    center_tol = max(6.0, 0.08 * max(size_a, size_b))
    return dc <= center_tol and size_rel <= 0.30

def _average_shapes(a, b):
    """
    Average two paired inner/outer edge fits into a single shape that
    represents the TRUE CENTRELINE of the drawn stroke — i.e. the position
    a CAD/CAM user would actually want to cut along.
    """
    if a['type'] == 'circle':
        cx = (a['cx'] + b['cx']) / 2.0
        cy = (a['cy'] + b['cy']) / 2.0
        r  = (a['r']  + b['r'])  / 2.0
        return {'type': 'circle', 'cx': cx, 'cy': cy, 'r': r,
                'err': max(a.get('err', 0.0), b.get('err', 0.0)),
                'area': math.pi * r * r}
    else:
        cx = (a['cx'] + b['cx']) / 2.0
        cy = (a['cy'] + b['cy']) / 2.0
        w  = (a['w']  + b['w'])  / 2.0
        h  = (a['h']  + b['h'])  / 2.0
        return {'type': 'rect', 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': w * h}

def pair_by_hierarchy(shapes, parents, children):
    """
    For every contour, check its hierarchy parent and first-child. If either
    is a "same stroke, offset edge" match (per _shapes_similar_for_pairing),
    average the pair into one shape and mark both as consumed. Any contour
    left unconsumed is passed through standalone (e.g. a filled circle drawn
    without a separate hole contour, or a shape whose pair didn't survive
    classification).
    """
    n = len(shapes)
    consumed = [False] * n
    output = []

    for i in range(n):
        if consumed[i] or shapes[i] is None:
            continue
        s_i = shapes[i]
        partner = -1

        c = children[i]
        if c != -1 and 0 <= c < n and not consumed[c] and shapes[c] is not None \
                and _shapes_similar_for_pairing(s_i, shapes[c]):
            partner = c

        if partner == -1:
            p = parents[i]
            if p != -1 and 0 <= p < n and not consumed[p] and shapes[p] is not None \
                    and _shapes_similar_for_pairing(s_i, shapes[p]):
                partner = p

        if partner != -1:
            output.append(_average_shapes(s_i, shapes[partner]))
            consumed[i] = True
            consumed[partner] = True
        else:
            output.append(s_i)
            consumed[i] = True

    return output


# ════════════════════════════════════════════════════════════════════════════
# STEP 10 — RESIDUAL DEDUP (proximity-average safety net) + ABSOLUTE FILTER
# ════════════════════════════════════════════════════════════════════════════

def _dedup_residual(shapes, kind):
    """
    Safety net for any near-duplicate shapes that the hierarchy pass didn't
    catch (e.g. extra contour fragments from a thicker stroke). Groups
    same-type shapes by proximity and AVERAGES each group into one shape
    (instead of arbitrarily discarding the larger one — averaging keeps the
    result centred on the true geometry rather than biased toward whichever
    edge happened to be "smaller").
    """
    used = [False] * len(shapes)
    result = []
    for i, s in enumerate(shapes):
        if used[i]:
            continue
        grp = [s]
        used[i] = True
        for j in range(i + 1, len(shapes)):
            if used[j]:
                continue
            s2 = shapes[j]
            if s2['type'] != kind:
                continue
            dc = math.hypot(s['cx'] - s2['cx'], s['cy'] - s2['cy'])
            if kind == 'circle':
                size_a, size_b = s['r'], s2['r']
            else:
                size_a, size_b = max(s['w'], s['h']), max(s2['w'], s2['h'])
            size_rel   = abs(size_a - size_b) / max(size_a, size_b, 1e-9)
            center_tol = max(8.0, 0.10 * max(size_a, size_b))
            if dc <= center_tol and size_rel <= 0.30:
                grp.append(s2)
                used[j] = True

        if len(grp) == 1:
            result.append(grp[0])
        else:
            if kind == 'circle':
                cx = float(np.mean([g['cx'] for g in grp]))
                cy = float(np.mean([g['cy'] for g in grp]))
                r  = float(np.mean([g['r']  for g in grp]))
                result.append({'type': 'circle', 'cx': cx, 'cy': cy, 'r': r,
                                'err': max(g.get('err', 0.0) for g in grp),
                                'area': math.pi * r * r})
            else:
                cx = float(np.mean([g['cx'] for g in grp]))
                cy = float(np.mean([g['cy'] for g in grp]))
                w  = float(np.mean([g['w']  for g in grp]))
                h  = float(np.mean([g['h']  for g in grp]))
                result.append({'type': 'rect', 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': w * h})
    return result


def _circles_overlap(a, b):
    """
    True if circle `a` and circle `b` substantially overlap — i.e. one
    circle's centre lies inside the other's disk. This catches the case
    where a single wobbly hand-drawn ring produces an outer-edge contour
    and an inner-edge ("hole") contour whose CENTRES differ by more than
    `_shapes_similar_for_pairing`'s tight hierarchy-pairing tolerance (so
    they were never paired/averaged) and also differ enough in radius that
    `_dedup_residual`'s tolerance didn't merge them either — leaving two
    near-duplicate circles of the SAME physical hole, one slightly smaller
    and offset from the other (the "smallest diameter circle" symptom).
    """
    dc = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
    return dc < max(a['r'], b['r'])

def _dedup_overlapping_circles(circles):
    """
    Cluster circles whose disks overlap (per `_circles_overlap`) and keep
    only ONE representative per cluster — the one with the lowest algebraic
    fit error (`err`), tie-broken by the larger radius (the outer edge is
    the more representative boundary of a stroke). All other members of the
    cluster are dropped as duplicates of the same physical hole.
    """
    n = len(circles)
    used = [False] * n
    result = []
    # Process largest-first so a big "true" circle absorbs smaller
    # overlapping duplicates rather than the reverse.
    order = sorted(range(n), key=lambda i: circles[i]['r'], reverse=True)
    for i in order:
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            if _circles_overlap(circles[i], circles[j]):
                cluster.append(j)
                used[j] = True
        best = min(cluster, key=lambda k: (circles[k].get('err', 0.0), -circles[k]['r']))
        result.append(circles[best])
    return result

def _circle_overlaps_any_rect(c, rects, center_inside_frac=0.5):
    """
    True if circle `c`'s centre lies inside (or within `center_inside_frac`
    of its radius from) any rect's bounding box.

    In sheet-layout sketches a real bore/hole is drawn in open material, not
    stacked on top of a separate cut-out rectangle. When a rectangle's
    "hole" contour has rounded/eroded corners (common after MORPH_OPEN on a
    hand-drawn line of uneven thickness), its aspect ratio can exceed the
    circle-fit threshold and the Kasa fit happily returns a "circle" sitting
    on top of that rectangle — the "circle at lower-bottom-left" symptom.
    Such circles are rejected here.
    """
    for r in rects:
        x0 = r['cx'] - r['w'] / 2.0
        x1 = r['cx'] + r['w'] / 2.0
        y0 = r['cy'] - r['h'] / 2.0
        y1 = r['cy'] + r['h'] / 2.0
        dx = max(x0 - c['cx'], 0.0, c['cx'] - x1)
        dy = max(y0 - c['cy'], 0.0, c['cy'] - y1)
        dist = math.hypot(dx, dy)
        if dist < c['r'] * center_inside_frac:
            return True
    return False


def build_clean_shapes(simplified_contours, parents, children, img_w, img_h,
                       min_shape_area=None, rect_aspect_min=0.15,
                       rect_rel_area_min=0.01, circle_rel_radius_min=0.4,
                       circle_rect_overlap_frac=0.5):
    """
    Full v10 pipeline:
      a. Classify every contour as circle or rect.
      b. Pair inner/outer stroke edges via hierarchy and AVERAGE -> true
         centreline geometry (replaces v9.1's "discard the larger" hack).
      c. Residual proximity dedup (averaging) as a safety net.
      d. Absolute-area noise filter (drops any leftover speckle-derived
         micro-shapes).
      e. Annotation filter (scale-invariant — does NOT depend on image
         resolution):
           - Rect aspect filter: drops thin line-like contours
             (dimension lines / arrow shafts) whose min(w,h)/max(w,h)
             falls below `rect_aspect_min`. A real cut-rectangle is never
             that thin; a dimension line is.
           - Rect relative-area filter: drops rects whose area is below
             `rect_rel_area_min` of the LARGEST rect found (the outer
             sheet/board boundary). Annotation text blocks (e.g. "900mm")
             are small relative to the board; real cut-outs are not.
      f. Spurious-circle filters (run BEFORE the relative-radius filter,
         since these artefacts can be similar in size to the real circle):
           - Circle-circle overlap dedup: collapses near-duplicate circles
             whose disks overlap (offset inner/outer edges of one wobbly
             stroke) down to a single best-fit circle.
           - Circle-rect overlap rejection: drops any circle centred inside
             a detected rectangle's bounding box (a rectangle "hole"
             contour misclassified as round).
      g. Circle relative-radius filter: drops circles whose radius is below
         `circle_rel_radius_min` of the largest remaining circle (text
         glyph circles like the "0" in "900mm").
      h. NO re-centering. Every shape keeps its measured (cx, cy) — output
         geometry matches the Canny image pixel-for-pixel.
    Returns final_shapes, circles_final, rects_final.
    """
    classified = []
    for cnt in simplified_contours:
        pts_xy = np.array([[float(p[0][0]), float(p[0][1])] for p in cnt], dtype=float)
        if len(pts_xy) < 2:
            classified.append(None)
            continue
        classified.append(_classify_contour(pts_xy))

    # Step 1: hierarchy-based offset-pair averaging
    paired = pair_by_hierarchy(classified, parents, children)
    paired = [s for s in paired if s is not None]

    # Step 2: residual proximity dedup, per type
    circles = [s for s in paired if s['type'] == 'circle']
    rects   = [s for s in paired if s['type'] == 'rect']
    circles = _dedup_residual(circles, 'circle')
    rects   = _dedup_residual(rects,   'rect')

    # Step 3: absolute-area noise filter
    if min_shape_area is None:
        min_shape_area = max(16.0, 0.00003 * img_w * img_h)

    circles = [c for c in circles if (math.pi * c['r'] * c['r']) >= min_shape_area]
    rects   = [r for r in rects   if (r['w'] * r['h'])           >= min_shape_area]

    # Step 4a: rect aspect-ratio filter (drop thin dimension-line shafts)
    rects = [r for r in rects
             if max(r['w'], r['h']) > 0
             and (min(r['w'], r['h']) / max(r['w'], r['h'])) >= rect_aspect_min]

    # Step 4b: rect relative-area filter (drop annotation text blocks,
    # measured against the largest surviving rect — typically the board)
    if rects:
        max_rect_area = max(r['w'] * r['h'] for r in rects)
        rects = [r for r in rects if (r['w'] * r['h']) >= rect_rel_area_min * max_rect_area]

    # Step 4c: circle-circle overlap dedup (collapse offset-duplicate rings)
    circles = _dedup_overlapping_circles(circles)

    # Step 4d: circle-rect overlap rejection (drop rect-corner misfits).
    # The largest rect is treated as the outer board/boundary — real holes
    # are legitimately drawn INSIDE it, so it's excluded from this check.
    # Only the smaller "cut-out" rects are considered: a circle centred on
    # top of one of those is almost certainly that rect's hole-contour
    # misclassified as round, not a separate real hole.
    if len(rects) > 1:
        rects_sorted_desc = sorted(rects, key=lambda r: r['w'] * r['h'], reverse=True)
        cutout_rects = rects_sorted_desc[1:]
    else:
        cutout_rects = []
    circles = [c for c in circles if not _circle_overlaps_any_rect(c, cutout_rects, circle_rect_overlap_frac)]

    # Step 4e: circle relative-radius filter (drop text-character circles,
    # measured against the largest surviving circle)
    if circles:
        max_r = max(c['r'] for c in circles)
        circles = [c for c in circles if c['r'] >= circle_rel_radius_min * max_r]

    circles_final = circles
    rects_final   = rects
    final_shapes  = list(rects_final) + list(circles_final)
    return final_shapes, circles_final, rects_final


# ════════════════════════════════════════════════════════════════════════════
# STEP 11 — CLEAN DXF EXPORT (one entity per shape, true detected position)
# ════════════════════════════════════════════════════════════════════════════

def build_clean_dxf(final_shapes, img_w, img_h, out_path):
    """
    Write a DXF containing only true closed shapes:
      - CIRCLE entities for circles
      - Closed LWPOLYLINE (rectangle) entities for rects
    Coordinates are the shapes' measured (cx, cy) — identical to their
    position in the Canny/cleaned-mask image (origin top-left, Y-down,
    same convention as `analysis.coordSystem`). No additional offset or
    re-centering is applied.
    """
    if not HAS_DXF or not final_shapes:
        return None, 0, 0

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
            msp.add_circle(
                (s['cx'], s['cy'], 0.0),
                s['r'],
                dxfattribs={"layer": "CIRCLES", "color": 256}
            )
            entity_count += 1

        elif s['type'] == 'rect':
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
# STEP 12 — PNG PREVIEW (side-by-side comparison)
# ════════════════════════════════════════════════════════════════════════════

def build_comparison_png(edges, final_shapes, img_w, img_h, out_path):
    if not HAS_CV or not np:
        return False

    try:
        left = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        left[:] = (15, 12, 10)
        left[edges > 0] = (255, 255, 255)

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

        font   = cv2.FONT_HERSHEY_SIMPLEX
        n_circ = sum(1 for s in final_shapes if s['type'] == 'circle')
        n_rect = sum(1 for s in final_shapes if s['type'] == 'rect')
        cv2.putText(left,  "Canny Edge Detection",
                    (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(right,
                    f"Final Shapes: {n_rect} rect(s) + {n_circ} circle(s)",
                    (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        sep   = np.full((img_h, 4, 3), 40, dtype=np.uint8)
        panel = np.concatenate([left, sep, right], axis=1)

        ok = cv2.imwrite(str(out_path), panel)
        return bool(ok and out_path.exists())
    except Exception as e:
        sys.stderr.write(f"PNG preview error: {e}\n")
        return False


# ════════════════════════════════════════════════════════════════════════════
# STEP 13 — PDF EXPORT
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
        c.drawString(margin, page_h - 24, "SheetForge — Hierarchy-Aware Shape Fitting Preview")
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
                            "SheetForge v10  •  Hierarchy-Aware Offset-Pair Averaging  •  Clean DXF")
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
    canny_low      = int(opts.get("cannyLow",    20))
    canny_high     = int(opts.get("cannyHigh",   80))
    epsilon_factor = float(opts.get("epsilonFactor", 0.5))
    min_blob_area  = int(opts.get("minBlobArea", 50))    # px-area: kills isolated dots/specks
    min_shape_area = opts.get("minShapeArea", None)       # px-area: absolute noise filter
    if min_shape_area is not None:
        min_shape_area = float(min_shape_area)
    rect_aspect_min       = float(opts.get("rectAspectMin", 0.15))     # drop thin dimension lines
    rect_rel_area_min     = float(opts.get("rectRelAreaMin", 0.01))    # drop annotation text blocks
    circle_rel_radius_min = float(opts.get("circleRelRadiusMin", 0.4)) # drop text-glyph circles
    circle_rect_overlap_frac = float(opts.get("circleRectOverlapFrac", 0.5))  # drop circles sitting on rects

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
    opened    = morph_clean(binary)
    opened_px = int(np.count_nonzero(opened))
    steps.append(step_record("CV-4: MORPH_OPEN", f"{white_px - opened_px} spur px removed", t0))

    # ── STEP 5: Connected-Component Speckle Removal ──────────────────────
    t0 = now_ms()
    cleaned, removed_blobs, removed_px = remove_small_blobs(opened, min_blob_area)
    steps.append(step_record(
        f"CV-5: Connected-Component Filter (minBlobArea={min_blob_area}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px) — 100% dot-free mask", t0))

    # ── STEP 5b: Zhang-Suen Thinning (1px skeleton, for Hough hole-circles)
    t0 = now_ms()
    skeleton    = zhang_suen_thinning(cleaned)
    skeleton_px = int(np.count_nonzero(skeleton))
    steps.append(step_record(
        "CV-5b: Zhang-Suen Thinning",
        f"{skeleton_px}px 1-pixel-wide skeleton (centreline) extracted", t0))

    # ── STEP 6: Canny (visualisation only) ────────────────────────────────
    t0 = now_ms()
    edges_raw = canny_edges(cleaned, canny_low, canny_high)
    edges, edge_dot_blobs, edge_dot_px = clean_edge_preview(edges_raw, min_blob_area=3)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(f"CV-6: Canny (lo={canny_low}, hi={canny_high}) + dot cleanup",
                              f"{edge_px} edge px (preview only), {edge_dot_blobs} stray dot(s) removed", t0))

    # ── STEP 7: Contour + hierarchy extraction on CLEANED MASK ────────────
    t0 = now_ms()
    simplified_contours, parents, children = extract_contours_with_hierarchy(cleaned, epsilon_factor)
    total_pts = sum(len(c) for c in simplified_contours)
    steps.append(step_record(
        f"CV-7: findContours(RETR_TREE) on cleaned mask + approxPolyDP (ε={epsilon_factor})",
        f"{len(simplified_contours)} contours  |  {total_pts} vertices", t0))

    # ── STEP 8-10: classify, pair, dedup, filter ───────────────────────────
    t0 = now_ms()
    final_shapes, circles_f, rects_f = build_clean_shapes(
        simplified_contours, parents, children, img_w, img_h, min_shape_area,
        rect_aspect_min=rect_aspect_min,
        rect_rel_area_min=rect_rel_area_min,
        circle_rel_radius_min=circle_rel_radius_min,
        circle_rect_overlap_frac=circle_rect_overlap_frac)
    n_circles = len(circles_f)
    n_rects   = len(rects_f)
    steps.append(step_record(
        "LS-8/9/10: LS classification + hierarchy offset-pair averaging + residual dedup + area filter",
        f"{len(simplified_contours)} raw contours → {n_rects} rect(s) + {n_circles} circle(s)  "
        f"(true measured positions, no re-centering)",
        t0))

    # ── Output dir ────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str   = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    pdf_name = f"design_{ts_str}.pdf"
    png_name = f"preview_{ts_str}.png"

    dxf_path = server_out_dir / dxf_name
    pdf_path = server_out_dir / pdf_name
    png_path = server_out_dir / png_name

    # ── STEP 11: Clean DXF export ─────────────────────────────────────────
    t0 = now_ms()
    _, entity_count, dxf_size = build_clean_dxf(final_shapes, img_w, img_h, dxf_path)

    dxf_content_str = ""
    if dxf_size and dxf_size > 0:
        try:
            with open(dxf_path, encoding="utf-8", errors="replace") as f:
                dxf_content_str = f.read(200_000)
        except Exception:
            pass

    steps.append(step_record(
        "DXF-11: Clean export (CIRCLE + closed LWPOLYLINE, one entity per shape, true position)",
        f"{entity_count} entities  |  {dxf_size // 1024 if dxf_size else 0} KB", t0))

    # ── STEP 11b: Real-millimetre outline + Hough-circle holes DXF ────────
    t0 = now_ms()
    mm_dxf_name = f"design_mm_{ts_str}.dxf"
    mm_dxf_path = server_out_dir / mm_dxf_name

    outline_mm, holes_mm, mm_scale = extract_main_outline_and_holes(
        cleaned, skeleton, dpi, img_h)
    _, mm_entity_count, mm_dxf_size = build_mm_dxf(outline_mm, holes_mm, mm_dxf_path)

    mm_dxf_content_str = ""
    if mm_dxf_size and mm_dxf_size > 0:
        try:
            with open(mm_dxf_path, encoding="utf-8", errors="replace") as f:
                mm_dxf_content_str = f.read(200_000)
        except Exception:
            pass

    steps.append(step_record(
        "DXF-11b: Real-mm export (main external contour, tight approxPolyDP, "
        "Hough-circle holes from skeleton, $INSUNITS=mm)",
        f"{len(outline_mm)}-pt outline + {len(holes_mm)} hole(s)  |  "
        f"{mm_dxf_size // 1024 if mm_dxf_size else 0} KB  |  scale={mm_scale:.4f} mm/px", t0))

    # ── STEP 12: PNG comparison preview ───────────────────────────────────
    t0 = now_ms()
    png_ok   = build_comparison_png(edges, final_shapes, img_w, img_h, png_path)
    png_size = png_path.stat().st_size if png_ok and png_path.exists() else 0
    steps.append(step_record(
        "PNG-12: Side-by-side comparison preview (Canny vs final shapes)",
        f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # ── STEP 13: PDF export ───────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-13: Export edge preview", "OK" if pdf_ok else "FAILED", t0))

    # ── Analysis summary ──────────────────────────────────────────────────
    n_circ_final = sum(1 for s in final_shapes if s['type'] == 'circle')
    n_rect_final = sum(1 for s in final_shapes if s['type'] == 'rect')
    analysis = {
        "width"          : float(img_w),
        "height"         : float(img_h),
        "dpi"            : dpi,
        "edgePixels"     : edge_px,
        "edges"          : entity_count,
        "contours"       : len(simplified_contours),
        "mergedContours" : len(final_shapes),
        "closedContours" : len(final_shapes),
        "totalVertices"  : total_pts,
        "blurKsize"      : blur_ksize,
        "cannyLow"       : canny_low,
        "cannyHigh"      : canny_high,
        "epsilonFactor"  : epsilon_factor,
        "minBlobArea"    : min_blob_area,
        "speckleBlobsRemoved": removed_blobs,
        "imgW"           : img_w,
        "imgH"           : img_h,
        "scaleMmPerDu"   : round(25.4 / dpi, 4),
        "coordSystem"    : "origin=top-left px, Y-down (image convention), no offset/re-centering",
        "circlesDetected": n_circ_final,
        "rectsDetected"  : n_rect_final,
        "shapeSummary"   : (
            f"{n_rect_final} rectangle(s), {n_circ_final} circle(s)"
            f" — speckle-free mask, hierarchy offset-pairs averaged to centreline,"
            f" true measured positions"
        ),
        "skeletonPixels" : skeleton_px,
        "mmScale"        : round(mm_scale, 6),
        "mmOutlinePoints": len(outline_mm),
        "mmHolesDetected": len(holes_mm),
        "mmCoordSystem"  : "origin=bottom-left, Y-up, millimetres ($INSUNITS=4)",
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
            "mmDxfFilename"   : mm_dxf_name if mm_dxf_size else "",
            "mmDxfAbsPath"    : str(mm_dxf_path) if mm_dxf_size else "",
            "mmDxfEntities"   : mm_entity_count,
            "mmDxfFileSize"   : mm_dxf_size or 0,
            "gcodeFiles"      : {},
            "gcodeFilePaths"  : {},
        },
        "dxfContent"   : dxf_content_str,
        "mmDxfContent" : mm_dxf_content_str,
        "dxfAvailable" : bool(dxf_size and dxf_size > 0),
        "mmDxfAvailable": bool(mm_dxf_size and mm_dxf_size > 0),
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
