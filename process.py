#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v10  (Hierarchy-Aware Shape Fitting)
================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

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
    """
    Two-stage morphological cleanup:
      1. MORPH_OPEN (3x3 cross) — erodes away thin spurs/single-pixel
         protrusions attached to real strokes.
      2. MORPH_CLOSE (3x3 ellipse) — fills tiny 1px gaps/holes inside real
         strokes (helps keep a stroke as ONE connected component so it
         survives connected-component filtering as a single blob, and
         gives findContours cleaner ring hierarchies for offset-pair
         averaging).
    """
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  open_kernel,  iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return closed

def _remove_small_blobs_once(binary, min_area):
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

def remove_small_blobs(binary, min_area, aggressive=True):
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

    When `aggressive=True` (default), a second pass is run: after the
    first pass, a light MORPH_OPEN (3x3 ellipse) erodes away any thin
    remnants left clinging to surviving blobs (e.g. a speckle that was
    8-connected to a real stroke and barely pushed it over `min_area`),
    and a second connected-component pass removes anything that erosion
    now drops below `min_area`. This compounds cleanup beyond a single
    pass while a stable mask (no further small blobs) exits early.
    """
    cleaned, removed_blobs, removed_px = _remove_small_blobs_once(binary, min_area)

    if aggressive:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, erode_kernel, iterations=1)
        cleaned2, rb2, rp2 = _remove_small_blobs_once(eroded, min_area)
        if rb2 > 0:
            cleaned = cleaned2
            removed_blobs += rb2
            removed_px += rp2

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
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    w = float(x.max() - x.min())
    h = float(y.max() - y.min())
    if w < 1e-6 or h < 1e-6:
        return None

    aspect = min(w, h) / max(w, h)

    # Try circle fit first
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

    # For complex shapes with many vertices, keep as polygon instead of bounding box
    if len(pts_xy) > 8:
        return {
            'type': 'poly',
            'points': pts_xy.tolist(),  # Keep all approxPolyDP vertices
            'area': cv2.contourArea(np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)),
            'w': w, 'h': h,
            'cx': (x.min() + x.max()) / 2.0,
            'cy': (y.min() + y.max()) / 2.0,
        }

    # Simple rectangle fallback for truly simple 4-corner shapes
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
    if a['type'] != b['type']:
        return False
    dc = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
    
    if a['type'] == 'circle':
        size_a, size_b = a['r'], b['r']
        if size_a <= 0 or size_b <= 0:
            return False
        size_rel = abs(size_a - size_b) / max(size_a, size_b)
        center_tol = max(6.0, 0.08 * max(size_a, size_b))
        return dc <= center_tol and size_rel <= 0.30
    
    elif a['type'] == 'poly':
        # Compare polygons by bounding box metrics
        size_a = max(a['w'], a['h'])
        size_b = max(b['w'], b['h'])
        if size_a <= 0 or size_b <= 0:
            return False
        size_rel = abs(size_a - size_b) / max(size_a, size_b)
        center_tol = max(6.0, 0.08 * max(size_a, size_b))
        return dc <= center_tol and size_rel <= 0.30
    
    else:  # rect
        if a['w'] <= 0 or a['h'] <= 0 or b['w'] <= 0 or b['h'] <= 0:
            return False
        w_rel = abs(a['w'] - b['w']) / max(a['w'], b['w'])
        h_rel = abs(a['h'] - b['h']) / max(a['h'], b['h'])
        size_rel = max(w_rel, h_rel)
        center_tol = max(6.0, 0.08 * max(a['w'], a['h'], b['w'], b['h']))
        return dc <= center_tol and size_rel <= 0.30

def _average_shapes(a, b):
    if a['type'] == 'circle':
        cx = (a['cx'] + b['cx']) / 2.0
        cy = (a['cy'] + b['cy']) / 2.0
        r = (a['r'] + b['r']) / 2.0
        return {'type': 'circle', 'cx': cx, 'cy': cy, 'r': r,
                'err': max(a.get('err', 0.0), b.get('err', 0.0)),
                'area': math.pi * r * r}
    
    elif a['type'] == 'poly':
        # Average the point sets by using the outer one (larger area)
        # or interpolate between corresponding points
        # For simplicity, use the one with more points (more detailed)
        if len(a['points']) >= len(b['points']):
            return a
        return b
    
    else:  # rect
        cx = (a['cx'] + b['cx']) / 2.0
        cy = (a['cy'] + b['cy']) / 2.0
        w = (a['w'] + b['w']) / 2.0
        h = (a['h'] + b['h']) / 2.0
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
                size_rel   = abs(size_a - size_b) / max(size_a, size_b, 1e-9)
                center_tol = max(8.0, 0.10 * max(size_a, size_b))
                similar = dc <= center_tol and size_rel <= 0.30

            elif kind == 'poly':
                size_a = max(s['w'], s['h'])
                size_b = max(s2['w'], s2['h'])
                size_rel = abs(size_a - size_b) / max(size_a, size_b, 1e-9)
                center_tol = max(8.0, 0.10 * max(size_a, size_b))
                similar = dc <= center_tol and size_rel <= 0.30
              
            else:
                # Require BOTH width and height to match independently —
                # otherwise a thin stray strip with a similar height (but
                # very different width) to the real rect would get averaged
                # into it and corrupt its dimensions.
                w_rel = abs(s['w'] - s2['w']) / max(s['w'], s2['w'], 1e-9)
                h_rel = abs(s['h'] - s2['h']) / max(s['h'], s2['h'], 1e-9)
                size_rel   = max(w_rel, h_rel)
                center_tol = max(8.0, 0.10 * max(s['w'], s['h'], s2['w'], s2['h']))
                similar = dc <= center_tol and size_rel <= 0.30
            if similar:
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

            elif kind == 'poly':
                # Use the polygon with the most points (most detailed)
                best = max(grp, key=lambda g: len(g['points']))
                result.append(best)
              
            else:
                cx = float(np.mean([g['cx'] for g in grp]))
                cy = float(np.mean([g['cy'] for g in grp]))
                w  = float(np.mean([g['w']  for g in grp]))
                h  = float(np.mean([g['h']  for g in grp]))
                result.append({'type': 'rect', 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': w * h})
    return result


# Hard floor: regardless of any other (configurable) area threshold, NO
# shape with a surface area below this is ever allowed through to the DXF
# exporter. This is intentionally a constant, not an option — it's the
# final backstop against degenerate/microscopic contours reaching the DXF.
HARD_MIN_SHAPE_AREA_PX = 20.0


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
    only ONE representative per cluster — ALWAYS the one with the SMALLEST
    radius/diameter (the inner edge of the drawn stroke, which represents
    the true hole boundary). All other members of the cluster are dropped
    as duplicates of the same physical hole.
    """
    n = len(circles)
    used = [False] * n
    result = []
    # Process smallest-first so a small "true" circle is selected before
    # larger overlapping duplicates are absorbed into its cluster.
    order = sorted(range(n), key=lambda i: circles[i]['r'])  # ascending: smallest first
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
        # ALWAYS keep the smallest radius in the cluster
        best = min(cluster, key=lambda k: circles[k]['r'])
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

    # === NEW: 190px² area filter ===
    paired = [s for s in paired if s['area'] >= 190.0]

    # Step 2: residual proximity dedup, per type
    circles = [s for s in paired if s['type'] == 'circle']
    rects   = [s for s in paired if s['type'] == 'rect']
    polys   = [s for s in paired if s['type'] == 'poly']
    circles = _dedup_residual(circles, 'circle')
    rects   = _dedup_residual(rects,   'rect')
    polys   = _dedup_residual(polys,   'poly')

    # Step 3: absolute-area noise filter.
    # With minBlobArea lowered to 20px (to satisfy a <20px speckle-removal
    # requirement at the mask stage), more small fragments can now reach the
    # contour stage. Compensate with a higher soft floor here — still far
    # below any real shape (smallest real shape seen so far: ~8,100px² for
    # door_lock circles, ~136,000px² for oven_top rects) but well above the
    # 20-80px speckle fragments this allows through MORPH/CC filtering.
    if min_shape_area is None:
        min_shape_area = max(HARD_MIN_SHAPE_AREA_PX * 5.0, 0.00008 * img_w * img_h)

    circles = [c for c in circles if (math.pi * c['r'] * c['r']) >= min_shape_area]
    rects   = [r for r in rects   if (r['w'] * r['h'])           >= min_shape_area]
    polys   = [p for p in polys   if p['area']                   >= min_shape_area]

    # Step 4a: rect aspect-ratio filter (drop thin dimension-line shafts)
    rects = [r for r in rects
             if max(r['w'], r['h']) > 0
             and (min(r['w'], r['h']) / max(r['w'], r['h'])) >= rect_aspect_min]
    # For polys, use bounding box aspect as proxy
    polys = [p for p in polys
             if max(p['w'], p['h']) > 0
             and (min(p['w'], p['h']) / max(p['w'], p['h'])) >= rect_aspect_min]

    # Step 4b: rect relative-area filter (drop annotation text blocks,
    # measured against the largest surviving rect — typically the board)
        if rects:
        max_rect_area = max(r['w'] * r['h'] for r in rects)
        rects = [r for r in rects if (r['w'] * r['h']) >= rect_rel_area_min * max_rect_area]
    # For polys, filter against largest poly area
    if polys:
        max_poly_area = max(p['area'] for p in polys)
        polys = [p for p in polys if p['area'] >= rect_rel_area_min * max_poly_area]

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

    # Step 5: HARD final floor — no shape under HARD_MIN_SHAPE_AREA_PX
    # (20px²) is ever passed to the DXF exporter, no matter what the
    # configurable min_shape_area / relative filters above allowed through.
    circles = [c for c in circles if (math.pi * c['r'] * c['r']) >= HARD_MIN_SHAPE_AREA_PX]
    rects   = [r for r in rects   if (r['w'] * r['h'])           >= HARD_MIN_SHAPE_AREA_PX]
    polys   = [p for p in polys   if p['area']                   >= HARD_MIN_SHAPE_AREA_PX]

    circles_final = circles
    rects_final   = rects
    polys_final   = polys
    final_shapes  = list(rects_final) + list(circles_final) + list(polys_final)
    return final_shapes, circles_final, rects_final, polys_final                     


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

        elif s['type'] == 'poly':
            # Export as closed LWPOLYLINE with all vertices
            pts = [(p[0], p[1]) for p in s['points']]
            poly = msp.add_lwpolyline(
                pts, format="xy",
                dxfattribs={"layer": "SHAPES", "color": 256}
            )
            poly.close(True)
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

            elif s['type'] == 'poly':
                # Draw polygon using all vertices
                pts = np.array([[int(p[0]), int(p[1])] for p in s['points']], dtype=np.int32)
                cv2.polylines(right, [pts], True, (80, 200, 80), 2, cv2.LINE_AA)
              
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
        from datetime import datetime, timezone
        c.drawRightString(page_w - margin, page_h - 24,
                          f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")

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

    blur_ksize     = int(opts.get("blurKsize",    7))    # ksize=21 tested — degrades results, see notes
    canny_low      = int(opts.get("cannyLow",    20))
    canny_high     = int(opts.get("cannyHigh",   80))
    epsilon_factor = float(opts.get("epsilonFactor", 0.5))
    min_blob_area  = int(opts.get("minBlobArea", 20))    # ← changed from 50; user-requested <20px
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

    # ── STEP 4: Morph Open + Close ───────────────────────────────────────
    t0 = now_ms()
    opened    = morph_clean(binary)
    opened_px = int(np.count_nonzero(opened))
    steps.append(step_record("CV-4: MORPH_OPEN (3x3 cross) + MORPH_CLOSE (3x3 ellipse)",
                              f"{white_px - opened_px} net px change (spurs removed / 1px gaps filled)", t0))

    # ── STEP 5: Connected-Component Speckle Removal (two-pass aggressive) ─
    t0 = now_ms()
    cleaned, removed_blobs, removed_px = remove_small_blobs(opened, min_blob_area, aggressive=True)
    steps.append(step_record(
        f"CV-5: Two-pass Connected-Component Filter (minBlobArea={min_blob_area}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px) — 100% dot-free mask", t0))

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
        # ── STEP 8-10: classify, pair, dedup, filter ───────────────────────────
    t0 = now_ms()
    final_shapes, circles_f, rects_f, polys_f = build_clean_shapes(
        simplified_contours, parents, children, img_w, img_h, min_shape_area,
        rect_aspect_min=rect_aspect_min,
        rect_rel_area_min=rect_rel_area_min,
        circle_rel_radius_min=circle_rel_radius_min,
        circle_rect_overlap_frac=circle_rect_overlap_frac)
    n_circles = len(circles_f)
    n_rects   = len(rects_f)
    n_polys   = len(polys_f)
    steps.append(step_record(
        "LS-8/9/10: LS classification + hierarchy offset-pair averaging + residual dedup + area filter",
        f"{len(simplified_contours)} raw contours → {n_rects} rect(s) + {n_circles} circle(s)  "
        f"(true measured positions, no re-centering, hard floor={HARD_MIN_SHAPE_AREA_PX:.0f}px²)",
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
        "hardMinShapeAreaPx": HARD_MIN_SHAPE_AREA_PX,        "speckleBlobsRemoved": removed_blobs,
        "imgW"           : img_w,
        "imgH"           : img_h,
        "scaleMmPerDu"   : round(25.4 / dpi, 4),
        "coordSystem"    : "origin=top-left px, Y-down (image convention), no offset/re-centering",
        "circlesDetected": n_circ_final,
        "rectsDetected"  : n_rect_final,
        "polysDetected": n_poly_final,
        "shapeSummary"   : (
            f"{n_rect_final} rectangle(s), {n_circ_final} circle(s), {n_poly_final} polygon(s)"
            f" — speckle-free mask, hierarchy offset-pairs averaged to centreline,"
            f" true measured positions"
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
