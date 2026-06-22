#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v12  (Adaptive Neural-Style Preprocessing + Rect Angle Detector)
================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

WHAT CHANGED IN v12 AND WHY
-----------------------------
Six targeted changes over v11:

  1. MORPH_OPEN REMOVED — MORPH_CLOSE ONLY
     cv2.MORPH_OPEN was silently eroding thin corners and short spurs on
     lighter/noisier hand-drawn inputs, destroying real geometry before
     contour extraction ever saw it. Speckle removal is owned entirely by
     the connected-component filter (Step 5), which makes exact decisions
     on blob area without any erosion side-effects. MORPH_CLOSE is kept
     (fills 1px gaps, keeps stroke as one connected blob). Applies in both
     morph_clean() and the aggressive-pass inside remove_small_blobs().

  2. EPSILON_FACTOR 0.5 → 0.03
     The approxPolyDP simplification factor was far too aggressive: at 0.5
     it was collapsing real polygon vertices away, making multi-sided cuts
     look like rectangles. At 0.03 it preserves fine corner detail while
     still removing trivially-collinear duplicate points.

  3. HARD AREA FILTER 190px² → 145px²
     The post-hierarchy-pair area floor was dropping some small genuine
     features (bolt holes near the edge of a part). 145px² sits comfortably
     above real speckle (≤20px²) while letting more real shapes through.

  4. DUPLICATE POLY DRAWING BLOCK REMOVED
     build_comparison_png() had a dead (unreachable) second `elif s['type']
     == 'poly':` block — an identical branch that could never execute
     because Python short-circuits at the first matching elif. Removed.

  5. RECTANGLE DETECTOR: 4-8 VERTICES + ~90° ANGLE TEST
     The old classifier forced any contour with ≤8 points into a 'rect'
     regardless of whether its angles were actually right-angles. L-shaped
     notches, hexagons, and other multi-vertex shapes were being silently
     crushed into wrong bounding-box rectangles. The new _is_rectangle_like()
     helper checks that ≥70% of the interior angles are within 25° of 90°
     (OR near 180° — forgiving near-collinear approxPolyDP artefact points
     along a straight edge). Only contours that pass that test are classified
     as 'rect'; everything else stays as 'poly' with its real vertices intact.

  6. ADAPTIVE NEURAL-STYLE OPENCV PREPROCESSING PIPELINE (new Steps 1.5–2.5)
     Three new enhancement stages run before median blur / thresholding:
       a. CLAHE (Step 1.5) — contrast-limited adaptive histogram equalisation
          on 8×8 tiles. Evens out faint/shadowed regions of the scan so
          downstream stages see every stroke at comparable brightness.
       b. Bilateral Denoise (Step 2) — edge-preserving noise reduction.
          Blurs flat noisy areas (background grain, JPEG block artefacts)
          without softening genuine stroke edges.
       c. Unsharp Mask (Step 2.5) — boosts high-frequency edge contrast
          just before thresholding. Faint pencil strokes that would have
          broken into dashes now survive as continuous lines.
     Steps 3-5 are now a CLOSED ADAPTIVE LOOP (adaptive_binarize_and_clean):
     measures foreground density + surviving blob count after each
     threshold/morph/clean pass, and automatically retunes blockSize, C,
     and minBlobArea if the result is outside the sane range (0.3%-8%
     density, ≤60 blobs). Converges in 1 pass for clean scans; retries
     up to 4× for difficult inputs. The attempt with the lowest "badness"
     score is always kept even if perfect convergence is never reached.

Pipeline:
  1.   Load image
  1.5  CLAHE adaptive contrast enhancement
  2.   Bilateral edge-preserving denoise + median blur
  2.5  Unsharp-mask edge sharpening
  3-5. Adaptive closed-loop: threshold + MORPH_CLOSE + CC speckle filter
  6.   Canny (visualisation / PDF preview only)
  7.   Contour + Hierarchy (RETR_TREE on cleaned mask)
  8.   Shape Classification (Kasa circle LS + angle-aware rect detector)
  9.   Hierarchy Pairing (inner/outer → averaged centreline)
  10.  Residual Dedup + Filter
  10.5 Rectilinear Geometric Primitive Fitting (dedup + corner snap)
  11.  DXF Export
  12.  PNG Preview
  13.  PDF Export
"""

import sys, os, json, time, traceback, math
from pathlib import Path
from itertools import combinations

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
# STEP 1.5-2.5 — ADAPTIVE CONTRAST / DENOISE / SHARPEN
# ════════════════════════════════════════════════════════════════════════════

def enhance_contrast_clahe(gray, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Adaptive contrast enhancement (CLAHE — Contrast Limited Adaptive
    Histogram Equalisation). Unlike a single global histogram equalisation,
    CLAHE operates on local tiles and clips each tile's histogram before
    redistributing it, so it boosts contrast in faint/unevenly-lit regions
    of a scan or phone photo (a corner of the page caught in shadow, a
    light pencil stroke) WITHOUT blowing out regions that already have
    strong contrast or amplifying flat-noise into visible speckle. This
    runs first, before any blur/denoise/threshold stage, so every
    downstream stage sees a more evenly-lit, higher-contrast image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def denoise_edge_preserving(gray, d=7, sigma_color=50, sigma_space=50):
    """
    Edge-preserving denoise (bilateral filter). Knocks down sensor/scan
    grain and JPEG-style noise while keeping real stroke edges sharp —
    a plain Gaussian blur would soften genuine corners along with noise;
    bilateral filtering only blurs across pixels that are also spatially
    AND tonally close, so it smooths flat noisy regions but leaves a
    pencil-stroke-to-background edge crisp. Runs ahead of the median blur,
    which still handles any remaining salt-and-pepper outliers.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

def sharpen_unsharp_mask(gray, amount=1.5, blur_ksize=5, blur_sigma=1.0):
    """
    Unsharp-mask edge sharpening: subtract a Gaussian-blurred copy of the
    image from the (weighted) original to boost high-frequency edge
    contrast right before binarisation. This is what lets faint pencil
    strokes — already evened-out by CLAHE and cleaned by the bilateral
    filter — survive adaptiveThreshold as a continuous line instead of
    breaking into dashes.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return sharpened


# ════════════════════════════════════════════════════════════════════════════
# STEPS 2-5 — DENOISE / BINARISE / SPECKLE REMOVAL
# ════════════════════════════════════════════════════════════════════════════

def median_blur(gray, ksize=5):
    if ksize % 2 == 0: ksize += 1
    return cv2.medianBlur(gray, ksize)

def adaptive_threshold_binarize(blurred, block_size=15, c_val=4):
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size, C=c_val,
    )

def morph_clean(binary):
    """
    Single-stage morphological cleanup: MORPH_CLOSE only (3x3 ellipse).

    MORPH_OPEN was removed — on noisier / lighter hand-drawn strokes the
    open stage was eroding away legitimate thin corners and short spurs
    before contour extraction ever saw them, which is a worse failure mode
    than the speckle it was meant to catch (speckle removal is now owned
    entirely by the exact connected-component filter in Step 5, which does
    not erode real geometry). MORPH_CLOSE alone still fills tiny 1px
    gaps/holes inside real strokes — keeping each stroke as ONE connected
    component so it survives connected-component filtering as a single
    blob, and giving findContours cleaner ring hierarchies for offset-pair
    averaging — without removing or eroding any real geometry.
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
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
    first pass, a light MORPH_CLOSE (3x3 ellipse) is applied to the
    surviving mask and a second connected-component pass re-checks it
    against `min_area`. MORPH_OPEN (erosion-based) is no longer used
    anywhere in this pipeline — every actual blob-removal decision here is
    still made by the exact connected-component filter, never by erosion,
    so no real geometry can be eaten away. This compounds cleanup beyond a
    single pass while a stable mask (no further small blobs) exits early.
    """
    cleaned, removed_blobs, removed_px = _remove_small_blobs_once(binary, min_area)

    if aggressive:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        reclosed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        cleaned2, rb2, rp2 = _remove_small_blobs_once(reclosed, min_area)
        if rb2 > 0:
            cleaned = cleaned2
            removed_blobs += rb2
            removed_px += rp2

    return cleaned, removed_blobs, removed_px


def adaptive_binarize_and_clean(sharpened, min_blob_area, max_iterations=4):
    """
    STEPS 3-5, CLOSED LOOP: runs adaptiveThreshold -> MORPH_CLOSE ->
    connected-component speckle filtering, then measures how "sane" the
    resulting mask is and — if it isn't — automatically retunes
    adaptiveThreshold's blockSize/C and the blob-area floor and reruns,
    instead of trusting one fixed set of parameters to work for every
    input image.

    Quality signal: foreground pixel DENSITY (stroke px / total px) and
    surviving BLOB COUNT after speckle filtering. A clean hand-drawn
    sketch on a sheet typically has strokes covering roughly 0.3%-8% of
    the frame in a modest, stable number of blobs once real speckle is
    gone:
      - density too LOW  -> threshold was too strict, faint/legitimate
        strokes were lost as background. Loosen it (lower C / smaller
        blockSize pulls more grey pixels into "foreground").
      - density too HIGH, or still many small blobs -> threshold (or the
        blob-area floor) was too loose, background grain/shadow was
        captured as foreground. Tighten both.
    The loop stops as soon as the mask lands in the sane range, so most
    clean scans converge in a single pass — and it is hard-capped at
    `max_iterations` so a pathological image can never hang the pipeline;
    whichever iteration produced the best (closest-to-sane) result is kept
    even if true convergence was never reached.

    Returns: cleaned, binary, removed_blobs, removed_px, history,
             final_block_size, final_c, final_blob_floor
    """
    img_area    = float(sharpened.shape[0] * sharpened.shape[1])
    block_size  = 15
    c_val       = 4
    blob_floor  = float(min_blob_area)

    history = []
    best         = None
    best_badness = None

    for it in range(max(1, max_iterations)):
        binary = adaptive_threshold_binarize(sharpened, block_size=block_size, c_val=c_val)
        closed = morph_clean(binary)
        cleaned, removed_blobs, removed_px = remove_small_blobs(closed, blob_floor, aggressive=True)

        fg_px       = int(np.count_nonzero(cleaned))
        density     = fg_px / img_area
        n_labels, _ = cv2.connectedComponents(cleaned, connectivity=8)
        n_blobs     = n_labels - 1

        # "Badness" = 0 when fully inside the sane window, otherwise how
        # far outside it we are (used only to pick the least-bad attempt
        # if every iteration falls short of the convergence test).
        if density < 0.003:
            density_badness = (0.003 - density) / 0.003
        elif density > 0.08:
            density_badness = (density - 0.08) / 0.08
        else:
            density_badness = 0.0
        blob_badness = max(0.0, (n_blobs - 60) / 60.0)
        badness = density_badness + blob_badness

        history.append({
            "iteration": it, "blockSize": block_size, "C": c_val,
            "minBlobArea": round(blob_floor, 1),
            "density": round(density, 5), "blobs": n_blobs,
            "badness": round(badness, 4),
        })

        if best is None or badness < best_badness:
            best = (cleaned, binary, removed_blobs, removed_px, block_size, c_val, blob_floor)
            best_badness = badness

        converged = (0.003 <= density <= 0.08) and (n_blobs <= 60)
        if converged or it == max_iterations - 1:
            break

        if density < 0.003:
            c_val      = max(1, c_val - 2)
            block_size = max(7, block_size - 2)
        else:
            c_val      = c_val + 2
            block_size = block_size + 2
            blob_floor = blob_floor * 1.5

    cleaned, binary, removed_blobs, removed_px, final_block, final_c, final_floor = best
    return cleaned, binary, removed_blobs, removed_px, history, final_block, final_c, final_floor


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

def extract_contours_with_hierarchy(cleaned_mask, epsilon_factor=0.03):
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

def _is_rectangle_like(pts_xy, angle_tol_deg=25.0, min_fraction_near_90=0.7):
    """
    True if a (closed) contour's vertices form a rectangle-like shape —
    i.e. MOST interior corner angles cluster around 90°. This deliberately
    does NOT require exactly 4 vertices: approxPolyDP frequently splits one
    true 90° corner of a hand-drawn or slightly noisy rectangle into two
    nearby points (or leaves an extra near-collinear point along a long
    edge), so a real rectangle can legitimately come through with 4-8
    vertices. What it must NOT have is a genuinely different polygon shape
    (an L-bracket, notch, hexagon, etc.) masquerading as a rectangle just
    because it happens to have <=8 points — those have several angles far
    from 90° and are correctly rejected here so they stay classified as
    'poly' instead of being collapsed into a wrong bounding-box rectangle.
    """
    n = len(pts_xy)
    if n < 4 or n > 8:
        return False

    near_90 = 0
    counted = 0
    for i in range(n):
        p_prev = pts_xy[(i - 1) % n]
        p_curr = pts_xy[i]
        p_next = pts_xy[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-6 or n2 < 1e-6:
            continue  # degenerate/duplicate vertex — skip, don't count against it
        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle_deg = math.degrees(math.acos(cos_a))
        counted += 1
        # Accept angles near 90° (a true corner) OR near 180° (a spurious
        # near-collinear extra point sitting along an otherwise-straight
        # edge — common approxPolyDP artefact, not a real corner).
        if abs(angle_deg - 90.0) <= angle_tol_deg or angle_deg >= (180.0 - angle_tol_deg):
            near_90 += 1

    if counted == 0:
        return False
    return (near_90 / counted) >= min_fraction_near_90


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

    # Rectangle detector: 4-8 vertices is enough for a real rectangle —
    # approxPolyDP can legitimately split one true 90° corner into two
    # close points, or leave an extra near-collinear point on a long edge,
    # without the shape being anything other than a rectangle. So instead
    # of requiring exactly 4 points, we check the actual corner geometry
    # via `_is_rectangle_like` (interior angles clustering around 90°).
    if _is_rectangle_like(pts_xy):
        cx_b, cy_b, w_b, h_b = _fit_rect_algebraic(pts_xy)
        return {
            'type': 'rect',
            'cx': cx_b, 'cy': cy_b, 'w': w_b, 'h': h_b,
            'area': w_b * h_b,
        }

    # Everything else — true complex/many-vertex shapes, and any 4-8 vertex
    # contour whose angles DON'T cluster around 90° (so it isn't actually
    # rectangular: L-brackets, notches, hexagons, etc.) — is kept as a
    # polygon with its real vertices, instead of being forced into a
    # bounding-box rectangle that would misrepresent its true shape.
    return {
        'type': 'poly',
        'points': pts_xy.tolist(),  # Keep all approxPolyDP vertices
        'area': cv2.contourArea(np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)),
        'w': w, 'h': h,
        'cx': (x.min() + x.max()) / 2.0,
        'cy': (y.min() + y.max()) / 2.0,
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
        # FIX v10.1: Actually merge both polygons by combining point sets
        # rather than arbitrarily picking the larger one
        all_pts = np.array(a['points'] + b['points'], dtype=np.float32)
        cx = (all_pts[:,0].min() + all_pts[:,0].max()) / 2.0
        cy = (all_pts[:,1].min() + all_pts[:,1].max()) / 2.0
        w = all_pts[:,0].max() - all_pts[:,0].min()
        h = all_pts[:,1].max() - all_pts[:,1].min()
        return {
            'type': 'poly',
            'points': all_pts.tolist(),
            'area': cv2.contourArea(np.array(all_pts, dtype=np.float32).reshape(-1, 1, 2)),
            'w': float(w), 'h': float(h),
            'cx': float(cx), 'cy': float(cy),
        }
    
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

    # === NEW: 145px² area filter ===
    paired = [s for s in paired if s['area'] >= 145.0]

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
# STEP 10.5 — RECTILINEAR GEOMETRIC PRIMITIVE FITTING
# ════════════════════════════════════════════════════════════════════════════

def _dedup_parallel_segments(raw_segments, tol=15.0, slack=20.0):
    """
    v11 — Robust duplicate-edge collapsing (replaces the old fixed-grid
    round(coord/8)*8 bucketing from v10.2).

    The CV trace frequently reports the same physical edge as 2-4 near-
    parallel segments a few pixels apart (sub-pixel skeleton noise, slightly
    different approxPolyDP corner picks across nearby vertices, inner vs
    outer stroke edge survivors, etc). Two segments of the SAME orientation
    (both 'H' or both 'V') are treated as traces of the SAME physical edge
    iff BOTH:
        a) their constant coordinate (y for H, x for V) differs by <= tol
        b) their [start, end] spans overlap, or nearly touch within `slack`
    Condition (b) is what keeps this safe on staircase / multi-step
    features: two genuinely different edges that happen to sit at a similar
    height (e.g. the two steps of a notch) do NOT get merged, because their
    spans don't overlap — only condition (a) being true is not enough.

    This is checked PAIRWISE (not just between sorted neighbours) with a
    Union-Find, so 3+ mutually-close traces collapse transitively into one
    cluster even when not every pair in the cluster individually satisfies
    (a) on its own.

    The representative segment for a cluster takes the MEDIAN constant
    coordinate and MEDIAN start/end across all traces in it — robust to a
    single noisy outlier trace, unlike the old approach's hard grid-snap
    (which could just as easily split a real duplicate pair that straddled
    a bucket boundary, or merge two real edges that happened to round into
    the same bucket).
    """
    h_segs = [s for s in raw_segments if s['type'] == 'H']
    v_segs = [s for s in raw_segments if s['type'] == 'V']

    def cluster(segs):
        n = len(segs)
        if n == 0:
            return []
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j in combinations(range(n), 2):
            c1, s1, e1 = segs[i]['coord'], segs[i]['start'], segs[i]['end']
            c2, s2, e2 = segs[j]['coord'], segs[j]['start'], segs[j]['end']
            if abs(c1 - c2) <= tol:
                overlap = min(e1, e2) - max(s1, s2)
                if overlap > -slack:
                    union(i, j)

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        med = lambda a: a[len(a) // 2] if len(a) % 2 else (a[len(a)//2 - 1] + a[len(a)//2]) / 2.0
        reps = []
        for idxs in groups.values():
            consts = sorted(segs[i]['coord'] for i in idxs)
            starts = sorted(segs[i]['start'] for i in idxs)
            ends   = sorted(segs[i]['end']   for i in idxs)
            reps.append({'coord': med(consts), 'start': med(starts), 'end': med(ends),
                         'n_traces': len(idxs)})
        return reps

    merged = []
    for r in cluster(h_segs):
        merged.append({'type': 'H', 'coord': r['coord'], 'start': r['start'], 'end': r['end'],
                       'n_traces': r['n_traces']})
    for r in cluster(v_segs):
        merged.append({'type': 'V', 'coord': r['coord'], 'start': r['start'], 'end': r['end'],
                       'n_traces': r['n_traces']})
    return merged


def _snap_segment_corners(merged_segments, tol=60.0):
    """
    v11 — Extend-to-intersect corner snapping (NEW — there was no equivalent
    stage in v10.2; segments were exported with whatever loose endpoints
    Step 3 happened to produce, so adjacent H/V edges routinely fell a few
    pixels short of actually touching at the corner).

    After _dedup_parallel_segments there is exactly one segment per real
    edge, but consecutive H/V segments still don't necessarily terminate at
    the same point. Every corner of a rectilinear shape is, by construction,
    the intersection of one horizontal line y=y_h and one vertical line
    x=x_v — i.e. the point (x_v, y_h). This:
        1. Collects every open endpoint of every H and V segment.
        2. Greedily nearest-neighbour-matches H endpoints to V endpoints by
           Euclidean distance (closest pairs first), capped at `tol` so the
           match can never bridge two unrelated corners.
        3. Overwrites each matched pair's endpoint with the shared
           intersection (v.coord, h.coord) — literally extending or
           trimming each line until it meets its neighbour, then stopping.

    Mutates the segment dicts in place and returns the same list. Segments
    with no match within `tol` (e.g. a genuinely open/dangling edge) are
    left untouched.
    """
    h_segs = [s for s in merged_segments if s['type'] == 'H']
    v_segs = [s for s in merged_segments if s['type'] == 'V']
    if not h_segs or not v_segs:
        return merged_segments

    h_endpoints = []
    for hi, s in enumerate(h_segs):
        h_endpoints.append((hi, 'start', s['start'], s['coord']))
        h_endpoints.append((hi, 'end',   s['end'],   s['coord']))

    v_endpoints = []
    for vi, s in enumerate(v_segs):
        v_endpoints.append((vi, 'start', s['coord'], s['start']))
        v_endpoints.append((vi, 'end',   s['coord'], s['end']))

    pairs = []
    for h in h_endpoints:
        for v in v_endpoints:
            d = math.hypot(h[2] - v[2], h[3] - v[3])
            if d <= tol:
                pairs.append((d, h, v))
    pairs.sort(key=lambda t: t[0])

    used_h, used_v, n_snapped = set(), set(), 0
    for d, h, v in pairs:
        hkey, vkey = (h[0], h[1]), (v[0], v[1])
        if hkey in used_h or vkey in used_v:
            continue
        used_h.add(hkey)
        used_v.add(vkey)

        hi, h_end, _hx, hy = h
        vi, v_end, vx, _vy = v
        corner_x, corner_y = vx, hy
        h_segs[hi][h_end] = corner_x
        v_segs[vi][v_end] = corner_y
        n_snapped += 1

    # Defensive: corner correction should never flip a segment's direction,
    # but guard against degenerate/near-zero-length edges after snapping.
    for s in h_segs + v_segs:
        if s['start'] > s['end']:
            s['start'], s['end'] = s['end'], s['start']

    return h_segs + v_segs


def fit_rectilinear_to_polygon(poly_shape, original_contour, 
                                approx_eps_factor=0.003, 
                                orientation_ratio=1.2,
                                dedup_tol=15.0,
                                dedup_slack=20.0,
                                corner_snap_tol=60.0):
    """
    v11: Extracts separate horizontal and vertical line segments, collapses
    duplicate parallel traces of the same edge down to ONE line per side
    (_dedup_parallel_segments), then extends every remaining line along its
    axis until it intersects its perpendicular neighbour and stops there
    (_snap_segment_corners) — so the exported segments form a fully
    connected, closed rectilinear chain instead of a pile of loose,
    near-duplicate strokes.

    Returns: (updated_poly_shape, list_of_segments)
    Each segment: {'type': 'H', 'coord': y, 'start': x1, 'end': x2}
               or {'type': 'V', 'coord': x, 'start': y1, 'end': y2}
    """
    if isinstance(original_contour, list):
        original_contour = np.array(original_contour, dtype=np.float32)
    if original_contour.ndim == 3:
        original_contour = original_contour.reshape(-1, 2)
    
    # Step 1: Get corners using approxPolyDP with tight epsilon
    arc_len = cv2.arcLength(original_contour.reshape(-1, 1, 2), True)
    epsilon = max(0.0005 * arc_len, 0.5)
    approx = cv2.approxPolyDP(original_contour.reshape(-1, 1, 2), epsilon, True)
    vertices = np.array([pt[0] for pt in approx], dtype=np.float32)
    
    if len(vertices) < 4:
        poly_shape['rectilinear'] = False
        return poly_shape, []
    
    # Step 2: Extract orthogonal edge segments from corner-to-corner edges
    raw_segments = []
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        
        if length < 15:  # Skip very short noise segments
            continue
        
        if dx > dy * 1.3:  # Clearly horizontal
            y_avg = (p1[1] + p2[1]) / 2.0
            x1 = min(p1[0], p2[0])
            x2 = max(p1[0], p2[0])
            raw_segments.append({'type': 'H', 'coord': float(y_avg), 'start': float(x1), 'end': float(x2)})
        elif dy > dx * 1.3:  # Clearly vertical
            x_avg = (p1[0] + p2[0]) / 2.0
            y1 = min(p1[1], p2[1])
            y2 = max(p1[1], p2[1])
            raw_segments.append({'type': 'V', 'coord': float(x_avg), 'start': float(y1), 'end': float(y2)})
        elif max(dx, dy) > 80:  # Long diagonal - force to dominant axis
            if dx > dy:
                y_avg = (p1[1] + p2[1]) / 2.0
                x1 = min(p1[0], p2[0])
                x2 = max(p1[0], p2[0])
                raw_segments.append({'type': 'H', 'coord': float(y_avg), 'start': float(x1), 'end': float(x2)})
            else:
                x_avg = (p1[0] + p2[0]) / 2.0
                y1 = min(p1[1], p2[1])
                y2 = max(p1[1], p2[1])
                raw_segments.append({'type': 'V', 'coord': float(x_avg), 'start': float(y1), 'end': float(y2)})
    
    # Step 3: Collapse duplicate parallel traces of the same physical edge
    # down to a single representative line (v11 — see _dedup_parallel_segments
    # for the union-find + median-cluster logic; replaces the old fixed-grid
    # round(coord/8)*8 bucketing).
    n_raw = len(raw_segments)
    deduped_segments = _dedup_parallel_segments(raw_segments, tol=dedup_tol, slack=dedup_slack)

    # Step 4: Extend every remaining line along its axis until it intersects
    # its perpendicular neighbour, then stop (v11 — new stage; see
    # _snap_segment_corners). This is what actually closes the chain into a
    # connected outline instead of a pile of loose, almost-touching strokes.
    final_segments = _snap_segment_corners(deduped_segments, tol=corner_snap_tol)

    # Update poly_shape with segment info + cleaning telemetry
    poly_shape['rectilinear'] = True
    poly_shape['segments'] = final_segments
    poly_shape['n_segments'] = len(final_segments)
    poly_shape['n_raw_segments'] = n_raw
    poly_shape['n_traces_collapsed'] = n_raw - len(final_segments)

    return poly_shape, final_segments


def apply_rectilinear_fitting(final_shapes, simplified_contours,
                               dedup_tol=15.0, dedup_slack=20.0, corner_snap_tol=60.0):
    """
    v11: Apply rectilinear geometric primitive fitting to all polygon shapes
    — duplicate-edge collapsing + extend-to-intersect corner snapping.
    Returns shapes with 'segments' list containing connected H/V line segments.
    Circles and rectangles are passed through unchanged.
    """
    fitted_shapes = []
    
    for shape in final_shapes:
        if shape['type'] == 'poly':
            # Find the corresponding original contour by matching center
            best_contour = None
            best_score = 0
            
            for cnt in simplified_contours:
                cnt_pts = np.array([pt[0] for pt in cnt], dtype=np.float32)
                if len(cnt_pts) < 3:
                    continue
                
                cnt_cx = (cnt_pts[:,0].min() + cnt_pts[:,0].max()) / 2
                cnt_cy = (cnt_pts[:,1].min() + cnt_pts[:,1].max()) / 2
                shape_cx = shape.get('cx', 0)
                shape_cy = shape.get('cy', 0)
                
                dist = math.hypot(cnt_cx - shape_cx, cnt_cy - shape_cy)
                if dist < 50:  # Within 50px
                    score = len(cnt_pts)
                    if score > best_score:
                        best_score = score
                        best_contour = cnt_pts
            
            if best_contour is not None and len(best_contour) > 10:
                fitted, segments = fit_rectilinear_to_polygon(
                    shape.copy(), best_contour,
                    dedup_tol=dedup_tol, dedup_slack=dedup_slack, corner_snap_tol=corner_snap_tol)
                fitted['segments'] = segments
                fitted_shapes.append(fitted)
            else:
                fitted_shapes.append(shape)
        else:
            fitted_shapes.append(shape)
    
    return fitted_shapes

# ════════════════════════════════════════════════════════════════════════════
# STEP 11 — CLEAN DXF EXPORT (one entity per shape, true detected position)
# ════════════════════════════════════════════════════════════════════════════

def build_clean_dxf(final_shapes, img_w, img_h, out_path):
    """
    FIXED v10.2: Exports each rectilinear segment as a separate 2-point LWPOLYLINE.
    Circles remain as CIRCLE entities. Each line segment is individually editable.
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
    doc.layers.new("H_LINES", dxfattribs={"color": 5,  "linetype": "CONTINUOUS"})
    doc.layers.new("V_LINES", dxfattribs={"color": 6,  "linetype": "CONTINUOUS"})

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
            # FIXED v10.2: Export each segment as a separate 2-point LWPOLYLINE
            segments = s.get('segments', [])
            if segments:
                for seg in segments:
                    if seg['type'] == 'H':
                        # Horizontal line: (x1, y) -> (x2, y)
                        y = seg['coord']
                        x1 = seg['start']
                        x2 = seg['end']
                        pts = [(x1, y), (x2, y)]
                        layer = "H_LINES"
                    else:
                        # Vertical line: (x, y1) -> (x, y2)
                        x = seg['coord']
                        y1 = seg['start']
                        y2 = seg['end']
                        pts = [(x, y1), (x, y2)]
                        layer = "V_LINES"
                    
                    line = msp.add_lwpolyline(
                        pts, format="xy",
                        dxfattribs={"layer": layer, "color": 256}
                    )
                    # Don't close - these are open line segments
                    entity_count += 1
            else:
                # Fallback: export as single polygon if no segments
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
                            "SheetForge v11  •  Dedup + Extend-to-Intersect Corner Snap  •  Clean DXF")
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
    epsilon_factor = float(opts.get("epsilonFactor", 0.03))
    min_blob_area  = int(opts.get("minBlobArea", 20))    # ← changed from 50; user-requested <20px
    min_shape_area = opts.get("minShapeArea", None)       # px-area: absolute noise filter
    if min_shape_area is not None:
        min_shape_area = float(min_shape_area)
    rect_aspect_min       = float(opts.get("rectAspectMin", 0.10))     # drop thin dimension lines
    rect_rel_area_min     = float(opts.get("rectRelAreaMin", 0.001))   # drop annotation text blocks (lowered: 0.1% of max, not 1%)
    circle_rel_radius_min = float(opts.get("circleRelRadiusMin", 0.2)) # drop text-glyph circles (lowered)
    circle_rect_overlap_frac = float(opts.get("circleRectOverlapFrac", 0.5))  # drop circles sitting on rects
    dedup_tol         = float(opts.get("dedupTol", 15.0))       # max px gap between duplicate parallel edges
    dedup_slack        = float(opts.get("dedupSlack", 20.0))     # negative-overlap slack for near-touching spans
    corner_snap_tol    = float(opts.get("cornerSnapTol", 60.0))  # max px distance to snap a corner pair

    steps = []

    # ── STEP 1: Load ──────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record("CV-1: Load Image", f"{img_w}×{img_h}px  DPI={dpi:.0f}", t0))

    # ── STEP 1.5: Adaptive Contrast Enhancement (CLAHE) ─────────────────────
    t0 = now_ms()
    contrast_enhanced = enhance_contrast_clahe(gray)
    steps.append(step_record("CV-1.5: Adaptive Contrast Enhancement (CLAHE)",
                              "Local contrast normalised — faint strokes evened out", t0))

    # ── STEP 2: Edge-Preserving Denoise + Median Blur ───────────────────────
    t0 = now_ms()
    denoised = denoise_edge_preserving(contrast_enhanced)
    blurred  = median_blur(denoised, ksize=blur_ksize)
    steps.append(step_record(f"CV-2: Bilateral Denoise + Median Blur (ksize={blur_ksize})",
                              "Grain/salt-and-pepper noise reduced, edges preserved", t0))

    # ── STEP 2.5: Unsharp-Mask Edge Sharpening ──────────────────────────────
    t0 = now_ms()
    sharpened = sharpen_unsharp_mask(blurred)
    steps.append(step_record("CV-2.5: Unsharp Mask Sharpening",
                              "High-frequency stroke edges boosted before thresholding", t0))

    # ── STEPS 3-5: Adaptive Threshold + Morph Close + Speckle Removal ──────
    # Closed loop: automatically retunes blockSize/C/minBlobArea and reruns
    # until the cleaned mask's foreground density + blob count settle into
    # the sane range (or the iteration cap is hit), instead of trusting one
    # fixed set of parameters for every image.
    t0 = now_ms()
    (cleaned, binary, removed_blobs, removed_px, adapt_history,
     final_block, final_c, final_floor) = adaptive_binarize_and_clean(
        sharpened, min_blob_area, max_iterations=4)
    white_px = int(np.count_nonzero(binary))
    final_density = adapt_history[-1]["density"] if adapt_history else 0.0
    steps.append(step_record(
        f"CV-3/4/5: Adaptive Threshold + MORPH_CLOSE + Speckle Filter "
        f"({len(adapt_history)} pass(es), converged blockSize={final_block} C={final_c} "
        f"minBlobArea={final_floor:.0f}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px), "
        f"final foreground density {final_density*100:.2f}% — adaptive convergence", t0))



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
  
    # ── STEP 10.5: Rectilinear Geometric Primitive Fitting ───────────────
    t0 = now_ms()
    final_shapes = apply_rectilinear_fitting(
        final_shapes, simplified_contours,
        dedup_tol=dedup_tol, dedup_slack=dedup_slack, corner_snap_tol=corner_snap_tol)
    n_rectilinear = sum(1 for s in final_shapes if s.get('rectilinear'))
    n_raw_total = sum(s.get('n_raw_segments', 0) for s in final_shapes if s.get('rectilinear'))
    n_final_total = sum(s.get('n_segments', 0) for s in final_shapes if s.get('rectilinear'))
    steps.append(step_record(
        "GEO-10.5: Rectilinear Primitive Fitting (dedup + extend-to-intersect corner snap)",
        f"{n_rectilinear} polygon(s) → {n_raw_total} raw traces collapsed to {n_final_total} "
        f"connected H/V segments", t0))
  
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
    n_poly_final = sum(1 for s in final_shapes if s['type'] == 'poly')
    analysis = {
        "width"                  : float(img_w),
        "height"                 : float(img_h),
        "dpi"                    : dpi,
        "edgePixels"             : edge_px,
        "edges"                  : entity_count,
        "contours"               : len(simplified_contours),
        "mergedContours"         : len(final_shapes),
        "closedContours"         : len(final_shapes),
        "totalVertices"          : total_pts,
        "blurKsize"              : blur_ksize,
        "cannyLow"               : canny_low,
        "cannyHigh"              : canny_high,
        "epsilonFactor"          : epsilon_factor,
        "minBlobArea"            : min_blob_area,
        "hardMinShapeAreaPx"     : HARD_MIN_SHAPE_AREA_PX,
        "speckleBlobsRemoved"    : removed_blobs,
        "adaptivePipelinePasses" : len(adapt_history),
        "adaptiveHistory"        : adapt_history,
        "finalBlockSize"         : final_block,
        "finalC"                 : final_c,
        "finalBlobFloor"         : round(final_floor, 1),
        "imgW"                   : img_w,
        "imgH"                   : img_h,
        "scaleMmPerDu"           : round(25.4 / dpi, 4),
        "coordSystem"            : "origin=top-left px, Y-down (image convention), no offset/re-centering",
        "circlesDetected"        : n_circ_final,
        "rectsDetected"          : n_rect_final,
        "polysDetected"          : n_poly_final,
        "shapeSummary"           : (
            f"{n_rect_final} rectangle(s), {n_circ_final} circle(s), {n_poly_final} polygon(s)"
            f" — CLAHE + bilateral denoise + unsharp sharpen + adaptive closed-loop threshold,"
            f" speckle-free mask, hierarchy offset-pairs averaged to centreline, true measured positions"
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
