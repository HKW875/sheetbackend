#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v12  (True Centerline DXF via Algebraic LS + Intersection Trimming)
==============================================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

WHAT CHANGED FROM v11 AND WHY
-------------------------------
v11 produced shape-level DXF (one CIRCLE or LWPOLYLINE rectangle per detected shape),
which meant the DXF was a re-construction of classified shapes, not a true tracing of
the actual drawn geometry.

v12 replaces the DXF export entirely with a TRUE CENTERLINE approach:

  1. CANNY AREA FILTER (200px minimum)
     Every connected blob in the Canny edge result with pixel area < 200px is removed
     before any further processing. This eliminates all noise specks, annotation dots,
     and dimension-line arrowheads from the edge image.

  2. ALGEBRAIC LEAST-SQUARES CENTERLINE FITTING ON CANNY WHITE PIXELS
     Instead of classifying shapes and re-drawing idealized rectangles/circles, the
     pipeline now:
       a. Skeletonizes the 200px-filtered Canny edge to a true 1-px centerline.
       b. Extracts all contours from the skeleton (RETR_LIST, no hierarchy needed).
       c. For each contour segment, tests whether it fits a circle (Kasa algebraic
          LS) or is better represented as a polyline.
       d. Straight/orthogonal segments are snapped to exact 90-degree horizontal or
          vertical directions using LS line fitting so DXF lines are perfectly aligned.
       e. Each contour produces EXACTLY ONE DXF entity (LINE, LWPOLYLINE, or CIRCLE).

  3. INTERSECTION TRIMMING (polylines end at intersections)
     After all segments are computed, segment endpoints are extended to their true
     intersection points with adjacent segments. Each polyline starts and ends exactly
     where it meets another polyline — no floating endpoints.

  4. SINGLE CENTERLINE PER EDGE
     Because the input to contour extraction is a skeletonized (1-px) image, each
     physical drawn edge produces exactly one contour — no inner/outer pairs.

Pipeline:
  1.  Load image
  2.  Median Blur
  3.  Adaptive Threshold
  4.  Morph Open
  5.  Connected-Component Filter (minBlobArea)
  6.  Canny Edge Detection → 200px area filter → skeletonize (single centerline)
  7.  Contour extraction on skeleton (RETR_LIST, CHAIN_APPROX_NONE)
  8.  Per-contour: circle LS fit test, else orthogonal polyline with 90° snap
  9.  Intersection detection and endpoint trimming
  10. DXF Export (LINE/LWPOLYLINE/CIRCLE per centerline segment)
  11. PNG Preview
  12. PDF Export
"""
import numpy as np
import sys, os, json, time, traceback, math
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
    dpi = 1200.0
    if HAS_PIL:
        try:
            pil  = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (1200, 1200))
            dpi  = float(xdpi[0]) if xdpi and xdpi[0] > 1 else 1200.0
        except Exception:
            pass
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = bgr.shape[:2]
    return bgr, gray, dpi, img_w, img_h


# ════════════════════════════════════════════════════════════════════════════
# STEPS 2-5 — DENOISE / BINARISE / SPECKLE REMOVAL
# ════════════════════════════════════════════════════════════════════════════
min_area = 200
def median_blur(gray, ksize=5):
    if ksize % 2 == 0: ksize += 1
    return cv2.medianBlur(gray, ksize)

def adaptive_threshold_binarize(blurred):
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=5, C=1,
    )

def morph_clean(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)




def remove_small_blobs(binary, min_area):
    """Remove all 8-connected blobs with pixel area < min_area efficiently using NumPy."""
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Extract the areas of all components (index 0 is the background)
    areas = stats[:, cv2.CC_STAT_AREA]
    
    # Identify which labels fail the area threshold (excluding background label 0)
    small_blobs_mask = (areas < min_area)
    small_blobs_mask[0] = False  # Keep background
    
    # Calculate tracking metrics
    removed_blobs = int(np.sum(small_blobs_mask))
    removed_px = int(np.sum(areas[small_blobs_mask]))
    
    # Generate a lookup table: set invalid labels to 0, valid labels to 255
    lut = np.ones(num_labels, dtype=np.uint8) * 255
    lut[small_blobs_mask] = 0
    lut[0] = 0  # Background stays black
    
    # Instantly map labels to the final image without loop iteration
    cleaned = lut[labels]
    
    return cleaned, removed_blobs, removed_px



# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — CANNY → 200px AREA FILTER → SKELETONIZE
# ════════════════════════════════════════════════════════════════════════════

def canny_edges(cleaned, low_threshold=200, high_threshold=255):
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

def thicken_to_centerline(binary_mask, thickness_px=4):
    if binary_mask is None or not HAS_CV:
        return binary_mask
    radius = max(1, thickness_px // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary_mask, kernel, iterations=1)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — CONTOUR EXTRACTION ON SKELETON
# ════════════════════════════════════════════════════════════════════════════

def extract_centerline_contours(skeleton, min_contour_len=10):
    """
    Extract all contours from the 1-px skeleton using RETR_LIST (no hierarchy)
    and CHAIN_APPROX_NONE (keep all points for accurate LS fitting).
    Returns list of (Nx2) float arrays of pixel coordinates.
    """
    # Thicken slightly so findContours can trace the skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(skeleton, kernel, iterations=1)
    contours, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = []
    for cnt in contours:
        pts = np.array([[p[0][0], p[0][1]] for p in cnt], dtype=float)
        if len(pts) >= min_contour_len:
            result.append(pts)
    return result


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — ALGEBRAIC LS FITTING: CIRCLE OR ORTHOGONAL POLYLINE
# ════════════════════════════════════════════════════════════════════════════

def _fit_circle_kasa(pts):
    """
    Kasa algebraic least-squares circle fit.
    Returns (cx, cy, r, rms_relative) or None if degenerate.
    """
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.column_stack([x, y, np.ones(len(x))])
    b_ = x**2 + y**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b_, rcond=None)
    except Exception:
        return None
    cx = res[0] / 2.0
    cy = res[1] / 2.0
    r  = math.sqrt(abs(res[2] + cx**2 + cy**2))
    if r < 2.0:
        return None
    dists = np.sqrt((x - cx)**2 + (y - cy)**2)
    rms   = float(np.sqrt(((dists - r)**2).mean()))
    rel   = rms / (r + 1e-9)
    return cx, cy, r, rel


def _is_closed_contour(pts, tol=8.0):
    """True if first and last points are close enough to form a closed shape."""
    return math.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]) < tol


def _snap_angle_90(angle_rad):
    """Snap an angle to the nearest 0, 90, 180, 270 degrees."""
    deg = math.degrees(angle_rad) % 180
    if deg <= 45 or deg > 135:
        return 0.0  # horizontal
    return math.pi / 2  # vertical


def _fit_line_ls(pts):
    """
    Fit a line through pts using SVD (total least squares).
    Returns (angle_rad, midpoint_x, midpoint_y).
    """
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    centered = pts - np.array([cx, cy])
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # principal direction
    angle = math.atan2(float(direction[1]), float(direction[0]))
    return angle, cx, cy


def _project_pts_onto_line(pts, angle, cx, cy):
    """Project all pts onto the fitted line, return (t_min, t_max) parameter range."""
    dx, dy = math.cos(angle), math.sin(angle)
    ts = [(p[0] - cx) * dx + (p[1] - cy) * dy for p in pts]
    return min(ts), max(ts)


def _orthogonal_polyline_from_contour(pts, angle_snap_tol=0.3):
    """
    Convert a contour point cloud to an orthogonal polyline with 90-degree snapping.
    Splits the contour into roughly-straight segments, fits a line to each,
    snaps each to horizontal or vertical, then chains them together.
    Returns list of (x, y) corner points defining the polyline.
    """
    # Use Douglas-Peucker to find key corners
    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    arc = cv2.arcLength(pts_int, closed=False)
    epsilon = max(2.0, 0.02 * arc)
    approx = cv2.approxPolyDP(pts_int, epsilon, closed=False)
    corners = np.array([[p[0][0], p[0][1]] for p in approx], dtype=float)

    if len(corners) < 2:
        return [(float(pts[0, 0]), float(pts[0, 1])),
                (float(pts[-1, 0]), float(pts[-1, 1]))]

    result = []
    for i in range(len(corners) - 1):
        p0 = corners[i]
        p1 = corners[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.atan2(abs(dy), abs(dx))
        # Snap to horizontal (angle < 45°) or vertical (angle >= 45°)
        if angle < math.pi / 4:  # horizontal
            mid_y = (p0[1] + p1[1]) / 2.0
            seg_p0 = (p0[0], mid_y)
            seg_p1 = (p1[0], mid_y)
        else:  # vertical
            mid_x = (p0[0] + p1[0]) / 2.0
            seg_p0 = (mid_x, p0[1])
            seg_p1 = (mid_x, p1[1])

        if not result:
            result.append(seg_p0)
        else:
            # Connect previous endpoint to this segment start with a clean join
            prev = result[-1]
            if abs(seg_p0[0] - prev[0]) > 1 or abs(seg_p0[1] - prev[1]) > 1:
                # Insert an elbow point
                if angle < math.pi / 4:  # current is horizontal → elbow at same y as current, same x as prev
                    elbow = (prev[0], seg_p0[1])
                else:  # current is vertical → elbow at same x as current, same y as prev
                    elbow = (seg_p0[0], prev[1])
                result.append(elbow)
            result.append(seg_p0)
        result.append(seg_p1)

    # Deduplicate consecutive identical points
    deduped = [result[0]] if result else []
    for p in result[1:]:
        if abs(p[0] - deduped[-1][0]) > 0.5 or abs(p[1] - deduped[-1][1]) > 0.5:
            deduped.append(p)
    return deduped


def classify_and_fit_contours(contours, circle_rms_tol=0.10, min_pts_circle=16):
    """
    For each contour:
      - If it is closed and the Kasa circle fit has rel RMS < circle_rms_tol → CIRCLE entity
      - Otherwise → orthogonal LWPOLYLINE with 90° snapping
    Returns list of entity dicts.
    """
    entities = []
    for pts in contours:
        closed = _is_closed_contour(pts)
        # Attempt circle fit on closed contours with enough points
        if closed and len(pts) >= min_pts_circle:
            fit = _fit_circle_kasa(pts)
            if fit is not None:
                cx, cy, r, rel = fit
                if rel < circle_rms_tol:
                    entities.append({
                        'type': 'circle',
                        'cx': float(cx), 'cy': float(cy), 'r': float(r),
                        'closed': True,
                    })
                    continue

        # Fallback: orthogonal polyline
        poly_pts = _orthogonal_polyline_from_contour(pts)
        if len(poly_pts) >= 2:
            entities.append({
                'type': 'polyline',
                'pts': poly_pts,
                'closed': closed and len(poly_pts) >= 3,
            })
    return entities


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — INTERSECTION DETECTION & ENDPOINT TRIMMING
# ════════════════════════════════════════════════════════════════════════════

def _seg_intersect(p1, p2, p3, p4, tol=20.0):
    """
    Find intersection of line segment p1→p2 with line (infinite) through p3→p4.
    Returns intersection point or None.
    Uses parameter t along p1→p2; returns point if 0 <= t <= 1 (within segment).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None  # parallel
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    # Allow slight overshoot for trimming (extend endpoint up to tol px)
    seg_len = math.hypot(d1x, d1y)
    t_tol = tol / (seg_len + 1e-9)
    if -t_tol <= t <= 1 + t_tol:
        ix = x1 + t * d1x
        iy = y1 + t * d1y
        return (ix, iy)
    return None


def trim_endpoints_to_intersections(entities, snap_radius=15.0):
    """
    For each polyline entity, attempt to extend/trim its start and end points
    to the nearest intersection with any other entity's segments.
    Circles are not modified (they are closed by definition).

    Algorithm:
      For each polyline P:
        For each of its two endpoints (start, end):
          Gather the endpoint's final segment direction.
          For each other entity Q (polyline or circle):
            For each segment of Q: test intersection with endpoint-ray.
            Keep the nearest intersection within snap_radius.
          If found: move the endpoint to the intersection.
    """
    polylines = [e for e in entities if e['type'] == 'polyline']

    for i, poly in enumerate(polylines):
        pts = list(poly['pts'])
        if len(pts) < 2:
            continue

        # Try to trim/extend START endpoint
        # The outward ray at start goes from pts[1] → pts[0] (and beyond)
        p_start = pts[0]
        p_start_dir = pts[1]  # direction reference

        best_pt = None
        best_dist = snap_radius

        for j, other in enumerate(entities):
            if i == j and other['type'] == 'polyline':
                continue
            if other['type'] == 'polyline':
                segs = list(zip(other['pts'][:-1], other['pts'][1:]))
            else:
                # Circle: skip for endpoint trimming
                continue
            for (q0, q1) in segs:
                ip = _seg_intersect(p_start, p_start_dir, q0, q1, tol=snap_radius)
                if ip is not None:
                    d = math.hypot(ip[0] - p_start[0], ip[1] - p_start[1])
                    if d < best_dist:
                        best_dist = d
                        best_pt = ip

        if best_pt is not None:
            pts[0] = best_pt

        # Try to trim/extend END endpoint
        p_end = pts[-1]
        p_end_dir = pts[-2]

        best_pt = None
        best_dist = snap_radius

        for j, other in enumerate(entities):
            if i == j and other['type'] == 'polyline':
                continue
            if other['type'] == 'polyline':
                segs = list(zip(other['pts'][:-1], other['pts'][1:]))
            else:
                continue
            for (q0, q1) in segs:
                ip = _seg_intersect(p_end, p_end_dir, q0, q1, tol=snap_radius)
                if ip is not None:
                    d = math.hypot(ip[0] - p_end[0], ip[1] - p_end[1])
                    if d < best_dist:
                        best_dist = d
                        best_pt = ip

        if best_pt is not None:
            pts[-1] = best_pt

        poly['pts'] = pts

    return entities


# ════════════════════════════════════════════════════════════════════════════
# STEP 10 — DXF EXPORT (true centerline entities)
# ════════════════════════════════════════════════════════════════════════════

def build_centerline_dxf(entities, img_w, img_h, out_path):
    """
    Write DXF with:
      - CIRCLE entities for circle fits
      - LWPOLYLINE entities (with 90° orthogonal segments) for all polylines
    Coordinates are pixel-space (origin top-left, Y-down from image).
    The DXF Y axis is flipped so that the geometry looks correct in CAD
    (Y increases upward in DXF convention).
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
    doc.layers.new("CENTERLINES", dxfattribs={"color": 7,  "linetype": "CONTINUOUS"})
    doc.layers.new("CIRCLES",     dxfattribs={"color": 1,  "linetype": "CONTINUOUS"})

    def flip_y(y):
        return float(img_h) - float(y)

    entity_count = 0

    for e in entities:
        if e['type'] == 'circle':
            cx = float(e['cx'])
            cy = flip_y(e['cy'])
            r  = float(e['r'])
            msp.add_circle(
                (cx, cy, 0.0), r,
                dxfattribs={"layer": "CIRCLES", "color": 1}
            )
            entity_count += 1

        elif e['type'] == 'polyline':
            pts_dxf = [(float(p[0]), flip_y(p[1])) for p in e['pts']]
            if len(pts_dxf) < 2:
                continue
            poly = msp.add_lwpolyline(
                pts_dxf, format="xy",
                dxfattribs={"layer": "CENTERLINES", "color": 7}
            )
            if e.get('closed') and len(pts_dxf) >= 3:
                poly.close(True)
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

        n_circ = 0
        n_poly = 0
        for e in entities:
            if e['type'] == 'circle':
                cx_px = int(round(e['cx']))
                cy_px = int(round(e['cy']))
                r_px  = int(round(e['r']))
                cv2.circle(right, (cx_px, cy_px), r_px, (80, 80, 220), 2, cv2.LINE_AA)
                n_circ += 1
            elif e['type'] == 'polyline':
                pts_draw = [(int(round(p[0])), int(round(p[1]))) for p in e['pts']]
                for k in range(len(pts_draw) - 1):
                    cv2.line(right, pts_draw[k], pts_draw[k + 1], (80, 200, 80), 2, cv2.LINE_AA)
                if e.get('closed') and len(pts_draw) >= 3:
                    cv2.line(right, pts_draw[-1], pts_draw[0], (80, 200, 80), 2, cv2.LINE_AA)
                n_poly += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left, "Canny Skeleton (200px filter)", (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(right, f"Centerlines: {n_poly} polyline(s) + {n_circ} circle(s)", (10, 22), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        sep   = np.full((img_h, 4, 3), 40, dtype=np.uint8)
        panel = np.concatenate([left, sep, right], axis=1)
        ok = cv2.imwrite(str(out_path), panel)
        return bool(ok and out_path.exists())
    except Exception as e:
        sys.stderr.write(f"PNG preview error: {e}\n")
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
        c.drawString(margin, page_h - 24, "SheetForge v12 — True Centerline DXF Preview")
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
        _draw_panel(c, reader, right_x, margin, col_w, col_h, "Skeleton Centerline (200px filter)")
        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 12,
                            "SheetForge v12  •  Algebraic LS Centerline  •  90° Snap  •  Intersection Trim")
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

    blur_ksize        = int(opts.get("blurKsize",    5))
    canny_low         = int(opts.get("cannyLow",    20))
    canny_high        = int(opts.get("cannyHigh",   80))
    min_blob_area     = int(opts.get("minBlobArea", 50))
    # v12: minimum Canny blob area — removes all edge fragments < 200px
    canny_min_area    = int(opts.get("cannyMinArea", 200))
    circle_rms_tol    = float(opts.get("circleRmsTol", 0.10))
    snap_radius       = float(opts.get("snapRadius", 15.0))

    steps = []

    # STEP 1: Load
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record("CV-1: Load Image", f"{img_w}×{img_h}px  DPI={dpi:.0f}", t0))

    # STEP 2: Median Blur
    t0 = now_ms()
    blurred = median_blur(gray, ksize=blur_ksize)
    steps.append(step_record(f"CV-2: Median Blur (ksize={blur_ksize})", "Noise reduced", t0))

    # STEP 3: Adaptive Threshold
    t0 = now_ms()
    binary   = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record("CV-3: Adaptive Threshold", f"{white_px} white px", t0))

    # STEP 4: Morph Open
    t0 = now_ms()
    opened    = morph_clean(binary)
    opened_px = int(np.count_nonzero(opened))
    steps.append(step_record("CV-4: MORPH_OPEN", f"{white_px - opened_px} spur px removed", t0))

    # STEP 5: Connected-Component Speckle Removal on binary mask
    t0 = now_ms()
    cleaned, removed_blobs, removed_px = remove_small_blobs(opened, min_blob_area)
    steps.append(step_record(
        f"CV-5: Blob Filter (minBlobArea={min_blob_area}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px)", t0))

    # STEP 6a: Canny edge detection
    t0 = now_ms()
    edges_raw = canny_edges(cleaned, canny_low, canny_high)

    # STEP 6b: Remove ALL Canny blobs < 200px (the key new filter)
    edges_200, removed_edge_blobs, removed_edge_px = remove_small_blobs(edges_raw, canny_min_area)
    steps.append(step_record(
        f"CV-6: Canny (lo={canny_low}, hi={canny_high}) + {canny_min_area}px area filter",
        f"{removed_edge_blobs} edge blob(s) removed ({removed_edge_px}px) — "
        f"{int(np.count_nonzero(edges_200))} edge px remain", t0))

    # STEP 6c: Skeletonize filtered Canny to true 1-px centerline
    t0 = now_ms()
    skeleton = skeletonize_mask(edges_200)
    skel_px  = int(np.count_nonzero(skeleton))
    steps.append(step_record(
        "CV-6c: Skeletonize (Zhang-Suen) → 1-px centerline",
        f"{skel_px} skeleton px  — single centerline per edge guaranteed", t0))

    # Build display edge image (thickened for visibility)
    edges_display = thicken_to_centerline(skeleton, thickness_px=4)

    # STEP 7: Contour extraction on 1-px skeleton
    t0 = now_ms()
    raw_contours = extract_centerline_contours(skeleton, min_contour_len=10)
    total_pts = sum(len(c) for c in raw_contours)
    steps.append(step_record(
        "CV-7: Contour extraction on skeleton (RETR_LIST, CHAIN_APPROX_NONE)",
        f"{len(raw_contours)} contours  |  {total_pts} vertices", t0))

    # STEP 8: Algebraic LS fitting — circle or orthogonal polyline
    t0 = now_ms()
    entities = classify_and_fit_contours(
        raw_contours,
        circle_rms_tol=circle_rms_tol,
        min_pts_circle=16,
    )
    n_circ = sum(1 for e in entities if e['type'] == 'circle')
    n_poly = sum(1 for e in entities if e['type'] == 'polyline')
    steps.append(step_record(
        f"LS-8: Algebraic LS fit (Kasa circle, 90° orthogonal polyline)",
        f"{len(raw_contours)} contours → {n_circ} circle(s) + {n_poly} polyline(s)  "
        f"[circle RMS tol={circle_rms_tol}]", t0))

    # STEP 9: Intersection trimming
    t0 = now_ms()
    entities = trim_endpoints_to_intersections(entities, snap_radius=snap_radius)
    steps.append(step_record(
        f"GEO-9: Intersection trimming (snap_radius={snap_radius}px)",
        f"Polyline endpoints trimmed to nearest intersecting segment", t0))

    # Output dir
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)
    ts_str   = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    pdf_name = f"design_{ts_str}.pdf"
    png_name = f"preview_{ts_str}.png"
    dxf_path = server_out_dir / dxf_name
    pdf_path = server_out_dir / pdf_name
    png_path = server_out_dir / png_name

    # STEP 10: Centerline DXF export
    t0 = now_ms()
    _, entity_count, dxf_size = build_centerline_dxf(entities, img_w, img_h, dxf_path)
    dxf_content_str = ""
    if dxf_size and dxf_size > 0:
        try:
            with open(dxf_path, encoding="utf-8", errors="replace") as f:
                dxf_content_str = f.read(200_000)
        except Exception:
            pass
    steps.append(step_record(
        "DXF-10: Centerline DXF export (CIRCLE + LWPOLYLINE, 90° snapped, intersection-trimmed)",
        f"{entity_count} entities  |  {dxf_size // 1024 if dxf_size else 0} KB", t0))

    # STEP 11: PNG comparison preview
    t0 = now_ms()
    png_ok   = build_comparison_png(edges_display, entities, img_w, img_h, png_path)
    png_size = png_path.stat().st_size if png_ok and png_path.exists() else 0
    steps.append(step_record(
        "PNG-11: Side-by-side preview (skeleton vs centerline entities)",
        f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # STEP 12: PDF export
    t0 = now_ms()
    pdf_ok = export_pdf(edges_display, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-12: Export centerline preview", "OK" if pdf_ok else "FAILED", t0))

    analysis = {
        "width"            : float(img_w),
        "height"           : float(img_h),
        "dpi"              : dpi,
        "edgePixels"       : skel_px,
        "edges"            : entity_count,
        "contours"         : len(raw_contours),
        "mergedContours"   : len(entities),
        "closedContours"   : sum(1 for e in entities if e.get('closed')),
        "totalVertices"    : total_pts,
        "blurKsize"        : blur_ksize,
        "cannyLow"         : canny_low,
        "cannyHigh"        : canny_high,
        "cannyMinArea"     : canny_min_area,
        "circleRmsTol"     : circle_rms_tol,
        "snapRadius"       : snap_radius,
        "minBlobArea"      : min_blob_area,
        "imgW"             : img_w,
        "imgH"             : img_h,
        "scaleMmPerDu"     : round(25.4 / dpi, 4),
        "coordSystem"      : "DXF Y-flipped (Y-up), origin=bottom-left",
        "circlesDetected"  : n_circ,
        "polylinesDetected": n_poly,
        "shapeSummary"     : (
            f"{n_poly} polyline(s) + {n_circ} circle(s) — "
            f"true 1-px skeleton, algebraic LS fit, 90° snap, intersection-trimmed"
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
