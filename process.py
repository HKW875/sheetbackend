#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v12  (Algebraic LS + Contour H/V Fitting)
================================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

WHAT CHANGED IN v12 AND WHY
-----------------------------
v11 used approxPolyDP for corner detection and rectilinear fitting, which
produced yellow dots at every vertex (including collinear points) and could
not guarantee fully connected H/V chains. 

FIX: Complete rewrite of the shape extraction pipeline:
  1. SKIP approxPolyDP entirely — use algebraic least squares fitting instead.
  2. Detect circles simultaneously using contour analysis + Kasa algebraic
     circle fit on each contour's point cloud.
  3. For rectilinear contours: extract H/V edges by fitting lines to boundary
     points using algebraic median fitting, then deduplicate, snap corners,
     align symmetrically, and remove unconnected dangling segments.
  4. Yellow dots ONLY at H-V intersections (true direction changes), NEVER on
     circles. All H/V segments form one continuous unbroken contour.
  5. Multiple parallel lines are merged; only one representative line per
     physical edge is kept.

Pipeline:
  1.  Load image
  2.  Median Blur                (salt-and-pepper pre-clean)
  3.  Adaptive Threshold         (binarise, strokes = WHITE)
  4.  Morph Open + Close         (remove spurs, fill 1px gaps)
  5.  Connected-Component Filter (remove EVERY blob < minBlobArea)
  6.  Canny Edge Detection       (visualisation / PDF preview only)
  7.  Contour Extraction          (findContours on cleaned mask, NO approxPolyDP)
  8.  Circle Detection            (Kasa algebraic LS fit per contour)
  9.  H/V Edge Extraction         (algebraic median fitting on boundary points)
  10. Parallel Deduplication      (Union-Find clustering, median coords)
  11. Corner Snapping             (extend/trim to exact H-V intersections)
  12. Symmetric Alignment         (cluster + median align intersection points)
  13. Unconnected Filter         (remove dangling segments)
  14. Final Parallel Merge        (keep one line per physical edge)
  15. DXF Export — CLEAN         (CIRCLE + 2-point LWPOLYLINE per segment)
  16. PNG Preview                (side-by-side: Canny vs final shapes + yellow dots)
  17. PDF Export                 (reportlab)
"""

import sys, os, json, time, traceback, math
from pathlib import Path
from itertools import combinations
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
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == lbl] = 255
        else:
            removed_px += area
            removed_blobs += 1
    return cleaned, removed_blobs, removed_px

def remove_small_blobs(binary, min_area, aggressive=True):
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
    edges_clean, removed_blobs, removed_px = remove_small_blobs(edges, min_blob_area)
    return edges_clean, removed_blobs, removed_px


# ════════════════════════════════════════════════════════════════════════════
# STEPS 7-8 — CONTOUR EXTRACTION + CIRCLE DETECTION (Kasa Algebraic LS)
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------
# KASA FIT
# ---------------------------------------------------------

def _kasa_circle_fit(pts_xy):

    x = pts_xy[:, 0].astype(np.float64)
    y = pts_xy[:, 1].astype(np.float64)

    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2

    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = res[0] / 2.0
    cy = res[1] / 2.0

    r = math.sqrt(abs(res[2] + cx**2 + cy**2))

    return cx, cy, r


# ---------------------------------------------------------
# CIRCULARITY
# ---------------------------------------------------------

def contour_circularity(cnt):

    area = cv2.contourArea(cnt)

    if area <= 1:
        return 0.0

    peri = cv2.arcLength(cnt, True)

    if peri <= 1:
        return 0.0

    return (4.0 * math.pi * area) / (peri * peri)


# ---------------------------------------------------------
# RECTILINEAR DETECTION
# ---------------------------------------------------------

def is_rectilinear(cnt):

    pts = cnt.reshape(-1, 2)

    if len(pts) < 10:
        return False

    hv = 0

    for i in range(len(pts)):

        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]

        dx = abs(float(p2[0] - p1[0]))
        dy = abs(float(p2[1] - p1[1]))

        if dx < 3 or dy < 3:
            hv += 1

    return (hv / len(pts)) > 0.75


# ---------------------------------------------------------
# BBOX TEST
# ---------------------------------------------------------

def point_inside_bbox(px, py, bbox):

    x1, y1, x2, y2 = bbox

    return x1 <= px <= x2 and y1 <= py <= y2


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def detect_circles_and_rectilinear(cleaned_mask):

    contours, hierarchy = cv2.findContours(
        cleaned_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    circles = []
    rectilinear = []

    # -----------------------------------------------------
    # FIND RECTILINEAR CONTAINERS
    # -----------------------------------------------------

    for cnt in contours:

        if len(cnt) < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 20:
            continue

        if is_rectilinear(cnt):

            rectilinear.append({
                'cnt': cnt,
                'pts': cnt.reshape(-1, 2).astype(np.float32),
                'bbox': (x, y, x + w, y + h)
            })

    # -----------------------------------------------------
    # DETECT CIRCLES
    # -----------------------------------------------------

    for cnt in contours:

        if len(cnt) < 20:
            continue

        # ALWAYS DEFINE pts EARLY
        pts = cnt.reshape(-1, 2).astype(np.float32)

        if len(pts) < 5:
            continue

        area = cv2.contourArea(cnt)

        if area < 30:
            continue

        x = pts[:, 0]
        y = pts[:, 1]

        w_bbox = float(x.max() - x.min())
        h_bbox = float(y.max() - y.min())

        if w_bbox < 10 or h_bbox < 10:
            continue

        # -------------------------------------------------
        # CIRCULARITY
        # -------------------------------------------------

        circ = contour_circularity(cnt)

        if circ < 0.55:
            continue

        # -------------------------------------------------
        # ELLIPSE FIT
        # -------------------------------------------------

        try:

            ellipse = cv2.fitEllipse(cnt)

            (ecx, ecy), (MA, ma), angle = ellipse

            major = max(MA, ma)
            minor = min(MA, ma)

            ellipse_ratio = minor / (major + 1e-9)

            if ellipse_ratio < 0.65:
                continue

        except cv2.error:
            continue

        # -------------------------------------------------
        # KASA FIT
        # -------------------------------------------------

        try:

            cx, cy, r = _kasa_circle_fit(pts)

        except Exception:
            continue

        if r < 10 or r > 200:
            continue

        dists = np.sqrt((x - cx)**2 + (y - cy)**2)

        rms = np.sqrt(((dists - r)**2).mean())

        rel_err = rms / (r + 1e-9)

        # adaptive tolerance
        if r < 30:
            err_thresh = 0.35
        elif r < 60:
            err_thresh = 0.30
        else:
            err_thresh = 0.25

        if rel_err > err_thresh:
            continue

        # -------------------------------------------------
        # MUST BE INSIDE RECTILINEAR REGION
        # -------------------------------------------------

        inside = False

        for rect in rectilinear:

            if point_inside_bbox(cx, cy, rect['bbox']):

                inside = True
                break

        if not inside:
            continue

        circles.append({

            'cx': float(cx),
            'cy': float(cy),
            'r': float(r),

            'circularity': float(circ),

            'ellipse_ratio': float(ellipse_ratio),

            'rms': float(rel_err)
        })

    # -----------------------------------------------------
    # MERGE DUPLICATES
    # -----------------------------------------------------

    circles = sorted(circles, key=lambda c: c['rms'])

    merged = []

    used = [False] * len(circles)

    for i in range(len(circles)):

        if used[i]:
            continue

        c1 = circles[i]

        group = [c1]

        used[i] = True

        for j in range(i + 1, len(circles)):

            if used[j]:
                continue

            c2 = circles[j]

            dc = math.hypot(
                c1['cx'] - c2['cx'],
                c1['cy'] - c2['cy']
            )

            dr = abs(c1['r'] - c2['r']) / max(c1['r'], c2['r'], 1e-9)

            if dc < 40 and dr < 0.30:

                group.append(c2)

                used[j] = True

        merged.append({

            'cx': float(np.mean([c['cx'] for c in group])),

            'cy': float(np.mean([c['cy'] for c in group])),

            'r': float(np.mean([c['r'] for c in group])),

            'rms': float(min(c['rms'] for c in group)),

            'circularity': float(np.mean([c['circularity'] for c in group])),

            'ellipse_ratio': float(np.mean([c['ellipse_ratio'] for c in group]))
        })

    return merged, rectilinear


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — H/V EDGE EXTRACTION (Algebraic Median Fitting on Boundary Points)
# ════════════════════════════════════════════════════════════════════════════

def extract_hv_edges(rectilinear_contours):
    """
    Extract horizontal and vertical edges from rectilinear contours.
    For each contour, fit lines to the top/bottom/left/right boundary points
    using algebraic median fitting.
    """
    all_edges = []

    for r in rectilinear_contours:
        pts = r['pts']
        x, y = pts[:, 0], pts[:, 1]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())

        margin = max(15, min(x_max - x_min, y_max - y_min) * 0.15)

        # Top edge (horizontal)
        top_mask = y <= y_min + margin
        top_pts = pts[top_mask]
        if len(top_pts) > 5:
            y_fit = float(np.median(top_pts[:, 1]))
            x_start = float(np.min(top_pts[:, 0]))
            x_end = float(np.max(top_pts[:, 0]))
            all_edges.append({'type': 'H', 'coord': y_fit, 'start': x_start, 'end': x_end})

        # Bottom edge (horizontal)
        bot_mask = y >= y_max - margin
        bot_pts = pts[bot_mask]
        if len(bot_pts) > 5:
            y_fit = float(np.median(bot_pts[:, 1]))
            x_start = float(np.min(bot_pts[:, 0]))
            x_end = float(np.max(bot_pts[:, 0]))
            all_edges.append({'type': 'H', 'coord': y_fit, 'start': x_start, 'end': x_end})

        # Left edge (vertical)
        left_mask = x <= x_min + margin
        left_pts = pts[left_mask]
        if len(left_pts) > 5:
            x_fit = float(np.median(left_pts[:, 0]))
            y_start = float(np.min(left_pts[:, 1]))
            y_end = float(np.max(left_pts[:, 1]))
            all_edges.append({'type': 'V', 'coord': x_fit, 'start': y_start, 'end': y_end})

        # Right edge (vertical)
        right_mask = x >= x_max - margin
        right_pts = pts[right_mask]
        if len(right_pts) > 5:
            x_fit = float(np.median(right_pts[:, 0]))
            y_start = float(np.min(right_pts[:, 1]))
            y_end = float(np.max(right_pts[:, 1]))
            all_edges.append({'type': 'V', 'coord': x_fit, 'start': y_start, 'end': y_end})

    # Filter very short segments (noise)
    all_edges = [s for s in all_edges if (s['end'] - s['start']) >= 40]

    return all_edges


# ════════════════════════════════════════════════════════════════════════════
# STEPS 10-14 — DEDUP, SNAP, ALIGN, FILTER, MERGE
# ════════════════════════════════════════════════════════════════════════════

def dedup_parallel_segments(segments, tol=60, slack=80):
    """Union-Find clustering of duplicate parallel segments."""
    h_segs = [s.copy() for s in segments if s['type'] == 'H']
    v_segs = [s.copy() for s in segments if s['type'] == 'V']

    def cluster(segs_list):
        if not segs_list: return []
        n = len(segs_list)
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        for i, j in combinations(range(n), 2):
            if abs(segs_list[i]['coord'] - segs_list[j]['coord']) <= tol:
                overlap = min(segs_list[i]['end'], segs_list[j]['end']) - max(segs_list[i]['start'], segs_list[j]['start'])
                if overlap > -slack:
                    union(i, j)
        groups = defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        return [{'type': segs_list[0]['type'],
                 'coord': float(np.median([segs_list[i]['coord'] for i in g])),
                 'start': float(min(segs_list[i]['start'] for i in g)),
                 'end': float(max(segs_list[i]['end'] for i in g))} for g in groups.values()]

    return cluster(h_segs) + cluster(v_segs)

def snap_segment_corners(segments, tol=100):
    """Extend/trim H and V segments to meet at exact intersections."""
    h_segs = [s.copy() for s in segments if s['type'] == 'H']
    v_segs = [s.copy() for s in segments if s['type'] == 'V']
    if not h_segs or not v_segs: return segments

    h_ends = [(hi, 'start', s['start'], s['coord']) for hi, s in enumerate(h_segs)] + \
             [(hi, 'end', s['end'], s['coord']) for hi, s in enumerate(h_segs)]
    v_ends = [(vi, 'start', s['coord'], s['start']) for vi, s in enumerate(v_segs)] + \
             [(vi, 'end', s['coord'], s['end']) for vi, s in enumerate(v_segs)]

    pairs = []
    for he in h_ends:
        for ve in v_ends:
            d = math.hypot(he[2]-ve[2], he[3]-ve[3])
            if d <= tol: pairs.append((d, he, ve))
    pairs.sort(key=lambda t: t[0])

    used_h, used_v = set(), set()
    for d, he, ve in pairs:
        hk, vk = (he[0], he[1]), (ve[0], ve[1])
        if hk in used_h or vk in used_v: continue
        used_h.add(hk); used_v.add(vk)
        h_segs[he[0]][he[1]] = ve[2]
        v_segs[ve[0]][ve[1]] = he[3]

    for s in h_segs + v_segs:
        if s['start'] > s['end']: s['start'], s['end'] = s['end'], s['start']
    return h_segs + v_segs

def find_intersections(segments):
    """Find all H-V intersection points."""
    h_segs = [s for s in segments if s['type'] == 'H']
    v_segs = [s for s in segments if s['type'] == 'V']
    pts = []
    for hs in h_segs:
        for vs in v_segs:
            ix, iy = vs['coord'], hs['coord']
            if hs['start']-10 <= ix <= hs['end']+10 and vs['start']-10 <= iy <= vs['end']+10:
                pts.append((float(ix), float(iy)))
    unique = []
    for p in pts:
        if not any(math.hypot(p[0]-u[0], p[1]-u[1]) < 15 for u in unique):
            unique.append(p)
    return unique

def align_points_symmetrically(points, tol=40):
    """Align intersection points to be level and symmetrical via clustering."""
    if not points: return []
    pts = np.array(points)
    x_vals = sorted(pts[:, 0])
    y_vals = sorted(pts[:, 1])

    def cluster(vals):
        if not vals: return []
        groups = [[vals[0]]]
        for v in vals[1:]:
            if v - groups[-1][-1] <= tol: groups[-1].append(v)
            else: groups.append([v])
        return [np.median(g) for g in groups]

    x_groups = cluster(x_vals)
    y_groups = cluster(y_vals)

    def nearest(val, groups):
        return min(groups, key=lambda g: abs(g - val))

    aligned = [(nearest(p[0], x_groups), nearest(p[1], y_groups)) for p in pts]
    unique = []
    for p in aligned:
        if not any(math.hypot(p[0]-u[0], p[1]-u[1]) < 10 for u in unique):
            unique.append(p)
    return unique

def remove_unconnected_segments(segments, intersections, tol=30):
    """Remove segments whose endpoints don't connect to any intersection."""
    result = []
    for s in segments:
        if s['type'] == 'H':
            y, x1, x2 = s['coord'], s['start'], s['end']
            c1 = any(abs(p[0]-x1)<tol and abs(p[1]-y)<tol for p in intersections)
            c2 = any(abs(p[0]-x2)<tol and abs(p[1]-y)<tol for p in intersections)
            if c1 or c2: result.append(s)
        else:
            x, y1, y2 = s['coord'], s['start'], s['end']
            c1 = any(abs(p[0]-x)<tol and abs(p[1]-y1)<tol for p in intersections)
            c2 = any(abs(p[0]-x)<tol and abs(p[1]-y2)<tol for p in intersections)
            if c1 or c2: result.append(s)
    return result

def merge_final_parallel(segments, tol=25):
    """Final merge: keep only one line per unique physical edge."""
    return dedup_parallel_segments(segments, tol=tol, slack=50)


# ════════════════════════════════════════════════════════════════════════════
# STEP 15 — CLEAN DXF EXPORT (CIRCLE + 2-point LWPOLYLINE per segment)
# ════════════════════════════════════════════════════════════════════════════

def build_clean_dxf(circles, segments, img_w, img_h, out_path):
    """Export to DXF. Always counts entities even if ezdxf is not available."""
    entity_count = len(circles) + len(segments)

    if not HAS_DXF:
        return None, entity_count, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"] = (0.0, 0.0)
    doc.header["$LIMMAX"] = (float(img_w), float(img_h))

    msp = doc.modelspace()
    doc.layers.new("CIRCLES", dxfattribs={"color": 1, "linetype": "CONTINUOUS"})
    doc.layers.new("H_LINES", dxfattribs={"color": 5, "linetype": "CONTINUOUS"})
    doc.layers.new("V_LINES", dxfattribs={"color": 6, "linetype": "CONTINUOUS"})

    for c in circles:
        msp.add_circle((c['cx'], c['cy'], 0.0), c['r'],
                      dxfattribs={"layer": "CIRCLES", "color": 256})

    for s in segments:
        if s['type'] == 'H':
            pts = [(s['start'], s['coord']), (s['end'], s['coord'])]
            layer = "H_LINES"
        else:
            pts = [(s['coord'], s['start']), (s['coord'], s['end'])]
            layer = "V_LINES"
        msp.add_lwpolyline(pts, format="xy",
                          dxfattribs={"layer": layer, "color": 256})

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════
# STEP 16 — PNG PREVIEW (side-by-side comparison with yellow dots)
# ════════════════════════════════════════════════════════════════════════════

def build_comparison_png(edges, circles, segments, intersections, img_w, img_h, out_path):
    if not HAS_CV or not np: return False

    try:
        left = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        left[:] = (15, 12, 10)
        left[edges > 0] = (255, 255, 255)

        right = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        right[:] = (15, 12, 10)

        # Draw circles in blue (NO yellow dots on circles)
        for c in circles:
            cx, cy, r = int(round(c['cx'])), int(round(c['cy'])), int(round(c['r']))
            cv2.circle(right, (cx, cy), r, (220, 80, 80), 2, cv2.LINE_AA)

        # Draw H/V segments in green
        for s in segments:
            if s['type'] == 'H':
                y = int(round(s['coord']))
                x1 = int(round(s['start']))
                x2 = int(round(s['end']))
                cv2.line(right, (x1, y), (x2, y), (80, 200, 80), 2, cv2.LINE_AA)
            else:
                x = int(round(s['coord']))
                y1 = int(round(s['start']))
                y2 = int(round(s['end']))
                cv2.line(right, (x, y1), (x, y2), (80, 200, 80), 2, cv2.LINE_AA)

        # Draw YELLOW DOTS at intersections (direction changes ONLY)
        for p in intersections:
            px, py = int(round(p[0])), int(round(p[1]))
            cv2.circle(right, (px, py), 7, (0, 255, 255), -1)   # Yellow fill
            cv2.circle(right, (px, py), 9, (255, 128, 0), 2)     # Orange outline

        sep = np.full((img_h, 4, 3), 40, dtype=np.uint8)
        panel = np.concatenate([left, sep, right], axis=1)

        ok = cv2.imwrite(str(out_path), panel)
        return bool(ok and out_path.exists())
    except Exception as e:
        sys.stderr.write(f"PNG preview error: {e}\n")
        return False


# ════════════════════════════════════════════════════════════════════════════
# STEP 17 — PDF EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export_pdf(edges, out_path, orig_bgr=None):
    if not HAS_RL: return False

    import tempfile, os as _os
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    try:
        page_w, page_h = landscape(A4)
        c = rl_canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

        margin = 30
        col_w = (page_w - margin * 3) / 2
        col_h = page_h - margin * 2 - 40

        c.setFillColorRGB(0.04, 0.05, 0.06)
        c.rect(0, page_h - 36, page_w, 36, fill=1, stroke=0)
        c.setFillColorRGB(0.9, 0.91, 0.93)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, page_h - 24, "SheetForge — Algebraic LS + Contour H/V Fitting Preview")
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

        right_x = margin * 2 + col_w
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        reader, tname = _arr_to_reader(edges_bgr)
        tmp_files.append(tname)
        _draw_panel(c, reader, right_x, margin, col_w, col_h, "Canny Edge Detection")

        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 12,
                            "SheetForge v12  •  Algebraic LS + Contour H/V  •  Clean DXF")
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
    img_h = h - title_h
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
    opts = {}
    if len(sys.argv) > 2:
        try: opts = json.loads(sys.argv[2])
        except Exception: pass

    blur_ksize = int(opts.get("blurKsize", 7))
    canny_low = int(opts.get("cannyLow", 20))
    canny_high = int(opts.get("cannyHigh", 80))
    min_blob_area = int(opts.get("minBlobArea", 20))
    dedup_tol = float(opts.get("dedupTol", 60.0))
    dedup_slack = float(opts.get("dedupSlack", 80.0))
    corner_snap_tol = float(opts.get("cornerSnapTol", 100.0))
    align_tol = float(opts.get("alignTol", 40.0))
    filter_tol = float(opts.get("filterTol", 30.0))
    final_merge_tol = float(opts.get("finalMergeTol", 25.0))

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
    binary = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record("CV-3: Adaptive Threshold", f"{white_px} white px", t0))

    # ── STEP 4: Morph Open + Close ───────────────────────────────────────
    t0 = now_ms()
    opened = morph_clean(binary)
    opened_px = int(np.count_nonzero(opened))
    steps.append(step_record("CV-4: MORPH_OPEN + MORPH_CLOSE",
                              f"{white_px - opened_px} net px change", t0))

    # ── STEP 5: Connected-Component Filter ─────────────────────────────────
    t0 = now_ms()
    cleaned, removed_blobs, removed_px = remove_small_blobs(opened, min_blob_area, aggressive=True)
    steps.append(step_record(
        f"CV-5: Connected-Component Filter (minBlobArea={min_blob_area}px)",
        f"{removed_blobs} speckle blob(s) removed ({removed_px}px)", t0))

    # ── STEP 6: Canny (visualisation only) ────────────────────────────────
    t0 = now_ms()
    edges_raw = canny_edges(cleaned, canny_low, canny_high)
    edges, edge_dot_blobs, edge_dot_px = clean_edge_preview(edges_raw, min_blob_area=3)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(f"CV-6: Canny + dot cleanup",
                              f"{edge_px} edge px, {edge_dot_blobs} stray dot(s) removed", t0))

    # ── STEP 7: Contour Extraction + Circle Detection (Algebraic LS) ───────
    t0 = now_ms()
    circles, rectilinear = detect_circles_and_rectilinear(cleaned)
    steps.append(step_record(
        "LS-7: Contour Extraction + Algebraic Circle Detection (Kasa LS)",
        f"{len(circles)} circle(s), {len(rectilinear)} rectilinear contour(s)", t0))

    # ── STEP 8: H/V Edge Extraction ───────────────────────────────────────
    t0 = now_ms()
    segments = extract_hv_edges(rectilinear)
    steps.append(step_record("GEO-8: H/V Edge Extraction (algebraic median on boundary)",
                              f"{len(segments)} raw edges", t0))

    # ── STEP 9: Parallel Deduplication ────────────────────────────────────
    t0 = now_ms()
    segments = dedup_parallel_segments(segments, tol=dedup_tol, slack=dedup_slack)
    steps.append(step_record("GEO-9: Parallel Deduplication (Union-Find + median)",
                              f"{len(segments)} segments after dedup", t0))

    # ── STEP 10: Corner Snapping ────────────────────────────────────────────
    t0 = now_ms()
    segments = snap_segment_corners(segments, tol=corner_snap_tol)
    steps.append(step_record("GEO-10: Corner Snapping (extend/trim to intersect)",
                              "Segments snapped to exact intersections", t0))

    # ── STEP 11: Find Intersections ───────────────────────────────────────
    t0 = now_ms()
    intersections = find_intersections(segments)
    steps.append(step_record("GEO-11: Intersection Detection",
                              f"{len(intersections)} H-V intersection points", t0))

    # ── STEP 12: Symmetric Alignment ───────────────────────────────────────
    t0 = now_ms()
    intersections = align_points_symmetrically(intersections, tol=align_tol)
    steps.append(step_record("GEO-12: Symmetric Alignment (cluster + median)",
                              f"{len(intersections)} aligned points", t0))

    # ── STEP 13: Remove Unconnected ─────────────────────────────────────────
    t0 = now_ms()
    segments = remove_unconnected_segments(segments, intersections, tol=filter_tol)
    steps.append(step_record("GEO-13: Unconnected Segment Filter",
                              f"{len(segments)} connected segments", t0))

    # ── STEP 14: Final Parallel Merge ──────────────────────────────────────
    t0 = now_ms()
    segments = merge_final_parallel(segments, tol=final_merge_tol)
    steps.append(step_record("GEO-14: Final Parallel Merge (one line per edge)",
                              f"{len(segments)} final segments", t0))

    # ── STEP 15: Recalculate intersections after final merge ──────────────
    t0 = now_ms()
    intersections = find_intersections(segments)
    intersections = align_points_symmetrically(intersections, tol=align_tol)
    steps.append(step_record("GEO-15: Final Intersection Recalculation",
                              f"{len(intersections)} final corners (yellow dots)", t0))

    # ── Output dir ────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    pdf_name = f"design_{ts_str}.pdf"
    png_name = f"preview_{ts_str}.png"

    dxf_path = server_out_dir / dxf_name
    pdf_path = server_out_dir / pdf_name
    png_path = server_out_dir / png_name

    # ── STEP 16: DXF Export ───────────────────────────────────────────────
    t0 = now_ms()
    _, entity_count, dxf_size = build_clean_dxf(circles, segments, img_w, img_h, dxf_path)

    dxf_content_str = ""
    if dxf_size and dxf_size > 0:
        try:
            with open(dxf_path, encoding="utf-8", errors="replace") as f:
                dxf_content_str = f.read(200_000)
        except Exception:
            pass

    steps.append(step_record(
        "DXF-16: Clean export (CIRCLE + 2-point LWPOLYLINE, true position)",
        f"{entity_count} entities  |  {dxf_size // 1024 if dxf_size else 0} KB", t0))

    # ── STEP 17: PNG Preview ──────────────────────────────────────────────
    t0 = now_ms()
    png_ok = build_comparison_png(edges, circles, segments, intersections, img_w, img_h, png_path)
    png_size = png_path.stat().st_size if png_ok and png_path.exists() else 0
    steps.append(step_record(
        "PNG-17: Side-by-side preview (Canny vs final shapes + yellow dots)",
        f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # ── STEP 18: PDF Export ───────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-18: Export edge preview", "OK" if pdf_ok else "FAILED", t0))

    # ── Analysis summary ──────────────────────────────────────────────────
    n_h = sum(1 for s in segments if s['type'] == 'H')
    n_v = sum(1 for s in segments if s['type'] == 'V')
    analysis = {
        "width": float(img_w),
        "height": float(img_h),
        "dpi": dpi,
        "edgePixels": edge_px,
        "edges": entity_count,
        "circlesDetected": len(circles),
        "segmentsDetected": len(segments),
        "horizontalSegments": n_h,
        "verticalSegments": n_v,
        "intersections": len(intersections),
        "speckleBlobsRemoved": removed_blobs,
        "blurKsize": blur_ksize,
        "cannyLow": canny_low,
        "cannyHigh": canny_high,
        "minBlobArea": min_blob_area,
        "coordSystem": "origin=top-left px, Y-down, no approxPolyDP, algebraic LS fitting",
        "shapeSummary": (
            f"{len(circles)} circle(s), {n_h} horizontal + {n_v} vertical segments, "
            f"{len(intersections)} corner intersections (yellow dots) — "
            f"contour-based H/V extraction, Union-Find dedup, corner snap, symmetric align"
        ),
    }

    # Include segment and intersection data for verification
    result_data = {
        "steps": steps,
        "analysis": analysis,
        "circles": circles,
        "segments": segments,
        "intersections": intersections,
        "dwg": {
            "entities": entity_count,
            "fileSize": dxf_size or 0,
            "filename": dxf_name if dxf_size else "",
            "dxfAbsPath": str(dxf_path) if dxf_size else "",
            "pdfFilename": pdf_name if pdf_ok else "",
            "edgePngFilename": png_name if png_ok else "",
            "edgePngPath": str(png_path) if png_ok else "",
            "gcodeFiles": {},
            "gcodeFilePaths": {},
        },
        "dxfContent": dxf_content_str,
        "dxfAvailable": bool(dxf_size and dxf_size > 0),
        "pdfAvailable": pdf_ok,
        "pngAvailable": png_ok,
        "gcodeAvailable": False,
    }

    print(json.dumps(result_data, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "steps": [],
            "analysis": {},
            "dwg": {"entities": 0, "fileSize": 0},
            "dxfContent": "",
        }))
        sys.exit(1)
