#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v8.0
================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

Pipeline (precision contour → CAD → CNC):
  1. Load image              (cv2.imread)
  2. Median Blur             (cv2.medianBlur — salt-and-pepper noise reduction)
  3. Adaptive Threshold      (cv2.adaptiveThreshold — binarisation, lines=WHITE)
  4. Morph Clean             (cv2.morphologyEx MORPH_OPEN — speckle removal)
  5. Canny Edge Detection    (cv2.Canny — thin precise edges)
  6. Contour Extraction      (cv2.findContours → cv2.approxPolyDP simplification)
  7. DXF Export — RAW        (ezdxf — raw contours, one LWPOLYLINE per contour)
  8. DXF Merge + Heal        (NEW — chain-stitch fragmented segments into closed
                              LWPOLYLINE contours; output is the download DXF)
  9. G-Code Generation       (NEW — one .nc file per CNC machine type, from merged DXF)
 10. PDF Export              (reportlab — edge image rendered to PDF page)
 11. PNG Preview             (dark-bg canvas with white edges — saved as .png)
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


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD IMAGE
# ════════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — MEDIAN BLUR
# ════════════════════════════════════════════════════════════════════════════════

def median_blur(gray, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(gray, ksize)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — ADAPTIVE THRESHOLD
# ════════════════════════════════════════════════════════════════════════════════

def adaptive_threshold_binarize(blurred):
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — MORPHOLOGICAL OPEN
# ════════════════════════════════════════════════════════════════════════════════

def morph_clean(binary):
    kernel  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — CANNY EDGE DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def canny_edges(cleaned, low_threshold=30, high_threshold=100):
    return cv2.Canny(cleaned, low_threshold, high_threshold)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — CONTOUR EXTRACTION + approxPolyDP
# ════════════════════════════════════════════════════════════════════════════════

def extract_simplified_contours(edges, epsilon_factor=0.5):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE,
    )
    simplified = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        arc     = cv2.arcLength(cnt, closed=False)
        epsilon = epsilon_factor * arc / max(len(cnt), 1)
        epsilon = max(epsilon, 0.3)
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=False)
        if len(approx) >= 2:
            simplified.append(approx)
    return simplified


# ════════════════════════════════════════════════════════════════════════════════
# COORDINATE HELPER
# ════════════════════════════════════════════════════════════════════════════════

def px_to_cad(px_x, px_y, img_h):
    return float(px_x), float((img_h - 1) - px_y)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — DXF RAW EXPORT  (unchanged from v7 — intermediate file)
# ════════════════════════════════════════════════════════════════════════════════

def build_and_save_dxf(simplified_contours, img_w, img_h, out_path):
    if not HAS_DXF:
        return None, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"] = (0.0, 0.0)
    doc.header["$LIMMAX"] = (float(img_w), float(img_h))

    msp = doc.modelspace()
    doc.layers.new("EDGES", dxfattribs={"color": 7, "linetype": "CONTINUOUS"})

    entity_count = 0
    for cnt in simplified_contours:
        pts = []
        for pt in cnt:
            px_x = int(pt[0][0])
            px_y = int(pt[0][1])
            cx, cy = px_to_cad(px_x, px_y, img_h)
            pts.append((cx, cy))

        if len(pts) < 2:
            continue

        if len(pts) == 2:
            msp.add_line(start=pts[0], end=pts[1],
                         dxfattribs={"layer": "EDGES", "color": 256})
        else:
            msp.add_lwpolyline(pts, format="xy",
                               dxfattribs={"layer": "EDGES", "color": 256})
        entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — DXF MERGE + HEAL
# ════════════════════════════════════════════════════════════════════════════════
#
# Goal: turn the many short LINE and open LWPOLYLINE fragments that OpenCV
# produces into smooth, closed LWPOLYLINE contours suitable for CAM.
#
# Algorithm:
#   A. Collect all segment endpoints from the raw DXF.
#   B. Build a spatial adjacency graph: endpoint → list of (segment_id, end_idx).
#   C. Chain-stitch adjacent endpoints within SNAP_TOL drawing units.
#   D. Walk each chain: if start == end (within SNAP_TOL) mark as closed.
#   E. Chains shorter than MIN_CHAIN_PTS are discarded (noise).
#   F. Write merged chains as closed/open LWPOLYLINE entities on layer "CONTOURS".
#
# ════════════════════════════════════════════════════════════════════════════════

SNAP_TOL       = 3.0   # drawing units — endpoints closer than this are joined
MIN_CHAIN_PTS  = 4     # chains with fewer vertices are discarded
CLOSE_TOL      = 5.0   # if chain start-end gap ≤ this, close the polyline


def _pt_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _collect_segments_from_dxf(raw_dxf_path):
    """
    Read the raw DXF and return a list of segments, each being a list of
    (x, y) tuples forming either a LINE (2 pts) or LWPOLYLINE (N pts).
    """
    if not HAS_DXF:
        return []
    try:
        doc = ezdxf.readfile(str(raw_dxf_path))
    except Exception:
        return []

    segments = []
    for entity in doc.modelspace():
        etype = entity.dxftype()
        if etype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            segments.append([(s.x, s.y), (e.x, e.y)])
        elif etype in ("LWPOLYLINE", "POLYLINE"):
            pts = []
            try:
                for pt in entity.get_points():
                    pts.append((pt[0], pt[1]))
            except Exception:
                try:
                    pts = [(v[0], v[1]) for v in entity.points()]
                except Exception:
                    pass
            if len(pts) >= 2:
                segments.append(pts)
    return segments


def _build_endpoint_index(segments, snap_tol):
    """
    Return a dict: rounded_key → list of (seg_idx, end_which)
    where end_which is 0=start, 1=end of segment.
    Rounding by snap_tol buckets neighbouring endpoints together.
    """
    def key(x, y):
        # bucket to nearest snap_tol grid
        bx = round(x / snap_tol)
        by = round(y / snap_tol)
        return (bx, by)

    index = defaultdict(list)
    for si, seg in enumerate(segments):
        index[key(*seg[0])].append((si, 0))
        index[key(*seg[-1])].append((si, 1))
    return index


def _chain_stitch(segments, snap_tol):
    """
    Greedy chain-stitching: walk from an unused segment end, greedily
    snapping to the nearest unvisited neighbouring endpoint.

    Returns a list of chains, each a list of (x,y) and a boolean is_closed.
    """
    if not segments:
        return []

    index = _build_endpoint_index(segments, snap_tol)
    used  = [False] * len(segments)

    def bucket_neighbours(x, y):
        bx = round(x / snap_tol)
        by = round(y / snap_tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (bx + dx, by + dy)
                if k in index:
                    yield from index[k]

    def find_next(tip_x, tip_y, exclude_seg):
        best_d   = snap_tol + 1e-9
        best_hit = None
        for (si, end_which) in bucket_neighbours(tip_x, tip_y):
            if used[si] or si == exclude_seg:
                continue
            seg = segments[si]
            ep  = seg[0] if end_which == 0 else seg[-1]
            d   = _pt_dist((tip_x, tip_y), ep)
            if d < best_d:
                best_d   = d
                best_hit = (si, end_which, d)
        return best_hit

    chains = []

    for start_si in range(len(segments)):
        if used[start_si]:
            continue
        used[start_si] = True

        # Build chain forward from end of start_si
        seg       = segments[start_si]
        chain_pts = list(seg)        # start with all points of first segment
        tip_si    = start_si

        while True:
            tip = chain_pts[-1]
            hit = find_next(tip[0], tip[1], tip_si)
            if hit is None:
                break
            next_si, end_which, _ = hit
            used[next_si] = True
            tip_si        = next_si
            next_seg      = segments[next_si]
            if end_which == 0:
                # join forward: next segment goes in natural order
                chain_pts.extend(next_seg[1:])
            else:
                # join reversed: flip the segment before appending
                chain_pts.extend(reversed(next_seg[:-1]))

        is_closed = _pt_dist(chain_pts[0], chain_pts[-1]) <= CLOSE_TOL
        chains.append((chain_pts, is_closed))

    return chains


def merge_and_heal_dxf(raw_dxf_path, merged_dxf_path, img_w, img_h):
    """
    Stage 8: read raw_dxf_path, chain-stitch segments, write merged_dxf_path.

    Returns (merged_doc, n_contours, n_closed, file_size_bytes).
    """
    if not HAS_DXF:
        return None, 0, 0, 0

    segments = _collect_segments_from_dxf(raw_dxf_path)
    if not segments:
        return None, 0, 0, 0

    chains = _chain_stitch(segments, SNAP_TOL)

    # Write merged DXF
    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"]   = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"]   = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"]   = (0.0, 0.0)
    doc.header["$LIMMAX"]   = (float(img_w), float(img_h))

    msp = doc.modelspace()
    doc.layers.new("CONTOURS", dxfattribs={"color": 3, "linetype": "CONTINUOUS"})
    doc.layers.new("OPEN",     dxfattribs={"color": 1, "linetype": "CONTINUOUS"})

    n_closed   = 0
    n_contours = 0

    for chain_pts, is_closed in chains:
        if len(chain_pts) < MIN_CHAIN_PTS:
            continue  # discard noise

        layer = "CONTOURS" if is_closed else "OPEN"

        # Deduplicate consecutive identical/near-identical points
        deduped = [chain_pts[0]]
        for p in chain_pts[1:]:
            if _pt_dist(p, deduped[-1]) > 0.1:
                deduped.append(p)

        if len(deduped) < 2:
            continue

        poly = msp.add_lwpolyline(
            deduped,
            format="xy",
            dxfattribs={"layer": layer, "color": 256},
        )
        poly.close(is_closed)

        n_contours += 1
        if is_closed:
            n_closed += 1

    doc.saveas(str(merged_dxf_path))
    file_size = merged_dxf_path.stat().st_size
    return doc, n_contours, n_closed, file_size


# ════════════════════════════════════════════════════════════════════════════════
# STEP 9 — DXF SHAPE REFINEMENT PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
#
# Input : merged DXF from Step 8 (LWPOLYLINE + LINE fragments)
# Output: refined DXF with native CIRCLE and closed LWPOLYLINE entities,
#         centred on the outer boundary, plus a rendered PNG preview.
#
# Sub-steps:
#   9a. Parse all polyline vertices from the merged DXF
#   9b. Algebraic Least Squares + Kasa circle fitting per polyline
#       Solve  x² + y² = Ax + By + C  →  linear system for (cx,cy,r)
#       Accept as circle when RMS residual < CIRCLE_RMS_THRESH
#   9c. Classify remainder as rectangle / open path
#   9d. Deduplicate: group by centre proximity and size similarity,
#       keep one representative per group
#   9e. Size filter: discard shapes whose bounding box is smaller than
#       min_shape_size (user-supplied drawing units, 0 = off)
#   9f. Gemini AI text filter (optional): user describes shapes to eliminate
#       in plain English; Gemini inspects each shape's properties and returns
#       a keep/discard verdict
#   9g. Centre alignment: shift every shape so the group is centred on the
#       X-midpoint of the outer boundary rectangle
#   9h. Write refined DXF: native CIRCLE entities for circles,
#       closed LWPOLYLINE for rectangles / closed paths
#   9i. Render PNG: white shapes on dark background
# ════════════════════════════════════════════════════════════════════════════════

CIRCLE_RMS_THRESH  = 2.5   # drawing units — lower = stricter circle test
DEDUP_CENTER_TOL   = 8.0   # centres closer than this → same shape group
DEDUP_SIZE_REL     = 0.12  # relative size diff ≤ this → same group


# ── 9a-b: Kasa algebraic circle fit ─────────────────────────────────────────

def _kasa_fit(pts):
    """
    Algebraic Least Squares (Kasa method):
        x² + y² = Ax + By + C
    Solve for A,B,C then recover  cx = A/2,  cy = B/2,  r = sqrt(C + cx²+cy²).
    Returns (cx, cy, r, rms) or None if system is singular.
    """
    n = len(pts)
    if n < 5:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    # Build matrix A and vector b
    A_mat = np.column_stack([xs, ys, np.ones(n)])
    b_vec = xs**2 + ys**2
    try:
        result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        coeffs = result[0]
    except Exception:
        return None
    A_c, B_c, C_c = coeffs
    cx = A_c / 2.0
    cy = B_c / 2.0
    r2 = C_c + cx**2 + cy**2
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    # RMS residual
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    rms = float(np.sqrt(np.mean((dists - r)**2)))
    return cx, cy, r, rms


def _classify_shapes(chains):
    """
    For each (pts, is_closed) chain from the merged DXF, attempt Kasa fit.
    Returns list of shape dicts:
        type     : 'circle' | 'rect' | 'path'
        cx, cy   : centre (circles and rects)
        r        : radius (circles only)
        pts      : vertex list (rects / paths)
        bbox_w, bbox_h : bounding box
        is_closed: bool
    """
    shapes = []
    for pts, is_closed in chains:
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)

        base = {
            "pts"     : pts,
            "is_closed": is_closed,
            "bbox_w"  : bw,
            "bbox_h"  : bh,
            "cx"      : (min(xs) + max(xs)) / 2.0,
            "cy"      : (min(ys) + max(ys)) / 2.0,
        }

        # Try circle fit on closed polylines with enough points
        if is_closed and len(pts) >= 8:
            fit = _kasa_fit(pts)
            if fit is not None:
                cx_f, cy_f, r_f, rms = fit
                # Accept as circle if RMS is small and aspect ratio is near 1
                aspect = max(bw, bh) / (min(bw, bh) + 1e-9)
                if rms < CIRCLE_RMS_THRESH and aspect < 1.5 and r_f > 0:
                    shapes.append({**base, "type": "circle",
                                   "cx": cx_f, "cy": cy_f, "r": r_f, "rms": rms})
                    continue

        # Classify as rectangle if closed and roughly 4 vertices (after approx)
        if is_closed:
            shapes.append({**base, "type": "rect"})
        else:
            shapes.append({**base, "type": "path"})

    return shapes


# ── 9c: read chains from merged DXF ─────────────────────────────────────────

def _read_chains_from_dxf(dxf_path):
    """Return list of (pts, is_closed) from a DXF file."""
    if not HAS_DXF:
        return []
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:
        return []
    chains = []
    for ent in doc.modelspace():
        etype = ent.dxftype()
        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in ent.get_points()]
            if len(pts) >= 2:
                chains.append((pts, ent.closed))
        elif etype == "LINE":
            s, e = ent.dxf.start, ent.dxf.end
            chains.append([(s.x, s.y), (e.x, e.y)], False)
        elif etype == "CIRCLE":
            c = ent.dxf.center
            r = ent.dxf.radius
            # Re-approximate circle for classification
            n = max(32, int(2 * math.pi * r))
            pts = [(c.x + r * math.cos(2*math.pi*i/n),
                    c.y + r * math.sin(2*math.pi*i/n)) for i in range(n)]
            chains.append((pts, True))
    return chains


# ── 9d: deduplication ───────────────────────────────────────────────────────

def _dedup_shapes(shapes):
    """
    Group shapes by centre proximity and size similarity.
    Within each group keep the one with the largest bounding box (outermost).
    """
    used   = [False] * len(shapes)
    kept   = []

    def size_of(s):
        return s.get("r", 0) * 2 if s["type"] == "circle" else max(s["bbox_w"], s["bbox_h"])

    for i, s in enumerate(shapes):
        if used[i]:
            continue
        group = [i]
        sz_i  = size_of(s)
        for j in range(i + 1, len(shapes)):
            if used[j]:
                continue
            t = shapes[j]
            dc = math.hypot(s["cx"] - t["cx"], s["cy"] - t["cy"])
            sz_j = size_of(t)
            sz_max = max(sz_i, sz_j, 1e-9)
            if dc < DEDUP_CENTER_TOL and abs(sz_i - sz_j) / sz_max < DEDUP_SIZE_REL:
                group.append(j)
        # Keep the shape with the largest size in the group
        best = max(group, key=lambda idx: size_of(shapes[idx]))
        for idx in group:
            used[idx] = True
        kept.append(shapes[best])

    return kept


# ── 9e: numeric size filter ──────────────────────────────────────────────────

def _size_filter(shapes, min_size):
    """Discard shapes whose max bounding dimension is below min_size."""
    if min_size <= 0:
        return shapes
    def dim(s):
        return s.get("r", 0) * 2 if s["type"] == "circle" else max(s["bbox_w"], s["bbox_h"])
    return [s for s in shapes if dim(s) >= min_size]


# ── 9f: Gemini AI text filter ────────────────────────────────────────────────

def _gemini_text_filter(shapes, filter_description, gemini_api_key):
    """
    Ask Gemini to identify which shapes to discard based on a plain-English
    description from the user. Returns filtered list.

    Sends a JSON array of shape summaries; expects a JSON array of 0-based
    indices to KEEP.
    """
    if not filter_description or not gemini_api_key:
        return shapes

    import urllib.request, urllib.error

    summaries = []
    for i, s in enumerate(shapes):
        if s["type"] == "circle":
            summaries.append({"idx": i, "type": "circle",
                               "cx": round(s["cx"],1), "cy": round(s["cy"],1),
                               "diameter": round(s.get("r",0)*2, 1)})
        else:
            summaries.append({"idx": i, "type": s["type"],
                               "cx": round(s["cx"],1), "cy": round(s["cy"],1),
                               "width": round(s["bbox_w"],1), "height": round(s["bbox_h"],1)})

    prompt = (
        "You are a CAD drawing assistant. Here is a JSON list of shapes in a DXF drawing:\n"
        f"{json.dumps(summaries, indent=2)}\n\n"
        f"The user wants to REMOVE shapes matching this description: \"{filter_description}\"\n\n"
        "Return ONLY a JSON array of the integer indices (the 'idx' field) of shapes that should be KEPT "
        "(i.e. NOT matching the description). No markdown, no explanation, just the JSON array."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512}
    }).encode("utf-8")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-1.5-flash:generateContent?key={gemini_api_key}")

    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip markdown fences if present
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        keep_indices = set(json.loads(text))
        return [s for s in shapes if s in [shapes[i] for i in keep_indices if i < len(shapes)]]
    except Exception as e:
        sys.stderr.write(f"Gemini filter error (non-fatal): {e}\n")
        return shapes


# ── 9g: centre alignment ─────────────────────────────────────────────────────

def _centre_align(shapes, img_w):
    """
    Compute the outer bounding box of all shapes, find its X centre,
    then shift every shape so it is centred on img_w / 2.
    """
    if not shapes:
        return shapes

    all_cx = [s["cx"] for s in shapes]
    all_cy = [s["cy"] for s in shapes]

    # Find outermost shape (largest bounding box = outer boundary)
    def bbox_area(s):
        return s["bbox_w"] * s["bbox_h"]
    outer = max(shapes, key=bbox_area)

    # X midpoint of the outer boundary
    outer_cx = outer["cx"]
    target_cx = img_w / 2.0
    dx = target_cx - outer_cx

    shifted = []
    for s in shapes:
        ns = dict(s)
        ns["cx"] = s["cx"] + dx
        ns["cy"] = s["cy"]
        if "pts" in s:
            ns["pts"] = [(p[0] + dx, p[1]) for p in s["pts"]]
        shifted.append(ns)
    return shifted


# ── 9h: write refined DXF ────────────────────────────────────────────────────

def write_refined_dxf(shapes, img_w, img_h, out_path):
    """
    Write a DXF R2018 with:
      - native CIRCLE entities for circle shapes
      - closed LWPOLYLINE for rect / closed-path shapes
      - open  LWPOLYLINE for open-path shapes
    Returns (doc, n_circles, n_rects, file_size).
    """
    if not HAS_DXF:
        return None, 0, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 0
    doc.header["$EXTMIN"]   = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"]   = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"]   = (0.0, 0.0)
    doc.header["$LIMMAX"]   = (float(img_w), float(img_h))

    msp = doc.modelspace()

    # Layers
    for lname, col in [("CIRCLES", 5), ("RECTS", 3), ("PATHS", 1)]:
        doc.layers.new(lname, dxfattribs={"color": col, "linetype": "CONTINUOUS"})

    n_circles = n_rects = 0

    for s in shapes:
        if s["type"] == "circle":
            msp.add_circle(
                center=(s["cx"], s["cy"]),
                radius=s["r"],
                dxfattribs={"layer": "CIRCLES", "color": 256},
            )
            n_circles += 1

        elif s["type"] in ("rect", "path"):
            pts = s.get("pts", [])
            if len(pts) < 2:
                continue
            # Deduplicate consecutive near-identical points
            deduped = [pts[0]]
            for p in pts[1:]:
                if math.hypot(p[0]-deduped[-1][0], p[1]-deduped[-1][1]) > 0.1:
                    deduped.append(p)
            if len(deduped) < 2:
                continue
            layer  = "RECTS" if s["type"] == "rect" else "PATHS"
            closed = s.get("is_closed", s["type"] == "rect")
            poly   = msp.add_lwpolyline(
                deduped,
                format="xy",
                dxfattribs={"layer": layer, "color": 256},
            )
            poly.close(closed)
            n_rects += 1

    doc.saveas(str(out_path))
    return doc, n_circles, n_rects, out_path.stat().st_size


# ── 9i: render final PNG ─────────────────────────────────────────────────────

def render_refined_png(shapes, img_w, img_h, out_path):
    """
    Draw white shapes on a dark background using OpenCV.
    Circles → cv2.circle; rects/paths → cv2.polylines.
    Returns True on success.
    """
    if not HAS_CV:
        return False
    try:
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        canvas[:] = (12, 10, 15)   # very dark purple-black background

        for s in shapes:
            if s["type"] == "circle":
                cx = int(round(s["cx"]))
                # DXF Y is up; convert back to pixel Y
                cy = int(round((img_h - 1) - s["cy"]))
                r  = int(round(s["r"]))
                if r > 0:
                    cv2.circle(canvas, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                pts = s.get("pts", [])
                if len(pts) < 2:
                    continue
                # Convert DXF coords back to pixel coords
                px_pts = np.array(
                    [[[int(round(p[0])), int(round((img_h - 1) - p[1]))]]
                     for p in pts],
                    dtype=np.int32
                )
                closed = s.get("is_closed", s["type"] == "rect")
                cv2.polylines(canvas, [px_pts], isClosed=closed,
                              color=(255, 255, 255), thickness=1,
                              lineType=cv2.LINE_AA)

        result = cv2.imwrite(str(out_path), canvas)
        return result and out_path.exists()
    except Exception as e:
        sys.stderr.write(f"PNG render error: {e}\n")
        return False


# ── Master Step-9 orchestrator ───────────────────────────────────────────────

def refine_dxf_pipeline(merged_dxf_path, refined_dxf_path, refined_png_path,
                         img_w, img_h, min_shape_size, filter_description,
                         gemini_api_key):
    """
    Run the full shape-refinement pipeline on the merged DXF.
    Returns dict with counts and file sizes.
    """
    # 9a-c: read + classify
    chains = _read_chains_from_dxf(merged_dxf_path)
    shapes = _classify_shapes(chains)

    # 9d: deduplicate
    shapes = _dedup_shapes(shapes)

    # 9e: numeric size filter
    shapes = _size_filter(shapes, min_shape_size)

    # 9f: Gemini text filter (optional)
    if filter_description and gemini_api_key:
        shapes = _gemini_text_filter(shapes, filter_description, gemini_api_key)

    # 9g: centre alignment
    shapes = _centre_align(shapes, img_w)

    # 9h: write refined DXF
    _, n_circles, n_rects, dxf_size = write_refined_dxf(
        shapes, img_w, img_h, refined_dxf_path)

    # 9i: render PNG
    png_ok = render_refined_png(shapes, img_w, img_h, refined_png_path)

    n_circles_found = sum(1 for s in shapes if s["type"] == "circle")
    n_rects_found   = sum(1 for s in shapes if s["type"] in ("rect", "path"))

    return {
        "total"    : len(shapes),
        "circles"  : n_circles_found,
        "rects"    : n_rects_found,
        "dxf_size" : dxf_size,
        "png_ok"   : png_ok,
    }


# ════════════════════════════════════════════════════════════════════════════════
# STEP 10 — G-CODE GENERATION
# ════════════════════════════════════════════════════════════════════════════════
#
# Reads the merged DXF (closed LWPOLYLINE contours) and generates standard
# RS-274 / ISO G-Code for the following CNC machine families:
#
#   laser     — CO₂ / fiber laser cutter
#   plasma    — plasma arc cutter
#   waterjet  — abrasive waterjet
#   oxyfuel   — oxy-fuel flame cutter
#   mill      — CNC milling machine (contour milling)
#   router    — CNC router (wood/plastic/thin aluminium)
#
# Each machine gets its own .nc file.  The G-Code is parameterised by
# feedRate, plungeRate, spindleRpm (mill/router only), cutDepth, and safeZ.
#
# Coordinate mapping:
#   DXF units are pixel-scale drawing units (1 px = 1 du).
#   If the image DPI is known, scale = 25.4 / dpi converts du → mm.
#   Default: dpi=96  →  scale = 0.2646 mm/du
# ════════════════════════════════════════════════════════════════════════════════

MACHINE_PROFILES = {
    "laser": {
        "label"       : "Laser Cutter",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
            "M5  (Laser off)\n"
            "G28 (Home)\n"
        ),
        "cut_on"      : "M3 S{power} (Laser ON)",
        "cut_off"     : "M5 (Laser OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "postamble"   : "M5\nG28\nM2\n",
        "power"       : 1000,
        "feed"        : 3000,
        "pierce_dwell": 0,      # ms — laser pierces instantly
        "safe_z"      : None,   # 2-axis machine; no Z moves
    },
    "plasma": {
        "label"       : "Plasma Cutter",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
            "G28 (Home)\n"
        ),
        "cut_on"      : "M3 (Arc ON)",
        "cut_off"     : "M5 (Arc OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f} Z{safe_z:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "cut_z"       : "G1 Z{cut_z:.4f} F{plunge}",
        "safe_z_move" : "G0 Z{safe_z:.4f}",
        "postamble"   : "M5\nG28\nM2\n",
        "feed"        : 2500,
        "plunge"      : 300,
        "safe_z"      : 5.0,
        "cut_z"       : 0.0,
        "pierce_dwell": 500,    # ms — plasma needs pierce dwell
    },
    "waterjet": {
        "label"       : "Waterjet",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
        ),
        "cut_on"      : "M7 (Waterjet ON)",
        "cut_off"     : "M9 (Waterjet OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "postamble"   : "M9\nG28\nM2\n",
        "feed"        : 800,
        "pierce_dwell": 1000,   # ms — waterjet piercing
        "safe_z"      : None,
    },
    "oxyfuel": {
        "label"       : "Oxy-Fuel Cutter",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
        ),
        "cut_on"      : "M3 (Flame ON / preheat complete)",
        "cut_off"     : "M5 (Flame OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "postamble"   : "M5\nG28\nM2\n",
        "feed"        : 400,
        "pierce_dwell": 3000,   # ms — oxy preheat dwell
        "safe_z"      : None,
    },
    "mill": {
        "label"       : "CNC Mill",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
            "G17 (XY plane)\n"
            "T1 M6 (Tool change)\n"
            "G43 H1 (Tool length compensation)\n"
        ),
        "cut_on"      : "M3 S{rpm} (Spindle CW)",
        "cut_off"     : "M5 (Spindle OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f} Z{safe_z:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "cut_z"       : "G1 Z{cut_z:.4f} F{plunge}",
        "safe_z_move" : "G0 Z{safe_z:.4f}",
        "postamble"   : "M5\nG28\nM2\n",
        "feed"        : 1000,
        "plunge"      : 200,
        "rpm"         : 12000,
        "safe_z"      : 5.0,
        "cut_z"       : -2.0,
        "pierce_dwell": 0,
    },
    "router": {
        "label"       : "CNC Router",
        "ext"         : "nc",
        "preamble"    : (
            "G21 (Metric)\n"
            "G90 (Absolute)\n"
            "G94 (Feed per minute)\n"
            "G17 (XY plane)\n"
            "T1 M6 (Tool change)\n"
        ),
        "cut_on"      : "M3 S{rpm} (Spindle CW)",
        "cut_off"     : "M5 (Spindle OFF)",
        "rapid"       : "G0 X{x:.4f} Y{y:.4f} Z{safe_z:.4f}",
        "feed_move"   : "G1 X{x:.4f} Y{y:.4f} F{feed}",
        "cut_z"       : "G1 Z{cut_z:.4f} F{plunge}",
        "safe_z_move" : "G0 Z{safe_z:.4f}",
        "postamble"   : "M5\nG28\nM2\n",
        "feed"        : 2000,
        "plunge"      : 500,
        "rpm"         : 18000,
        "safe_z"      : 5.0,
        "cut_z"       : -3.0,
        "pierce_dwell": 0,
    },
}


def _read_merged_contours(merged_dxf_path):
    """
    Read the merged DXF and return a list of contours.
    Each contour: {"pts": [(x,y), ...], "closed": bool}
    """
    if not HAS_DXF:
        return []
    try:
        doc = ezdxf.readfile(str(merged_dxf_path))
    except Exception:
        return []

    contours = []
    for entity in doc.modelspace():
        etype = entity.dxftype()
        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
            is_closed = entity.closed
            if len(pts) >= 2:
                contours.append({"pts": pts, "closed": is_closed})
        elif etype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            contours.append({"pts": [(s.x, s.y), (e.x, e.y)], "closed": False})
    return contours


def _format_rapid(profile, x, y):
    safe_z = profile.get("safe_z")
    if safe_z is not None:
        return profile["rapid"].format(x=x, y=y, safe_z=safe_z)
    return profile["rapid"].format(x=x, y=y)


def generate_gcode_for_machine(machine_key, contours, opts, scale, ts_str, out_dir):
    """
    Generate a .nc G-Code file for machine_key from the list of contours.
    Returns the output file path (str) or '' on failure.
    """
    profile = MACHINE_PROFILES.get(machine_key)
    if not profile:
        return ''

    # Merge opts overrides into profile defaults
    feed     = opts.get("feedRate",    profile.get("feed", 1000))
    plunge   = opts.get("plungeRate",  profile.get("plunge", 300))
    rpm      = opts.get("spindleRpm",  profile.get("rpm", 12000))
    safe_z   = profile.get("safe_z",  5.0)
    cut_z    = profile.get("cut_z",   -2.0)
    power    = profile.get("power",   1000)
    pierce_ms= profile.get("pierce_dwell", 0)

    lines = []
    lines.append(f"; SheetForge v8 — {profile['label']} G-Code")
    lines.append(f"; Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"; Scale: {scale:.4f} mm/du | Contours: {len(contours)}")
    lines.append(f"; Feed: {feed} mm/min | Safe Z: {safe_z} mm")
    lines.append("")
    lines.append(profile["preamble"])

    # Spindle / laser / jet on (use first contour as reference)
    cut_on_str = profile["cut_on"]
    if "{power}" in cut_on_str:
        cut_on_str = cut_on_str.format(power=power)
    elif "{rpm}" in cut_on_str:
        cut_on_str = cut_on_str.format(rpm=rpm)

    lines.append(cut_on_str)
    lines.append("")

    for ci, contour in enumerate(contours):
        pts    = contour["pts"]
        closed = contour["closed"]

        if len(pts) < 2:
            continue

        # Scale from drawing units to mm
        scaled = [(round(p[0] * scale, 4), round(p[1] * scale, 4)) for p in pts]

        lines.append(f"; --- Contour {ci + 1} ({'closed' if closed else 'open'}) ---")

        # Rapid to start position
        lines.append(_format_rapid(profile, scaled[0][0], scaled[0][1]))

        # Plunge / cut engagement
        if "cut_z" in profile and profile.get("safe_z") is not None:
            lines.append(profile["cut_z"].format(cut_z=cut_z, plunge=plunge))

        # Pierce dwell
        if pierce_ms > 0:
            lines.append(f"G4 P{pierce_ms / 1000:.3f} (Pierce dwell)")

        # Cut along contour
        for pt in scaled[1:]:
            lines.append(profile["feed_move"].format(x=pt[0], y=pt[1], feed=feed))

        # Close contour
        if closed:
            lines.append(profile["feed_move"].format(
                x=scaled[0][0], y=scaled[0][1], feed=feed))

        # Retract
        if "safe_z_move" in profile and profile.get("safe_z") is not None:
            lines.append(profile["safe_z_move"].format(safe_z=safe_z))

        lines.append("")

    lines.append(profile["cut_off"])
    lines.append(profile["postamble"])

    gcode_str = "\n".join(lines)

    fname    = f"design_{ts_str}_{machine_key}.nc"
    out_path = out_dir / fname
    try:
        out_path.write_text(gcode_str, encoding="utf-8")
        return str(out_path), fname, gcode_str
    except Exception as e:
        sys.stderr.write(f"G-Code write error ({machine_key}): {e}\n")
        return '', '', ''


def generate_all_gcode(merged_dxf_path, opts, dpi, ts_str, out_dir):
    """
    Run G-Code generation for all machine types.
    Returns dict of machine_key → {path, filename, size}.
    """
    scale    = 25.4 / dpi          # drawing units → mm
    contours = _read_merged_contours(merged_dxf_path)

    if not contours:
        return {}

    results = {}
    for machine_key in MACHINE_PROFILES:
        try:
            path_str, fname, _ = generate_gcode_for_machine(
                machine_key, contours, opts, scale, ts_str, out_dir
            )
            if path_str:
                results[machine_key] = {
                    "path"    : path_str,
                    "filename": fname,
                    "size"    : Path(path_str).stat().st_size,
                }
        except Exception as e:
            sys.stderr.write(f"G-Code error ({machine_key}): {e}\n")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# STEP 10 — PDF EXPORT
# ════════════════════════════════════════════════════════════════════════════════

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
        c.drawString(margin, page_h - 24, "SheetForge — Edge Detection Preview")
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
                            "SheetForge v8.0  •  Merged DXF contours + Multi-machine G-Code")
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


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    opts       = {}
    if len(sys.argv) > 2:
        try: opts = json.loads(sys.argv[2])
        except Exception: pass

    blur_ksize         = int(opts.get("blurKsize",      5))
    canny_low          = int(opts.get("cannyLow",      30))
    canny_high         = int(opts.get("cannyHigh",    100))
    epsilon_factor     = float(opts.get("epsilonFactor", 0.5))
    min_shape_size     = float(opts.get("minShapeSize",   0.0))   # drawing units; 0 = off
    filter_description = str(opts.get("filterDescription", "")).strip()
    gemini_api_key     = os.environ.get("GEMINI_API_KEY", "")
    gcode_opts         = opts.get("gcodeOptions", {})

    steps = []

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record("CV-1: Load Image", f"{img_w}×{img_h}px  DPI={dpi:.0f}", t0))

    # ── STEP 2: Median Blur ───────────────────────────────────────────────────
    t0 = now_ms()
    blurred = median_blur(gray, ksize=blur_ksize)
    steps.append(step_record(f"CV-2: Median Blur (ksize={blur_ksize})", "Noise reduced", t0))

    # ── STEP 3: Adaptive Threshold ───────────────────────────────────────────
    t0 = now_ms()
    binary   = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record("CV-3: Adaptive Threshold", f"{white_px} white px", t0))

    # ── STEP 4: Morph Open ───────────────────────────────────────────────────
    t0 = now_ms()
    cleaned     = morph_clean(binary)
    cleaned_px  = int(np.count_nonzero(cleaned))
    steps.append(step_record("CV-4: MORPH_OPEN", f"{white_px - cleaned_px} speckles removed", t0))

    # ── STEP 5: Canny ────────────────────────────────────────────────────────
    t0 = now_ms()
    edges   = canny_edges(cleaned, canny_low, canny_high)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(f"CV-5: Canny (lo={canny_low}, hi={canny_high})", f"{edge_px} edge px", t0))

    # ── STEP 6: Contour extraction ───────────────────────────────────────────
    t0 = now_ms()
    simplified_contours = extract_simplified_contours(edges, epsilon_factor)
    total_pts = sum(len(c) for c in simplified_contours)
    steps.append(step_record(
        f"CV-6: findContours + approxPolyDP (ε={epsilon_factor})",
        f"{len(simplified_contours)} contours  |  {total_pts} vertices", t0))

    # ── Output dir ────────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str    = int(time.time())
    raw_dxf_name      = f"design_{ts_str}_raw.dxf"
    merged_dxf_name   = f"design_{ts_str}_merged.dxf"
    refined_dxf_name  = f"design_{ts_str}.dxf"
    pdf_name          = f"design_{ts_str}.pdf"
    canny_png_name    = f"canny_{ts_str}.png"
    refined_png_name  = f"preview_{ts_str}.png"

    raw_dxf_path      = server_out_dir / raw_dxf_name
    merged_dxf_path   = server_out_dir / merged_dxf_name
    refined_dxf_path  = server_out_dir / refined_dxf_name
    pdf_path          = server_out_dir / pdf_name
    canny_png_path    = server_out_dir / canny_png_name
    refined_png_path  = server_out_dir / refined_png_name

    # ── STEP 7: Raw DXF export ────────────────────────────────────────────────
    t0 = now_ms()
    _, entity_count, raw_size = build_and_save_dxf(
        simplified_contours, img_w, img_h, raw_dxf_path)
    steps.append(step_record(
        "DXF-7: Raw export (LWPOLYLINE/LINE per contour)",
        f"{entity_count} entities  |  {raw_size // 1024 if raw_size else 0} KB", t0))

    # ── STEP 8: DXF Merge + Heal ──────────────────────────────────────────────
    t0 = now_ms()
    _, n_contours, n_closed, merged_size = merge_and_heal_dxf(
        raw_dxf_path, merged_dxf_path, img_w, img_h)

    # Read merged DXF content for frontend preview
    merged_dxf_str = ""
    if merged_size > 0:
        try:
            with open(merged_dxf_path, encoding="utf-8") as f:
                merged_dxf_str = f.read()
        except Exception:
            pass

    steps.append(step_record(
        "DXF-8: Merge & Heal (chain-stitch → closed LWPOLYLINE)",
        (f"{n_contours} merged contours  |  {n_closed} closed  |"
         f"  {(n_contours - n_closed)} open  |  {merged_size // 1024 if merged_size else 0} KB"),
        t0))

    # ── STEP 9: Shape Refinement Pipeline ────────────────────────────────────
    t0 = now_ms()
    refine_result = {"total": 0, "circles": 0, "rects": 0,
                     "dxf_size": 0, "png_ok": False}
    if merged_size > 0:
        try:
            refine_result = refine_dxf_pipeline(
                merged_dxf_path, refined_dxf_path, refined_png_path,
                img_w, img_h,
                min_shape_size, filter_description, gemini_api_key,
            )
        except Exception as e:
            sys.stderr.write(f"Step 9 refine error: {e}\n{traceback.format_exc()}\n")

    refined_size   = refine_result.get("dxf_size", 0)
    refined_png_ok = refine_result.get("png_ok", False)

    # Read refined DXF content for frontend (download)
    refined_dxf_str = ""
    if refined_size > 0:
        try:
            with open(refined_dxf_path, encoding="utf-8") as f:
                refined_dxf_str = f.read()
        except Exception:
            pass

    gemini_used = bool(filter_description and gemini_api_key)
    steps.append(step_record(
        "DXF-9: Shape Refinement (Kasa fit → dedup → filter → centre → native entities)",
        (f"{refine_result.get('total',0)} shapes  |  "
         f"{refine_result.get('circles',0)} circles  |  "
         f"{refine_result.get('rects',0)} rects/paths  |  "
         f"{'Gemini filter applied' if gemini_used else 'no AI filter'}  |  "
         f"{refined_size // 1024 if refined_size else 0} KB"),
        t0))

    # ── STEP 10: G-Code generation ────────────────────────────────────────────
    # G-code is generated from the REFINED DXF (native circles + closed polys)
    t0 = now_ms()
    gcode_files = {}
    gcode_src = refined_dxf_path if refined_size > 0 else merged_dxf_path
    if refined_size > 0 or merged_size > 0:
        try:
            gcode_files = generate_all_gcode(
                gcode_src, gcode_opts, dpi, ts_str, server_out_dir)
        except Exception as e:
            sys.stderr.write(f"G-code generation error: {e}\n")

    machine_summary = ", ".join(
        f"{k}({v['size']//1024}KB)" for k, v in gcode_files.items()) or "none"
    steps.append(step_record(
        "NC-10: G-Code generation (laser/plasma/waterjet/oxyfuel/mill/router)",
        f"{len(gcode_files)} files — {machine_summary}", t0))

    # ── STEP 11: PDF export ───────────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-11: Export edge preview", "OK" if pdf_ok else "FAILED", t0))

    # ── STEP 12: Canny PNG (intermediate reference) ───────────────────────────
    t0 = now_ms()
    canny_png_ok   = False
    canny_png_size = 0
    try:
        canvas_ = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        canvas_[:] = (15, 12, 10)
        canvas_[edges > 0] = (255, 255, 255)
        if cv2.imwrite(str(canny_png_path), canvas_) and canny_png_path.exists():
            canny_png_ok   = True
            canny_png_size = canny_png_path.stat().st_size
    except Exception as e:
        sys.stderr.write(f"Canny PNG error: {e}\n")
    steps.append(step_record(
        "PNG-12: Canny edge reference",
        f"{canny_png_size // 1024 if canny_png_size else 0} KB" if canny_png_ok else "FAILED",
        t0))

    # ── Analysis summary ──────────────────────────────────────────────────────
    analysis = {
        "width"           : float(img_w),
        "height"          : float(img_h),
        "dpi"             : dpi,
        "edgePixels"      : edge_px,
        "edges"           : entity_count,
        "contours"        : len(simplified_contours),
        "mergedContours"  : n_contours,
        "closedContours"  : n_closed,
        "refinedShapes"   : refine_result.get("total", 0),
        "refinedCircles"  : refine_result.get("circles", 0),
        "refinedRects"    : refine_result.get("rects", 0),
        "totalVertices"   : total_pts,
        "blurKsize"       : blur_ksize,
        "cannyLow"        : canny_low,
        "cannyHigh"       : canny_high,
        "epsilonFactor"   : epsilon_factor,
        "minShapeSize"    : min_shape_size,
        "filterDescription": filter_description,
        "imgW"            : img_w,
        "imgH"            : img_h,
        "scaleMmPerDu"    : round(25.4 / dpi, 4),
        "coordSystem"     : "origin=bottom-left, 1px=1du, Y-up (CAD convention)",
    }

    # Build gcodeFiles map (machine → filename) for schema
    gcode_files_map = {k: v["filename"] for k, v in gcode_files.items()}

    # The refined DXF is the download target; refined PNG is the View Drawing target
    out_dxf_name = refined_dxf_name if refined_size > 0 else merged_dxf_name
    out_dxf_path = str(refined_dxf_path) if refined_size > 0 else str(merged_dxf_path)
    out_dxf_size = refined_size if refined_size > 0 else merged_size
    out_dxf_str  = refined_dxf_str if refined_dxf_str else ""

    print(json.dumps({
        "steps"        : steps,
        "analysis"     : analysis,
        "dwg": {
            "entities"         : entity_count,
            "fileSize"         : out_dxf_size,
            "filename"         : out_dxf_name if out_dxf_size else "",
            "dxfAbsPath"       : out_dxf_path if out_dxf_size else "",
            "rawDxfFilename"   : raw_dxf_name if raw_size else "",
            "pdfFilename"      : pdf_name if pdf_ok else "",
            # refined PNG = the "View Drawing" image
            "edgePngFilename"  : refined_png_name if refined_png_ok else (canny_png_name if canny_png_ok else ""),
            "edgePngPath"      : str(refined_png_path) if refined_png_ok else (str(canny_png_path) if canny_png_ok else ""),
            "cannyPngFilename" : canny_png_name if canny_png_ok else "",
            "cannyPngPath"     : str(canny_png_path) if canny_png_ok else "",
            "gcodeFiles"       : gcode_files_map,
            "gcodeFilePaths"   : {k: v["path"] for k, v in gcode_files.items()},
            "refinedShapes"    : refine_result.get("total", 0),
            "refinedCircles"   : refine_result.get("circles", 0),
            "refinedRects"     : refine_result.get("rects", 0),
        },
        "dxfContent"   : out_dxf_str[:50000] if out_dxf_str else "",
        "dxfAvailable" : out_dxf_size > 0,
        "pdfAvailable" : pdf_ok,
        "pngAvailable" : refined_png_ok or canny_png_ok,
        "refinedPngAvailable" : refined_png_ok,
        "gcodeAvailable": len(gcode_files) > 0,
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
        }))
        sys.exit(1)
