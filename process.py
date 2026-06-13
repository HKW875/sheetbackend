#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v9.0
================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

Pipeline (precision contour → CAD → CNC):
  1. Load image              (cv2.imread)
  2. Median Blur             (cv2.medianBlur — salt-and-pepper noise reduction)
  3. Adaptive Threshold      (cv2.adaptiveThreshold — binarisation, lines=WHITE)
  4. Morph Clean             (cv2.morphologyEx MORPH_OPEN — speckle removal)
  5. Canny Edge Detection    (cv2.Canny — thin precise edges)
  6. Skeleton Centerline     (NEW — rasterise DXF LWPOLYLINEs → 1-px skeleton;
                              eliminates double-traced contour bands so that
                              findContours returns a single centerline per edge)
     6a. Close 1-px gaps     (cv2.dilate 3×3 → bridge sub-pixel breaks)
     6b. Zhang-Suen thin     (cv2.ximgproc.thinning — single-pixel skeleton)
     6c. findContours        (on skeleton → RETR_LIST / CHAIN_APPROX_TC89_L1)
     6d. Douglas-Peucker     (cv2.approxPolyDP ε=2 px — simplify vertices)
     6e. Noise / open filter (drop fragments < MIN_SKEL_PTS; discard open chains)
  7. DXF Export — RAW        (ezdxf — one LWPOLYLINE per closed centerline contour)
  8. DXF Merge + Heal        (chain-stitch fragmented segments → closed LWPOLYLINE)
  9. G-Code Generation       (one .nc file per CNC machine type, from merged DXF)
 10. PDF Export              (reportlab — edge image rendered to PDF page)
 11. PNG Preview             (dark-bg canvas with white skeleton — saved as .png)
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
# STEP 6 — SKELETON CENTERLINE  (replaces old approxPolyDP-only step)
#
# Problem: Canny + findContours on raw Canny output returns two parallel
# contours (inner + outer edge of each 2-3 px Canny band). The result is a
# doubled DXF where every physical edge is drawn twice.
#
# Fix (per-pixel approach, no DXF round-trip needed):
#   6a. Close 1-px gaps  — dilate Canny output with a 3×3 kernel so hairline
#       breaks in the edge image are bridged before thinning.
#   6b. Zhang-Suen thin  — cv2.ximgproc.thinning() reduces every band to a
#       single-pixel-wide skeleton (medial axis). Falls back to iterative
#       morphological thinning when ximgproc is unavailable.
#   6c. findContours     — run on the 1-px skeleton; RETR_LIST returns one
#       chain per connected component with no inner/outer duplication.
#   6d. Douglas-Peucker  — approxPolyDP with ε=2 px compresses vertices while
#       preserving corner geometry.
#   6e. Noise/open filter — fragments shorter than MIN_SKEL_PTS are noise;
#       open chains (start ≠ end) are dropped so only closed loops survive
#       into the DXF.  The closed flag uses CLOSE_GAP_PX tolerance so a
#       nearly-closed chain is snapped shut automatically.
# ════════════════════════════════════════════════════════════════════════════════

# Tuning constants
DP_EPSILON_PX  = 2.0   # Douglas-Peucker simplification (drawing units = px)
MIN_SKEL_PTS   = 4     # minimum vertices after DP; shorter chains = noise
CLOSE_GAP_PX   = 4.0   # if chain start-end ≤ this, auto-close the polyline


def _zhang_suen_thin(binary):
    """
    Reduce a binary image to a 1-px skeleton.
    Uses cv2.ximgproc.thinning() (Zhang-Suen, available in opencv-contrib).
    Falls back to repeated cv2.morphologyEx(MORPH_ERODE) thinning if contrib
    is not present (slower but functionally equivalent for our use-case).
    """
    # ximgproc.thinning expects uint8 with 0/255 values
    src = binary.copy()
    src[src > 0] = 255

    try:
        thin = cv2.ximgproc.thinning(src, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin
    except AttributeError:
        pass  # ximgproc not available; use fallback

    # Morphological thinning fallback
    k    = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = src.copy()
    for _ in range(40):           # max 40 erosion passes
        eroded = cv2.erode(prev, k)
        delta  = cv2.subtract(prev, eroded)
        skel   = cv2.bitwise_or(delta, prev)
        prev   = eroded
        if cv2.countNonZero(eroded) == 0:
            break
    return skel


def skeletonize_edges(edges):
    """
    Step 6a-6b: close 1-px gaps then thin Canny output to a 1-px skeleton.
    Returns the skeleton image (uint8, same shape as edges).
    """
    # 6a — close hairline gaps with a single 3×3 dilation pass
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed  = cv2.dilate(edges, k_close, iterations=1)

    # 6b — Zhang-Suen / morphological thinning
    skeleton = _zhang_suen_thin(closed)
    return skeleton


def extract_centerline_contours(skeleton, img_h):
    """
    Steps 6c-6e: findContours on 1-px skeleton → Douglas-Peucker ε=2 px →
    discard noise and open chains → return list of (approx_array, is_closed).

    Only CLOSED contours (or nearly-closed within CLOSE_GAP_PX) are kept;
    open fragments are treated as noise and discarded.
    """
    # 6c — find contours on the skeleton
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1,
    )

    results = []
    for cnt in contours:
        if len(cnt) < 2:
            continue

        # 6d — Douglas-Peucker with fixed ε=2 px
        approx = cv2.approxPolyDP(cnt, DP_EPSILON_PX, closed=True)

        if len(approx) < MIN_SKEL_PTS:
            continue   # 6e — noise fragment: too few vertices

        # 6e — determine closure: compare start and end in image coords
        p0 = approx[0][0]
        p1 = approx[-1][0]
        gap = math.hypot(float(p0[0]) - float(p1[0]),
                         float(p0[1]) - float(p1[1]))

        is_closed = (gap <= CLOSE_GAP_PX)

        # Discard open chains — they are either edge fragments or noise
        if not is_closed:
            continue

        results.append(approx)

    return results


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

        if len(pts) < 3:          # closed polyline needs ≥3 pts
            continue

        # Deduplicate consecutive near-identical points
        deduped = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > 0.1:
                deduped.append(p)
        if len(deduped) < 3:
            continue

        poly = msp.add_lwpolyline(deduped, format="xy",
                                  dxfattribs={"layer": "EDGES", "color": 256})
        poly.close(True)          # all centerline contours are closed loops
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
# STEP 9 — G-CODE GENERATION
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

    blur_ksize      = int(opts.get("blurKsize",    5))
    canny_low       = int(opts.get("cannyLow",    30))
    canny_high      = int(opts.get("cannyHigh",  100))
    gcode_opts      = opts.get("gcodeOptions", {})

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

    # ── STEP 6: Skeleton centerline ──────────────────────────────────────────
    # 6a+6b — close 1-px gaps then Zhang-Suen thin to single-pixel skeleton
    t0 = now_ms()
    skeleton    = skeletonize_edges(edges)
    skel_px     = int(np.count_nonzero(skeleton))
    band_ratio  = round(edge_px / max(skel_px, 1), 2)
    steps.append(step_record(
        "CV-6a-6b: Skeleton (close gaps + Zhang-Suen thin)",
        f"{skel_px} skeleton px  |  band→skeleton ratio {band_ratio}×", t0))

    # 6c-6e — findContours on skeleton → D-P ε=2px → closed loops only
    t0 = now_ms()
    simplified_contours = extract_centerline_contours(skeleton, img_h)
    total_pts = sum(len(c) for c in simplified_contours)
    steps.append(step_record(
        f"CV-6c-6e: findContours on skeleton + D-P (ε={DP_EPSILON_PX}px) + close filter",
        f"{len(simplified_contours)} closed centerline contours  |  {total_pts} vertices", t0))

    # ── Output dir ────────────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str    = int(time.time())
    raw_dxf_name    = f"design_{ts_str}_raw.dxf"
    merged_dxf_name = f"design_{ts_str}.dxf"
    pdf_name        = f"design_{ts_str}.pdf"
    png_name        = f"preview_{ts_str}.png"

    raw_dxf_path    = server_out_dir / raw_dxf_name
    merged_dxf_path = server_out_dir / merged_dxf_name
    pdf_path        = server_out_dir / pdf_name
    png_path        = server_out_dir / png_name

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

    # ── STEP 9: G-Code generation ─────────────────────────────────────────────
    t0 = now_ms()
    gcode_files = {}
    if merged_size > 0:
        gcode_files = generate_all_gcode(
            merged_dxf_path, gcode_opts, dpi, ts_str, server_out_dir)

    machine_summary = ", ".join(
        f"{k}({v['size']//1024}KB)" for k, v in gcode_files.items()) or "none"
    steps.append(step_record(
        "NC-9: G-Code generation (laser/plasma/waterjet/oxyfuel/mill/router)",
        f"{len(gcode_files)} files — {machine_summary}", t0))

    # ── STEP 10: PDF export ───────────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record("PDF-10: Export edge preview", "OK" if pdf_ok else "FAILED", t0))

    # ── STEP 11: PNG preview ──────────────────────────────────────────────────
    t0 = now_ms()
    png_ok   = False
    png_size = 0
    try:
        canvas_ = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        canvas_[:] = (15, 12, 10)
        # Show skeleton (single-pixel centerlines) in bright white,
        # and original Canny bands underneath in dim grey for reference
        canvas_[edges > 0]    = (60, 60, 60)     # Canny band — dim grey
        canvas_[skeleton > 0] = (255, 255, 255)  # skeleton centerline — white
        if cv2.imwrite(str(png_path), canvas_) and png_path.exists():
            png_ok   = True
            png_size = png_path.stat().st_size
    except Exception as e:
        sys.stderr.write(f"PNG error: {e}\n")
    steps.append(step_record("PNG-11: Save skeleton preview", f"{png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED", t0))

    # ── Analysis summary ──────────────────────────────────────────────────────
    analysis = {
        "width"          : float(img_w),
        "height"         : float(img_h),
        "dpi"            : dpi,
        "edgePixels"     : edge_px,
        "skeletonPixels" : skel_px,
        "bandToSkelRatio": band_ratio,
        "edges"          : entity_count,
        "contours"       : len(simplified_contours),
        "mergedContours" : n_contours,
        "closedContours" : n_closed,
        "totalVertices"  : total_pts,
        "blurKsize"      : blur_ksize,
        "cannyLow"       : canny_low,
        "cannyHigh"      : canny_high,
        "dpEpsilonPx"    : DP_EPSILON_PX,
        "imgW"           : img_w,
        "imgH"           : img_h,
        "scaleMmPerDu"   : round(25.4 / dpi, 4),
        "coordSystem"    : "origin=bottom-left, 1px=1du, Y-up (CAD convention)",
    }

    # Build gcodeFiles map (machine → filename) for schema
    gcode_files_map = {k: v["filename"] for k, v in gcode_files.items()}

    print(json.dumps({
        "steps"        : steps,
        "analysis"     : analysis,
        "dwg": {
            "entities"        : entity_count,
            "fileSize"        : merged_size,
            "filename"        : merged_dxf_name if merged_size else "",
            "dxfAbsPath"      : str(merged_dxf_path) if merged_size else "",
            "rawDxfFilename"  : raw_dxf_name if raw_size else "",
            "pdfFilename"     : pdf_name if pdf_ok else "",
            "edgePngFilename" : png_name if png_ok else "",
            "edgePngPath"     : str(png_path) if png_ok else "",
            "gcodeFiles"      : gcode_files_map,
            "gcodeFilePaths"  : {k: v["path"] for k, v in gcode_files.items()},
        },
        "dxfContent"   : merged_dxf_str[:50000] if merged_dxf_str else "",
        "dxfAvailable" : merged_size > 0,
        "pdfAvailable" : pdf_ok,
        "pngAvailable" : png_ok,
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
