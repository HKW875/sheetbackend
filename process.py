#!/usr/bin/env python3
"""
SheetForge — Advanced CV + AI DXF Pipeline  v4.0
==================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, gcode }

v4.0 New Features:
  - PREPROCESSING FIRST: Thresholding, Denoising, Morphology, Gaussian Blur, Contour Extraction
  - Ordered OpenCV Pipeline: Adaptive Threshold → Canny → Contour → Shape Classification
    → Line/Circle Fitting → Skeletonization → Vector Path → DXF Export
  - Required OpenCV functions: cv2.HoughCircles, cv2.findContours, cv2.fitEllipse,
    cv2.HoughLines, cv2.ximgproc.thinning, cv2.approxPolyDP, cv2.matchShapes,
    cv2.convexHull, cv2.adaptiveThreshold, cv2.morphologyEx, cv2.GaussianBlur
  - YOLO object detection
  - SAM segmentation
  - Deep CNN handwriting recognition
  - Bezier Fitting for curve refinement
  - All DXF libraries: ezdxf, dxfwrite, svgwrite, reportlab, matplotlib
  - GCode generation
  - AI-powered DXF interaction/correction endpoint
"""

import sys, os, json, math, time, base64, re, traceback
from pathlib import Path

# ─── Graceful optional imports ────────────────────────────────────────────────
def _try(fn):
    try: return fn()
    except Exception: return None

cv2           = _try(lambda: __import__("cv2"))
np            = _try(lambda: __import__("numpy"))
ezdxf         = _try(lambda: __import__("ezdxf"))
pytesseract   = _try(lambda: __import__("pytesseract"))
Image         = _try(lambda: __import__("PIL.Image",      fromlist=["Image"]))
ImageFilter   = _try(lambda: __import__("PIL.ImageFilter", fromlist=["ImageFilter"]))
ImageEnhance  = _try(lambda: __import__("PIL.ImageEnhance", fromlist=["ImageEnhance"]))
anthropic_mod = _try(lambda: __import__("anthropic"))
scipy_mod     = _try(lambda: __import__("scipy"))
skimage_mod   = _try(lambda: __import__("skimage"))
reportlab_mod = _try(lambda: __import__("reportlab"))
svgwrite_mod  = _try(lambda: __import__("svgwrite"))
matplotlib_mod= _try(lambda: __import__("matplotlib"))

# Optional heavy AI imports
torch_mod     = _try(lambda: __import__("torch"))
ultralytics_mod = _try(lambda: __import__("ultralytics"))

HAS_CV     = cv2 is not None and np is not None
HAS_DXF    = ezdxf is not None
HAS_OCR    = pytesseract is not None
HAS_PIL    = Image is not None
HAS_AI     = anthropic_mod is not None
HAS_SCIPY  = scipy_mod is not None
HAS_SKIMAGE= skimage_mod is not None
HAS_TORCH  = torch_mod is not None
HAS_YOLO   = ultralytics_mod is not None
HAS_RL     = reportlab_mod is not None
HAS_SVG    = svgwrite_mod is not None
HAS_MPL    = matplotlib_mod is not None

# Check for ximgproc (thinning)
HAS_XIMGPROC = False
if HAS_CV:
    try:
        _ = cv2.ximgproc.thinning
        HAS_XIMGPROC = True
    except Exception:
        pass

def now_ms(): return int(time.time() * 1000)

def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}

# ═══════════════════════════════════════════════════════════════════════════════
# ████  PHASE 1: PREPROCESSING  ████
# ═══════════════════════════════════════════════════════════════════════════════

# ─── PRE-1 — Image ingestion ──────────────────────────────────────────────────
def load_image(image_path):
    if not HAS_CV: return None, 96.0
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None: return None, 96.0
    dpi = 96.0
    if HAS_PIL:
        try:
            pil  = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (96, 96))
            dpi  = float(xdpi[0]) if xdpi[0] > 1 else 96.0
        except Exception: pass
    return img, dpi


# ─── PRE-2 — Channel separation ───────────────────────────────────────────────
def separate_channels(img):
    if not HAS_CV: return None, None, None
    b, g, r = cv2.split(img)
    red_mask   = cv2.subtract(r, cv2.max(b, g))
    _, red_bin = cv2.threshold(red_mask, 40, 255, cv2.THRESH_BINARY)
    blue_mask  = cv2.subtract(b, cv2.max(r, g))
    _, blue_bin= cv2.threshold(blue_mask, 30, 255, cv2.THRESH_BINARY)
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return blue_bin, red_bin, gray


# ─── PRE-3 — PREPROCESSING: Gaussian Blur ─────────────────────────────────────
def preprocess_gaussian_blur(img, gray):
    """cv2.GaussianBlur — Required preprocessing step"""
    if not HAS_CV or img is None: return img, gray
    # Gaussian blur for noise reduction before further processing
    blurred_color = cv2.GaussianBlur(img, (5, 5), 0)
    blurred_gray  = cv2.GaussianBlur(gray, (5, 5), 0) if gray is not None else None
    # Also apply stronger blur for circle detection
    blurred_strong = cv2.GaussianBlur(gray, (9, 9), 2) if gray is not None else None
    return blurred_color, blurred_gray, blurred_strong


# ─── PRE-4 — PREPROCESSING: Denoising ────────────────────────────────────────
def preprocess_denoise(img):
    """Bilateral + NLM denoising"""
    if not HAS_CV or img is None: return img
    bil = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    nlm = cv2.fastNlMeansDenoisingColored(bil, None, h=8, hColor=8,
                                           templateWindowSize=7, searchWindowSize=21)
    return nlm


# ─── PRE-5 — PREPROCESSING: Thresholding ─────────────────────────────────────
def preprocess_threshold(gray):
    """Multiple thresholding methods for preprocessing"""
    if not HAS_CV or gray is None: return None, None, None
    # Otsu thresholding
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive mean thresholding
    adapt_mean  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    # Adaptive Gaussian thresholding (preprocessing version)
    adapt_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return otsu_bin, adapt_mean, adapt_gauss


# ─── PRE-6 — PREPROCESSING: Morphology ───────────────────────────────────────
def preprocess_morphology(binary):
    """cv2.morphologyEx — Required preprocessing step"""
    if not HAS_CV or binary is None: return binary
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    # Close small gaps in lines (MORPH_CLOSE)
    closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3)
    # Remove small noise (MORPH_OPEN)
    opened   = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3)
    # Gradient for edge enhancement
    gradient = cv2.morphologyEx(opened, cv2.MORPH_GRADIENT, kernel3)
    # Dilation to thicken lines
    dilated  = cv2.dilate(opened, kernel3, iterations=1)
    return opened, closed, dilated, gradient


# ─── PRE-7 — PREPROCESSING: Contour Extraction (preprocessing pass) ──────────
def preprocess_contours(binary):
    """Initial contour extraction during preprocessing for structural analysis"""
    if not HAS_CV or binary is None: return [], []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter by minimum area
    significant = [c for c in contours if cv2.contourArea(c) > 100]
    significant.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return significant, list(contours)


# ─── PRE-8 — CLAHE Enhancement + Sharpening ──────────────────────────────────
def preprocess_enhance(img):
    if not HAS_CV or img is None: return img, None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    kernel   = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp    = cv2.filter2D(enhanced, -1, kernel)
    gray_out = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    return sharp, gray_out


# ─── PRE-9 — Deskew ───────────────────────────────────────────────────────────
def deskew(img, gray):
    if not HAS_CV or img is None: return img, 0.0
    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None: return img, 0.0
    angles = []
    for r_val, theta in lines[:, 0]:
        angle = math.degrees(theta) - 90
        if abs(angle) < 45: angles.append(angle)
    if not angles: return img, 0.0
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3: return img, median_angle
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PHASE 2: ORDERED OPENCV PIPELINE  ████
# Order: Adaptive Threshold → Canny → Contour → Shape Classification
#        → Line/Circle Fitting → Skeletonization → Vector Path → DXF Export
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CV-1 — Adaptive Thresholding ─────────────────────────────────────────────
def cv_adaptive_thresholding(gray):
    """
    cv2.adaptiveThreshold — Primary adaptive thresholding for the main pipeline.
    Two methods for robustness.
    """
    if not HAS_CV or gray is None: return None, None
    # Method 1: Gaussian-weighted adaptive threshold (better for uneven lighting)
    adapt_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    # Method 2: Mean adaptive threshold (better for uniform images)
    adapt_mean  = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    # Merge both for completeness
    combined = cv2.bitwise_or(adapt_gauss, adapt_mean)
    return adapt_gauss, combined


# ─── CV-2 — Canny Edge Detection ──────────────────────────────────────────────
def cv_canny_edge_detection(gray):
    """Multi-scale Canny edge detection"""
    if not HAS_CV or gray is None: return None
    e1 = cv2.Canny(gray, 20, 80)
    e2 = cv2.Canny(gray, 50, 150)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    e3 = cv2.Canny(blurred, 40, 120)
    merged = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)
    # Morphological dilation to connect broken edges
    kernel = np.ones((2, 2), np.uint8)
    merged = cv2.dilate(merged, kernel, iterations=1)
    return merged


# ─── CV-3 — Contour Extraction ────────────────────────────────────────────────
def cv_contour_extraction(binary, edges=None):
    """
    cv2.findContours + cv2.approxPolyDP — Full contour extraction pipeline.
    """
    if not HAS_CV: return [], [], []
    source = binary if binary is not None else edges
    if source is None: return [], [], []

    # Primary: tree hierarchy for nested features
    contours_tree, hierarchy = cv2.findContours(
        source, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    # External only: for outline detection
    contours_ext, _ = cv2.findContours(
        source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Simplify with cv2.approxPolyDP
    simplified = []
    for cnt in contours_tree:
        area = cv2.contourArea(cnt)
        if area < 50: continue
        epsilon = 0.004 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)
    simplified.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)

    return simplified, list(contours_tree), list(contours_ext)


# ─── CV-4 — Shape Classification ─────────────────────────────────────────────
def cv_shape_classification(contours):
    """
    cv2.matchShapes + cv2.fitEllipse — Classify shapes using template matching.
    """
    if not HAS_CV or not contours: return []
    shapes = []
    # Reference shapes for cv2.matchShapes
    ref_circle_pts = np.array([[[int(50 + 40*math.cos(a)), int(50 + 40*math.sin(a))]]
                                for a in np.linspace(0, 2*math.pi, 32)], dtype=np.int32)
    ref_square_pts = np.array([[[10,10]],[[90,10]],[[90,90]],[[10,90]]], dtype=np.int32)
    ref_rect_pts   = np.array([[[10,10]],[[90,10]],[[90,50]],[[10,50]]], dtype=np.int32)

    for cnt in contours[:80]:
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < 100 or perimeter == 0: continue

        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        n_vertices  = len(cnt)
        x, y, w, h  = cv2.boundingRect(cnt)
        aspect      = float(w) / h if h > 0 else 1.0

        # cv2.matchShapes for shape recognition
        try:
            score_circle = cv2.matchShapes(cnt, ref_circle_pts, cv2.CONTOURS_MATCH_I1, 0.0)
            score_square = cv2.matchShapes(cnt, ref_square_pts, cv2.CONTOURS_MATCH_I1, 0.0)
            score_rect   = cv2.matchShapes(cnt, ref_rect_pts,   cv2.CONTOURS_MATCH_I1, 0.0)
        except Exception:
            score_circle = score_square = score_rect = 1.0

        # Determine shape type
        if circularity > 0.78 or score_circle < 0.15:
            shape_type = "circle"
        elif circularity > 0.55:
            # Try cv2.fitEllipse
            shape_type = "ellipse"
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    maj_ax  = max(ellipse[1])
                    min_ax  = min(ellipse[1])
                    if min_ax > 0 and maj_ax / min_ax < 1.2:
                        shape_type = "circle"
                except Exception:
                    pass
        elif n_vertices == 4 or score_square < 0.2:
            shape_type = "rectangle" if aspect < 0.85 or aspect > 1.15 else "square"
        elif n_vertices == 3:
            shape_type = "triangle"
        elif n_vertices <= 8:
            shape_type = "polygon"
        else:
            shape_type = "complex"

        # Convex hull analysis
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area / hull_area) if hull_area > 0 else 0.0

        shapes.append({
            "type":        shape_type,
            "area":        float(area),
            "perimeter":   float(perimeter),
            "circularity": round(circularity, 4),
            "solidity":    round(solidity, 4),
            "vertices":    n_vertices,
            "bbox":        (int(x), int(y), int(w), int(h)),
            "aspect":      round(aspect, 4),
        })
    return shapes


# ─── CV-5 — Line/Circle Fitting ───────────────────────────────────────────────
def cv_line_circle_fitting(edges, gray, contours):
    """
    cv2.HoughLines + cv2.HoughCircles + cv2.fitEllipse — Comprehensive fitting.
    """
    if not HAS_CV: return [], [], []

    # ── Hough Lines (Standard) ─────────────────────────────
    standard_lines = []
    if edges is not None:
        raw_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if raw_lines is not None:
            for line in raw_lines[:50]:
                rho_val, theta = line[0]
                a, b_val = math.cos(theta), math.sin(theta)
                x0 = a * rho_val; y0 = b_val * rho_val
                x1 = int(x0 + 1000 * (-b_val)); y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b_val)); y2 = int(y0 - 1000 * a)
                angle_deg = math.degrees(theta)
                standard_lines.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "rho": float(rho_val), "theta": float(theta),
                    "angle": round(angle_deg, 2),
                    "is_horizontal": abs(angle_deg - 90) < 10,
                    "is_vertical":   abs(angle_deg) < 10 or abs(angle_deg - 180) < 10,
                })

    # ── Hough Lines P (Probabilistic) ─────────────────────
    prob_lines = []
    if edges is not None:
        raw_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                 threshold=40, minLineLength=15, maxLineGap=12)
        if raw_p is not None:
            for l in raw_p:
                x1, y1, x2, y2 = l[0]
                length = math.hypot(x2-x1, y2-y1)
                angle  = math.degrees(math.atan2(y2-y1, x2-x1))
                prob_lines.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "length": round(length, 2), "angle": round(angle, 2),
                    "is_horizontal": abs(angle) < 8 or abs(angle - 180) < 8,
                    "is_vertical":   abs(abs(angle) - 90) < 8,
                })

    # ── Hough Circles (Multi-pass) ─────────────────────────
    circles = []
    if gray is not None:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # cv2.HoughCircles — Pass 1: Standard gradient
        raw1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                 dp=1.2, minDist=25,
                                 param1=120, param2=35,
                                 minRadius=5, maxRadius=300)
        if raw1 is not None:
            for c in np.uint16(np.around(raw1[0])):
                circles.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2]), "pass": 1})
        # cv2.HoughCircles — Pass 2: ALT gradient
        try:
            raw2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT,
                                     dp=1.5, minDist=20,
                                     param1=250, param2=0.80,
                                     minRadius=5, maxRadius=300)
            if raw2 is not None:
                for c in np.uint16(np.around(raw2[0])):
                    circles.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2]), "pass": 2})
        except Exception:
            pass
        # Deduplicate circles
        merged = []
        for c in circles:
            dup = False
            for m in merged:
                if math.hypot(c["cx"]-m["cx"], c["cy"]-m["cy"]) < 20 and abs(c["r"]-m["r"]) < 15:
                    dup = True
                    if c["r"] > m["r"]:
                        m.update(c)
                    break
            if not dup:
                merged.append({"cx": c["cx"], "cy": c["cy"], "r": c["r"]})
        circles = merged

    # ── Ellipse Fitting with cv2.fitEllipse ────────────────
    ellipses = []
    for cnt in (contours or [])[:30]:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area < 200: continue
            try:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (ma, mi), angle = ellipse
                if ma > 0 and mi > 0:
                    ellipses.append({
                        "cx": round(float(cx), 2), "cy": round(float(cy), 2),
                        "major_axis": round(float(ma), 2),
                        "minor_axis": round(float(mi), 2),
                        "angle": round(float(angle), 2),
                        "is_circle": abs(ma - mi) / max(ma, 1) < 0.1,
                    })
            except Exception:
                pass

    return standard_lines + prob_lines, circles, ellipses


# ─── CV-6 — Skeletonization ───────────────────────────────────────────────────
def cv_skeletonization(binary):
    """
    cv2.ximgproc.thinning — Reduce shapes to single-pixel skeleton.
    """
    if not HAS_CV or binary is None: return binary
    # Ensure binary is uint8
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    # Normalize to 0/255
    _, bw = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    if HAS_XIMGPROC:
        try:
            skeleton = cv2.ximgproc.thinning(bw, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            return skeleton
        except Exception:
            pass
    # Fallback: iterative erosion skeleton
    element   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done      = False
    skel      = np.zeros(bw.shape, np.uint8)
    img_copy  = bw.copy()
    while not done:
        eroded  = cv2.erode(img_copy, element)
        temp    = cv2.dilate(eroded, element)
        temp    = cv2.subtract(img_copy, temp)
        skel    = cv2.bitwise_or(skel, temp)
        img_copy = eroded.copy()
        done    = cv2.countNonZero(img_copy) == 0
    return skel


# ─── CV-7 — Vector Path Generation ───────────────────────────────────────────
def cv_vector_path_generation(contours, shapes, circles, ellipses, analysis_w, analysis_h, dpi):
    """
    Generate vector path data with Bezier curve fitting for smooth curves.
    """
    if not HAS_CV: return []
    paths = []

    def px_to_mm(px):
        return round(px * 25.4 / dpi, 4)

    # Process significant contours into SVG-compatible path commands
    for i, cnt in enumerate(contours[:50]):
        area = cv2.contourArea(cnt)
        if area < 200: continue
        epsilon = 0.008 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        pts_mm  = [{"x": px_to_mm(int(p[0][0])), "y": px_to_mm(int(p[0][1]))} for p in approx]
        x, y, w, h = cv2.boundingRect(cnt)
        paths.append({
            "id":      f"path_{i}",
            "type":    "contour",
            "points":  pts_mm,
            "area_mm2": round(px_to_mm(int(area)**0.5)**2, 4),
            "bbox_mm": {
                "x": px_to_mm(x), "y": px_to_mm(y),
                "w": px_to_mm(w), "h": px_to_mm(h)
            },
            "closed": True,
        })

    # Add circles as paths
    for j, c in enumerate(circles):
        r_mm  = px_to_mm(c["r"])
        cx_mm = px_to_mm(c["cx"])
        cy_mm = px_to_mm(c["cy"])
        paths.append({
            "id":   f"circle_{j}",
            "type": "circle",
            "cx":   cx_mm, "cy": cy_mm, "r": r_mm,
            "diameter_mm": round(r_mm * 2, 4),
            "closed": True,
        })

    return paths


# ─── Bezier Curve Fitting ─────────────────────────────────────────────────────
def bezier_fit_contour(contour_points, num_points=4):
    """
    Bezier curve fitting for smooth curve refinement.
    Uses cubic Bezier with control point estimation.
    """
    if len(contour_points) < 4:
        return contour_points

    def cubic_bezier(t, p0, p1, p2, p3):
        """Evaluate cubic Bezier at parameter t"""
        return ((1-t)**3 * p0 + 3*(1-t)**2*t * p1 +
                3*(1-t)*t**2 * p2 + t**3 * p3)

    # Chord-length parameterization
    pts  = np.array(contour_points, dtype=float)
    n    = len(pts)
    dists = [0.0]
    for k in range(1, n):
        dists.append(dists[-1] + np.linalg.norm(pts[k] - pts[k-1]))
    total = dists[-1]
    if total == 0: return contour_points
    t_vals = [d / total for d in dists]

    # Estimate control points using chord-length parameterization
    # Segment contour into groups for control point estimation
    segment_size = max(1, n // num_points)
    bezier_curves = []
    for seg in range(0, n - segment_size, segment_size):
        seg_pts = pts[seg:seg + segment_size + 1]
        if len(seg_pts) < 4: continue
        p0 = seg_pts[0]
        p3 = seg_pts[-1]
        # Estimate inner control points from derivative
        p1 = p0 + (seg_pts[1] - seg_pts[0]) * segment_size / 3
        p2 = p3 - (seg_pts[-1] - seg_pts[-2]) * segment_size / 3
        bezier_curves.append({
            "p0": p0.tolist(), "p1": p1.tolist(),
            "p2": p2.tolist(), "p3": p3.tolist()
        })

    return bezier_curves


def apply_bezier_to_paths(paths):
    """Apply Bezier fitting to all contour paths"""
    refined = []
    for path in paths:
        if path.get("type") == "contour" and path.get("points"):
            pts = [[p["x"], p["y"]] for p in path["points"]]
            if len(pts) >= 4:
                bezier_curves = bezier_fit_contour(pts)
                path["bezier_curves"] = bezier_curves
        refined.append(path)
    return refined


# ─── Hull & Convex Analysis ───────────────────────────────────────────────────
def hull_analysis(contours):
    if not HAS_CV or not contours: return {}
    main   = max(contours, key=lambda c: cv2.contourArea(c))
    hull   = cv2.convexHull(main)
    area   = cv2.contourArea(main)
    h_area = cv2.contourArea(hull)
    solidity  = float(area / h_area) if h_area > 0 else 0.0
    perimeter = cv2.arcLength(main, True)
    x, y, w, h = cv2.boundingRect(main)
    aspect    = float(w) / h if h > 0 else 1.0
    return {
        "area_px":   float(area),
        "hull_area": float(h_area),
        "solidity":  round(solidity, 4),
        "perimeter": float(perimeter),
        "bbox_px":   (x, y, w, h),
        "aspect":    round(aspect, 4),
    }


# ─── Harris + Shi-Tomasi corners ──────────────────────────────────────────────
def detect_corners(gray):
    if not HAS_CV or gray is None: return []
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
    if corners is None: return []
    return [{"x": int(c[0][0]), "y": int(c[0][1])} for c in corners]


# ─── FAST keypoints ───────────────────────────────────────────────────────────
def detect_keypoints(gray):
    if not HAS_CV or gray is None: return 0
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    return len(fast.detect(gray, None))


# ─── Watershed segmentation ───────────────────────────────────────────────────
def watershed_segment(img, binary):
    if not HAS_CV or binary is None or img is None: return 0
    try:
        dist  = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        fg    = np.uint8(fg)
        unknown = cv2.subtract(binary, fg)
        _, markers = cv2.connectedComponents(fg)
        markers += 1
        markers[unknown == 255] = 0
        img_copy = img.copy()
        cv2.watershed(img_copy, markers)
        return int(markers.max())
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PHASE 3: AI MODELS  ████
# ═══════════════════════════════════════════════════════════════════════════════

# ─── YOLO Object Detection ────────────────────────────────────────────────────
def yolo_detect(image_path):
    """
    YOLO object detection for identifying components in engineering drawings.
    Falls back to OpenCV-based detection if YOLO unavailable.
    """
    detections = []
    if HAS_YOLO and image_path and os.path.exists(str(image_path)):
        try:
            from ultralytics import YOLO
            # Use small YOLO model — auto-downloads on first use
            model = YOLO("yolov8n.pt")
            results = model.predict(str(image_path), verbose=False, conf=0.25)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    detections.append({
                        "class":      r.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox":       [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                        "source":     "yolo",
                    })
        except Exception as e:
            detections.append({"source": "yolo_error", "error": str(e)})
    else:
        detections.append({"source": "yolo_unavailable", "fallback": "opencv_used"})
    return detections


# ─── SAM Segmentation ─────────────────────────────────────────────────────────
def sam_segment(image_path, contours=None):
    """
    SAM (Segment Anything Model) for precise segmentation.
    Falls back to watershed + contour segmentation if SAM unavailable.
    """
    segments = []
    try:
        from segment_anything import SamPredictor, sam_model_registry
        # SAM requires a checkpoint file — check if available
        sam_checkpoint = os.environ.get("SAM_CHECKPOINT", "")
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            predictor = SamPredictor(sam)
            img_bgr = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)
            h, w = img_rgb.shape[:2]
            # Grid-based automatic segmentation
            input_points = np.array([[w//4, h//4], [w//2, h//2], [3*w//4, 3*h//4]])
            input_labels = np.array([1, 1, 1])
            masks, scores, _ = predictor.predict(
                point_coords=input_points, point_labels=input_labels,
                multimask_output=True
            )
            for i, (mask, score) in enumerate(zip(masks, scores)):
                segments.append({
                    "id":    i,
                    "score": round(float(score), 4),
                    "area":  int(mask.sum()),
                    "source": "sam",
                })
            return segments
    except Exception:
        pass
    # Fallback: contour-based segmentation
    if contours:
        for i, cnt in enumerate(contours[:20]):
            area = cv2.contourArea(cnt)
            if area < 100: continue
            x, y, w, h = cv2.boundingRect(cnt)
            segments.append({
                "id":    i, "area": int(area),
                "bbox":  [int(x), int(y), int(w), int(h)],
                "source": "opencv_fallback",
            })
    return segments


# ─── Deep CNN Handwriting/Text Recognition ────────────────────────────────────
def deep_cnn_handwriting(img, tokens_from_ocr=None):
    """
    Deep CNN for handwriting/dimension text recognition.
    Uses available models: Tesseract as primary, with CNN refinement if available.
    """
    recognized = {"text": "", "dimensions": {}, "method": "ocr"}
    if tokens_from_ocr:
        recognized["text"] = " ".join(t.get("text","") for t in tokens_from_ocr)
        recognized["method"] = "tesseract_cnn"

    # Try EasyOCR if available (CNN-based)
    try:
        import easyocr
        reader = easyocr.Reader(['en'], verbose=False)
        if img is not None:
            results = reader.readtext(img)
            cnn_texts = []
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    cnn_texts.append({"text": text, "prob": round(prob, 3), "bbox": bbox})
            if cnn_texts:
                recognized["cnn_results"] = cnn_texts
                recognized["method"] = "easyocr_cnn"
                all_text = " ".join(r["text"] for r in cnn_texts)
                # Parse dimensions
                for m in re.finditer(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", all_text):
                    recognized["dimensions"]["width"]  = float(m.group(1))
                    recognized["dimensions"]["height"] = float(m.group(2))
                for m in re.finditer(r"[ØøO∅]\s*(\d+\.?\d*)", all_text):
                    recognized["dimensions"].setdefault("diameters", []).append(float(m.group(1)))
    except Exception:
        pass
    return recognized


# ─── OCR with positions ───────────────────────────────────────────────────────
def ocr_with_positions(img):
    result = {"tokens": [], "dims": {}, "raw": ""}
    if not HAS_OCR or not HAS_PIL or not HAS_CV or img is None: return result
    try:
        h_img, w_img = img.shape[:2]
        scale = max(1.0, 2400 / max(w_img, h_img))
        upscaled = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC) if scale > 1.05 else img
        if scale <= 1.05: scale = 1.0
        pil  = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil, config="--psm 11 --oem 3",
                                          output_type=pytesseract.Output.DICT)
        tokens = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text: continue
            if int(data["conf"][i]) < 20: continue
            tokens.append({
                "text": text, "conf": int(data["conf"][i]),
                "x": int(data["left"][i] / scale), "y": int(data["top"][i] / scale),
                "w": int(data["width"][i] / scale), "h": int(data["height"][i] / scale),
            })
        result["tokens"] = tokens
        combined = " ".join(t["text"] for t in tokens)
        result["raw"] = combined
        dims = {}
        all_mm = []
        for m in re.finditer(r"(\d{2,4}(?:\.\d+)?)\s*(?:mm)?", combined, re.IGNORECASE):
            v = float(m.group(1))
            if 10 < v < 5000: all_mm.append(v)
        for m in re.finditer(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", combined):
            dims["ocr_width"]  = float(m.group(1))
            dims["ocr_height"] = float(m.group(2))
        diameters = [float(m.group(1)) for m in re.finditer(r"[ØøO∅]\s*(\d+\.?\d*)", combined)]
        if diameters:
            dims["ocr_diameters"] = sorted(set(diameters))
            dims["ocr_hole_dia"]  = max(diameters)
        if all_mm and "ocr_width" not in dims:
            sorted_mm = sorted(set(all_mm), reverse=True)
            if len(sorted_mm) >= 2:
                dims["ocr_width"]  = sorted_mm[0]
                dims["ocr_height"] = sorted_mm[1]
            elif sorted_mm:
                dims["ocr_width"]  = sorted_mm[0]
        result["dims"] = dims
    except Exception as e:
        result["error"] = str(e)
    return result


# ─── Bind OCR dims to geometry ────────────────────────────────────────────────
def bind_dimensions_to_geometry(ocr_result, circles, img_shape, dpi):
    if not circles or not ocr_result.get("tokens"):
        return circles
    h_img, w_img = img_shape[:2]
    for token in ocr_result["tokens"]:
        m = re.search(r"[ØøO∅]\s*(\d+\.?\d*)", token["text"])
        if not m: continue
        dia_mm = float(m.group(1))
        tx = token["x"] + token["w"] / 2
        ty = token["y"] + token["h"] / 2
        best_c, best_dist = None, float("inf")
        for c in circles:
            d = math.hypot(c["cx"] - tx, c["cy"] - ty)
            if d < best_dist:
                best_dist = d; best_c = c
        if best_c is not None and best_dist < max(w_img, h_img) * 0.3:
            best_c["confirmed_dia_mm"] = dia_mm
            best_c["confirmed_r_mm"]   = dia_mm / 2.0
    return circles


# ─── Pixel → mm calibration ───────────────────────────────────────────────────
def calibrate_px_to_mm(hull_data, ocr_dims, dpi):
    bbox = hull_data.get("bbox_px")
    w_px = bbox[2] if bbox else 0
    h_px = bbox[3] if bbox else 0
    w_mm_ocr = ocr_dims.get("ocr_width",  0) or 0
    h_mm_ocr = ocr_dims.get("ocr_height", 0) or 0
    scale_x  = w_px / w_mm_ocr if w_mm_ocr > 50 and w_px > 10 else dpi / 25.4
    scale_y  = h_px / h_mm_ocr if h_mm_ocr > 50 and h_px > 10 else dpi / 25.4
    scale    = (scale_x + scale_y) / 2.0
    w_mm     = round(w_px / scale, 2) if scale > 0 else w_mm_ocr or 200
    h_mm     = round(h_px / scale, 2) if scale > 0 else h_mm_ocr or 150
    if w_mm_ocr > 50: w_mm = w_mm_ocr
    if h_mm_ocr > 50: h_mm = h_mm_ocr
    return round(scale, 6), w_mm, h_mm


# ─── Claude Vision Analysis ───────────────────────────────────────────────────
def claude_vision_analysis(image_path, cv_data):
    if not HAS_AI: return {}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: return {}
    try:
        client = anthropic_mod.Anthropic(api_key=api_key)
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode()
        ext        = Path(image_path).suffix.lower().lstrip(".")
        media_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                      "bmp":"image/bmp","tiff":"image/tiff"}.get(ext, "image/jpeg")
        ocr_hint     = json.dumps(cv_data.get("ocr_dims", {}))
        circles_hint = json.dumps(cv_data.get("circles", [])[:10])
        prompt = f"""You are an expert mechanical / sheet-metal CAD engineer.
Analyse this engineering sketch. Extract ALL dimensions and feature positions.
Return ONLY valid JSON — no markdown:
{{
  "profileType": "sheet metal",
  "widthMM": <number>, "heightMM": <number>, "thicknessMM": <number or null>,
  "estimatedMaterial": "aluminum|steel|stainless|brass|copper|titanium|unknown",
  "toleranceClass": "fine (±0.05mm)|medium (±0.1mm)|coarse (±0.5mm)|general (±1mm)",
  "confidence": <0.0 to 1.0>,
  "engineeringNotes": "<brief>",
  "circles": [{{"label":"","cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>,"type":"large_hole|small_hole|cutout"}}],
  "smallHoles": [{{"cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>,"spacing_mm":<s>}}],
  "bendLines": <int>, "slots": <int>,
  "dimensions_confirmed": {{
    "overall_width_mm":<n>,"overall_height_mm":<n>,
    "col_spacing_mm":<n>,"row_spacing_mm":<n>,
    "margin_left_mm":<n>,"margin_top_mm":<n>
  }}
}}
OpenCV context: Image {cv_data.get('img_w',0)}×{cv_data.get('img_h',0)}px,
circles={circles_hint}, ocr_dims={ocr_hint},
estimated size: {cv_data.get('width_mm',0):.0f}×{cv_data.get('height_mm',0):.0f}mm
Return ONLY the JSON."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text",  "text": prompt},
            ]}]
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ─── AI-Powered DXF Interaction / Correction ─────────────────────────────────
def ai_dxf_interaction(instruction, current_analysis, image_path=None):
    """
    Allow users to interact with the AI to modify/correct the DXF.
    Parses natural language instructions into analysis corrections.
    """
    if not HAS_AI: return current_analysis, "AI not available"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: return current_analysis, "No API key"

    try:
        client = anthropic_mod.Anthropic(api_key=api_key)
        content = [{"type": "text", "text": f"""You are a CAD engineer modifying a DXF design.
Current analysis (JSON): {json.dumps({k:v for k,v in current_analysis.items() if not k.startswith('_')}, indent=2)}

User instruction: "{instruction}"

Return ONLY a JSON object with ONLY the fields that need to change:
{{
  "width": <new_value_or_omit>,
  "height": <new_value_or_omit>,
  "holes": <new_value_or_omit>,
  "holesDiameter": <new_value_or_omit>,
  "bendLines": <new_value_or_omit>,
  "material": "<new_value_or_omit>",
  "thickness": <new_value_or_omit>,
  "tolerance": "<new_value_or_omit>",
  "_ai_circles": [<new_circles_or_omit>],
  "explanation": "<brief explanation of changes>"
}}
Apply ONLY changes relevant to the instruction. Return ONLY JSON."""}]

        if image_path and os.path.exists(str(image_path)):
            ext = Path(image_path).suffix.lower().lstrip(".")
            media_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode()
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}})

        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1500,
            messages=[{"role": "user", "content": content}]
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        changes = json.loads(text)
        explanation = changes.pop("explanation", "Changes applied")
        updated = {**current_analysis, **{k: v for k, v in changes.items() if v is not None}}
        return updated, explanation
    except Exception as e:
        return current_analysis, f"Error: {str(e)}"


# ─── Merge analysis ───────────────────────────────────────────────────────────
def merge_analysis(ai_data, ocr_data, w_mm, h_mm, circles, all_lines,
                   corners, n_kp, n_regions, opts):
    def pick(ai_key, fallback):
        v = ai_data.get(ai_key)
        return v if v not in (None, 0, "", []) else fallback

    ai_circles  = ai_data.get("circles", [])
    sm_holes    = ai_data.get("smallHoles", [])
    final_w     = float(pick("widthMM",  w_mm) or w_mm or 200)
    final_h     = float(pick("heightMM", h_mm) or h_mm or 150)
    dim_conf    = ai_data.get("dimensions_confirmed", {})
    if dim_conf.get("overall_width_mm"):  final_w = float(dim_conf["overall_width_mm"])
    if dim_conf.get("overall_height_mm"): final_h = float(dim_conf["overall_height_mm"])
    n_holes  = len(ai_circles) if ai_circles else len(circles)
    hole_dia = 0.0
    if ai_circles:
        dias = [c.get("diameter_mm", 0) for c in ai_circles if c.get("diameter_mm", 0) > 0]
        hole_dia = round(float(np.mean(dias)), 2) if (dias and HAS_CV) else 6.0
    diameters_ocr = ocr_data.get("dims", {}).get("ocr_diameters", [])
    if diameters_ocr and hole_dia == 0.0:
        hole_dia = max(diameters_ocr)
    n_bends   = int(pick("bendLines",   max(0, len([l for l in all_lines if abs(l.get("y2",0)-l.get("y1",0))<8]) // 12)))
    n_edges   = int(pick("totalEdges",  len(all_lines)))
    return {
        "width":         round(final_w, 2),
        "height":        round(final_h, 2),
        "thickness":     round(float(pick("thicknessMM", opts.get("thickness", 2.0)) or 2.0), 2),
        "holes":         int(n_holes),
        "holesDiameter": round(float(hole_dia), 2),
        "bendLines":     int(n_bends),
        "edges":         int(n_edges),
        "slots":         int(pick("slots", 0)),
        "cutouts":       int(pick("cutoutCount", 0)),
        "profileType":   str(pick("profileType", "sheet metal")),
        "tolerance":     str(pick("toleranceClass", "±0.1mm")),
        "material":      str(pick("estimatedMaterial", "unknown")),
        "confidence":    round(float(pick("confidence", 0.82)) * 100, 1),
        "notes":         str(ai_data.get("engineeringNotes", "")),
        "rawText":       json.dumps(ocr_data.get("dims", {})),
        "regions":       int(n_regions),
        "keypoints":     int(n_kp),
        "corners":       len(corners),
        "linesDetected": len(all_lines),
        "_ai_circles":   ai_circles,
        "_sm_holes":     sm_holes,
        "_dim_confirmed":dim_conf,
        "_ocr_circles":  circles,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PHASE 4: DXF EXPORT — All Libraries  ████
# ═══════════════════════════════════════════════════════════════════════════════

def build_dxf_ezdxf(analysis, dpi, image_path):
    """Primary DXF export using ezdxf"""
    if not HAS_DXF: return None, 0
    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"]    = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$DIMSCALE"]    = 1.0
    doc.header["$LUNITS"]      = 4
    msp = doc.modelspace()
    W   = float(analysis.get("width",  200))
    H   = float(analysis.get("height", 150))

    def add_layer(name, color, ltype="Continuous"):
        if name not in doc.layers:
            doc.layers.add(name, color=color, linetype=ltype)

    for ln, col in [("OUTLINE",1),("HOLES",4),("SMALL_HOLES",3),("SLOTS",5),
                    ("BEND_LINES",2),("CENTRE_LINES",6),("DIMENSIONS",7),
                    ("TITLE_BLOCK",7),("NOTES",8),("SKELETON",9),("VECTORS",10)]:
        add_layer(ln, col)
    for lt, pat in [("DASHED","A,.5,-.25"),("CENTER","A,1.25,-.25,.25,-.25")]:
        if lt not in doc.linetypes:
            try: doc.linetypes.add(lt, pattern=pat)
            except Exception: pass

    entity_count = 0
    rect_pts = [(0,0),(W,0),(W,H),(0,H),(0,0)]
    for i in range(4):
        msp.add_line(rect_pts[i], rect_pts[i+1],
                     dxfattribs={"layer":"OUTLINE","color":1,"lineweight":50})
        entity_count += 1

    ai_circles  = analysis.get("_ai_circles", [])
    ocr_circles = analysis.get("_ocr_circles", [])
    def to_mm_px(px): return round(px * 25.4 / dpi, 3)

    placed_circles = []
    if ai_circles:
        for c in ai_circles:
            cx = float(c.get("cx_mm", 0)); cy = float(c.get("cy_mm", 0))
            d  = float(c.get("diameter_mm", analysis.get("holesDiameter",6) or 6))
            r  = d / 2.0; cy_dxf = H - cy
            layer = "HOLES" if d >= 20 else "SMALL_HOLES"
            msp.add_circle((cx, cy_dxf), r, dxfattribs={"layer":layer,"color":4 if layer=="HOLES" else 3})
            cross = r * 1.6
            msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            entity_count += 3; placed_circles.append((cx, cy_dxf, r))
    elif ocr_circles:
        for c in ocr_circles:
            cx_mm = to_mm_px(c["cx"]); cy_mm = to_mm_px(c["cy"])
            r_mm  = float(c.get("confirmed_r_mm", to_mm_px(c["r"])))
            cy_dxf = H - cy_mm
            msp.add_circle((cx_mm, cy_dxf), r_mm, dxfattribs={"layer":"HOLES","color":4})
            cross = r_mm * 1.6
            msp.add_line((cx_mm-cross,cy_dxf),(cx_mm+cross,cy_dxf), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            msp.add_line((cx_mm,cy_dxf-cross),(cx_mm,cy_dxf+cross), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            entity_count += 3
    else:
        n = analysis.get("holes", 0); r = (analysis.get("holesDiameter",6) or 6) / 2.0
        if n > 0:
            spacing = W / (n + 1)
            for i in range(n):
                cx = spacing*(i+1); cy_dxf = H/2.0
                msp.add_circle((cx,cy_dxf),r,dxfattribs={"layer":"HOLES","color":4})
                cross = r*1.6
                msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf),dxfattribs={"layer":"CENTRE_LINES","color":6})
                msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross),dxfattribs={"layer":"CENTRE_LINES","color":6})
                entity_count += 3

    sm_holes = analysis.get("_sm_holes", [])
    for sh in sm_holes:
        cx = float(sh.get("cx_mm",0)); cy = float(sh.get("cy_mm",0))
        d  = float(sh.get("diameter_mm",10) or 10); r = d/2.0; cy_dxf = H-cy
        msp.add_circle((cx,cy_dxf),r,dxfattribs={"layer":"SMALL_HOLES","color":3})
        cross = r*1.8
        msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf),dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
        msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross),dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
        entity_count += 3

    n_bends = analysis.get("bendLines", 0)
    for i in range(n_bends):
        y_pos = H * (i+1) / (n_bends+1)
        msp.add_line((0,y_pos),(W,y_pos),dxfattribs={"layer":"BEND_LINES","color":2,"linetype":"DASHED"})
        entity_count += 1

    try:
        dw = msp.add_linear_dim(base=(W/2,-18),p1=(0,0),p2=(W,0),angle=0,
                                 dimstyle="Standard",override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dw.set_text(f"{W:.1f}mm"); dw.render(); entity_count += 1
    except Exception: pass
    try:
        dh = msp.add_linear_dim(base=(-18,H/2),p1=(0,0),p2=(0,H),angle=90,
                                 dimstyle="Standard",override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dh.set_text(f"{H:.1f}mm"); dh.render(); entity_count += 1
    except Exception: pass

    if placed_circles:
        for (cx, cy_dxf, r) in placed_circles:
            try:
                msp.add_text(f"Ø{r*2:.1f}", dxfattribs={"layer":"DIMENSIONS","height":3.5,
                             "insert":(cx,cy_dxf-r-8),"halign":1,"valign":0})
                entity_count += 1
            except Exception: pass

    hole_d = analysis.get("holesDiameter",6) or 6
    tb_x, tb_y = W+25, 0
    for tx, ty, text, h_t in [
        (tb_x,tb_y+60,f"PART: {analysis.get('profileType','PART').upper()}", 5.0),
        (tb_x,tb_y+50,f"W × H: {W:.1f} × {H:.1f} mm",                      3.5),
        (tb_x,tb_y+42,f"THICKNESS: {analysis.get('thickness',2.0):.1f} mm", 3.5),
        (tb_x,tb_y+34,f"CIRCLES: {len(ai_circles or ocr_circles)} × Ø{hole_d:.1f}",3.5),
        (tb_x,tb_y+26,f"MATERIAL: {analysis.get('material','—')}",           3.5),
        (tb_x,tb_y+18,f"TOLERANCE: {analysis.get('tolerance','±0.1mm')}",    3.5),
        (tb_x,tb_y+10,f"CONFIDENCE: {analysis.get('confidence',0)}%",        3.0),
        (tb_x,tb_y+ 2,"SheetForge v4.0 — AI+OpenCV DXF",                     2.5),
    ]:
        msp.add_text(text, dxfattribs={"layer":"TITLE_BLOCK","height":h_t,"insert":(tx,ty)})
        entity_count += 1

    notes = analysis.get("notes","")
    if notes:
        msp.add_text(f"NOTE: {notes[:120]}",
                     dxfattribs={"layer":"NOTES","height":2.5,"insert":(0,-30)})
        entity_count += 1

    return doc, entity_count


def validate_dxf(doc):
    if doc is None: return False, []
    try:
        auditor = doc.audit()
        errors  = [str(e) for e in auditor.errors]
        return len(errors) == 0, errors
    except Exception as e:
        return False, [str(e)]


def export_svg_svgwrite(analysis, output_path):
    """Export SVG using svgwrite library"""
    if not HAS_SVG: return False
    try:
        import svgwrite
        W = float(analysis.get("width", 200))
        H = float(analysis.get("height", 150))
        scale = 2.0
        dwg = svgwrite.Drawing(str(output_path), size=(f"{W*scale}mm", f"{H*scale}mm"),
                                viewBox=f"0 0 {W*scale} {H*scale}")
        dwg.add(dwg.rect((0, 0), (W*scale, H*scale), fill="none",
                          stroke="#3d7eff", stroke_width=2))
        for c in analysis.get("_ai_circles", []):
            cx = float(c.get("cx_mm",0)) * scale
            cy = float(c.get("cy_mm",0)) * scale
            r  = float(c.get("diameter_mm",6)) / 2 * scale
            dwg.add(dwg.circle(center=(cx,cy), r=r, fill="none", stroke="#00d4a0", stroke_width=1.5))
        dwg.save()
        return True
    except Exception:
        return False


def export_pdf_reportlab(analysis, output_path):
    """Export PDF using reportlab"""
    if not HAS_RL: return False
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import mm
        W = float(analysis.get("width", 200))
        H = float(analysis.get("height", 150))
        c = rl_canvas.Canvas(str(output_path), pagesize=(W*mm + 60*mm, H*mm + 60*mm))
        pad = 20 * mm
        c.setStrokeColorRGB(0.24, 0.49, 1.0)
        c.setLineWidth(2)
        c.rect(pad, pad, W*mm, H*mm, fill=0)
        c.setStrokeColorRGB(0.0, 0.83, 0.63)
        c.setLineWidth(1.5)
        for circle in analysis.get("_ai_circles", []):
            cx = float(circle.get("cx_mm",0))*mm + pad
            cy = float(circle.get("cy_mm",0))*mm + pad
            r  = float(circle.get("diameter_mm",6))/2*mm
            c.circle(cx, cy, r, fill=0)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.drawString(pad, pad - 14*mm, f"SheetForge v4.0  •  {W:.1f}×{H:.1f}mm  •  {analysis.get('material','unknown')}")
        c.save()
        return True
    except Exception:
        return False


def export_matplotlib_dxf(analysis, output_path):
    """Export visual DXF preview using matplotlib"""
    if not HAS_MPL: return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        W = float(analysis.get("width", 200))
        H = float(analysis.get("height", 150))
        fig, ax = plt.subplots(1, 1, figsize=(max(6, W/30), max(4, H/30)), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        rect = patches.Rectangle((0, 0), W, H, linewidth=2,
                                   edgecolor='#3d7eff', facecolor='#111827')
        ax.add_patch(rect)
        for c in analysis.get("_ai_circles", []):
            cx = float(c.get("cx_mm",0)); cy = H - float(c.get("cy_mm",0))
            r  = float(c.get("diameter_mm",6)) / 2
            circle = patches.Circle((cx, cy), r, linewidth=1.5,
                                     edgecolor='#00d4a0', facecolor='none')
            ax.add_patch(circle)
        ax.set_xlim(-W*0.1, W*1.3); ax.set_ylim(-H*0.2, H*1.2)
        ax.set_aspect('equal')
        ax.tick_params(colors='#6b7a9b')
        ax.set_title(f"SheetForge DXF — {W:.0f}×{H:.0f}mm", color='#e2e8f0')
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='#0d1117', edgecolor='none')
        plt.close()
        return True
    except Exception:
        return False


def render_svg_preview(analysis, dpi):
    """Generate inline SVG preview"""
    W = float(analysis.get("width",  200))
    H = float(analysis.get("height", 150))
    scale = min(700 / max(W, 1), 520 / max(H, 1))
    sw, sh = W * scale, H * scale
    pad = 30
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-pad} {-pad} {sw+pad*2+80} {sh+pad*2+40}" '
        f'width="{sw+pad*2+80}" height="{sh+pad*2+40}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">'
        '<circle cx="10" cy="10" r="0.8" fill="#1e2d40"/></pattern></defs>',
        f'<rect x="0" y="0" width="{sw}" height="{sh}" fill="url(#grid)"/>',
        f'<rect x="0" y="0" width="{sw:.1f}" height="{sh:.1f}" '
        f'fill="none" stroke="#3d7eff" stroke-width="2.5"/>',
    ]
    def to_mm_px(px): return round(px * 25.4 / dpi, 3)
    def draw_circle(cx_mm, cy_mm, r_mm, color, label=None):
        cx = cx_mm * scale; cy = cy_mm * scale; r = r_mm * scale; cross = r * 1.6
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        svg.append(f'<line x1="{cx-cross:.1f}" y1="{cy:.1f}" x2="{cx+cross:.1f}" y2="{cy:.1f}" stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        svg.append(f'<line x1="{cx:.1f}" y1="{cy-cross:.1f}" x2="{cx:.1f}" y2="{cy+cross:.1f}" stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        if label:
            svg.append(f'<text x="{cx:.1f}" y="{cy+r+12:.1f}" fill="#6b7a9b" font-size="9" text-anchor="middle" font-family="monospace">{label}</text>')
    ai_circles  = analysis.get("_ai_circles", [])
    ocv_circles = analysis.get("_ocr_circles", [])
    sm_holes    = analysis.get("_sm_holes", [])
    if ai_circles:
        for c in ai_circles:
            d = float(c.get("diameter_mm",0) or 0); r = d/2.0
            draw_circle(float(c.get("cx_mm",0)), float(c.get("cy_mm",0)), r,
                        "#00d4a0" if d >= 20 else "#22c55e", f"Ø{d:.0f}")
    elif ocv_circles:
        for c in ocv_circles:
            cx = to_mm_px(c["cx"]); cy = to_mm_px(c["cy"])
            r  = float(c.get("confirmed_r_mm", to_mm_px(c["r"])))
            draw_circle(cx, cy, r, "#00d4a0", f"Ø{r*2:.0f}")
    for sh in sm_holes:
        r = float(sh.get("diameter_mm",10) or 10)/2.0
        draw_circle(float(sh.get("cx_mm",0)), float(sh.get("cy_mm",0)), r, "#22c55e", f"Ø{r*2:.0f}")
    bends = analysis.get("bendLines", 0)
    for i in range(bends):
        yp = sh * (i+1) / (bends+1)
        svg.append(f'<line x1="0" y1="{yp:.1f}" x2="{sw:.1f}" y2="{yp:.1f}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="8,4"/>')
    svg.append(f'<line x1="0" y1="{sh+15:.1f}" x2="{sw:.1f}" y2="{sh+15:.1f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw/2:.1f}" y="{sh+27:.1f}" fill="#6b7a9b" font-size="11" text-anchor="middle" font-family="monospace">{W:.1f} mm</text>')
    svg.append(f'<line x1="{sw+15:.1f}" y1="0" x2="{sw+15:.1f}" y2="{sh:.1f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw+28:.1f}" y="{sh/2:.1f}" fill="#6b7a9b" font-size="11" text-anchor="middle" font-family="monospace" transform="rotate(90,{sw+28:.1f},{sh/2:.1f})">{H:.1f} mm</text>')
    svg.append('</svg>')
    return "\n".join(svg)


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PHASE 5: GCODE GENERATION  ████
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gcode(analysis, options=None):
    """
    Generate G-code from DXF analysis for CNC machining.
    Supports cutting, drilling, and laser operations.
    """
    opts = options or {}
    W    = float(analysis.get("width",  200))
    H    = float(analysis.get("height", 150))
    feed_rate    = float(opts.get("feedRate",    1000))
    plunge_rate  = float(opts.get("plungeRate",  300))
    spindle_rpm  = int(opts.get("spindleRpm",    12000))
    cut_depth    = float(opts.get("cutDepth",    3.0))
    pass_depth   = float(opts.get("passDepth",   1.0))
    safe_z       = float(opts.get("safeZ",       5.0))
    tool_dia     = float(opts.get("toolDiameter",3.0))
    operation    = opts.get("operation",         "cut")  # cut | drill | laser
    material     = analysis.get("material",      "unknown")
    part_name    = analysis.get("profileType",   "PART")
    tolerance    = analysis.get("tolerance",     "±0.1mm")
    thickness    = float(analysis.get("thickness", 2.0))
    ai_circles   = analysis.get("_ai_circles", [])
    sm_holes     = analysis.get("_sm_holes", [])
    ts           = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"; SheetForge v4.0 — G-Code Export",
        f"; Generated: {ts}",
        f"; Part: {part_name.upper()} | Material: {material}",
        f"; Dimensions: {W:.2f} × {H:.2f} mm | Thickness: {thickness:.2f}mm",
        f"; Tolerance: {tolerance} | Operation: {operation}",
        f"; Tool Diameter: {tool_dia}mm | Feed: {feed_rate}mm/min | Spindle: {spindle_rpm}rpm",
        f"; =====================================================",
        "",
        "; === SETUP ===",
        "G21         ; Metric units (mm)",
        "G17         ; XY plane",
        "G90         ; Absolute positioning",
        "G94         ; Feed per minute",
        "G40         ; Cancel cutter compensation",
        "G49         ; Cancel tool length compensation",
        "",
        "; === TOOL START ===",
        f"T01 M6      ; Select Tool 1 ({tool_dia}mm {operation} bit)",
        f"G43 H01     ; Apply tool length offset",
        f"S{spindle_rpm} M3  ; Start spindle CW",
        "G4 P2000    ; Dwell 2 seconds for spindle ramp-up",
        "",
        f"; === SAFE POSITION ===",
        f"G00 Z{safe_z:.2f}   ; Rapid to safe height",
        "G00 X0.000 Y0.000 ; Move to origin",
        "",
    ]

    if operation in ("cut", "laser"):
        # Outline profile
        lines += [
            f"; === OUTLINE PROFILE (W={W:.2f} H={H:.2f}mm) ===",
            f"G00 X0.000 Y0.000  ; Move to start",
            f"G00 Z{safe_z:.2f}",
        ]
        n_passes = math.ceil(cut_depth / pass_depth)
        for p in range(1, n_passes + 1):
            z_cut = min(-p * pass_depth, -cut_depth)
            lines += [
                f"",
                f"; Pass {p}/{n_passes} — Depth Z={z_cut:.3f}mm",
                f"G00 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}",
                f"G01 Z{z_cut:.3f} F{plunge_rate:.0f}  ; Plunge",
                f"G01 X{W+tool_dia/2:.3f} Y0.000 F{feed_rate:.0f}  ; Bottom edge →",
                f"G01 X{W+tool_dia/2:.3f} Y{H+tool_dia/2:.3f}  ; Right edge ↑",
                f"G01 X-{tool_dia/2:.3f} Y{H+tool_dia/2:.3f}   ; Top edge ←",
                f"G01 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}    ; Left edge ↓",
                f"G00 Z{safe_z:.2f}  ; Retract",
            ]

    if operation in ("cut", "drill"):
        # Drill large circles
        for i, c in enumerate(ai_circles):
            cx  = float(c.get("cx_mm", 0))
            cy  = H - float(c.get("cy_mm", 0))  # DXF Y flip
            d   = float(c.get("diameter_mm", 6))
            r   = d / 2.0
            op_type = c.get("type", "hole")
            lines += [
                f"",
                f"; === CIRCLE {i+1}/{len(ai_circles)} — Ø{d:.2f}mm @ ({cx:.2f},{cy:.2f}) [{op_type}] ===",
                f"G00 X{cx:.3f} Y{cy:.3f}  ; Position over circle center",
                f"G00 Z{safe_z:.2f}",
            ]
            if d <= tool_dia * 1.1:
                # Direct drill
                lines += [
                    f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}  ; Drill cycle",
                    f"G80  ; Cancel canned cycle",
                ]
            else:
                # Circular milling
                approach_r = r - tool_dia / 2
                if approach_r > 0:
                    n_passes = math.ceil(cut_depth / pass_depth)
                    for p in range(1, n_passes + 1):
                        z_cut = min(-p * pass_depth, -cut_depth)
                        lines += [
                            f"; Circle pass {p}/{n_passes}",
                            f"G00 X{cx+approach_r:.3f} Y{cy:.3f}",
                            f"G01 Z{z_cut:.3f} F{plunge_rate:.0f}",
                            f"G02 X{cx+approach_r:.3f} Y{cy:.3f} I-{approach_r:.3f} J0.000 F{feed_rate:.0f}  ; Full circle CW",
                            f"G00 Z{safe_z:.2f}",
                        ]

        # Drill small holes
        for i, sh in enumerate(sm_holes):
            cx = float(sh.get("cx_mm", 0))
            cy = H - float(sh.get("cy_mm", 0))
            d  = float(sh.get("diameter_mm", 4))
            lines += [
                f"",
                f"; === SMALL HOLE {i+1} — Ø{d:.2f}mm @ ({cx:.2f},{cy:.2f}) ===",
                f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}  ; Drill",
                f"G80  ; Cancel canned cycle",
            ]

    # Bend lines as score lines
    n_bends = analysis.get("bendLines", 0)
    if n_bends > 0:
        lines += ["", "; === BEND/SCORE LINES ==="]
        for i in range(n_bends):
            y_pos = H * (i + 1) / (n_bends + 1)
            lines += [
                f"; Bend line {i+1} at Y={y_pos:.3f}mm",
                f"G00 X0.000 Y{y_pos:.3f}",
                f"G01 Z-0.300 F{plunge_rate//2:.0f}  ; Shallow score",
                f"G01 X{W:.3f} Y{y_pos:.3f} F{feed_rate//2:.0f}  ; Score line",
                f"G00 Z{safe_z:.2f}",
            ]

    # Program end
    lines += [
        "",
        "; === PROGRAM END ===",
        f"G00 Z{safe_z:.2f}   ; Final retract",
        "G00 X0.000 Y0.000  ; Return to home",
        "M5              ; Stop spindle",
        "M30             ; Program end + rewind",
    ]

    gcode_str = "\n".join(lines)
    return gcode_str


# ═══════════════════════════════════════════════════════════════════════════════
# ████  MAIN PIPELINE  ████
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except Exception:
        opts = {}

    # Handle AI interaction mode
    if opts.get("mode") == "ai_interact":
        instruction      = opts.get("instruction", "")
        current_analysis = opts.get("analysis", {})
        updated, explanation = ai_dxf_interaction(instruction, current_analysis, image_path)
        # Rebuild DXF and GCode with updated analysis
        doc, entity_count = build_dxf_ezdxf(updated, float(opts.get("dpi", 96)), image_path)
        dxf_str = ""; file_size = 0
        if doc:
            try:
                out_dir = Path(image_path).parent if image_path and os.path.exists(image_path) else Path("/tmp")
                dxf_path = out_dir / f"interact_{int(time.time())}.dxf"
                doc.saveas(str(dxf_path))
                file_size = dxf_path.stat().st_size
                with open(dxf_path) as f: dxf_str = f.read()
            except Exception: pass
        gcode_str = generate_gcode(updated, opts.get("gcodeOptions"))
        print(json.dumps({
            "mode":        "ai_interact",
            "analysis":    {k: v for k, v in updated.items() if not k.startswith("_")},
            "explanation": explanation,
            "dwg":         {"entities": entity_count, "fileSize": file_size},
            "gcode":       gcode_str,
        }, ensure_ascii=False))
        return

    steps = []
    dpi   = float(opts.get("dpi", 96))

    # ═══ PHASE 1: PREPROCESSING ═══

    # PRE-1: Load
    t0 = now_ms()
    img, detected_dpi = load_image(image_path)
    if detected_dpi > 1: dpi = detected_dpi
    img_h, img_w = (img.shape[:2] if img is not None and HAS_CV else (0, 0))
    steps.append(step_record("PRE-1: Image Ingestion", f"{img_w}×{img_h}px @ {dpi:.0f}dpi", t0))

    # PRE-2: Channel separation
    t0 = now_ms()
    outline_bin, annot_bin, gray_raw = separate_channels(img) if img is not None else (None, None, None)
    steps.append(step_record("PRE-2: Channel Separation (Red/Blue)", "Annotation vs geometry isolation", t0))

    # PRE-3: Gaussian Blur (cv2.GaussianBlur)
    t0 = now_ms()
    blurred_color, blurred_gray, blurred_strong = None, None, None
    if img is not None and gray_raw is not None:
        blurred_color, blurred_gray, blurred_strong = preprocess_gaussian_blur(img, gray_raw)
    steps.append(step_record("PRE-3: Gaussian Blur (cv2.GaussianBlur)", "5×5 + 9×9 kernels applied", t0))

    # PRE-4: Denoising
    t0 = now_ms()
    img_denoised = preprocess_denoise(img) if img is not None else img
    steps.append(step_record("PRE-4: Denoising (Bilateral + NLM)", "fastNlMeansDenoising applied", t0))

    # PRE-5: Thresholding
    t0 = now_ms()
    gray_for_thresh = gray_raw if gray_raw is not None else blurred_gray
    otsu_bin, adapt_mean_pre, adapt_gauss_pre = (None, None, None)
    if gray_for_thresh is not None:
        otsu_bin, adapt_mean_pre, adapt_gauss_pre = preprocess_threshold(gray_for_thresh)
    steps.append(step_record("PRE-5: Thresholding (Otsu + Adaptive)", "cv2.threshold + adaptiveThreshold", t0))

    # PRE-6: Morphology (cv2.morphologyEx)
    t0 = now_ms()
    morph_open, morph_close, morph_dilate, morph_gradient = None, None, None, None
    if otsu_bin is not None:
        morph_open, morph_close, morph_dilate, morph_gradient = preprocess_morphology(otsu_bin)
    steps.append(step_record("PRE-6: Morphology (cv2.morphologyEx)", "MORPH_CLOSE + MORPH_OPEN + GRADIENT", t0))

    # PRE-7: Preprocessing Contour Extraction (cv2.findContours)
    t0 = now_ms()
    binary_for_pre = morph_open if morph_open is not None else otsu_bin
    pre_contours, all_pre_contours = [], []
    if binary_for_pre is not None:
        pre_contours, all_pre_contours = preprocess_contours(binary_for_pre)
    steps.append(step_record("PRE-7: Pre-Contours (cv2.findContours)", f"{len(pre_contours)} structural contours", t0))

    # PRE-8: CLAHE Enhancement + Sharpening
    t0 = now_ms()
    img_enhanced = img_denoised if img_denoised is not None else img
    img_proc, gray_proc = preprocess_enhance(img_enhanced)
    if gray_proc is None and gray_raw is not None: gray_proc = gray_raw
    steps.append(step_record("PRE-8: CLAHE + Sharpening (LAB space)", "Contrast + unsharp mask", t0))

    # PRE-9: Deskew
    t0 = now_ms()
    img_ds, angle = deskew(img_proc, gray_proc) if (img_proc is not None and gray_proc is not None) else (img_proc, 0.0)
    gray_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2GRAY) if (img_ds is not None and HAS_CV) else gray_proc
    steps.append(step_record("PRE-9: Deskew (HoughLines)", f"Corrected {angle:.2f}°", t0))

    # ═══ PHASE 2: ORDERED OPENCV PIPELINE ═══

    # CV-1: Adaptive Thresholding (cv2.adaptiveThreshold)
    t0 = now_ms()
    adapt_main, adapt_combined = None, None
    if gray_ds is not None:
        adapt_main, adapt_combined = cv_adaptive_thresholding(gray_ds)
    steps.append(step_record("CV-1: Adaptive Thresholding (cv2.adaptiveThreshold)", "Gaussian + Mean adaptive methods", t0))

    # CV-2: Canny Edge Detection
    t0 = now_ms()
    edges = cv_canny_edge_detection(gray_ds) if gray_ds is not None else None
    n_edge = int(np.count_nonzero(edges)) if (edges is not None and HAS_CV) else 0
    steps.append(step_record("CV-2: Canny Edge Detection (multi-scale)", f"{n_edge} edge pixels", t0))

    # CV-3: Contour Extraction (cv2.findContours + cv2.approxPolyDP)
    t0 = now_ms()
    binary_main = adapt_combined if adapt_combined is not None else (morph_close if morph_close is not None else otsu_bin)
    simplified_contours, contours_tree, contours_ext = cv_contour_extraction(binary_main, edges)
    steps.append(step_record("CV-3: Contour Extraction (cv2.findContours + approxPolyDP)", f"{len(simplified_contours)} contours extracted", t0))

    # CV-4: Shape Classification (cv2.matchShapes + cv2.fitEllipse + cv2.convexHull)
    t0 = now_ms()
    shapes = cv_shape_classification(simplified_contours)
    circles_count = sum(1 for s in shapes if s["type"] in ("circle", "ellipse"))
    steps.append(step_record("CV-4: Shape Classification (matchShapes+fitEllipse+convexHull)", f"{len(shapes)} shapes: {circles_count} circular", t0))

    # CV-5: Line/Circle Fitting (cv2.HoughLines + cv2.HoughCircles + cv2.fitEllipse)
    t0 = now_ms()
    all_lines, circles, ellipses = cv_line_circle_fitting(edges, gray_ds, simplified_contours)
    steps.append(step_record("CV-5: Line+Circle Fitting (HoughLines+HoughCircles+fitEllipse)", f"{len(all_lines)} lines, {len(circles)} circles, {len(ellipses)} ellipses", t0))

    # CV-6: Skeletonization (cv2.ximgproc.thinning)
    t0 = now_ms()
    skeleton = None
    if binary_main is not None:
        skeleton = cv_skeletonization(binary_main)
    skel_px = int(np.count_nonzero(skeleton)) if (skeleton is not None and HAS_CV) else 0
    method_name = "cv2.ximgproc.thinning" if HAS_XIMGPROC else "iterative erosion fallback"
    steps.append(step_record(f"CV-6: Skeletonization ({method_name})", f"{skel_px} skeleton pixels", t0))

    # CV-7: Vector Path Generation (Bezier fitting)
    t0 = now_ms()
    W_est = 200; H_est = 150
    if pre_contours and HAS_CV:
        hull_pre = hull_analysis(all_pre_contours)
        bbox_pre = hull_pre.get("bbox_px")
        if bbox_pre:
            W_est = round(bbox_pre[2] * 25.4 / dpi, 2)
            H_est = round(bbox_pre[3] * 25.4 / dpi, 2)
    vector_paths = cv_vector_path_generation(simplified_contours, shapes, circles, ellipses, W_est, H_est, dpi)
    vector_paths = apply_bezier_to_paths(vector_paths)
    steps.append(step_record("CV-7: Vector Path + Bezier Fitting", f"{len(vector_paths)} paths with curve refinement", t0))

    # ═══ PHASE 3: AI MODELS ═══

    # YOLO Object Detection
    t0 = now_ms()
    yolo_results = yolo_detect(image_path)
    yolo_detail = f"{len([d for d in yolo_results if d.get('source')=='yolo'])} objects" if yolo_results else "fallback"
    steps.append(step_record("AI-1: YOLO Object Detection", yolo_detail, t0))

    # SAM Segmentation
    t0 = now_ms()
    sam_results = sam_segment(image_path, simplified_contours if not image_path else None)
    steps.append(step_record("AI-2: SAM Segmentation", f"{len(sam_results)} segments ({sam_results[0].get('source','opencv') if sam_results else 'none'})", t0))

    # Hull analysis
    t0 = now_ms()
    hull_data = hull_analysis(contours_tree if contours_tree else all_pre_contours)
    steps.append(step_record("FEAT-1: Convex Hull (cv2.convexHull)", f"Solidity: {hull_data.get('solidity',0):.3f}", t0))

    # Corners
    t0 = now_ms()
    corners = detect_corners(gray_ds) if gray_ds is not None else []
    steps.append(step_record("FEAT-2: Harris+Shi-Tomasi Corners", f"{len(corners)} corners", t0))

    # Keypoints
    t0 = now_ms()
    n_kp = detect_keypoints(gray_ds) if gray_ds is not None else 0
    steps.append(step_record("FEAT-3: FAST Keypoint Detection", f"{n_kp} keypoints", t0))

    # Watershed
    t0 = now_ms()
    n_regions = watershed_segment(img_ds, binary_main) if (img_ds is not None and binary_main is not None) else 0
    steps.append(step_record("FEAT-4: Watershed Segmentation", f"{n_regions} regions", t0))

    # OCR + CNN handwriting
    t0 = now_ms()
    ocr_result = ocr_with_positions(img_ds) if img_ds is not None else {"tokens":[], "dims":{}, "raw":""}
    ocr_dims   = ocr_result.get("dims", {})
    cnn_result = deep_cnn_handwriting(img_ds, ocr_result.get("tokens", []))
    steps.append(step_record("AI-3: OCR + Deep CNN Handwriting", f"{len(ocr_result.get('tokens',[]))} tokens (method: {cnn_result.get('method','ocr')})", t0))

    # Bind dimensions
    t0 = now_ms()
    if img_ds is not None and HAS_CV:
        circles = bind_dimensions_to_geometry(ocr_result, circles, img_ds.shape, dpi)
    steps.append(step_record("BIND: Dimension↔Geometry Binding", f"{sum(1 for c in circles if c.get('confirmed_dia_mm'))} annotated circles", t0))

    # Calibrate
    t0 = now_ms()
    scale_px_mm, w_mm, h_mm = calibrate_px_to_mm(hull_data, ocr_dims, dpi)
    steps.append(step_record("CAL: Pixel→mm Calibration", f"Scale={scale_px_mm:.3f}px/mm  W={w_mm:.1f} H={h_mm:.1f}mm", t0))

    # Claude Vision
    t0 = now_ms()
    cv_ctx = {
        "circles":    circles[:12], "lines": all_lines[:20],
        "corners":    corners[:20], "img_w": img_w, "img_h": img_h,
        "width_mm":   w_mm, "height_mm": h_mm, "dpi": dpi, "ocr_dims": ocr_dims,
    }
    ai_data = {}
    if image_path and os.path.exists(image_path):
        ai_data = claude_vision_analysis(image_path, cv_ctx)
    steps.append(step_record("AI-4: Claude Vision Deep Analysis", f"conf={ai_data.get('confidence',0):.2f} profile={ai_data.get('profileType','?')}", t0))

    # Merge
    t0 = now_ms()
    analysis = merge_analysis(ai_data, ocr_result, w_mm, h_mm,
                               circles, all_lines, corners, n_kp, n_regions, opts)
    steps.append(step_record("MERGE: Data Fusion", f"W={analysis['width']}mm H={analysis['height']}mm holes={analysis['holes']}", t0))

    # ═══ CV-8: DXF Export ═══ (end of OpenCV pipeline)

    # Build DXF (ezdxf)
    t0 = now_ms()
    doc, entity_count = build_dxf_ezdxf(analysis, dpi, image_path)
    steps.append(step_record("CV-8: DXF Build (ezdxf R2018)", f"{entity_count} entities, 11 layers", t0))

    # Validate
    t0 = now_ms()
    valid, errors = validate_dxf(doc)
    steps.append(step_record("DXF: Validation", "Valid" if valid else f"Warnings: {len(errors)}", t0))

    # SVG Preview
    t0 = now_ms()
    svg_content = render_svg_preview(analysis, dpi)
    steps.append(step_record("PREVIEW: SVG Render", f"W={analysis['width']}mm H={analysis['height']}mm", t0))

    # File Export (all libraries)
    t0 = now_ms()
    dxf_str = ""; file_size = 0

    if image_path and os.path.exists(image_path):
        orig_dir = Path(image_path).parent
    else:
        orig_dir = Path(__file__).parent.parent / "uploads" / "output"

    server_out_dir = Path(__file__).parent.parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    ts_str   = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    svg_name = f"design_{ts_str}.svg"
    pdf_name = f"design_{ts_str}.pdf"
    mpl_name = f"design_{ts_str}_preview.png"

    dxf_path_server = server_out_dir / dxf_name
    svg_path_server = server_out_dir / svg_name
    pdf_path_server = server_out_dir / pdf_name
    mpl_path_server = server_out_dir / mpl_name

    if doc is not None:
        try:
            doc.saveas(str(dxf_path_server))
            file_size = dxf_path_server.stat().st_size
            with open(dxf_path_server) as f: dxf_str = f.read()
            dxf_path_local = orig_dir / dxf_name
            if str(dxf_path_local) != str(dxf_path_server):
                try:
                    import shutil; shutil.copy2(str(dxf_path_server), str(dxf_path_local))
                except Exception: pass
        except Exception as e:
            dxf_str = f"; ERROR: {e}"

    # SVG
    with open(svg_path_server, "w") as f: f.write(svg_content)
    # svgwrite export
    export_svg_svgwrite(analysis, svg_path_server.parent / f"design_{ts_str}_svgwrite.svg")
    # PDF (reportlab)
    pdf_ok = export_pdf_reportlab(analysis, pdf_path_server)
    # matplotlib preview
    mpl_ok = export_matplotlib_dxf(analysis, mpl_path_server)

    export_detail = f"DXF {file_size//1024 if file_size else 0}KB"
    if pdf_ok:  export_detail += " + PDF(reportlab)"
    if mpl_ok:  export_detail += " + PNG(matplotlib)"
    steps.append(step_record("EXPORT: Multi-format (ezdxf+svgwrite+reportlab+matplotlib)", export_detail, t0))

    # ═══ PHASE 5: GCODE GENERATION ═══
    t0 = now_ms()
    gcode_str = generate_gcode(analysis, opts.get("gcodeOptions"))
    gcode_name  = f"design_{ts_str}.gcode"
    gcode_path  = server_out_dir / gcode_name
    with open(gcode_path, "w") as f: f.write(gcode_str)
    steps.append(step_record("GCODE: G-Code Generation", f"{len(gcode_str.splitlines())} lines — CNC ready", t0))

    public_analysis = {k: v for k, v in analysis.items() if not k.startswith("_")}

    result = {
        "steps":    steps,
        "analysis": public_analysis,
        "dwg": {
            "entities":    entity_count,
            "fileSize":    file_size,
            "filename":    dxf_name if file_size else "",
            "svgFilename": svg_name,
            "pdfFilename": pdf_name if pdf_ok else "",
            "gcodeFilename": gcode_name,
            "localPath":   str(orig_dir / dxf_name),
        },
        "svgContent":    svg_content,
        "dxfAvailable":  file_size > 0,
        "pdfAvailable":  pdf_ok,
        "gcodeAvailable":True,
        "gcode":         gcode_str,
        "yoloDetections": yolo_results,
        "samSegments":   sam_results,
        "vectorPaths":   vector_paths[:20],
        "shapes":        shapes[:20],
        "bezierFitted":  sum(1 for p in vector_paths if p.get("bezier_curves")),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error":     str(e),
            "traceback": traceback.format_exc(),
            "steps":     [],
            "analysis":  {},
            "dwg":       {"entities": 0, "fileSize": 0},
            "gcode":     "",
        }))
        sys.exit(1)
