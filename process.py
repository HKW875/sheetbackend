#!/usr/bin/env python3
"""
SheetForge — CV + AI DXF Pipeline  v5.0
========================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, gcode }

v5.0 — Leaner, faster, more accurate:
  • Single-pass preprocessing (no redundant blur/denoise/morph chains)
  • Accurate px→mm calibration via DPI + OCR + hull cross-check
  • Precise circle/line/corner extraction feeding exact DXF coordinates
  • Tight Gemini prompt with full CV context for maximum accuracy
  • PDF rendered from actual DXF geometry (reportlab)
  • svgContent feeds <div class="dwg-viewer" id="dwg-main-viewer">
  • GCode generation preserved
  • All Gemini / ai_dxf_interaction functions kept identical
"""

import sys, os, json, math, time, re, traceback
from pathlib import Path

# ── Graceful optional imports ────────────────────────────────────────────────
def _try(fn):
    try: return fn()
    except Exception: return None

cv2           = _try(lambda: __import__("cv2"))
np            = _try(lambda: __import__("numpy"))
ezdxf         = _try(lambda: __import__("ezdxf"))
pytesseract   = _try(lambda: __import__("pytesseract"))
Image         = _try(lambda: __import__("PIL.Image",       fromlist=["Image"]))
genai_mod     = _try(lambda: __import__("google.genai",    fromlist=["genai"]))
reportlab_mod = _try(lambda: __import__("reportlab"))
svgwrite_mod  = _try(lambda: __import__("svgwrite"))
matplotlib_mod= _try(lambda: __import__("matplotlib"))

HAS_CV   = cv2 is not None and np is not None
HAS_DXF  = ezdxf is not None
HAS_OCR  = pytesseract is not None
HAS_PIL  = Image is not None
HAS_AI   = genai_mod is not None
HAS_RL   = reportlab_mod is not None
HAS_SVG  = svgwrite_mod is not None
HAS_MPL  = matplotlib_mod is not None

HAS_XIMGPROC = False
if HAS_CV:
    try: _ = cv2.ximgproc.thinning; HAS_XIMGPROC = True
    except Exception: pass

def now_ms(): return int(time.time() * 1000)
def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PREPROCESSING  (single-pass, no wasted work)
# ════════════════════════════════════════════════════════════════════════════════

def load_image(image_path):
    """Load image, detect DPI, return (bgr, gray, dpi)."""
    if not HAS_CV or not image_path or not os.path.exists(image_path):
        return None, None, 96.0
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None, None, 96.0
    dpi = 96.0
    if HAS_PIL:
        try:
            pil = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (96, 96))
            dpi  = float(xdpi[0]) if xdpi[0] > 1 else 96.0
        except Exception: pass
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray, dpi


def preprocess(img, gray):
    """
    Single-pass preprocessing:
    1. GaussianBlur (noise reduction)
    2. CLAHE (contrast normalisation)
    3. adaptiveThreshold (paper/ink separation)
    4. morphologyEx CLOSE (seal broken strokes)
    5. morphologyEx OPEN  (remove specks)
    Returns: (binary, gray_clean)
    """
    if not HAS_CV or img is None: return None, gray

    # Resize if very large (speeds up all downstream ops)
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        scale = 2400 / max(h, w)
        img  = cv2.resize(img,  (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # GaussianBlur for noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clean = clahe.apply(blurred)

    # Adaptive threshold — inverted (strokes = 255)
    binary = cv2.adaptiveThreshold(
        gray_clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # morphologyEx: close small gaps, then remove speckle
    k3 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3)
    return binary, gray_clean


def deskew(img, gray):
    """Correct skew using HoughLines on Canny edges."""
    if not HAS_CV or img is None: return img, gray, 0.0
    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None: return img, gray, 0.0
    angles = []
    for r_val, theta in lines[:, 0]:
        angle = math.degrees(theta) - 90
        if abs(angle) < 45: angles.append(angle)
    if not angles: return img, gray, 0.0
    med = float(np.median(angles))
    if abs(med) < 0.3: return img, gray, med
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), med, 1.0)
    img_r  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gray_r = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img_r, gray_r, med


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_contours(binary):
    """
    cv2.findContours + cv2.approxPolyDP
    Returns: (simplified, tree_list, ext_list)
    """
    if not HAS_CV or binary is None: return [], [], []
    contours_tree, _ = cv2.findContours(binary, cv2.RETR_TREE,     cv2.CHAIN_APPROX_NONE)
    contours_ext, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified = []
    for cnt in contours_tree:
        area = cv2.contourArea(cnt)
        if area < 50: continue
        eps   = 0.004 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        simplified.append(approx)
    simplified.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    return simplified, list(contours_tree), list(contours_ext)


def extract_lines(edges, gray):
    """
    cv2.HoughLines (standard) + cv2.HoughLinesP (probabilistic)
    Returns combined list of line dicts with pixel coordinates.
    """
    if not HAS_CV: return []
    lines_out = []

    if edges is not None:
        # Standard Hough → full-length lines
        raw = cv2.HoughLines(edges, 1, np.pi / 180, threshold=90)
        if raw is not None:
            for line in raw[:60]:
                rho, theta = line[0]
                a, b = math.cos(theta), math.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b)); y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b)); y2 = int(y0 - 1000 * a)
                ang = math.degrees(theta)
                lines_out.append({
                    "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                    "rho":float(rho),"theta":float(theta),"angle":round(ang,2),
                    "is_horizontal": abs(ang-90)<10,
                    "is_vertical":   abs(ang)<10 or abs(ang-180)<10,
                    "type":"hough",
                })

        # Probabilistic Hough → segment lines with real endpoints
        raw_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                                 minLineLength=25, maxLineGap=12)
        if raw_p is not None:
            for seg in raw_p[:80]:
                x1,y1,x2,y2 = seg[0]
                length = math.hypot(x2-x1, y2-y1)
                angle  = math.degrees(math.atan2(y2-y1, x2-x1))
                lines_out.append({
                    "x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                    "length":round(length,2),"angle":round(angle,2),
                    "is_horizontal": abs(angle)<8 or abs(angle-180)<8,
                    "is_vertical":   abs(abs(angle)-90)<8,
                    "type":"prob",
                })
    return lines_out


def extract_circles(gray):
    """
    cv2.HoughCircles (two passes: GRADIENT + GRADIENT_ALT), deduplicated.
    Returns list of {cx, cy, r} in pixels.
    """
    if not HAS_CV or gray is None: return []
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    found = []

    raw1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                             dp=1.2, minDist=20,
                             param1=100, param2=30,
                             minRadius=4, maxRadius=0)
    if raw1 is not None:
        for c in np.uint16(np.around(raw1[0])):
            found.append({"cx":int(c[0]),"cy":int(c[1]),"r":int(c[2])})

    try:
        raw2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT,
                                 dp=1.5, minDist=18,
                                 param1=220, param2=0.82,
                                 minRadius=4, maxRadius=0)
        if raw2 is not None:
            for c in np.uint16(np.around(raw2[0])):
                found.append({"cx":int(c[0]),"cy":int(c[1]),"r":int(c[2])})
    except Exception: pass

    # Deduplicate
    merged = []
    for c in found:
        dup = False
        for m in merged:
            if math.hypot(c["cx"]-m["cx"], c["cy"]-m["cy"]) < 18 and abs(c["r"]-m["r"]) < 12:
                dup = True
                if c["r"] > m["r"]: m.update(c)
                break
        if not dup:
            merged.append({"cx":c["cx"],"cy":c["cy"],"r":c["r"]})
    return merged


def classify_shapes(contours):
    """
    cv2.matchShapes + cv2.fitEllipse + cv2.convexHull
    Returns list of shape dicts.
    """
    if not HAS_CV or not contours: return []
    # Reference shapes for matchShapes
    ref_circle = np.array([[[int(50+40*math.cos(a)),int(50+40*math.sin(a))]]
                            for a in np.linspace(0,2*math.pi,32)], dtype=np.int32)
    ref_square = np.array([[[10,10]],[[90,10]],[[90,90]],[[10,90]]], dtype=np.int32)
    ref_rect   = np.array([[[10,10]],[[90,10]],[[90,50]],[[10,50]]], dtype=np.int32)

    shapes = []
    for cnt in contours[:80]:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 100 or peri == 0: continue
        circ   = 4 * math.pi * area / (peri * peri)
        n_v    = len(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect = float(w)/h if h > 0 else 1.0

        try:
            sc = cv2.matchShapes(cnt, ref_circle, cv2.CONTOURS_MATCH_I1, 0.0)
            ss = cv2.matchShapes(cnt, ref_square, cv2.CONTOURS_MATCH_I1, 0.0)
            sr = cv2.matchShapes(cnt, ref_rect,   cv2.CONTOURS_MATCH_I1, 0.0)
        except Exception:
            sc = ss = sr = 1.0

        if circ > 0.80 or sc < 0.12:
            stype = "circle"
        elif circ > 0.55 and len(cnt) >= 5:
            stype = "ellipse"
            try:
                (ex,ey),(ma,mi),_ = cv2.fitEllipse(cnt)
                if mi > 0 and ma/mi < 1.15: stype = "circle"
            except Exception: pass
        elif n_v == 4 or ss < 0.18:
            stype = "rectangle" if aspect < 0.85 or aspect > 1.15 else "square"
        elif n_v == 3:
            stype = "triangle"
        elif n_v <= 8:
            stype = "polygon"
        else:
            stype = "complex"

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area/hull_area) if hull_area > 0 else 0.0

        shapes.append({
            "type":stype, "area":float(area), "perimeter":float(peri),
            "circularity":round(circ,4), "solidity":round(solidity,4),
            "vertices":n_v, "bbox":(int(x),int(y),int(w),int(h)), "aspect":round(aspect,4),
        })
    return shapes


def detect_corners(gray):
    """cv2.goodFeaturesToTrack (Shi-Tomasi) corner detection."""
    if not HAS_CV or gray is None: return []
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts is None: return []
    return [{"x":int(p[0][0]),"y":int(p[0][1])} for p in pts]


def skeletonize(binary):
    """cv2.ximgproc.thinning or iterative erosion fallback."""
    if not HAS_CV or binary is None: return binary
    _, bw = cv2.threshold(binary.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    if HAS_XIMGPROC:
        try:
            return cv2.ximgproc.thinning(bw, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except Exception: pass
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skel = np.zeros(bw.shape, np.uint8)
    img  = bw.copy()
    while True:
        eroded = cv2.erode(img, elem)
        temp   = cv2.subtract(img, cv2.dilate(eroded, elem))
        skel   = cv2.bitwise_or(skel, temp)
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0: break
    return skel


def hull_analysis(contours):
    """cv2.convexHull analysis on the largest contour."""
    if not HAS_CV or not contours: return {}
    main   = max(contours, key=lambda c: cv2.contourArea(c))
    hull   = cv2.convexHull(main)
    area   = cv2.contourArea(main)
    h_area = cv2.contourArea(hull)
    x,y,w,h = cv2.boundingRect(main)
    return {
        "area_px":  float(area),
        "hull_area":float(h_area),
        "solidity": round(float(area/h_area) if h_area > 0 else 0, 4),
        "perimeter":float(cv2.arcLength(main, True)),
        "bbox_px":  (x,y,w,h),
        "aspect":   round(float(w)/h if h > 0 else 1.0, 4),
    }


def build_vector_paths(contours, circles, dpi):
    """Convert contours and circles to mm-space vector paths with Bezier hints."""
    if not HAS_CV: return []
    def px2mm(px): return round(px * 25.4 / dpi, 4)
    paths = []
    for i, cnt in enumerate(contours[:50]):
        area = cv2.contourArea(cnt)
        if area < 200: continue
        eps   = 0.008 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts   = [{"x":px2mm(int(p[0][0])),"y":px2mm(int(p[0][1]))} for p in approx]
        x,y,w,h = cv2.boundingRect(cnt)
        bezier = _bezier_hints(pts)
        paths.append({"id":f"path_{i}","type":"contour","points":pts,"closed":True,
                       "bbox_mm":{"x":px2mm(x),"y":px2mm(y),"w":px2mm(w),"h":px2mm(h)},
                       "bezier_curves":bezier})
    for j, c in enumerate(circles):
        r_mm = px2mm(c["r"])
        paths.append({"id":f"circle_{j}","type":"circle",
                       "cx":px2mm(c["cx"]),"cy":px2mm(c["cy"]),"r":r_mm,
                       "diameter_mm":round(r_mm*2,4),"closed":True})
    return paths


def _bezier_hints(pts, seg=4):
    """Lightweight cubic Bezier control-point hints from polyline points."""
    n = len(pts)
    if n < 4: return []
    curves = []
    step = max(1, n // seg)
    for s in range(0, n - step, step):
        p0 = [pts[s]["x"],   pts[s]["y"]]
        p3 = [pts[s+step]["x"], pts[s+step]["y"]]
        p1 = [p0[0]+(p3[0]-p0[0])/3, p0[1]+(p3[1]-p0[1])/3]
        p2 = [p0[0]+2*(p3[0]-p0[0])/3, p0[1]+2*(p3[1]-p0[1])/3]
        curves.append({"p0":p0,"p1":p1,"p2":p2,"p3":p3})
    return curves


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — OCR + DIMENSION BINDING + CALIBRATION
# ════════════════════════════════════════════════════════════════════════════════

def ocr_with_positions(img):
    """Tesseract OCR with positional tokens, dim extraction."""
    result = {"tokens":[], "dims":{}, "raw":""}
    if not HAS_OCR or not HAS_PIL or img is None: return result
    try:
        h, w = img.shape[:2]
        scale = max(1.0, 2400 / max(w, h))
        up    = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.05 else img
        if scale <= 1.05: scale = 1.0
        pil  = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil, config="--psm 11 --oem 3",
                                          output_type=pytesseract.Output.DICT)
        tokens = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text or int(data["conf"][i]) < 20: continue
            tokens.append({"text":text,"conf":int(data["conf"][i]),
                            "x":int(data["left"][i]/scale),"y":int(data["top"][i]/scale),
                            "w":int(data["width"][i]/scale),"h":int(data["height"][i]/scale)})
        result["tokens"] = tokens
        combined = " ".join(t["text"] for t in tokens)
        result["raw"] = combined
        dims = {}
        all_mm = [float(m.group(1)) for m in re.finditer(r"(\d{2,4}(?:\.\d+)?)\s*(?:mm)?", combined, re.I)
                  if 10 < float(m.group(1)) < 5000]
        for m in re.finditer(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", combined):
            dims["ocr_width"] = float(m.group(1)); dims["ocr_height"] = float(m.group(2))
        diameters = [float(m.group(1)) for m in re.finditer(r"[ØøO∅]\s*(\d+\.?\d*)", combined)]
        if diameters:
            dims["ocr_diameters"] = sorted(set(diameters))
            dims["ocr_hole_dia"]  = max(diameters)
        if all_mm and "ocr_width" not in dims:
            sm = sorted(set(all_mm), reverse=True)
            if len(sm) >= 2: dims["ocr_width"], dims["ocr_height"] = sm[0], sm[1]
            elif sm:          dims["ocr_width"] = sm[0]
        result["dims"] = dims
    except Exception as e:
        result["error"] = str(e)
    return result


def bind_dimensions_to_geometry(ocr_result, circles, img_shape, dpi):
    """Attach OCR-read diameter values to nearest detected circles."""
    if not circles or not ocr_result.get("tokens"): return circles
    h_img, w_img = img_shape[:2]
    for tok in ocr_result["tokens"]:
        m = re.search(r"[ØøO∅]\s*(\d+\.?\d*)", tok["text"])
        if not m: continue
        dia_mm = float(m.group(1))
        tx, ty = tok["x"] + tok["w"]/2, tok["y"] + tok["h"]/2
        best, best_d = None, float("inf")
        for c in circles:
            d = math.hypot(c["cx"]-tx, c["cy"]-ty)
            if d < best_d: best_d, best = d, c
        if best is not None and best_d < max(w_img, h_img) * 0.3:
            best["confirmed_dia_mm"] = dia_mm
            best["confirmed_r_mm"]   = dia_mm / 2.0
    return circles


def calibrate_px_to_mm(hull_data, ocr_dims, dpi):
    """Derive accurate px→mm scale using hull bbox, OCR dims, and DPI."""
    bbox  = hull_data.get("bbox_px")
    w_px  = bbox[2] if bbox else 0
    h_px  = bbox[3] if bbox else 0
    w_ocr = ocr_dims.get("ocr_width",  0) or 0
    h_ocr = ocr_dims.get("ocr_height", 0) or 0
    sx = w_px / w_ocr if w_ocr > 50 and w_px > 10 else dpi / 25.4
    sy = h_px / h_ocr if h_ocr > 50 and h_px > 10 else dpi / 25.4
    scale = (sx + sy) / 2.0
    w_mm  = round(w_px / scale, 2) if scale > 0 else (w_ocr or 200)
    h_mm  = round(h_px / scale, 2) if scale > 0 else (h_ocr or 150)
    if w_ocr > 50: w_mm = w_ocr
    if h_ocr > 50: h_mm = h_ocr
    return round(scale, 6), w_mm, h_mm


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — GEMINI VISION ANALYSIS  (unchanged API surface)
# ════════════════════════════════════════════════════════════════════════════════

def gemini_vision_analysis(image_path, cv_data):
    """
    Google Gemini vision analysis with full CV pre-analysis context.
    Identical external API to v4.0; enhanced prompt for higher accuracy.
    """
    if not HAS_AI: return {}
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return {}
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)

        with open(image_path, "rb") as f: img_bytes = f.read()
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                "bmp":"image/bmp","tiff":"image/tiff","pdf":"application/pdf"}.get(ext,"image/jpeg")

        circles_hint = json.dumps(cv_data.get("circles",[])[:10])
        lines_hint   = json.dumps([
            {"angle":l.get("angle"),"is_h":l.get("is_horizontal"),"is_v":l.get("is_vertical")}
            for l in cv_data.get("lines",[])[:12]
        ])
        corners_hint = json.dumps(cv_data.get("corners",[])[:12])
        ocr_hint     = json.dumps(cv_data.get("ocr_dims",{}))

        prompt = f"""You are an expert mechanical / sheet-metal CAD engineer.
Analyse this hand-drawn engineering sketch and extract ALL dimensions and feature positions precisely.

OpenCV pre-analysis context (use this to anchor your measurements):
  Image: {cv_data.get("img_w",0)}×{cv_data.get("img_h",0)}px @ {cv_data.get("dpi",96):.0f}dpi
  Estimated size: {cv_data.get("width_mm",0):.1f}×{cv_data.get("height_mm",0):.1f}mm
  CV circles (pixel coords + radii): {circles_hint}
  CV lines (angles): {lines_hint}
  CV corners: {corners_hint}
  OCR dimensions: {ocr_hint}

Rules:
- widthMM / heightMM must be the OUTER boundary of the part, not internal features.
- For each circle/hole include exact cx_mm, cy_mm measured from the top-left corner.
- If OCR dimensions conflict with your visual estimate, prefer the OCR value when plausible.
- Report confidence 0.0–1.0 honestly.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "profileType": "sheet metal",
  "widthMM": <number>, "heightMM": <number>, "thicknessMM": <number or null>,
  "estimatedMaterial": "aluminum|steel|stainless|brass|copper|titanium|unknown",
  "toleranceClass": "fine (±0.05mm)|medium (±0.1mm)|coarse (±0.5mm)|general (±1mm)",
  "confidence": <0.0-1.0>,
  "engineeringNotes": "<brief>",
  "circles": [{{"label":"","cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>,"type":"large_hole|small_hole|cutout"}}],
  "smallHoles": [{{"cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>,"spacing_mm":<s>}}],
  "bendLines": <int>, "slots": <int>,
  "dimensions_confirmed": {{
    "overall_width_mm":<n>,"overall_height_mm":<n>,
    "col_spacing_mm":<n>,"row_spacing_mm":<n>,
    "margin_left_mm":<n>,"margin_top_mm":<n>
  }}
}}"""

        image_part = genai_types.Part.from_bytes(data=img_bytes, mime_type=mime)
        text_part  = genai_types.Part.from_text(text=prompt)
        response   = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[genai_types.Content(parts=[image_part, text_part], role="user")],
        )
        text = response.text or ""
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# Alias kept for server.js compatibility
claude_vision_analysis = gemini_vision_analysis


def ai_dxf_interaction(instruction, current_analysis, image_path=None):
    """
    Natural-language DXF correction via Gemini.
    Identical signature to v4.0.
    """
    if not HAS_AI: return current_analysis, "AI not available"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return current_analysis, "No GEMINI_API_KEY set"
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)

        clean = {k:v for k,v in current_analysis.items() if not k.startswith("_")}
        prompt = f"""You are a CAD engineer modifying a DXF design.
Current analysis (JSON):
{json.dumps(clean, indent=2)}

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
Apply ONLY changes relevant to the instruction. Return ONLY JSON."""

        parts = [genai_types.Part.from_text(text=prompt)]
        if image_path and os.path.exists(str(image_path)):
            ext  = Path(image_path).suffix.lower().lstrip(".")
            mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}.get(ext,"image/jpeg")
            with open(image_path,"rb") as f: parts.insert(0, genai_types.Part.from_bytes(data=f.read(), mime_type=mime))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[genai_types.Content(parts=parts, role="user")],
        )
        text = response.text or ""
        text = re.sub(r"```[a-z]*","",text).strip().strip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        changes     = json.loads(text)
        explanation = changes.pop("explanation","Changes applied")
        updated     = {**current_analysis, **{k:v for k,v in changes.items() if v is not None}}
        return updated, explanation
    except Exception as e:
        return current_analysis, f"Error: {str(e)}"


def merge_analysis(ai_data, ocr_data, w_mm, h_mm, circles, all_lines, corners, opts):
    """Merge AI, OCR, and CV data into a single analysis dict."""
    def pick(key, fallback):
        v = ai_data.get(key)
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
        dias = [c.get("diameter_mm",0) for c in ai_circles if c.get("diameter_mm",0) > 0]
        hole_dia = round(float(np.mean(dias)) if dias and HAS_CV else 6.0, 2)
    if not hole_dia:
        hole_dia = ocr_data.get("dims",{}).get("ocr_hole_dia", 0.0)

    n_bends = int(pick("bendLines", max(0, len([l for l in all_lines if l.get("is_horizontal")]) // 10)))
    return {
        "width":         round(final_w, 2),
        "height":        round(final_h, 2),
        "thickness":     round(float(pick("thicknessMM", opts.get("thickness", 2.0)) or 2.0), 2),
        "holes":         int(n_holes),
        "holesDiameter": round(float(hole_dia), 2),
        "bendLines":     int(n_bends),
        "edges":         len(all_lines),
        "slots":         int(pick("slots", 0)),
        "cutouts":       int(pick("cutoutCount", 0)),
        "profileType":   str(pick("profileType","sheet metal")),
        "tolerance":     str(pick("toleranceClass","±0.1mm")),
        "material":      str(pick("estimatedMaterial","unknown")),
        "confidence":    round(float(pick("confidence", 0.82)) * 100, 1),
        "notes":         str(ai_data.get("engineeringNotes","")),
        "rawText":       json.dumps(ocr_data.get("dims",{})),
        "corners":       len(corners),
        "linesDetected": len(all_lines),
        "_ai_circles":   ai_circles,
        "_sm_holes":     sm_holes,
        "_dim_confirmed":dim_conf,
        "_ocr_circles":  circles,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — DXF BUILD (ezdxf)
# ════════════════════════════════════════════════════════════════════════════════

def build_dxf_ezdxf(analysis, dpi, image_path):
    """Build a precise DXF from merged analysis data."""
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
        if name not in doc.layers: doc.layers.add(name, color=color, linetype=ltype)
    for ln,col in [("OUTLINE",1),("HOLES",4),("SMALL_HOLES",3),("SLOTS",5),
                   ("BEND_LINES",2),("CENTRE_LINES",6),("DIMENSIONS",7),
                   ("TITLE_BLOCK",7),("NOTES",8),("SKELETON",9),("VECTORS",10)]:
        add_layer(ln, col)
    for lt, pat in [("DASHED","A,.5,-.25"),("CENTER","A,1.25,-.25,.25,-.25")]:
        if lt not in doc.linetypes:
            try: doc.linetypes.add(lt, pattern=pat)
            except Exception: pass

    entity_count = 0

    # Outline rectangle
    for p1, p2 in [((0,0),(W,0)),((W,0),(W,H)),((W,H),(0,H)),((0,H),(0,0))]:
        msp.add_line(p1, p2, dxfattribs={"layer":"OUTLINE","color":1,"lineweight":50})
        entity_count += 1

    # Circles
    ai_circles  = analysis.get("_ai_circles", [])
    ocr_circles = analysis.get("_ocr_circles", [])
    def px2mm(px): return round(px * 25.4 / dpi, 3)

    placed = []
    if ai_circles:
        for c in ai_circles:
            cx = float(c.get("cx_mm",0)); cy_dxf = H - float(c.get("cy_mm",0))
            d  = float(c.get("diameter_mm", analysis.get("holesDiameter",6) or 6)); r = d/2.0
            layer = "HOLES" if d >= 20 else "SMALL_HOLES"
            msp.add_circle((cx,cy_dxf), r, dxfattribs={"layer":layer,"color":4 if layer=="HOLES" else 3})
            cross = r*1.6
            msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            entity_count += 3; placed.append((cx,cy_dxf,r))
    elif ocr_circles:
        for c in ocr_circles:
            cx_mm  = px2mm(c["cx"]); cy_dxf = H - px2mm(c["cy"])
            r_mm   = float(c.get("confirmed_r_mm", px2mm(c["r"])))
            msp.add_circle((cx_mm,cy_dxf), r_mm, dxfattribs={"layer":"HOLES","color":4})
            cross = r_mm*1.6
            msp.add_line((cx_mm-cross,cy_dxf),(cx_mm+cross,cy_dxf), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            msp.add_line((cx_mm,cy_dxf-cross),(cx_mm,cy_dxf+cross), dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            entity_count += 3; placed.append((cx_mm,cy_dxf,r_mm))
    else:
        n = analysis.get("holes",0); r = (analysis.get("holesDiameter",6) or 6)/2.0
        if n > 0:
            sp = W/(n+1)
            for i in range(n):
                cx = sp*(i+1); cy_dxf = H/2.0
                msp.add_circle((cx,cy_dxf),r,dxfattribs={"layer":"HOLES","color":4})
                cross = r*1.6
                msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf),dxfattribs={"layer":"CENTRE_LINES","color":6})
                msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross),dxfattribs={"layer":"CENTRE_LINES","color":6})
                entity_count += 3; placed.append((cx,cy_dxf,r))

    # Small holes
    for sh in analysis.get("_sm_holes",[]):
        cx = float(sh.get("cx_mm",0)); d = float(sh.get("diameter_mm",10) or 10); r = d/2.0
        cy_dxf = H - float(sh.get("cy_mm",0))
        msp.add_circle((cx,cy_dxf),r,dxfattribs={"layer":"SMALL_HOLES","color":3})
        cross = r*1.8
        msp.add_line((cx-cross,cy_dxf),(cx+cross,cy_dxf),dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
        msp.add_line((cx,cy_dxf-cross),(cx,cy_dxf+cross),dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
        entity_count += 3

    # Bend lines
    n_bends = analysis.get("bendLines",0)
    for i in range(n_bends):
        yp = H*(i+1)/(n_bends+1)
        msp.add_line((0,yp),(W,yp),dxfattribs={"layer":"BEND_LINES","color":2,"linetype":"DASHED"})
        entity_count += 1

    # Dimensions
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

    # Diameter labels
    for (cx,cy_dxf,r) in placed:
        try:
            msp.add_text(f"Ø{r*2:.1f}",dxfattribs={"layer":"DIMENSIONS","height":3.5,"insert":(cx,cy_dxf-r-8),"halign":1,"valign":0})
            entity_count += 1
        except Exception: pass

    # Title block
    hole_d = analysis.get("holesDiameter",6) or 6
    tb_x, tb_y = W+25, 0
    for tx,ty,text,ht in [
        (tb_x,tb_y+60,f"PART: {analysis.get('profileType','PART').upper()}",5.0),
        (tb_x,tb_y+50,f"W × H: {W:.1f} × {H:.1f} mm",3.5),
        (tb_x,tb_y+42,f"THICKNESS: {analysis.get('thickness',2.0):.1f} mm",3.5),
        (tb_x,tb_y+34,f"CIRCLES: {len(ai_circles or ocr_circles)} × Ø{hole_d:.1f}",3.5),
        (tb_x,tb_y+26,f"MATERIAL: {analysis.get('material','—')}",3.5),
        (tb_x,tb_y+18,f"TOLERANCE: {analysis.get('tolerance','±0.1mm')}",3.5),
        (tb_x,tb_y+10,f"CONFIDENCE: {analysis.get('confidence',0)}%",3.0),
        (tb_x,tb_y+ 2,"SheetForge v5.0 — Gemini+OpenCV DXF",2.5),
    ]:
        msp.add_text(text,dxfattribs={"layer":"TITLE_BLOCK","height":ht,"insert":(tx,ty)})
        entity_count += 1

    if analysis.get("notes"):
        msp.add_text(f"NOTE: {str(analysis['notes'])[:120]}",
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


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 6 — EXPORTS  (SVG inline viewer, PDF, svgwrite, matplotlib)
# ════════════════════════════════════════════════════════════════════════════════

def render_svg_preview(analysis, dpi):
    """
    Inline SVG for <div class="dwg-viewer" id="dwg-main-viewer">.
    Renders actual DXF geometry: outline, circles, bend lines, dimensions.
    """
    W = float(analysis.get("width",  200))
    H = float(analysis.get("height", 150))
    scale = min(700 / max(W,1), 520 / max(H,1))
    sw, sh = W*scale, H*scale
    pad = 32

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-pad} {-pad} {sw+pad*2+90} {sh+pad*2+50}" '
        f'width="{sw+pad*2+90:.1f}" height="{sh+pad*2+50:.1f}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<defs>'
        '<pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">'
        '<circle cx="10" cy="10" r="0.7" fill="#1e2d40"/></pattern>'
        '</defs>',
        f'<rect x="0" y="0" width="{sw:.1f}" height="{sh:.1f}" fill="url(#grid)"/>',
        f'<rect x="0" y="0" width="{sw:.1f}" height="{sh:.1f}" fill="none" stroke="#3d7eff" stroke-width="2.5"/>',
    ]

    def px2mm(px): return round(px * 25.4 / dpi, 3)

    def draw_circle(cx_mm, cy_mm, r_mm, color, label=None):
        cx = cx_mm*scale; cy = cy_mm*scale; r = max(r_mm*scale, 2)
        cross = r*1.55
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        svg.append(f'<line x1="{cx-cross:.2f}" y1="{cy:.2f}" x2="{cx+cross:.2f}" y2="{cy:.2f}" stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        svg.append(f'<line x1="{cx:.2f}" y1="{cy-cross:.2f}" x2="{cx:.2f}" y2="{cy+cross:.2f}" stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        if label:
            svg.append(f'<text x="{cx:.2f}" y="{cy+r+11:.2f}" fill="#6b7a9b" font-size="9" text-anchor="middle" font-family="monospace">{label}</text>')

    ai_circles  = analysis.get("_ai_circles",  [])
    ocv_circles = analysis.get("_ocr_circles", [])
    sm_holes    = analysis.get("_sm_holes",    [])

    if ai_circles:
        for c in ai_circles:
            d = float(c.get("diameter_mm",0) or 0)
            draw_circle(float(c.get("cx_mm",0)), float(c.get("cy_mm",0)), d/2,
                        "#00d4a0" if d >= 20 else "#22c55e", f"Ø{d:.0f}")
    elif ocv_circles:
        for c in ocv_circles:
            cx = px2mm(c["cx"]); cy = px2mm(c["cy"])
            r  = float(c.get("confirmed_r_mm", px2mm(c["r"])))
            draw_circle(cx, cy, r, "#00d4a0", f"Ø{r*2:.0f}")
    for sh in sm_holes:
        r = float(sh.get("diameter_mm",10) or 10)/2.0
        draw_circle(float(sh.get("cx_mm",0)), float(sh.get("cy_mm",0)), r, "#22c55e", f"Ø{r*2:.0f}")

    bends = analysis.get("bendLines",0)
    for i in range(bends):
        yp = sh * (i+1)/(bends+1)
        svg.append(f'<line x1="0" y1="{yp:.2f}" x2="{sw:.2f}" y2="{yp:.2f}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="8,4"/>')

    # Width dimension
    svg.append(f'<line x1="0" y1="{sh+14:.2f}" x2="{sw:.2f}" y2="{sh+14:.2f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw/2:.2f}" y="{sh+26:.2f}" fill="#6b7a9b" font-size="11" text-anchor="middle" font-family="monospace">{W:.1f} mm</text>')
    # Height dimension
    svg.append(f'<line x1="{sw+14:.2f}" y1="0" x2="{sw+14:.2f}" y2="{sh:.2f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw+27:.2f}" y="{sh/2:.2f}" fill="#6b7a9b" font-size="11" text-anchor="middle" font-family="monospace" '
               f'transform="rotate(90,{sw+27:.2f},{sh/2:.2f})">{H:.1f} mm</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def export_pdf_reportlab(analysis, output_path):
    """
    Export a detailed PDF — this is what gets displayed in the DWG viewer.
    Renders exact DXF geometry: outline, circles with centre-lines, bend lines,
    title block, dimension annotations.
    """
    if not HAS_RL: return False
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, black, white

        W = float(analysis.get("width",  200))
        H = float(analysis.get("height", 150))

        # Page layout: drawing + margins + title block strip
        margin   = 20*mm
        tb_width = 60*mm
        page_w   = W*mm + 2*margin + tb_width
        page_h   = H*mm + 2*margin + 15*mm   # extra for dimension below

        c = rl_canvas.Canvas(str(output_path), pagesize=(page_w, page_h))
        origin_x = margin
        origin_y = margin + 12*mm   # room for bottom dimension text

        def dxf_x(x_mm): return origin_x + x_mm*mm
        def dxf_y(y_mm): return origin_y + y_mm*mm   # DXF Y = PDF Y here (already flipped in analysis)

        # Background
        c.setFillColor(HexColor("#0d1117"))
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        # Grid dots
        c.setFillColor(HexColor("#1e2d40"))
        grid_step = 10*mm
        gx = origin_x
        while gx <= origin_x + W*mm:
            gy = origin_y
            while gy <= origin_y + H*mm:
                c.circle(gx, gy, 0.7*mm, fill=1, stroke=0)
                gy += grid_step
            gx += grid_step

        # Outline
        c.setStrokeColor(HexColor("#3d7eff"))
        c.setLineWidth(2)
        c.rect(dxf_x(0), dxf_y(0), W*mm, H*mm, fill=0)

        # Circles
        ai_circles  = analysis.get("_ai_circles",  [])
        ocr_circles = analysis.get("_ocr_circles", [])
        sm_holes    = analysis.get("_sm_holes",    [])

        def draw_circle_pdf(cx_mm, cy_mm, r_mm, color_hex, label=None):
            cx = dxf_x(cx_mm); cy = dxf_y(cy_mm); r = max(r_mm*mm, 1.5*mm)
            cross = r*1.55
            c.setStrokeColor(HexColor(color_hex)); c.setLineWidth(1.5)
            c.circle(cx, cy, r, fill=0)
            c.setStrokeColor(HexColor("#a855f7")); c.setLineWidth(0.6)
            c.setDash(3, 2)
            c.line(cx-cross, cy, cx+cross, cy)
            c.line(cx, cy-cross, cx, cy+cross)
            c.setDash()
            if label:
                c.setFont("Helvetica", 6); c.setFillColor(HexColor("#6b7a9b"))
                c.drawCentredString(cx, cy-r-4*mm, label)
                c.setFillColor(white)

        if ai_circles:
            for ci in ai_circles:
                d = float(ci.get("diameter_mm",0) or 0)
                draw_circle_pdf(float(ci.get("cx_mm",0)), float(ci.get("cy_mm",0)), d/2,
                                "#00d4a0" if d >= 20 else "#22c55e", f"Ø{d:.0f}")
        elif ocr_circles:
            def px2mm(px): return round(px * 25.4 / float(analysis.get("_dpi",96)), 3)
            for ci in ocr_circles:
                cx = float(ci["cx"]) * 25.4 / 96; cy = float(ci["cy"]) * 25.4 / 96
                r  = float(ci.get("confirmed_r_mm", float(ci["r"]) * 25.4 / 96))
                draw_circle_pdf(cx, cy, r, "#00d4a0", f"Ø{r*2:.0f}")
        for sh in sm_holes:
            r = float(sh.get("diameter_mm",10) or 10)/2.0
            draw_circle_pdf(float(sh.get("cx_mm",0)), float(sh.get("cy_mm",0)), r, "#22c55e", f"Ø{r*2:.0f}")

        # Bend lines
        n_bends = analysis.get("bendLines", 0)
        c.setStrokeColor(HexColor("#f59e0b")); c.setLineWidth(1.2); c.setDash(6, 3)
        for i in range(n_bends):
            yp = H * (i+1)/(n_bends+1)
            c.line(dxf_x(0), dxf_y(yp), dxf_x(W), dxf_y(yp))
        c.setDash()

        # Width dimension
        c.setStrokeColor(HexColor("#6b7a9b")); c.setLineWidth(0.7)
        dim_y = dxf_y(0) - 9*mm
        c.line(dxf_x(0), dim_y, dxf_x(W), dim_y)
        c.setFont("Helvetica", 7); c.setFillColor(HexColor("#6b7a9b"))
        c.drawCentredString(dxf_x(W/2), dim_y - 4*mm, f"{W:.1f} mm")

        # Height dimension
        dim_x = dxf_x(W) + 8*mm
        c.line(dim_x, dxf_y(0), dim_x, dxf_y(H))
        c.saveState()
        c.translate(dim_x + 4*mm, dxf_y(H/2))
        c.rotate(90)
        c.drawCentredString(0, 0, f"{H:.1f} mm")
        c.restoreState()

        # Title block (right strip)
        tb_x = origin_x + W*mm + 8*mm
        tb_y = dxf_y(H) - 5*mm
        c.setFillColor(HexColor("#111827"))
        c.rect(tb_x-3*mm, dxf_y(0)-2*mm, tb_width, H*mm+4*mm, fill=1, stroke=0)
        c.setStrokeColor(HexColor("#1e3a5f")); c.setLineWidth(0.5)
        c.rect(tb_x-3*mm, dxf_y(0)-2*mm, tb_width, H*mm+4*mm, fill=0)

        hole_d = analysis.get("holesDiameter",6) or 6
        lines_tb = [
            ("Helvetica-Bold", 8, HexColor("#3d7eff"), f"{analysis.get('profileType','PART').upper()}"),
            ("Helvetica", 7, HexColor("#e2e8f0"), f"W × H: {W:.1f} × {H:.1f} mm"),
            ("Helvetica", 7, HexColor("#e2e8f0"), f"Thickness: {analysis.get('thickness',2.0):.1f} mm"),
            ("Helvetica", 7, HexColor("#e2e8f0"), f"Holes: {len(ai_circles or ocr_circles)} × Ø{hole_d:.1f}"),
            ("Helvetica", 7, HexColor("#e2e8f0"), f"Material: {analysis.get('material','—')}"),
            ("Helvetica", 7, HexColor("#e2e8f0"), f"Tolerance: {analysis.get('tolerance','±0.1mm')}"),
            ("Helvetica", 7, HexColor("#94a3b8"), f"Confidence: {analysis.get('confidence',0):.0f}%"),
            ("Helvetica", 6, HexColor("#475569"), "SheetForge v5.0"),
        ]
        ty = tb_y
        for font, size, color, text in lines_tb:
            c.setFont(font, size); c.setFillColor(color)
            c.drawString(tb_x, ty, text)
            ty -= (size + 3)*mm

        c.save()
        return True
    except Exception:
        return False


def export_svg_svgwrite(analysis, output_path):
    if not HAS_SVG: return False
    try:
        import svgwrite
        W = float(analysis.get("width",200)); H = float(analysis.get("height",150)); sc=2.0
        dwg = svgwrite.Drawing(str(output_path), size=(f"{W*sc}mm",f"{H*sc}mm"),
                               viewBox=f"0 0 {W*sc} {H*sc}")
        dwg.add(dwg.rect((0,0),(W*sc,H*sc), fill="none", stroke="#3d7eff", stroke_width=2))
        for ci in analysis.get("_ai_circles",[]):
            cx = float(ci.get("cx_mm",0))*sc; cy = float(ci.get("cy_mm",0))*sc
            r  = float(ci.get("diameter_mm",6))/2*sc
            dwg.add(dwg.circle(center=(cx,cy),r=r,fill="none",stroke="#00d4a0",stroke_width=1.5))
        dwg.save(); return True
    except Exception: return False


def export_matplotlib_dxf(analysis, output_path):
    if not HAS_MPL: return False
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        W = float(analysis.get("width",200)); H = float(analysis.get("height",150))
        fig, ax = plt.subplots(figsize=(max(6,W/30),max(4,H/30)), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.add_patch(patches.Rectangle((0,0),W,H,lw=2,ec="#3d7eff",fc="#111827"))
        for ci in analysis.get("_ai_circles",[]):
            cx = float(ci.get("cx_mm",0)); cy = H - float(ci.get("cy_mm",0))
            r  = float(ci.get("diameter_mm",6))/2
            ax.add_patch(patches.Circle((cx,cy),r,lw=1.5,ec="#00d4a0",fc="none"))
        ax.set_xlim(-W*0.1,W*1.35); ax.set_ylim(-H*0.2,H*1.2); ax.set_aspect("equal")
        ax.tick_params(colors="#6b7a9b"); ax.set_title(f"SheetForge DXF — {W:.0f}×{H:.0f}mm",color="#e2e8f0")
        plt.tight_layout()
        plt.savefig(str(output_path),dpi=150,bbox_inches="tight",facecolor="#0d1117",edgecolor="none")
        plt.close(); return True
    except Exception: return False


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 7 — GCODE GENERATION  (unchanged from v4.0)
# ════════════════════════════════════════════════════════════════════════════════

def generate_gcode(analysis, options=None):
    opts       = options or {}
    W          = float(analysis.get("width",  200))
    H          = float(analysis.get("height", 150))
    feed_rate  = float(opts.get("feedRate",   1000))
    plunge_rate= float(opts.get("plungeRate", 300))
    spindle_rpm= int(opts.get("spindleRpm",   12000))
    cut_depth  = float(opts.get("cutDepth",   3.0))
    pass_depth = float(opts.get("passDepth",  1.0))
    safe_z     = float(opts.get("safeZ",      5.0))
    tool_dia   = float(opts.get("toolDiameter",3.0))
    operation  = opts.get("operation","cut")
    ai_circles = analysis.get("_ai_circles",[])
    sm_holes   = analysis.get("_sm_holes",[])
    thickness  = float(analysis.get("thickness",2.0))
    part_name  = analysis.get("profileType","PART")
    material   = analysis.get("material","unknown")
    tolerance  = analysis.get("tolerance","±0.1mm")
    ts         = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"; SheetForge v5.0 — G-Code Export",
        f"; Generated: {ts}",
        f"; Part: {part_name.upper()} | Material: {material}",
        f"; Dimensions: {W:.2f} × {H:.2f} mm | Thickness: {thickness:.2f}mm",
        f"; Tolerance: {tolerance} | Operation: {operation}",
        f"; Tool Diameter: {tool_dia}mm | Feed: {feed_rate}mm/min | Spindle: {spindle_rpm}rpm",
        f"; =====================================================",
        "","G21","G17","G90","G94","G40","G49","",
        f"T01 M6","G43 H01",f"S{spindle_rpm} M3","G4 P2000","",
        f"G00 Z{safe_z:.2f}","G00 X0.000 Y0.000","",
    ]

    if operation in ("cut","laser"):
        lines += [f"; === OUTLINE ===","G00 X0.000 Y0.000",f"G00 Z{safe_z:.2f}"]
        for p in range(1, math.ceil(cut_depth/pass_depth)+1):
            z = min(-p*pass_depth, -cut_depth)
            lines += [
                f"","G00 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}",
                f"G01 Z{z:.3f} F{plunge_rate:.0f}",
                f"G01 X{W+tool_dia/2:.3f} Y0.000 F{feed_rate:.0f}",
                f"G01 X{W+tool_dia/2:.3f} Y{H+tool_dia/2:.3f}",
                f"G01 X-{tool_dia/2:.3f} Y{H+tool_dia/2:.3f}",
                f"G01 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}",
                f"G00 Z{safe_z:.2f}",
            ]

    if operation in ("cut","drill"):
        for i, ci in enumerate(ai_circles):
            cx = float(ci.get("cx_mm",0)); cy = H-float(ci.get("cy_mm",0))
            d  = float(ci.get("diameter_mm",6)); r = d/2
            lines += [f"","G00 X{cx:.3f} Y{cy:.3f}",f"G00 Z{safe_z:.2f}"]
            if d <= tool_dia*1.1:
                lines += [f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}","G80"]
            elif r-tool_dia/2 > 0:
                ar = r-tool_dia/2
                for p in range(1,math.ceil(cut_depth/pass_depth)+1):
                    z = min(-p*pass_depth,-cut_depth)
                    lines += [f"G00 X{cx+ar:.3f} Y{cy:.3f}",
                              f"G01 Z{z:.3f} F{plunge_rate:.0f}",
                              f"G02 X{cx+ar:.3f} Y{cy:.3f} I-{ar:.3f} J0.000 F{feed_rate:.0f}",
                              f"G00 Z{safe_z:.2f}"]
        for sh in sm_holes:
            cx = float(sh.get("cx_mm",0)); cy = H-float(sh.get("cy_mm",0))
            lines += [f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}","G80"]

    n_bends = analysis.get("bendLines",0)
    if n_bends:
        lines.append("; === BEND LINES ===")
        for i in range(n_bends):
            yp = H*(i+1)/(n_bends+1)
            lines += [f"G00 X0.000 Y{yp:.3f}",f"G01 Z-0.300 F{plunge_rate//2:.0f}",
                      f"G01 X{W:.3f} Y{yp:.3f} F{feed_rate//2:.0f}",f"G00 Z{safe_z:.2f}"]

    lines += ["",f"G00 Z{safe_z:.2f}","G00 X0.000 Y0.000","M5","M30"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except Exception:
        opts = {}

    # ── AI interact mode (unchanged) ──────────────────────────────────────────
    if opts.get("mode") == "ai_interact":
        instruction      = opts.get("instruction","")
        current_analysis = opts.get("analysis",{})
        updated, explanation = ai_dxf_interaction(instruction, current_analysis, image_path)
        doc, entity_count = build_dxf_ezdxf(updated, float(opts.get("dpi",96)), image_path)
        dxf_str = ""; file_size = 0
        if doc:
            try:
                out_dir  = Path(image_path).parent if image_path and os.path.exists(image_path) else Path("/tmp")
                dxf_path = out_dir / f"interact_{int(time.time())}.dxf"
                doc.saveas(str(dxf_path))
                file_size = dxf_path.stat().st_size
                with open(dxf_path) as f: dxf_str = f.read()
            except Exception: pass
        print(json.dumps({
            "mode":"ai_interact",
            "analysis":{k:v for k,v in updated.items() if not k.startswith("_")},
            "explanation":explanation,
            "dwg":{"entities":entity_count,"fileSize":file_size},
            "gcode":generate_gcode(updated, opts.get("gcodeOptions")),
        }, ensure_ascii=False))
        return

    steps = []
    dpi   = float(opts.get("dpi", 96))

    # ── PRE-1: Load ────────────────────────────────────────────────────────────
    t0 = now_ms()
    img, gray, detected_dpi = load_image(image_path)
    if detected_dpi > 1: dpi = detected_dpi
    img_h, img_w = (img.shape[:2] if img is not None else (0,0))
    steps.append(step_record("PRE-1: Image Ingestion", f"{img_w}×{img_h}px @ {dpi:.0f}dpi", t0))

    # ── PRE-2: Single-pass preprocessing ──────────────────────────────────────
    t0 = now_ms()
    binary, gray_clean = preprocess(img, gray)
    steps.append(step_record("PRE-2: Preprocess (GaussianBlur+CLAHE+adaptiveThreshold+morphologyEx)", "One-pass preprocessing", t0))

    # ── PRE-3: Deskew ──────────────────────────────────────────────────────────
    t0 = now_ms()
    img_ds, gray_ds, angle = deskew(img, gray_clean if gray_clean is not None else gray)
    steps.append(step_record("PRE-3: Deskew (HoughLines)", f"Corrected {angle:.2f}°", t0))

    # After deskew, redo binary on the corrected gray
    if abs(angle) > 0.3 and gray_ds is not None:
        binary, gray_ds = preprocess(img_ds, gray_ds)

    # ── CV-1: Canny edges ─────────────────────────────────────────────────────
    t0 = now_ms()
    edges = None
    if gray_ds is not None and HAS_CV:
        blurred = cv2.GaussianBlur(gray_ds, (5,5), 0)
        e1 = cv2.Canny(blurred, 30, 100)
        e2 = cv2.Canny(blurred, 60, 150)
        edges = cv2.bitwise_or(e1, e2)
        edges = cv2.dilate(edges, np.ones((2,2),np.uint8), iterations=1)
    n_edge = int(np.count_nonzero(edges)) if (edges is not None and HAS_CV) else 0
    steps.append(step_record("CV-1: Canny Edge Detection (multi-scale)", f"{n_edge} edge pixels", t0))

    # ── CV-2: Contours ────────────────────────────────────────────────────────
    t0 = now_ms()
    simplified_contours, contours_tree, contours_ext = cv_contours = extract_contours(binary)
    steps.append(step_record("CV-2: Contour Extraction (findContours+approxPolyDP)", f"{len(simplified_contours)} contours", t0))

    # ── CV-3: Shape classification ────────────────────────────────────────────
    t0 = now_ms()
    shapes = classify_shapes(simplified_contours)
    circ_count = sum(1 for s in shapes if s["type"] in ("circle","ellipse"))
    steps.append(step_record("CV-3: Shape Classification (matchShapes+fitEllipse+convexHull)", f"{len(shapes)} shapes, {circ_count} circular", t0))

    # ── CV-4: Lines ───────────────────────────────────────────────────────────
    t0 = now_ms()
    all_lines = extract_lines(edges, gray_ds)
    steps.append(step_record("CV-4: Line Fitting (HoughLines+HoughLinesP)", f"{len(all_lines)} lines", t0))

    # ── CV-5: Circles ─────────────────────────────────────────────────────────
    t0 = now_ms()
    circles = extract_circles(gray_ds)
    steps.append(step_record("CV-5: Circle Fitting (HoughCircles ×2 passes)", f"{len(circles)} circles", t0))

    # ── CV-6: Corners ─────────────────────────────────────────────────────────
    t0 = now_ms()
    corners = detect_corners(gray_ds) if gray_ds is not None else []
    steps.append(step_record("CV-6: Corner Detection (goodFeaturesToTrack)", f"{len(corners)} corners", t0))

    # ── CV-7: Skeletonization ─────────────────────────────────────────────────
    t0 = now_ms()
    skeleton = skeletonize(binary) if binary is not None else None
    skel_px  = int(np.count_nonzero(skeleton)) if (skeleton is not None and HAS_CV) else 0
    method   = "cv2.ximgproc.thinning" if HAS_XIMGPROC else "iterative erosion"
    steps.append(step_record(f"CV-7: Skeletonization ({method})", f"{skel_px} skeleton pixels", t0))

    # ── CV-8: Vector paths (Bezier) ───────────────────────────────────────────
    t0 = now_ms()
    hull_data   = hull_analysis(contours_tree if contours_tree else simplified_contours)
    scale_px_mm, W_est, H_est = calibrate_px_to_mm(hull_data, {}, dpi)
    vector_paths = build_vector_paths(simplified_contours, circles, dpi)
    steps.append(step_record("CV-8: Vector Paths + Bezier Fitting", f"{len(vector_paths)} paths", t0))

    # ── OCR + dimension binding ───────────────────────────────────────────────
    t0 = now_ms()
    ocr_result = ocr_with_positions(img_ds if img_ds is not None else img)
    if img_ds is not None and HAS_CV:
        circles = bind_dimensions_to_geometry(ocr_result, circles, img_ds.shape, dpi)
    steps.append(step_record("OCR: Tesseract Dimension Extraction", f"{len(ocr_result.get('tokens',[]))} tokens", t0))

    # ── Calibration ───────────────────────────────────────────────────────────
    t0 = now_ms()
    scale_px_mm, w_mm, h_mm = calibrate_px_to_mm(hull_data, ocr_result.get("dims",{}), dpi)
    steps.append(step_record("CAL: Pixel→mm Calibration", f"Scale={scale_px_mm:.3f}px/mm  W={w_mm:.1f} H={h_mm:.1f}mm", t0))

    # ── Gemini Vision ─────────────────────────────────────────────────────────
    t0 = now_ms()
    cv_ctx = {
        "circles": circles[:12], "lines": all_lines[:20], "corners": corners[:20],
        "img_w": img_w, "img_h": img_h,
        "width_mm": w_mm, "height_mm": h_mm, "dpi": dpi,
        "ocr_dims": ocr_result.get("dims",{}),
    }
    ai_data = {}
    if image_path and os.path.exists(image_path):
        ai_data = gemini_vision_analysis(image_path, cv_ctx)
    steps.append(step_record("AI: Gemini Vision Deep Analysis", f"conf={ai_data.get('confidence',0):.2f} profile={ai_data.get('profileType','?')}", t0))

    # ── Merge ─────────────────────────────────────────────────────────────────
    t0 = now_ms()
    analysis = merge_analysis(ai_data, ocr_result, w_mm, h_mm, circles, all_lines, corners, opts)
    analysis["_dpi"] = dpi  # pass DPI to export functions
    steps.append(step_record("MERGE: Data Fusion", f"W={analysis['width']}mm H={analysis['height']}mm holes={analysis['holes']}", t0))

    # ── DXF Build ─────────────────────────────────────────────────────────────
    t0 = now_ms()
    doc, entity_count = build_dxf_ezdxf(analysis, dpi, image_path)
    steps.append(step_record("DXF: Build (ezdxf R2018)", f"{entity_count} entities", t0))

    t0 = now_ms()
    valid, errors = validate_dxf(doc)
    steps.append(step_record("DXF: Validation", "Valid" if valid else f"Warnings: {len(errors)}", t0))

    # ── SVG Preview ───────────────────────────────────────────────────────────
    t0 = now_ms()
    svg_content = render_svg_preview(analysis, dpi)
    steps.append(step_record("PREVIEW: SVG Render (dwg-viewer)", f"W={analysis['width']}mm H={analysis['height']}mm", t0))

    # ── File exports ──────────────────────────────────────────────────────────
    t0 = now_ms()
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

    dxf_path = server_out_dir / dxf_name
    svg_path = server_out_dir / svg_name
    pdf_path = server_out_dir / pdf_name
    mpl_path = server_out_dir / mpl_name

    dxf_str = ""; file_size = 0
    if doc is not None:
        try:
            doc.saveas(str(dxf_path))
            file_size = dxf_path.stat().st_size
            with open(dxf_path) as f: dxf_str = f.read()
            if str(orig_dir/dxf_name) != str(dxf_path):
                import shutil; shutil.copy2(str(dxf_path), str(orig_dir/dxf_name))
        except Exception as e:
            dxf_str = f"; ERROR: {e}"

    with open(svg_path,"w") as f: f.write(svg_content)
    export_svg_svgwrite(analysis, server_out_dir / f"design_{ts_str}_svgwrite.svg")
    pdf_ok = export_pdf_reportlab(analysis, pdf_path)
    mpl_ok = export_matplotlib_dxf(analysis, mpl_path)

    export_detail = f"DXF {file_size//1024 if file_size else 0}KB"
    if pdf_ok: export_detail += " + PDF(reportlab)"
    if mpl_ok: export_detail += " + PNG(matplotlib)"
    steps.append(step_record("EXPORT: Multi-format (ezdxf+svgwrite+reportlab+matplotlib)", export_detail, t0))

    # ── GCode ─────────────────────────────────────────────────────────────────
    t0 = now_ms()
    gcode_str  = generate_gcode(analysis, opts.get("gcodeOptions"))
    gcode_name = f"design_{ts_str}.gcode"
    with open(server_out_dir/gcode_name,"w") as f: f.write(gcode_str)
    steps.append(step_record("GCODE: G-Code Generation", f"{len(gcode_str.splitlines())} lines", t0))

    public_analysis = {k:v for k,v in analysis.items() if not k.startswith("_")}

    print(json.dumps({
        "steps":       steps,
        "analysis":    public_analysis,
        "dwg": {
            "entities":      entity_count,
            "fileSize":      file_size,
            "filename":      dxf_name if file_size else "",
            "svgFilename":   svg_name,
            "pdfFilename":   pdf_name if pdf_ok else "",
            "gcodeFilename": gcode_name,
            "localPath":     str(orig_dir/dxf_name),
        },
        "svgContent":     svg_content,
        "dxfAvailable":   file_size > 0,
        "pdfAvailable":   pdf_ok,
        "gcodeAvailable": True,
        "gcode":          gcode_str,
        "vectorPaths":    vector_paths[:20],
        "shapes":         shapes[:20],
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error":     str(e),
            "traceback": traceback.format_exc(),
            "steps":     [],
            "analysis":  {},
            "dwg":       {"entities":0,"fileSize":0},
            "gcode":     "",
        }))
        sys.exit(1)
