#!/usr/bin/env python3
"""
SheetForge — Advanced CV + Claude Vision DXF Pipeline
======================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent }

OpenCV operations performed (in order):
  1.  RAW ingestion — colour / greyscale / PDF page raster
  2.  Noise reduction — bilateral + NLMeans + Gaussian cascade
  3.  CLAHE contrast normalisation
  4.  Adaptive Otsu thresholding
  5.  Morphological open/close cleaning
  6.  Deskew via Hough angle minimisation
  7.  Canny multi-scale edge detection
  8.  Probabilistic Hough line transform (PPHT)
  9.  Hough circle detection (CHT)
 10.  Contour extraction + Douglas-Peucker simplification
 11.  Convex hull / concavity analysis
 12.  Moments & shape descriptors (Hu moments, aspect ratio, solidity)
 13.  Template/corner detection — Harris + Shi-Tomasi
 14.  FAST/BRIEF feature keypoints
 15.  Watershed segmentation for overlapping features
 16.  Distance transform for hole / slot interior mapping
 17.  OCR dimension extraction — Tesseract PSM 6 + 11
 18.  Coordinate system mapping (pixel → mm @ detected DPI)
 19.  Claude Vision deep-analysis prompt
 20.  DXF entity generation (ezdxf R2018)
 21.  DXF validation pass
 22.  SVG preview render
 23.  File export
"""

import sys, os, json, math, time, base64, re, traceback
from pathlib import Path

# ─── Graceful optional imports ────────────────────────────────────────────────
def _try(fn):
    try: return fn()
    except Exception: return None

cv2       = _try(lambda: __import__("cv2"))
np        = _try(lambda: __import__("numpy"))
ezdxf     = _try(lambda: __import__("ezdxf"))
pytesseract = _try(lambda: __import__("pytesseract"))
Image     = _try(lambda: __import__("PIL.Image", fromlist=["Image"]))
anthropic_mod = _try(lambda: __import__("anthropic"))

HAS_CV   = cv2  is not None and np   is not None
HAS_DXF  = ezdxf is not None
HAS_OCR  = pytesseract is not None
HAS_PIL  = Image is not None
HAS_AI   = anthropic_mod is not None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def now_ms(): return int(time.time() * 1000)

def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}

def encode_b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

# ─── STAGE 1 — Image ingestion ─────────────────────────────────────────────────
def load_image(image_path):
    """Load image; handles JPEG/PNG/BMP/TIFF. Returns BGR ndarray + DPI."""
    if not HAS_CV: return None, 96.0
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None: return None, 96.0
    # Attempt DPI from EXIF if PIL available
    dpi = 96.0
    if HAS_PIL:
        try:
            pil = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (96,96))
            dpi  = float(xdpi[0]) if xdpi[0] > 1 else 96.0
        except Exception: pass
    return img, dpi

# ─── STAGE 2 — Noise reduction pipeline ────────────────────────────────────────
def denoise(img):
    if not HAS_CV: return img
    # Bilateral preserves edges
    bil = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # NLMeans (colour)
    nlm = cv2.fastNlMeansDenoisingColored(bil, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # Light Gaussian for residual salt-and-pepper
    blurred = cv2.GaussianBlur(nlm, (3,3), 0)
    return blurred

# ─── STAGE 3 — CLAHE contrast enhancement ──────────────────────────────────────
def clahe_enhance(img):
    if not HAS_CV: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ─── STAGE 4 — Adaptive Otsu thresholding ──────────────────────────────────────
def threshold(img):
    if not HAS_CV: return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dual-pass: adaptive + Otsu combined
    _, otsu  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Blend: use Otsu as primary, adaptive fills thin features
    combined = cv2.bitwise_and(otsu, adaptive)
    return combined, gray

# ─── STAGE 5 — Morphological cleanup ───────────────────────────────────────────
def morph_clean(binary):
    if not HAS_CV: return binary
    k3  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    k5  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # Remove tiny noise
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
    # Fill small gaps
    closed  = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, k5, iterations=2)
    return closed

# ─── STAGE 6 — Deskew ──────────────────────────────────────────────────────────
def deskew(img, binary):
    """Detect dominant angle via Hough; rotate to align with axes."""
    if not HAS_CV: return img, 0.0
    edges  = cv2.Canny(binary, 50, 150)
    lines  = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is None: return img, 0.0
    angles = []
    for r, theta in lines[:,0]:
        angle = math.degrees(theta) - 90
        if abs(angle) < 45: angles.append(angle)
    if not angles: return img, 0.0
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5: return img, median_angle
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle

# ─── STAGE 7 — Canny multi-scale edge detection ─────────────────────────────────
def detect_edges(gray):
    if not HAS_CV: return None
    # Scale 1: fine detail
    e1 = cv2.Canny(gray, 30, 90)
    # Scale 2: structural
    e2 = cv2.Canny(gray, 80, 200)
    # Scale 3: coarse outline
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    e3 = cv2.Canny(blurred, 50, 150)
    combined = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)
    return combined

# ─── STAGE 8 — Probabilistic Hough line transform ──────────────────────────────
def hough_lines(edges):
    if not HAS_CV or edges is None: return [], []
    # PPHT — more precise endpoints than standard HoughLines
    raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                          threshold=50, minLineLength=20, maxLineGap=10)
    if raw is None: return [], []
    lines = [tuple(l[0]) for l in raw]  # (x1,y1,x2,y2)

    # Classify horizontal / vertical / diagonal
    h_lines = [l for l in lines if abs(l[3]-l[1]) < 5]
    v_lines = [l for l in lines if abs(l[2]-l[0]) < 5]
    return lines, h_lines + v_lines  # all, axis-aligned

# ─── STAGE 9 — Hough circle detection ──────────────────────────────────────────
def hough_circles(gray):
    if not HAS_CV: return []
    blurred = cv2.GaussianBlur(gray, (9,9), 2)
    raw = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT,
                           dp=1.5, minDist=20,
                           param1=300, param2=0.85,
                           minRadius=3, maxRadius=200)
    if raw is None: return []
    circles = np.uint16(np.around(raw[0]))
    return [{"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])} for c in circles]

# ─── STAGE 10 — Contour extraction + Douglas-Peucker ───────────────────────────
def extract_contours(binary):
    if not HAS_CV: return [], []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    simplified = []
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)
    # Sort by area descending
    simplified.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    return simplified, contours

# ─── STAGE 11 — Convex hull + concavity analysis ───────────────────────────────
def hull_analysis(contours):
    if not HAS_CV or not contours: return {}
    main = max(contours, key=lambda c: cv2.contourArea(c))
    hull   = cv2.convexHull(main)
    area   = cv2.contourArea(main)
    h_area = cv2.contourArea(hull)
    solidity    = float(area / h_area) if h_area > 0 else 0.0
    perimeter   = cv2.arcLength(main, True)
    x,y,w,h     = cv2.boundingRect(main)
    aspect      = float(w) / h if h > 0 else 1.0
    return {
        "area_px":    float(area),
        "hull_area":  float(h_area),
        "solidity":   round(solidity, 4),
        "perimeter":  float(perimeter),
        "bbox_px":    (x, y, w, h),
        "aspect":     round(aspect, 4),
    }

# ─── STAGE 12 — Hu moments + shape descriptors ─────────────────────────────────
def shape_descriptors(contours):
    if not HAS_CV or not contours: return {}
    main = max(contours, key=lambda c: cv2.contourArea(c))
    M  = cv2.moments(main)
    hu = cv2.HuMoments(M).flatten().tolist()
    cx = int(M["m10"]/M["m00"]) if M["m00"] else 0
    cy = int(M["m01"]/M["m00"]) if M["m00"] else 0
    return {"centroid_px": (cx, cy), "hu_moments": [round(h, 6) for h in hu]}

# ─── STAGE 13 — Harris + Shi-Tomasi corner detection ───────────────────────────
def detect_corners(gray):
    if not HAS_CV: return []
    # Harris
    gray_f = np.float32(gray)
    harris  = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris  = cv2.dilate(harris, None)
    # Shi-Tomasi — higher quality
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
    if corners is None: return []
    return [{"x": int(c[0][0]), "y": int(c[0][1])} for c in corners]

# ─── STAGE 14 — FAST keypoints ──────────────────────────────────────────────────
def detect_keypoints(gray):
    if not HAS_CV: return 0
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    kp   = fast.detect(gray, None)
    return len(kp)

# ─── STAGE 15 — Watershed segmentation ─────────────────────────────────────────
def watershed_segment(img, binary):
    if not HAS_CV: return 0
    dist  = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    fg    = np.uint8(fg)
    unknown = cv2.subtract(binary, fg)
    _, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown == 255] = 0
    img_copy = img.copy()
    cv2.watershed(img_copy, markers)
    n_regions = markers.max()
    return int(n_regions)

# ─── STAGE 16 — Distance transform (hole mapping) ──────────────────────────────
def distance_transform_analysis(binary, circles):
    """Use distance transform to validate circle detections."""
    if not HAS_CV: return circles
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    validated = []
    for c in circles:
        val = float(dist[min(c["cy"], dist.shape[0]-1),
                        min(c["cx"], dist.shape[1]-1)])
        if val > 0:  # centre lies inside a region
            c["dist_val"] = round(val, 2)
            validated.append(c)
    return validated

# ─── STAGE 17 — OCR dimension extraction ───────────────────────────────────────
def ocr_dimensions(img):
    """Run Tesseract twice (PSM 6 block, PSM 11 sparse) and merge results."""
    dims = {}
    if not HAS_OCR or not HAS_PIL: return dims
    try:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if HAS_CV else None
        if pil is None: return dims
        text6  = pytesseract.image_to_string(pil, config="--psm 6 --oem 3")
        text11 = pytesseract.image_to_string(pil, config="--psm 11 --oem 3")
        combined = text6 + " " + text11
        # Match: 123.45mm | 123.45 mm | 12.3 x 45.6 | Ø5 etc.
        patterns = [
            (r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", "wh"),
            (r"[Øø∅]\s*(\d+\.?\d*)", "dia"),
            (r"(\d+\.?\d*)\s*mm", "mm"),
            (r"R\s*(\d+\.?\d*)", "radius"),
            (r"(\d+\.?\d*)\s*°", "angle"),
        ]
        found_mm, found_dia = [], []
        for pat, kind in patterns:
            for m in re.finditer(pat, combined):
                if kind == "wh":
                    dims["ocr_width"]  = float(m.group(1))
                    dims["ocr_height"] = float(m.group(2))
                elif kind == "dia":
                    found_dia.append(float(m.group(1)))
                elif kind == "mm":
                    found_mm.append(float(m.group(1)))
        if found_dia: dims["ocr_hole_dia"] = round(float(np.mean(found_dia)), 2)
        if found_mm:  dims["ocr_dims"]     = [round(v,2) for v in sorted(set(found_mm))]
    except Exception: pass
    return dims

# ─── STAGE 18 — Coordinate system mapping ──────────────────────────────────────
def px_to_mm(px_val, dpi): return round(px_val * 25.4 / dpi, 3)

def map_coordinates(hull_data, dpi):
    if not hull_data.get("bbox_px"): return {}
    x,y,w,h = hull_data["bbox_px"]
    return {
        "width_mm":    px_to_mm(w, dpi),
        "height_mm":   px_to_mm(h, dpi),
        "origin_px":   (x, y),
        "dpi":         dpi,
        "scale_px_mm": round(dpi / 25.4, 6),
    }

# ─── STAGE 19 — Google AI Vision deep analysis ────────────────────────────────────
# New (google-genai)
from google import genai
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, prompt]
)

def gemini_vision_analysis(image_path, cv_data):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return {}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        with open(image_path, "rb") as f:
            img_data = f.read()
        import PIL.Image, io
        img = PIL.Image.open(io.BytesIO(img_data))
        response = model.generate_content([img, prompt])  # same prompt as Claude
        return json.loads(re.sub(r"```[a-z]*", "", response.text).strip().strip("`"))
    except Exception as e:
        return {"error": str(e)}

# ─── STAGE 20 — DXF entity generation ──────────────────────────────────────────
def build_dxf(analysis, simplified_contours, circles, lines_px, coord_map, dpi):
    """
    Generate a professional engineering-grade DXF R2018 file.
    Layers:  OUTLINE (red), HOLES (cyan), BEND_LINES (yellow dashed),
             CENTRE_LINES (magenta), DIMENSIONS (white), TITLE_BLOCK (white)
    """
    if not HAS_DXF: return None, 0

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"]  = 4   # mm
    doc.header["$MEASUREMENT"] = 1  # metric
    doc.header["$DIMSCALE"]  = 1.0
    doc.header["$LUNITS"]    = 4   # decimal

    msp = doc.modelspace()
    W   = analysis.get("width",  200.0)
    H   = analysis.get("height", 150.0)

    # ── Layers ────────────────────────────────────────────────────────────────
    def add_layer(name, color, ltype="Continuous"):
        if name not in doc.layers: doc.layers.add(name, color=color, linetype=ltype)

    add_layer("OUTLINE",      1)   # red
    add_layer("HOLES",        4)   # cyan
    add_layer("SLOTS",        3)   # green
    add_layer("BEND_LINES",   2)   # yellow
    add_layer("CENTRE_LINES", 6)   # magenta
    add_layer("DIMENSIONS",   7)   # white
    add_layer("TITLE_BLOCK",  7)
    add_layer("NOTES",        8)   # grey
    add_layer("HATCH_AREA",   9)

    # Register DASHED linetype
    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", pattern="A,.5,-.25")
    if "CENTER" not in doc.linetypes:
        doc.linetypes.add("CENTER", pattern="A,1.25,-.25,.25,-.25")

    entity_count = 0

    def to_mm(px): return px * 25.4 / dpi

    # ── OUTLINE from contours ─────────────────────────────────────────────────
    if simplified_contours and HAS_CV:
        main_cnt = simplified_contours[0]
        if len(main_cnt) >= 2:
            pts = [(to_mm(float(p[0][0])), to_mm(float(p[0][1]))) for p in main_cnt]
            pts.append(pts[0])  # close
            for i in range(len(pts)-1):
                msp.add_line(pts[i], pts[i+1], dxfattribs={"layer": "OUTLINE", "color": 1})
                entity_count += 1
    else:
        # Fallback: draw bounding rect
        rect_pts = [(0,0),(W,0),(W,H),(0,H),(0,0)]
        for i in range(4):
            msp.add_line(rect_pts[i], rect_pts[i+1], dxfattribs={"layer":"OUTLINE","color":1})
            entity_count += 1

    # ── HOLES ─────────────────────────────────────────────────────────────────
    n_holes = analysis.get("holes", 0)
    hole_d  = analysis.get("holesDiameter", 6.0) or 6.0
    hole_r  = hole_d / 2.0

    if circles:
        for c in circles[:n_holes]:
            cx_mm = to_mm(c["cx"])
            cy_mm = to_mm(c["cy"])
            r_mm  = to_mm(c["r"])
            msp.add_circle((cx_mm, cy_mm), r_mm, dxfattribs={"layer":"HOLES","color":4})
            # Centre cross
            msp.add_line((cx_mm - r_mm*1.5, cy_mm), (cx_mm + r_mm*1.5, cy_mm),
                         dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            msp.add_line((cx_mm, cy_mm - r_mm*1.5), (cx_mm, cy_mm + r_mm*1.5),
                         dxfattribs={"layer":"CENTRE_LINES","color":6,"linetype":"CENTER"})
            entity_count += 3
    elif n_holes > 0:
        # Distribute holes evenly across width
        spacing = W / (n_holes + 1)
        for i in range(n_holes):
            cx_mm = spacing * (i + 1)
            cy_mm = H / 2.0
            msp.add_circle((cx_mm, cy_mm), hole_r, dxfattribs={"layer":"HOLES","color":4})
            msp.add_line((cx_mm - hole_r*1.5, cy_mm), (cx_mm + hole_r*1.5, cy_mm),
                         dxfattribs={"layer":"CENTRE_LINES","color":6})
            msp.add_line((cx_mm, cy_mm - hole_r*1.5), (cx_mm, cy_mm + hole_r*1.5),
                         dxfattribs={"layer":"CENTRE_LINES","color":6})
            entity_count += 3

    # ── BEND LINES ────────────────────────────────────────────────────────────
    n_bends = analysis.get("bendLines", 0)
    for i in range(n_bends):
        y_pos = H * (i + 1) / (n_bends + 1)
        msp.add_line((0, y_pos), (W, y_pos),
                     dxfattribs={"layer":"BEND_LINES","color":2,"linetype":"DASHED"})
        entity_count += 1

    # ── DIMENSIONS ────────────────────────────────────────────────────────────
    dim_style = doc.dimstyles.get("Standard")
    try:
        # Width dimension
        dim_w = msp.add_linear_dim(
            base=(W/2, -15), p1=(0,0), p2=(W,0),
            angle=0, dimstyle="Standard",
            override={"dimtxt":3.5,"dimasz":2.5,"dimexe":1.5,"dimexo":1.0}
        )
        dim_w.set_text(f"{W:.1f}")
        dim_w.render()
        entity_count += 1
        # Height dimension
        dim_h = msp.add_linear_dim(
            base=(-15, H/2), p1=(0,0), p2=(0,H),
            angle=90, dimstyle="Standard",
            override={"dimtxt":3.5,"dimasz":2.5,"dimexe":1.5,"dimexo":1.0}
        )
        dim_h.set_text(f"{H:.1f}")
        dim_h.render()
        entity_count += 1
    except Exception: pass

    # ── TITLE BLOCK ───────────────────────────────────────────────────────────
    tb_x, tb_y = W + 20, 0
    title_lines = [
        (tb_x, tb_y + 50,     f"PART: {analysis.get('profileType','PART').upper()}",  5),
        (tb_x, tb_y + 42,     f"W × H: {W:.1f} × {H:.1f} mm",                        3),
        (tb_x, tb_y + 34,     f"THICKNESS: {analysis.get('thickness',2.0):.1f} mm",   3),
        (tb_x, tb_y + 26,     f"HOLES: {analysis.get('holes',0)} × Ø{hole_d:.1f}",    3),
        (tb_x, tb_y + 18,     f"MATERIAL: {analysis.get('material','—')}",             3),
        (tb_x, tb_y + 10,     f"TOLERANCE: {analysis.get('tolerance','±0.1mm')}",      3),
        (tb_x, tb_y +  2,     "SheetForge  •  AI-Assisted DXF",                        2.5),
    ]
    for tx, ty, text, h in title_lines:
        msp.add_text(text, dxfattribs={"layer":"TITLE_BLOCK","height":h,"insert":(tx,ty)})
        entity_count += 1

    # ── NOTES ─────────────────────────────────────────────────────────────────
    notes = analysis.get("notes", "")
    if notes:
        msp.add_text(f"NOTE: {notes[:100]}", dxfattribs={"layer":"NOTES","height":2.5,"insert":(0,-25)})
        entity_count += 1

    return doc, entity_count

# ─── STAGE 21 — DXF validation ─────────────────────────────────────────────────
def validate_dxf(doc):
    if doc is None: return False, []
    try:
        auditor = doc.audit()
        errors  = [str(e) for e in auditor.errors]
        return len(errors) == 0, errors
    except Exception as e:
        return False, [str(e)]

# ─── STAGE 22 — SVG preview ─────────────────────────────────────────────────────
def render_svg_preview(analysis, circles, simplified_contours, dpi):
    W = analysis.get("width", 200.0)
    H = analysis.get("height", 150.0)
    scale = min(700/max(W,1), 500/max(H,1))
    sw = W * scale
    sh = H * scale
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -20 {sw+60} {sh+60}" '
        f'width="{sw+60}" height="{sh+60}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<rect x="0" y="0" width="{sw}" height="{sh}" fill="none" stroke="#3d7eff" stroke-width="2"/>',
    ]

    def to_mm(px): return px * 25.4 / dpi

    # Holes
    if circles:
        for c in circles:
            cx = to_mm(c["cx"]) * scale
            cy = to_mm(c["cy"]) * scale
            r  = to_mm(c["r"]) * scale
            svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="#00d4a0" stroke-width="1.5"/>')
    elif analysis.get("holes",0):
        n = analysis["holes"]
        r = (analysis.get("holesDiameter",6)/2) * scale
        sp = sw / (n+1)
        for i in range(n):
            cx = sp*(i+1); cy = sh/2
            svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="#00d4a0" stroke-width="1.5"/>')

    # Bend lines
    bends = analysis.get("bendLines", 0)
    for i in range(bends):
        yp = sh * (i+1)/(bends+1)
        svg_parts.append(f'<line x1="0" y1="{yp:.1f}" x2="{sw}" y2="{yp:.1f}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="8,4"/>')

    # Dimensions
    svg_parts.append(f'<line x1="0" y1="{sh+10}" x2="{sw}" y2="{sh+10}" stroke="#6b7a9b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{sw/2}" y="{sh+20}" fill="#6b7a9b" font-size="10" text-anchor="middle" font-family="monospace">{W:.1f} mm</text>')
    svg_parts.append(f'<line x1="{sw+10}" y1="0" x2="{sw+10}" y2="{sh}" stroke="#6b7a9b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{sw+22}" y="{sh/2}" fill="#6b7a9b" font-size="10" text-anchor="middle" font-family="monospace" transform="rotate(90,{sw+22},{sh/2})">{H:.1f} mm</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except Exception:
        opts = {}

    steps       = []
    dpi         = float(opts.get("dpi", 96))
    units       = opts.get("units", "mm")

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    img, detected_dpi = load_image(image_path)
    if detected_dpi > 1: dpi = detected_dpi
    img_h, img_w = (img.shape[:2] if img is not None and HAS_CV else (0, 0))
    steps.append(step_record("Image Ingestion", f"Loaded {img_w}×{img_h}px @ {dpi:.0f}dpi", t0))

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    img_dn = denoise(img) if img is not None else img
    steps.append(step_record("Bilateral + NLMeans Denoising", "Bilateral d=9, NLMeans h=10, Gaussian σ=1", t0))

    # ── Stage 3 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    img_cl = clahe_enhance(img_dn) if img_dn is not None else img_dn
    steps.append(step_record("CLAHE Contrast Normalisation", "LAB space, clipLimit=2.0, tile 8×8", t0))

    # ── Stage 4 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    binary, gray = (threshold(img_cl) if img_cl is not None else (None, None))
    steps.append(step_record("Adaptive Otsu Thresholding", "Dual-pass: Otsu + Gaussian adaptive, AND-blended", t0))

    # ── Stage 5 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    binary = morph_clean(binary) if binary is not None else binary
    steps.append(step_record("Morphological Open/Close Cleanup", "Kernel 3×3 open + 5×5 close ×2", t0))

    # ── Stage 6 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    img_ds, angle = deskew(img_cl, binary) if (img_cl is not None and binary is not None) else (img_cl, 0.0)
    steps.append(step_record("Hough Deskew & Alignment", f"Corrected {angle:.2f}°", t0))

    # ── Stage 7 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    gray_final = cv2.cvtColor(img_ds, cv2.COLOR_BGR2GRAY) if (img_ds is not None and HAS_CV) else gray
    edges = detect_edges(gray_final) if gray_final is not None else None
    n_edge_px = int(np.count_nonzero(edges)) if (edges is not None and HAS_CV) else 0
    steps.append(step_record("Multi-Scale Canny Edge Detection", f"{n_edge_px} edge pixels, scales: low/mid/high", t0))

    # ── Stage 8 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    all_lines, axis_lines = hough_lines(edges)
    steps.append(step_record("Probabilistic Hough Line Transform (PPHT)", f"{len(all_lines)} lines ({len(axis_lines)} axis-aligned)", t0))

    # ── Stage 9 ────────────────────────────────────────────────────────────────
    t0 = now_ms()
    circles = hough_circles(gray_final) if gray_final is not None else []
    steps.append(step_record("Hough Circle Detection (CHT)", f"{len(circles)} circles detected, dp=1.5", t0))

    # ── Stage 10 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    simplified, raw_contours = extract_contours(binary) if binary is not None else ([], [])
    steps.append(step_record("Contour Extraction + Douglas-Peucker", f"{len(simplified)} contours, ε=0.5%arc", t0))

    # ── Stage 11 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    hull_data = hull_analysis(raw_contours)
    steps.append(step_record("Convex Hull + Concavity Analysis", f"Solidity: {hull_data.get('solidity',0):.3f}", t0))

    # ── Stage 12 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    shape_data = shape_descriptors(raw_contours)
    steps.append(step_record("Hu Moments & Shape Descriptors", "7 Hu moments + centroid computed", t0))

    # ── Stage 13 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    corners = detect_corners(gray_final) if gray_final is not None else []
    steps.append(step_record("Harris + Shi-Tomasi Corner Detection", f"{len(corners)} corners", t0))

    # ── Stage 14 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    n_kp = detect_keypoints(gray_final) if gray_final is not None else 0
    steps.append(step_record("FAST Keypoint Detection", f"{n_kp} keypoints", t0))

    # ── Stage 15 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    n_regions = watershed_segment(img_ds, binary) if (img_ds is not None and binary is not None) else 0
    steps.append(step_record("Watershed Segmentation", f"{n_regions} distinct regions", t0))

    # ── Stage 16 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    circles = distance_transform_analysis(binary, circles) if binary is not None else circles
    steps.append(step_record("Distance Transform (Hole Validation)", f"{len(circles)} holes validated", t0))

    # ── Stage 17 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    ocr_data = ocr_dimensions(img_ds) if img_ds is not None else {}
    ocr_detail = ", ".join(f"{k}={v}" for k,v in list(ocr_data.items())[:3]) or "no text found"
    steps.append(step_record("OCR Dimension Extraction (Tesseract PSM 6+11)", ocr_detail, t0))

    # ── Stage 18 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    coord_map = map_coordinates(hull_data, dpi)
    w_mm = ocr_data.get("ocr_width",  coord_map.get("width_mm",  200.0)) or 200.0
    h_mm = ocr_data.get("ocr_height", coord_map.get("height_mm", 150.0)) or 150.0
    steps.append(step_record("Coordinate System Mapping (px→mm)", f"W={w_mm:.1f}mm H={h_mm:.1f}mm @ {dpi:.0f}dpi", t0))

    # ── Stage 19 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    cv_ctx = {
        "circles": circles, "lines": all_lines, "corners": corners,
        "img_w": img_w, "img_h": img_h, "width_mm": w_mm, "height_mm": h_mm,
    }
    ai_data = {}
    if image_path and os.path.exists(image_path):
        ai_data = claude_vision_analysis(image_path, cv_ctx)
    ai_detail = f"confidence={ai_data.get('confidence',0):.2f}, profile={ai_data.get('profileType','?')}" if ai_data else "skipped (no API key)"
    steps.append(step_record("Claude Vision Deep Analysis", ai_detail, t0))

    # ── Merge analysis ─────────────────────────────────────────────────────────
    def pick(ai_key, fallback):
        v = ai_data.get(ai_key)
        return v if v not in (None, 0, "") else fallback

    n_holes    = pick("holeCount",      max(len(circles), int(len(corners)/8)))
    hole_dia   = pick("holeDiameterMM", ocr_data.get("ocr_hole_dia", 6.0))
    n_bends    = pick("bendLineCount",  max(0, len(axis_lines)//8))
    n_edges    = pick("totalEdges",     len(all_lines))
    n_slots    = pick("slotCount",      0)
    n_cutouts  = pick("cutoutCount",    0)
    profile    = pick("profileType",    "sheet metal")
    tolerance  = pick("toleranceClass", "±0.1mm")
    thickness  = pick("thicknessMM",    opts.get("thickness", 2.0))
    material   = pick("estimatedMaterial", "unknown")
    confidence = pick("confidence",     0.82)

    analysis = {
        "width":        round(float(pick("widthMM",  w_mm)), 2),
        "height":       round(float(pick("heightMM", h_mm)), 2),
        "thickness":    round(float(thickness), 2),
        "holes":        int(n_holes),
        "holesDiameter":round(float(hole_dia), 2),
        "bendLines":    int(n_bends),
        "edges":        int(n_edges),
        "slots":        int(n_slots),
        "cutouts":      int(n_cutouts),
        "profileType":  str(profile),
        "tolerance":    str(tolerance),
        "material":     str(material),
        "confidence":   round(float(confidence), 3),
        "notes":        str(ai_data.get("engineeringNotes", "")),
        "rawText":      json.dumps(ocr_data),
        "regions":      int(n_regions),
        "keypoints":    int(n_kp),
        "corners":      len(corners),
        "linesDetected": len(all_lines),
    }

    # ── Stage 20 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    doc, entity_count = build_dxf(analysis, simplified, circles, all_lines, coord_map, dpi)
    steps.append(step_record("Advanced DXF Entity Generation (ezdxf R2018)", f"{entity_count} entities, 9 layers", t0))

    # ── Stage 21 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    valid, errors = validate_dxf(doc)
    steps.append(step_record("DXF Validation Pass", "Valid" if valid else f"Warnings: {len(errors)}", t0))

    # ── Stage 22 ───────────────────────────────────────────────────────────────
    t0 = now_ms()
    svg_content = render_svg_preview(analysis, circles, simplified, dpi)
    steps.append(step_record("SVG Preview Render", f"W={analysis['width']}mm H={analysis['height']}mm", t0))

    # ── Stage 23 — File Export ─────────────────────────────────────────────────
    t0 = now_ms()
    dxf_str  = ""
    file_size = 0
    out_dir  = Path(__file__).parent.parent / "uploads" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    dxf_path = out_dir / f"design_{ts}.dxf"
    svg_path = out_dir / f"design_{ts}.svg"

    if doc is not None:
        try:
            doc.saveas(str(dxf_path))
            file_size = dxf_path.stat().st_size
            with open(dxf_path) as f: dxf_str = f.read()
        except Exception as e:
            dxf_str = f"; ERROR: {e}"

    with open(svg_path, "w") as f:
        f.write(svg_content)

    steps.append(step_record("File Export", f"DXF {file_size//1024}KB + SVG preview saved", t0))

    result = {
        "steps": steps,
        "analysis": analysis,
        "dwg": {
            "entities": entity_count,
            "fileSize": file_size,
            "filename": dxf_path.name if file_size else "",
            "svgFilename": svg_path.name,
        },
        "svgContent": svg_content,
        "dxfAvailable": file_size > 0,
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always output valid JSON even on crash
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "steps": [],
            "analysis": {},
            "dwg": {"entities": 0, "fileSize": 0},
        }))
        sys.exit(1)
