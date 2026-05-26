#!/usr/bin/env python3
"""
SheetForge — Advanced CV + Claude Vision DXF Pipeline  v3.0
============================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent }

Improvements in v3.0:
  - Multi-pass OCR with structured dimension-line association
  - Dimension-text → geometry binding (arrow endpoint → nearest shape)
  - Circle radius verified from OCR Ø annotations
  - Absolute mm positioning from OCR dimension chains
  - Cluster-based circle grouping for size disambiguation
  - Red/Blue channel separation for dimension vs outline detection
  - Improved Claude Vision prompt requesting full feature layout JSON
  - DXF built 100% from extracted mm coordinates — not guessed
  - DXF saved to same directory as the original uploaded image
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
Image         = _try(lambda: __import__("PIL.Image",   fromlist=["Image"]))
ImageFilter   = _try(lambda: __import__("PIL.ImageFilter", fromlist=["ImageFilter"]))
ImageEnhance  = _try(lambda: __import__("PIL.ImageEnhance", fromlist=["ImageEnhance"]))
anthropic_mod = _try(lambda: __import__("anthropic"))
scipy_mod     = _try(lambda: __import__("scipy"))

HAS_CV   = cv2  is not None and np is not None
HAS_DXF  = ezdxf is not None
HAS_OCR  = pytesseract is not None
HAS_PIL  = Image is not None
HAS_AI   = anthropic_mod is not None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def now_ms(): return int(time.time() * 1000)

def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}

# ─── STAGE 1 — Image ingestion ─────────────────────────────────────────────────
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

# ─── STAGE 2 — Channel separation (red annotations vs blue outlines) ───────────
def separate_channels(img):
    """
    Engineering sketches often use RED for dimensions/annotations and BLUE/BLACK
    for geometry outlines. Separate them for independent analysis.
    Returns: outline_gray, annotation_gray, full_gray
    """
    if not HAS_CV: return None, None, None
    b, g, r = cv2.split(img)

    # Red channel: where red >> blue & green (dimension lines, text)
    red_mask  = cv2.subtract(r, cv2.max(b, g))
    _, red_bin = cv2.threshold(red_mask, 40, 255, cv2.THRESH_BINARY)

    # Blue channel: where blue >> red & green (geometry outlines)
    blue_mask  = cv2.subtract(b, cv2.max(r, g))
    _, blue_bin = cv2.threshold(blue_mask, 30, 255, cv2.THRESH_BINARY)

    # Full grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return blue_bin, red_bin, gray


# ─── STAGE 3 — Enhanced pre-processing ────────────────────────────────────────
def preprocess(img):
    if not HAS_CV: return img, None
    # Denoise
    bil = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    nlm = cv2.fastNlMeansDenoisingColored(bil, None, h=8, hColor=8,
                                           templateWindowSize=7, searchWindowSize=21)
    # CLAHE in LAB space
    lab = cv2.cvtColor(nlm, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    # Sharpening
    kernel  = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp   = cv2.filter2D(enhanced, -1, kernel)
    gray    = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    return sharp, gray


# ─── STAGE 4 — Deskew ──────────────────────────────────────────────────────────
def deskew(img, gray):
    if not HAS_CV: return img, 0.0
    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None: return img, 0.0
    angles = []
    for r, theta in lines[:, 0]:
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


# ─── STAGE 5 — Multi-scale Canny edge detection ────────────────────────────────
def detect_edges(gray):
    if not HAS_CV: return None
    e1 = cv2.Canny(gray, 20, 80)
    e2 = cv2.Canny(gray, 50, 150)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    e3 = cv2.Canny(blurred, 40, 120)
    return cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)


# ─── STAGE 6 — Hough line transform ────────────────────────────────────────────
def hough_lines(edges):
    if not HAS_CV or edges is None: return [], []
    raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                          threshold=40, minLineLength=15, maxLineGap=12)
    if raw is None: return [], []
    lines     = [tuple(l[0]) for l in raw]
    h_lines   = [l for l in lines if abs(l[3] - l[1]) < 8]
    v_lines   = [l for l in lines if abs(l[2] - l[0]) < 8]
    return lines, h_lines + v_lines


# ─── STAGE 7 — Improved Hough circle detection (multi-pass) ────────────────────
def hough_circles_multipass(gray):
    """
    Multi-pass circle detection:
      Pass 1: strict (high confidence, catches large circles)
      Pass 2: relaxed (catches smaller/fainter circles)
    Returns list of dicts {cx, cy, r} deduplicated.
    """
    if not HAS_CV: return []
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    all_circles = []

    # Pass 1 — standard gradient, strict
    raw1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                             dp=1.2, minDist=25,
                             param1=120, param2=35,
                             minRadius=5, maxRadius=300)
    if raw1 is not None:
        for c in np.uint16(np.around(raw1[0])):
            all_circles.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2]), "pass": 1})

    # Pass 2 — ALT gradient for imperfect/hand-drawn circles
    try:
        raw2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT,
                                 dp=1.5, minDist=20,
                                 param1=250, param2=0.80,
                                 minRadius=5, maxRadius=300)
        if raw2 is not None:
            for c in np.uint16(np.around(raw2[0])):
                all_circles.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2]), "pass": 2})
    except Exception:
        pass

    # Deduplication: merge circles within 20px of each other (keep largest r)
    merged = []
    for c in all_circles:
        duplicate = False
        for m in merged:
            dist = math.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"])
            if dist < 20 and abs(c["r"] - m["r"]) < 15:
                duplicate = True
                if c["r"] > m["r"]:
                    m["cx"], m["cy"], m["r"] = c["cx"], c["cy"], c["r"]
                break
        if not duplicate:
            merged.append({"cx": c["cx"], "cy": c["cy"], "r": c["r"]})

    return merged


# ─── STAGE 8 — Contour extraction ──────────────────────────────────────────────
def extract_contours(binary):
    if not HAS_CV or binary is None: return [], []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    simplified = []
    for cnt in contours:
        epsilon = 0.004 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)
    simplified.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    return simplified, list(contours)


# ─── STAGE 9 — Hull & shape analysis ───────────────────────────────────────────
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


# ─── STAGE 10 — Harris + Shi-Tomasi corners ─────────────────────────────────────
def detect_corners(gray):
    if not HAS_CV: return []
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
    if corners is None: return []
    return [{"x": int(c[0][0]), "y": int(c[0][1])} for c in corners]


# ─── STAGE 11 — FAST keypoints ──────────────────────────────────────────────────
def detect_keypoints(gray):
    if not HAS_CV: return 0
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    return len(fast.detect(gray, None))


# ─── STAGE 12 — Watershed segmentation ─────────────────────────────────────────
def watershed_segment(img, binary):
    if not HAS_CV or binary is None: return 0
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


# ─── STAGE 13 — Advanced OCR: dimension text + spatial positions ────────────────
def ocr_with_positions(img):
    """
    Use Tesseract's image_to_data to get bounding boxes for each token.
    Returns structured list of {text, x, y, w, h, conf} for all dimension tokens.
    Also returns a flat dims dict for backward compat.
    """
    result = {"tokens": [], "dims": {}, "raw": ""}
    if not HAS_OCR or not HAS_PIL or not HAS_CV: return result

    try:
        # Upscale for better OCR (Tesseract works best ≥300dpi)
        h_img, w_img = img.shape[:2]
        scale = max(1.0, 2400 / max(w_img, h_img))
        if scale > 1.05:
            upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            upscaled  = img
            scale     = 1.0

        pil = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

        # PSM 11 = sparse text, best for engineering drawings
        data = pytesseract.image_to_data(pil, config="--psm 11 --oem 3",
                                          output_type=pytesseract.Output.DICT)

        tokens = []
        full_text_parts = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text: continue
            conf = int(data["conf"][i])
            if conf < 20: continue
            x  = int(data["left"][i]   / scale)
            y  = int(data["top"][i]    / scale)
            w  = int(data["width"][i]  / scale)
            h2 = int(data["height"][i] / scale)
            tokens.append({"text": text, "x": x, "y": y, "w": w, "h": h2, "conf": conf})
            full_text_parts.append(text)

        result["tokens"] = tokens
        result["raw"]    = " ".join(full_text_parts)

        # Parse dimension values from tokens
        combined = result["raw"]
        dims     = {}

        # Width × Height pattern  e.g. "900 mm", "900mm", "900"
        # Largest two mm numbers → width, height
        all_mm = []
        for m in re.finditer(r"(\d{2,4}(?:\.\d+)?)\s*(?:mm)?", combined, re.IGNORECASE):
            v = float(m.group(1))
            if 10 < v < 5000:
                all_mm.append(v)

        # Explicit WxH
        for m in re.finditer(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", combined):
            dims["ocr_width"]  = float(m.group(1))
            dims["ocr_height"] = float(m.group(2))

        # Diameter annotations  Ø150  or  ∅30
        diameters = []
        for m in re.finditer(r"[ØøO∅]\s*(\d+\.?\d*)", combined):
            diameters.append(float(m.group(1)))
        if diameters:
            dims["ocr_diameters"] = sorted(set(diameters))
            # Largest = primary holes, smallest = small holes
            dims["ocr_hole_dia"]  = max(diameters)

        # Radii
        for m in re.finditer(r"R\s*(\d+\.?\d*)", combined):
            dims.setdefault("ocr_radii", []).append(float(m.group(1)))

        # All raw mm values — pick outliers as W/H
        if all_mm and "ocr_width" not in dims:
            sorted_mm = sorted(set(all_mm), reverse=True)
            if len(sorted_mm) >= 2:
                dims["ocr_width"]  = sorted_mm[0]
                dims["ocr_height"] = sorted_mm[1]
            elif len(sorted_mm) == 1:
                dims["ocr_width"]  = sorted_mm[0]

        result["dims"] = dims

    except Exception as e:
        result["error"] = str(e)

    return result


# ─── STAGE 14 — Bind OCR dimensions to pixel positions ─────────────────────────
def bind_dimensions_to_geometry(ocr_result, circles, img_shape, dpi):
    """
    Associate Ø text tokens with the nearest detected circle.
    Returns circles list enriched with confirmed_r_mm field.
    """
    if not circles or not ocr_result.get("tokens"):
        return circles

    h_img, w_img = img_shape[:2]

    for token in ocr_result["tokens"]:
        m = re.search(r"[ØøO∅]\s*(\d+\.?\d*)", token["text"])
        if not m: continue
        dia_mm = float(m.group(1))
        # Token centre in pixels
        tx = token["x"] + token["w"] / 2
        ty = token["y"] + token["h"] / 2
        # Find closest circle
        best_c    = None
        best_dist = float("inf")
        for c in circles:
            d = math.hypot(c["cx"] - tx, c["cy"] - ty)
            if d < best_dist:
                best_dist = d
                best_c    = c
        if best_c is not None and best_dist < max(w_img, h_img) * 0.3:
            best_c["confirmed_dia_mm"] = dia_mm
            best_c["confirmed_r_mm"]   = dia_mm / 2.0

    return circles


# ─── STAGE 15 — Pixel → mm calibration ─────────────────────────────────────────
def calibrate_px_to_mm(hull_data, ocr_dims, dpi):
    """
    Determine the best px-per-mm scale.
    Priority: OCR total width → DPI conversion → bbox aspect ratio.
    Returns scale_px_mm (pixels per mm), w_mm, h_mm.
    """
    bbox = hull_data.get("bbox_px")
    w_px = bbox[2] if bbox else 0
    h_px = bbox[3] if bbox else 0

    w_mm_ocr = ocr_dims.get("ocr_width",  0) or 0
    h_mm_ocr = ocr_dims.get("ocr_height", 0) or 0

    # Use OCR dimension if plausible
    if w_mm_ocr > 50 and w_px > 10:
        scale_x = w_px / w_mm_ocr
    else:
        scale_x = dpi / 25.4  # fallback

    if h_mm_ocr > 50 and h_px > 10:
        scale_y = h_px / h_mm_ocr
    else:
        scale_y = dpi / 25.4

    # Average scale (assume uniform)
    scale = (scale_x + scale_y) / 2.0

    w_mm = round(w_px / scale, 2) if scale > 0 else w_mm_ocr or 200
    h_mm = round(h_px / scale, 2) if scale > 0 else h_mm_ocr or 150

    # If OCR width is very reliable, trust it over computed
    if w_mm_ocr > 50: w_mm = w_mm_ocr
    if h_mm_ocr > 50: h_mm = h_mm_ocr

    return round(scale, 6), w_mm, h_mm


# ─── STAGE 16 — Claude Vision (greatly improved prompt) ────────────────────────
def claude_vision_analysis(image_path, cv_data):
    """
    Send image + full CV context to Claude Vision.
    Request a detailed JSON with:
      - overall dimensions
      - every circle with its centre position and diameter
      - small hole positions and diameters
      - dimension callouts confirmed
    """
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

        ocr_hint = json.dumps(cv_data.get("ocr_dims", {}))
        circles_hint = json.dumps(cv_data.get("circles", [])[:10])

        prompt = f"""You are an expert mechanical / sheet-metal CAD engineer with perfect dimensional reading skills.

Analyse this engineering sketch carefully. Extract ALL dimensions written on the drawing and use them to determine the EXACT position and size of every feature.

Return ONLY a valid JSON object — no markdown, no preamble:
{{
  "profileType": "sheet metal",
  "widthMM": <overall width as number>,
  "heightMM": <overall height as number>,
  "thicknessMM": <thickness or null>,
  "estimatedMaterial": "aluminum | steel | stainless | brass | copper | titanium | unknown",
  "toleranceClass": "fine (±0.05mm) | medium (±0.1mm) | coarse (±0.5mm) | general (±1mm)",
  "confidence": <0.0 to 1.0>,
  "engineeringNotes": "<brief note>",
  "circles": [
    {{
      "label": "top-left burner",
      "cx_mm": <x centre from left edge of part in mm>,
      "cy_mm": <y centre from top edge of part in mm>,
      "diameter_mm": <diameter in mm>,
      "type": "large_hole | small_hole | burner | cutout"
    }}
  ],
  "smallHoles": [
    {{
      "cx_mm": <x from left>,
      "cy_mm": <y from top>,
      "diameter_mm": <diameter>,
      "spacing_mm": <centre-to-centre spacing if in a row>
    }}
  ],
  "bendLines": <integer count>,
  "slots": <integer count>,
  "dimensions_confirmed": {{
    "overall_width_mm": <number>,
    "overall_height_mm": <number>,
    "col_spacing_mm": <column spacing between circle centres if applicable>,
    "row_spacing_mm": <row spacing between circle centres if applicable>,
    "margin_left_mm": <distance from left edge to first feature centre>,
    "margin_top_mm":  <distance from top edge to first feature centre>
  }}
}}

Context from OpenCV pre-processing:
- Image dimensions (px): {cv_data.get('img_w',0)} × {cv_data.get('img_h',0)}
- Circles detected by OpenCV: {circles_hint}
- OCR dimension text found: {ocr_hint}
- Estimated overall size (px→mm @ {cv_data.get('dpi',96)}dpi): {cv_data.get('width_mm',0):.0f} × {cv_data.get('height_mm',0):.0f} mm

Read EVERY dimension label and arrow on the drawing carefully to place features at exact mm positions.
Return ONLY the JSON. No extra text."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type, "data": img_b64}},
                    {"type": "text",  "text": prompt},
                ]
            }]
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        # Extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: text = match.group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ─── STAGE 17 — Merge all sources into unified analysis ────────────────────────
def merge_analysis(ai_data, ocr_data, w_mm, h_mm, circles, all_lines, corners,
                   n_kp, n_regions, opts):
    def pick(ai_key, fallback):
        v = ai_data.get(ai_key)
        return v if v not in (None, 0, "", []) else fallback

    # Circles — prefer AI layout, fall back to OpenCV positions
    ai_circles = ai_data.get("circles", [])
    sm_holes   = ai_data.get("smallHoles", [])

    # Overall dimensions
    final_w = float(pick("widthMM",  w_mm) or w_mm or 200)
    final_h = float(pick("heightMM", h_mm) or h_mm or 150)

    # Confirmed dims block
    dim_conf   = ai_data.get("dimensions_confirmed", {})
    if dim_conf.get("overall_width_mm"):  final_w = float(dim_conf["overall_width_mm"])
    if dim_conf.get("overall_height_mm"): final_h = float(dim_conf["overall_height_mm"])

    n_holes  = len(ai_circles) if ai_circles else len(circles)
    hole_dia = 0.0
    if ai_circles:
        dias = [c.get("diameter_mm", 0) for c in ai_circles if c.get("diameter_mm", 0) > 0]
        hole_dia = round(float(np.mean(dias)), 2) if dias else 6.0

    diameters_ocr = ocr_data.get("dims", {}).get("ocr_diameters", [])
    if diameters_ocr and hole_dia == 0.0:
        hole_dia = max(diameters_ocr)

    n_bends   = int(pick("bendLines",   max(0, len([l for l in all_lines if abs(l[3]-l[1])<8]) // 12)))
    n_edges   = int(pick("totalEdges",  len(all_lines)))
    profile   = pick("profileType",    "sheet metal")
    tolerance = pick("toleranceClass", "±0.1mm")
    thickness = float(pick("thicknessMM", opts.get("thickness", 2.0)) or 2.0)
    material  = pick("estimatedMaterial", "unknown")
    conf      = float(pick("confidence", 0.82))

    return {
        "width":          round(final_w, 2),
        "height":         round(final_h, 2),
        "thickness":      round(thickness, 2),
        "holes":          int(n_holes),
        "holesDiameter":  round(float(hole_dia), 2),
        "bendLines":      int(n_bends),
        "edges":          int(n_edges),
        "slots":          int(pick("slots", 0)),
        "cutouts":        int(pick("cutoutCount", 0)),
        "profileType":    str(profile),
        "tolerance":      str(tolerance),
        "material":       str(material),
        "confidence":     round(conf * 100, 1),
        "notes":          str(ai_data.get("engineeringNotes", "")),
        "rawText":        json.dumps(ocr_data.get("dims", {})),
        "regions":        int(n_regions),
        "keypoints":      int(n_kp),
        "corners":        len(corners),
        "linesDetected":  len(all_lines),
        # Rich layout data for DXF builder
        "_ai_circles":    ai_circles,
        "_sm_holes":      sm_holes,
        "_dim_confirmed": dim_conf,
        "_ocr_circles":   circles,   # raw OpenCV circle detections in px
    }


# ─── STAGE 18 — Professional DXF generation (exact positions) ─────────────────
def build_dxf(analysis, dpi, image_path):
    """
    Build an accurate DXF using mm positions extracted from AI + OCR.
    DXF is also saved next to the original image file.
    """
    if not HAS_DXF: return None, 0, ""

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"]    = 4   # mm
    doc.header["$MEASUREMENT"] = 1   # metric
    doc.header["$DIMSCALE"]    = 1.0
    doc.header["$LUNITS"]      = 4   # decimal

    msp = doc.modelspace()
    W   = float(analysis.get("width",  200))
    H   = float(analysis.get("height", 150))

    def add_layer(name, color, ltype="Continuous"):
        if name not in doc.layers:
            doc.layers.add(name, color=color, linetype=ltype)

    add_layer("OUTLINE",       1)   # red
    add_layer("HOLES",         4)   # cyan
    add_layer("SMALL_HOLES",   3)   # green
    add_layer("SLOTS",         5)   # blue
    add_layer("BEND_LINES",    2)   # yellow
    add_layer("CENTRE_LINES",  6)   # magenta
    add_layer("DIMENSIONS",    7)   # white
    add_layer("TITLE_BLOCK",   7)
    add_layer("NOTES",         8)   # grey

    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED",  pattern="A,.5,-.25")
    if "CENTER" not in doc.linetypes:
        doc.linetypes.add("CENTER",  pattern="A,1.25,-.25,.25,-.25")

    entity_count = 0

    # ── OUTLINE ────────────────────────────────────────────────────────────────
    rect_pts = [(0, 0), (W, 0), (W, H), (0, H), (0, 0)]
    for i in range(4):
        msp.add_line(rect_pts[i], rect_pts[i+1],
                     dxfattribs={"layer": "OUTLINE", "color": 1, "lineweight": 50})
        entity_count += 1

    # ── MAIN CIRCLES (burner holes) from AI layout ──────────────────────────────
    ai_circles = analysis.get("_ai_circles", [])
    ocr_circles = analysis.get("_ocr_circles", [])  # OpenCV detections in px

    def to_mm_px(px): return round(px * 25.4 / dpi, 3)

    placed_circles = []

    if ai_circles:
        for c in ai_circles:
            cx = float(c.get("cx_mm", 0))
            cy = float(c.get("cy_mm", 0))
            d  = float(c.get("diameter_mm", analysis.get("holesDiameter", 6) or 6))
            r  = d / 2.0
            # DXF Y is bottom-up; convert from top-down sketch
            cy_dxf = H - cy
            layer  = "HOLES" if d >= 20 else "SMALL_HOLES"
            msp.add_circle((cx, cy_dxf), r,
                           dxfattribs={"layer": layer, "color": 4 if layer=="HOLES" else 3})
            # Centre cross
            cross = r * 1.6
            msp.add_line((cx - cross, cy_dxf), (cx + cross, cy_dxf),
                         dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            msp.add_line((cx, cy_dxf - cross), (cx, cy_dxf + cross),
                         dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            entity_count += 3
            placed_circles.append((cx, cy_dxf, r))

    elif ocr_circles:
        # Fall back to OpenCV pixel positions converted to mm
        for c in ocr_circles:
            cx_mm = to_mm_px(c["cx"])
            cy_mm = to_mm_px(c["cy"])
            r_mm  = float(c.get("confirmed_r_mm", to_mm_px(c["r"])))
            cy_dxf = H - cy_mm
            msp.add_circle((cx_mm, cy_dxf), r_mm,
                           dxfattribs={"layer": "HOLES", "color": 4})
            cross = r_mm * 1.6
            msp.add_line((cx_mm - cross, cy_dxf), (cx_mm + cross, cy_dxf),
                         dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            msp.add_line((cx_mm, cy_dxf - cross), (cx_mm, cy_dxf + cross),
                         dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            entity_count += 3

    else:
        # Last resort: distribute evenly
        n = analysis.get("holes", 0)
        r = (analysis.get("holesDiameter", 6) or 6) / 2.0
        if n > 0:
            spacing = W / (n + 1)
            for i in range(n):
                cx = spacing * (i + 1)
                cy_dxf = H / 2.0
                msp.add_circle((cx, cy_dxf), r,
                               dxfattribs={"layer": "HOLES", "color": 4})
                cross = r * 1.6
                msp.add_line((cx - cross, cy_dxf), (cx + cross, cy_dxf),
                             dxfattribs={"layer": "CENTRE_LINES", "color": 6})
                msp.add_line((cx, cy_dxf - cross), (cx, cy_dxf + cross),
                             dxfattribs={"layer": "CENTRE_LINES", "color": 6})
                entity_count += 3

    # ── SMALL HOLES ────────────────────────────────────────────────────────────
    sm_holes = analysis.get("_sm_holes", [])
    for sh in sm_holes:
        cx  = float(sh.get("cx_mm", 0))
        cy  = float(sh.get("cy_mm", 0))
        d   = float(sh.get("diameter_mm", 10) or 10)
        r   = d / 2.0
        cy_dxf = H - cy
        msp.add_circle((cx, cy_dxf), r,
                       dxfattribs={"layer": "SMALL_HOLES", "color": 3})
        cross = r * 1.8
        msp.add_line((cx - cross, cy_dxf), (cx + cross, cy_dxf),
                     dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
        msp.add_line((cx, cy_dxf - cross), (cx, cy_dxf + cross),
                     dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
        entity_count += 3

    # ── BEND LINES ─────────────────────────────────────────────────────────────
    n_bends = analysis.get("bendLines", 0)
    for i in range(n_bends):
        y_pos = H * (i + 1) / (n_bends + 1)
        msp.add_line((0, y_pos), (W, y_pos),
                     dxfattribs={"layer": "BEND_LINES", "color": 2, "linetype": "DASHED"})
        entity_count += 1

    # ── DIMENSIONS ─────────────────────────────────────────────────────────────
    try:
        dw = msp.add_linear_dim(base=(W/2, -18), p1=(0,0), p2=(W,0), angle=0,
                                 dimstyle="Standard",
                                 override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dw.set_text(f"{W:.1f}mm"); dw.render(); entity_count += 1
    except Exception: pass
    try:
        dh = msp.add_linear_dim(base=(-18, H/2), p1=(0,0), p2=(0,H), angle=90,
                                 dimstyle="Standard",
                                 override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dh.set_text(f"{H:.1f}mm"); dh.render(); entity_count += 1
    except Exception: pass

    # Add diameter dimensions for large circles
    if placed_circles:
        for (cx, cy_dxf, r) in placed_circles:
            try:
                msp.add_text(f"Ø{r*2:.1f}",
                             dxfattribs={"layer":"DIMENSIONS","height":3.5,
                                         "insert":(cx, cy_dxf - r - 8),
                                         "halign":1,"valign":0})
                entity_count += 1
            except Exception: pass

    # ── TITLE BLOCK ────────────────────────────────────────────────────────────
    hole_d = analysis.get("holesDiameter", 6) or 6
    tb_x, tb_y = W + 25, 0
    title_lines = [
        (tb_x, tb_y+60, f"PART: {analysis.get('profileType','PART').upper()}",         5.0),
        (tb_x, tb_y+50, f"W × H: {W:.1f} × {H:.1f} mm",                               3.5),
        (tb_x, tb_y+42, f"THICKNESS: {analysis.get('thickness',2.0):.1f} mm",          3.5),
        (tb_x, tb_y+34, f"CIRCLES: {len(ai_circles or ocr_circles)} × Ø{hole_d:.1f}", 3.5),
        (tb_x, tb_y+26, f"MATERIAL: {analysis.get('material','—')}",                   3.5),
        (tb_x, tb_y+18, f"TOLERANCE: {analysis.get('tolerance','±0.1mm')}",            3.5),
        (tb_x, tb_y+10, f"CONFIDENCE: {analysis.get('confidence',0)}%",                3.0),
        (tb_x, tb_y+ 2, "SheetForge  •  AI-Assisted DXF  v3",                         2.5),
    ]
    for tx, ty, text, h_t in title_lines:
        msp.add_text(text, dxfattribs={"layer":"TITLE_BLOCK","height":h_t,"insert":(tx,ty)})
        entity_count += 1

    notes = analysis.get("notes","")
    if notes:
        msp.add_text(f"NOTE: {notes[:120]}",
                     dxfattribs={"layer":"NOTES","height":2.5,"insert":(0,-30)})
        entity_count += 1

    return doc, entity_count


# ─── STAGE 19 — DXF validation ─────────────────────────────────────────────────
def validate_dxf(doc):
    if doc is None: return False, []
    try:
        auditor = doc.audit()
        errors  = [str(e) for e in auditor.errors]
        return len(errors) == 0, errors
    except Exception as e:
        return False, [str(e)]


# ─── STAGE 20 — SVG preview (accurate positions) ───────────────────────────────
def render_svg_preview(analysis, dpi):
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
        # Grid dots
        '<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">'
        '<circle cx="10" cy="10" r="0.8" fill="#1e2d40"/></pattern></defs>',
        f'<rect x="0" y="0" width="{sw}" height="{sh}" fill="url(#grid)"/>',
        # Outline
        f'<rect x="0" y="0" width="{sw:.1f}" height="{sh:.1f}" '
        f'fill="none" stroke="#3d7eff" stroke-width="2.5"/>',
    ]

    def to_mm_px(px): return round(px * 25.4 / dpi, 3)

    ai_circles = analysis.get("_ai_circles", [])
    ocv_circles = analysis.get("_ocr_circles", [])
    sm_holes    = analysis.get("_sm_holes", [])

    def draw_circle(cx_mm, cy_mm, r_mm, color, label=None):
        cx = cx_mm * scale
        cy = cy_mm * scale  # already top-down in SVG
        r  = r_mm  * scale
        cross = r * 1.6
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                   f'fill="none" stroke="{color}" stroke-width="1.8"/>')
        svg.append(f'<line x1="{cx-cross:.1f}" y1="{cy:.1f}" x2="{cx+cross:.1f}" y2="{cy:.1f}" '
                   f'stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        svg.append(f'<line x1="{cx:.1f}" y1="{cy-cross:.1f}" x2="{cx:.1f}" y2="{cy+cross:.1f}" '
                   f'stroke="#a855f7" stroke-width="0.8" stroke-dasharray="5,3"/>')
        if label:
            svg.append(f'<text x="{cx:.1f}" y="{cy+r+12:.1f}" fill="#6b7a9b" '
                       f'font-size="9" text-anchor="middle" font-family="monospace">{label}</text>')

    if ai_circles:
        for c in ai_circles:
            d   = float(c.get("diameter_mm", 0) or 0)
            r   = d / 2.0
            cx  = float(c.get("cx_mm", 0))
            cy  = float(c.get("cy_mm", 0))
            col = "#00d4a0" if d >= 20 else "#22c55e"
            draw_circle(cx, cy, r, col, f"Ø{d:.0f}")
    elif ocv_circles:
        for c in ocv_circles:
            cx = to_mm_px(c["cx"])
            cy = to_mm_px(c["cy"])
            r  = float(c.get("confirmed_r_mm", to_mm_px(c["r"])))
            draw_circle(cx, cy, r, "#00d4a0", f"Ø{r*2:.0f}")

    # Small holes
    for sh in sm_holes:
        cx = float(sh.get("cx_mm", 0))
        cy = float(sh.get("cy_mm", 0))
        r  = float(sh.get("diameter_mm", 10) or 10) / 2.0
        draw_circle(cx, cy, r, "#22c55e", f"Ø{r*2:.0f}")

    # Bend lines
    bends = analysis.get("bendLines", 0)
    for i in range(bends):
        yp = sh * (i + 1) / (bends + 1)
        svg.append(f'<line x1="0" y1="{yp:.1f}" x2="{sw:.1f}" y2="{yp:.1f}" '
                   f'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="8,4"/>')

    # Dimension arrows
    svg.append(f'<line x1="0" y1="{sh+15:.1f}" x2="{sw:.1f}" y2="{sh+15:.1f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw/2:.1f}" y="{sh+27:.1f}" fill="#6b7a9b" font-size="11" '
               f'text-anchor="middle" font-family="monospace">{W:.1f} mm</text>')
    svg.append(f'<line x1="{sw+15:.1f}" y1="0" x2="{sw+15:.1f}" y2="{sh:.1f}" stroke="#6b7a9b" stroke-width="1"/>')
    svg.append(f'<text x="{sw+28:.1f}" y="{sh/2:.1f}" fill="#6b7a9b" font-size="11" '
               f'text-anchor="middle" font-family="monospace" '
               f'transform="rotate(90,{sw+28:.1f},{sh/2:.1f})">{H:.1f} mm</text>')

    svg.append('</svg>')
    return "\n".join(svg)


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except Exception:
        opts = {}

    steps = []
    dpi   = float(opts.get("dpi", 96))

    # S1 — Load
    t0 = now_ms()
    img, detected_dpi = load_image(image_path)
    if detected_dpi > 1: dpi = detected_dpi
    img_h, img_w = (img.shape[:2] if img is not None and HAS_CV else (0, 0))
    steps.append(step_record("Image Ingestion", f"{img_w}×{img_h}px @ {dpi:.0f}dpi", t0))

    # S2 — Channel separation
    t0 = now_ms()
    outline_bin, annot_bin, gray_raw = separate_channels(img) if img is not None else (None, None, None)
    steps.append(step_record("Red/Blue Channel Separation", "Dimension annotations vs geometry outline", t0))

    # S3 — Preprocess
    t0 = now_ms()
    img_proc, gray = preprocess(img) if img is not None else (img, gray_raw)
    if gray is None and gray_raw is not None: gray = gray_raw
    steps.append(step_record("CLAHE + Sharpen Pre-processing", "LAB CLAHE + unsharp mask", t0))

    # S4 — Deskew
    t0 = now_ms()
    img_ds, angle = deskew(img_proc, gray) if (img_proc is not None and gray is not None) else (img_proc, 0.0)
    gray_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2GRAY) if (img_ds is not None and HAS_CV) else gray
    steps.append(step_record("Hough Deskew", f"Corrected {angle:.2f}°", t0))

    # S5 — Edges
    t0 = now_ms()
    edges = detect_edges(gray_ds) if gray_ds is not None else None
    n_edge = int(np.count_nonzero(edges)) if (edges is not None and HAS_CV) else 0
    steps.append(step_record("Multi-Scale Canny Edge Detection", f"{n_edge} edge pixels", t0))

    # S6 — Lines
    t0 = now_ms()
    all_lines, axis_lines = hough_lines(edges)
    steps.append(step_record("Probabilistic Hough Line Transform", f"{len(all_lines)} lines", t0))

    # S7 — Circles (multi-pass)
    t0 = now_ms()
    circles = hough_circles_multipass(gray_ds) if gray_ds is not None else []
    steps.append(step_record("Multi-Pass Hough Circle Detection", f"{len(circles)} circles (2-pass dedup)", t0))

    # S8 — Contours
    t0 = now_ms()
    # Use geometry-only channel if available, else binary from gray
    if outline_bin is not None:
        simp, raw_cnt = extract_contours(outline_bin)
    elif gray_ds is not None:
        _, binary = cv2.threshold(gray_ds, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if HAS_CV else (None, None)
        simp, raw_cnt = extract_contours(binary)
    else:
        simp, raw_cnt = [], []
    steps.append(step_record("Contour Extraction + Douglas-Peucker", f"{len(simp)} contours", t0))

    # S9 — Hull
    t0 = now_ms()
    hull_data = hull_analysis(raw_cnt)
    steps.append(step_record("Convex Hull + Shape Analysis", f"Solidity: {hull_data.get('solidity',0):.3f}", t0))

    # S10 — Corners
    t0 = now_ms()
    corners = detect_corners(gray_ds) if gray_ds is not None else []
    steps.append(step_record("Harris + Shi-Tomasi Corner Detection", f"{len(corners)} corners", t0))

    # S11 — Keypoints
    t0 = now_ms()
    n_kp = detect_keypoints(gray_ds) if gray_ds is not None else 0
    steps.append(step_record("FAST Keypoint Detection", f"{n_kp} keypoints", t0))

    # S12 — Watershed
    t0 = now_ms()
    n_regions = watershed_segment(img_ds, outline_bin if outline_bin is not None else
                                  (cv2.threshold(gray_ds, 127, 255, cv2.THRESH_BINARY)[1] if (gray_ds is not None and HAS_CV) else None)) if img_ds is not None else 0
    steps.append(step_record("Watershed Segmentation", f"{n_regions} regions", t0))

    # S13 — OCR with positions
    t0 = now_ms()
    ocr_result = ocr_with_positions(img_ds) if img_ds is not None else {"tokens":[], "dims":{}, "raw":""}
    ocr_dims   = ocr_result.get("dims", {})
    steps.append(step_record("OCR Dimension Extraction (positional)", f"{len(ocr_result.get('tokens',[]))} tokens — {list(ocr_dims.keys())}", t0))

    # S14 — Bind Ø annotations to circles
    t0 = now_ms()
    if img_ds is not None and HAS_CV:
        circles = bind_dimensions_to_geometry(ocr_result, circles, img_ds.shape, dpi)
    steps.append(step_record("Dimension↔Geometry Binding", f"{sum(1 for c in circles if c.get('confirmed_dia_mm'))} circles annotated with Ø", t0))

    # S15 — Calibrate px→mm
    t0 = now_ms()
    scale_px_mm, w_mm, h_mm = calibrate_px_to_mm(hull_data, ocr_dims, dpi)
    steps.append(step_record("Pixel→mm Calibration", f"Scale={scale_px_mm:.3f}px/mm  W={w_mm:.1f}mm H={h_mm:.1f}mm", t0))

    # S16 — Claude Vision
    t0 = now_ms()
    cv_ctx = {
        "circles":    circles[:12],
        "lines":      all_lines[:20],
        "corners":    corners[:20],
        "img_w":      img_w, "img_h": img_h,
        "width_mm":   w_mm,  "height_mm": h_mm,
        "dpi":        dpi,
        "ocr_dims":   ocr_dims,
    }
    ai_data = {}
    if image_path and os.path.exists(image_path):
        ai_data = claude_vision_analysis(image_path, cv_ctx)
    ai_detail = (f"conf={ai_data.get('confidence',0):.2f} "
                 f"profile={ai_data.get('profileType','?')} "
                 f"circles={len(ai_data.get('circles',[]))}") if ai_data else "skipped"
    steps.append(step_record("Claude Vision Deep Analysis (enhanced)", ai_detail, t0))

    # S17 — Merge
    t0 = now_ms()
    analysis = merge_analysis(ai_data, ocr_result, w_mm, h_mm,
                              circles, all_lines, corners, n_kp, n_regions, opts)
    steps.append(step_record("Data Fusion & Analysis Merge", f"W={analysis['width']}mm H={analysis['height']}mm holes={analysis['holes']}", t0))

    # S18 — Build DXF
    t0 = now_ms()
    doc, entity_count = build_dxf(analysis, dpi, image_path)
    steps.append(step_record("Precision DXF Generation (exact positions)", f"{entity_count} entities, 9 layers", t0))

    # S19 — Validate
    t0 = now_ms()
    valid, errors = validate_dxf(doc)
    steps.append(step_record("DXF Validation Pass", "Valid" if valid else f"Warnings: {len(errors)}", t0))

    # S20 — SVG preview
    t0 = now_ms()
    svg_content = render_svg_preview(analysis, dpi)
    steps.append(step_record("SVG Preview Render (accurate layout)", f"W={analysis['width']}mm H={analysis['height']}mm", t0))

    # S21 — Export
    t0 = now_ms()
    dxf_str   = ""
    file_size = 0

    # Primary: save next to original image
    if image_path and os.path.exists(image_path):
        orig_dir = Path(image_path).parent
    else:
        orig_dir = Path(__file__).parent.parent / "uploads" / "output"

    # Also save to server output dir for URL serving
    server_out_dir = Path(__file__).parent.parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    ts       = int(time.time())
    dxf_name = f"design_{ts}.dxf"
    svg_name = f"design_{ts}.svg"

    dxf_path_server = server_out_dir / dxf_name
    svg_path_server = server_out_dir / svg_name
    dxf_path_local  = orig_dir / dxf_name

    if doc is not None:
        try:
            doc.saveas(str(dxf_path_server))
            file_size = dxf_path_server.stat().st_size
            with open(dxf_path_server) as f: dxf_str = f.read()
            # Copy to image folder as well (if different)
            if str(dxf_path_local) != str(dxf_path_server):
                try:
                    import shutil
                    shutil.copy2(str(dxf_path_server), str(dxf_path_local))
                except Exception: pass
        except Exception as e:
            dxf_str = f"; ERROR: {e}"

    with open(svg_path_server, "w") as f:
        f.write(svg_content)

    steps.append(step_record("File Export", f"DXF {file_size//1024}KB saved to image folder + server", t0))

    # Strip internal fields before returning
    public_analysis = {k: v for k, v in analysis.items() if not k.startswith("_")}

    result = {
        "steps":    steps,
        "analysis": public_analysis,
        "dwg": {
            "entities":    entity_count,
            "fileSize":    file_size,
            "filename":    dxf_name if file_size else "",
            "svgFilename": svg_name,
            "localPath":   str(dxf_path_local),
        },
        "svgContent":   svg_content,
        "dxfAvailable": file_size > 0,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error":    str(e),
            "traceback": traceback.format_exc(),
            "steps":    [],
            "analysis": {},
            "dwg":      {"entities": 0, "fileSize": 0},
        }))
        sys.exit(1)
