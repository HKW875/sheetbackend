#!/usr/bin/env python3
"""
SheetForge — Lean OpenCV Pipeline  v6.0
========================================
Receives: image_path, options_json  (from Node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, gcode, pdfFilename }

Pipeline stages:
  PRE-1  Load image + detect DPI
  PRE-2  Grayscale  →  cv2.cvtColor()
  PRE-3  Binarise   →  cv2.adaptiveThreshold()  +  THRESH_BINARY_INV
  PRE-4  Smudge removal  →  cv2.morphologyEx()  (OPEN then CLOSE)
  CV-1   Canny edges  →  cv2.Canny() + cv2.dilate()
  CV-2   Line detection  →  cv2.HoughLines()
  CV-3   Circle detection  →  cv2.HoughCircles()
  CV-4   Contour / shape extraction  →  cv2.findContours() + cv2.approxPolyDP()
  CV-5   Symbol matching  →  cv2.matchTemplate()
  CV-6   Distance / dimension calculation  →  cv2.minAreaRect()  +  px→mm
  DXF    Build DXF  →  ezdxf
  PDF    Export PDF  →  reportlab
  GCODE  G-Code generation
"""

import sys, os, json, math, time, re, traceback
from pathlib import Path

# ── Graceful optional imports ─────────────────────────────────────────────────
def _try(fn):
    try: return fn()
    except Exception: return None

cv2           = _try(lambda: __import__("cv2"))
np            = _try(lambda: __import__("numpy"))
ezdxf         = _try(lambda: __import__("ezdxf"))
pytesseract   = _try(lambda: __import__("pytesseract"))
Image         = _try(lambda: __import__("PIL.Image", fromlist=["Image"]))
genai_mod     = _try(lambda: __import__("google.genai", fromlist=["genai"]))
reportlab_mod = _try(lambda: __import__("reportlab"))

HAS_CV  = cv2 is not None and np is not None
HAS_DXF = ezdxf is not None
HAS_OCR = pytesseract is not None
HAS_PIL = Image is not None
HAS_AI  = genai_mod is not None
HAS_RL  = reportlab_mod is not None

def now_ms(): return int(time.time() * 1000)
def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}


# ════════════════════════════════════════════════════════════════════════════════
# PRE-1  Load + DPI
# ════════════════════════════════════════════════════════════════════════════════

def load_image(image_path):
    """Load BGR image and detect DPI. Returns (bgr, dpi)."""
    if not HAS_CV or not image_path or not os.path.exists(image_path):
        return None, 96.0
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None, 96.0
    dpi = 96.0
    if HAS_PIL:
        try:
            pil = Image.open(str(image_path))
            xdpi = pil.info.get("dpi", (96, 96))
            dpi  = float(xdpi[0]) if xdpi[0] > 1 else 96.0
        except Exception: pass
    # Cap at 2400px on the long side — keeps all downstream ops fast
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        scale = 2400 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dpi *= scale          # adjust DPI proportionally
    return img, dpi


# ════════════════════════════════════════════════════════════════════════════════
# PRE-2  Grayscale
# ════════════════════════════════════════════════════════════════════════════════

def to_grayscale(img):
    """cv2.cvtColor — BGR → GRAY."""
    if not HAS_CV or img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ════════════════════════════════════════════════════════════════════════════════
# PRE-3  Binarisation  (paper / ink separation)
# ════════════════════════════════════════════════════════════════════════════════

def binarise(gray):
    """
    Separate dark lines from paper background.
      1. GaussianBlur  — reduce sensor noise
      2. CLAHE         — normalise local contrast
      3. adaptiveThreshold + THRESH_BINARY_INV  — ink → 255, paper → 0
    Returns binary image.
    """
    if not HAS_CV or gray is None: return None
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq      = clahe.apply(blurred)
    binary  = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    return binary


# ════════════════════════════════════════════════════════════════════════════════
# PRE-4  Smudge / noise removal
# ════════════════════════════════════════════════════════════════════════════════

def remove_smudges(binary):
    """
    cv2.morphologyEx:
      OPEN  — removes isolated speckle / smudges  (erode then dilate)
      CLOSE — seals small gaps in stroke lines    (dilate then erode)
    Returns cleaned binary.
    """
    if not HAS_CV or binary is None: return binary
    k3 = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k3, iterations=1)
    return cleaned


# ════════════════════════════════════════════════════════════════════════════════
# CV-1  Edge detection + dilation
# ════════════════════════════════════════════════════════════════════════════════

def detect_edges(gray):
    """
    Multi-scale Canny + cv2.dilate to connect near-adjacent edge fragments.
    Returns edge map.
    """
    if not HAS_CV or gray is None: return None
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    e1      = cv2.Canny(blurred, 30,  100)
    e2      = cv2.Canny(blurred, 60,  150)
    edges   = cv2.bitwise_or(e1, e2)
    edges   = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges


# ════════════════════════════════════════════════════════════════════════════════
# CV-2  Line detection  →  cv2.HoughLines
# ════════════════════════════════════════════════════════════════════════════════

def detect_lines(edges):
    """
    cv2.HoughLines — standard Hough transform for full-length lines.
    Returns list of line dicts with angle classification.
    """
    if not HAS_CV or edges is None: return []
    raw = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=90)
    if raw is None: return []
    lines = []
    for line in raw[:80]:
        rho, theta = line[0]
        a, b  = math.cos(theta), math.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1200 * (-b)), int(y0 + 1200 * a)
        x2, y2 = int(x0 - 1200 * (-b)), int(y0 - 1200 * a)
        ang = math.degrees(theta)
        lines.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "rho": round(float(rho), 2), "theta": round(float(theta), 4),
            "angle": round(ang, 2),
            "is_horizontal": abs(ang - 90) < 10,
            "is_vertical":   abs(ang) < 10 or abs(ang - 180) < 10,
            "type": "hough",
        })
    return lines


# ════════════════════════════════════════════════════════════════════════════════
# CV-3  Circle detection  →  cv2.HoughCircles
# ════════════════════════════════════════════════════════════════════════════════

def detect_circles(gray):
    """
    Two-pass cv2.HoughCircles (GRADIENT + GRADIENT_ALT), deduplicated.
    Returns list of {cx, cy, r} in pixels.
    """
    if not HAS_CV or gray is None: return []
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    found   = []

    # Pass 1 — standard gradient accumulator
    raw1 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                            dp=1.2, minDist=20,
                            param1=100, param2=30,
                            minRadius=4, maxRadius=0)
    if raw1 is not None:
        for c in np.uint16(np.around(raw1[0])):
            found.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])})

    # Pass 2 — GRADIENT_ALT (better for weak circles)
    try:
        raw2 = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT,
                                dp=1.5, minDist=18,
                                param1=220, param2=0.82,
                                minRadius=4, maxRadius=0)
        if raw2 is not None:
            for c in np.uint16(np.around(raw2[0])):
                found.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])})
    except Exception: pass

    # Deduplicate overlapping detections
    merged = []
    for c in found:
        dup = any(
            math.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"]) < 18
            and abs(c["r"] - m["r"]) < 12
            for m in merged
        )
        if not dup:
            merged.append(c)
    return merged


# ════════════════════════════════════════════════════════════════════════════════
# CV-4  Contour + shape extraction
# ════════════════════════════════════════════════════════════════════════════════

def extract_shapes(binary):
    """
    cv2.findContours + cv2.approxPolyDP to detect and classify drawing shapes.
    Returns (simplified_contours, shape_dicts).
    """
    if not HAS_CV or binary is None: return [], []
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    simplified = []
    shapes     = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80: continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0: continue

        # Simplify polygon
        eps    = 0.004 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        simplified.append(approx)

        # Classify
        circ   = 4 * math.pi * area / (peri * peri)
        n_v    = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 1.0

        if circ > 0.78 or (n_v > 8 and circ > 0.5):
            stype = "circle"
            if circ > 0.45 and n_v >= 5:
                try:
                    (ex, ey), (ma, mi), _ = cv2.fitEllipse(cnt)
                    stype = "circle" if (mi > 0 and ma / mi < 1.2) else "ellipse"
                except Exception: pass
        elif n_v == 3:
            stype = "triangle"
        elif n_v == 4:
            stype = "square" if 0.85 <= aspect <= 1.15 else "rectangle"
        elif n_v <= 8:
            stype = "polygon"
        else:
            stype = "complex"

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area / hull_area) if hull_area > 0 else 0.0

        shapes.append({
            "type": stype, "area": float(area), "perimeter": float(peri),
            "circularity": round(circ, 4), "solidity": round(solidity, 4),
            "vertices": n_v, "bbox": (int(x), int(y), int(w), int(h)),
            "aspect": round(aspect, 4),
        })

    simplified.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    return simplified, shapes


# ════════════════════════════════════════════════════════════════════════════════
# CV-5  Engineering symbol matching  →  cv2.matchTemplate
# ════════════════════════════════════════════════════════════════════════════════

# Built-in mini symbol templates (rendered into numpy arrays at import time)
def _make_circle_template(r=18):
    """Filled circle template."""
    sz = r * 2 + 4
    t  = np.zeros((sz, sz), np.uint8)
    cv2.circle(t, (sz // 2, sz // 2), r, 255, 2)
    return t

def _make_arrow_template():
    """Right-pointing arrow template."""
    t = np.zeros((24, 40), np.uint8)
    pts = np.array([[0,8],[28,8],[28,2],[39,12],[28,22],[28,16],[0,16]], np.int32)
    cv2.fillPoly(t, [pts], 255)
    return t

def _make_weld_template():
    """Simplified weld symbol (triangle)."""
    t = np.zeros((24, 24), np.uint8)
    pts = np.array([[12, 2], [22, 22], [2, 22]], np.int32)
    cv2.fillPoly(t, [pts], 255)
    return t

_SYMBOL_TEMPLATES = None

def _get_templates():
    global _SYMBOL_TEMPLATES
    if _SYMBOL_TEMPLATES is None and HAS_CV:
        _SYMBOL_TEMPLATES = {
            "circle_symbol": _make_circle_template(18),
            "arrow":         _make_arrow_template(),
            "weld":          _make_weld_template(),
        }
    return _SYMBOL_TEMPLATES or {}


def match_symbols(binary, threshold=0.55):
    """
    cv2.matchTemplate — slide each engineering symbol template over the binary
    image and return detections above the confidence threshold.
    Returns list of {symbol, x, y, w, h, confidence}.
    """
    if not HAS_CV or binary is None: return []
    detections = []
    templates  = _get_templates()
    for sym_name, tmpl in templates.items():
        th, tw = tmpl.shape[:2]
        if binary.shape[0] < th or binary.shape[1] < tw: continue
        result = cv2.matchTemplate(binary, tmpl, cv2.TM_CCOEFF_NORMED)
        locs   = np.where(result >= threshold)
        for (y, x) in zip(*locs):
            conf = float(result[y, x])
            # NMS: skip if a stronger detection already claimed this area
            duplicate = any(
                abs(d["x"] - x) < tw and abs(d["y"] - y) < th and d["confidence"] >= conf
                for d in detections if d["symbol"] == sym_name
            )
            if not duplicate:
                detections.append({
                    "symbol": sym_name, "x": int(x), "y": int(y),
                    "w": int(tw), "h": int(th), "confidence": round(conf, 3),
                })
    return detections


# ════════════════════════════════════════════════════════════════════════════════
# CV-6  Distance / dimension calculation  →  cv2.minAreaRect  +  px→mm
# ════════════════════════════════════════════════════════════════════════════════

def calculate_dimensions(simplified_contours, circles, dpi, img_shape):
    """
    Use cv2.minAreaRect on each contour to get the minimum bounding rectangle,
    then convert pixel measurements to real-world mm using the image's x=0, y=0
    pixel reference (top-left corner of the scanned image) and the DPI scale.

    Pixel origin (0, 0) corresponds to the physical top-left of the scanned sheet.
    Scale: 1 mm = dpi / 25.4  pixels.

    Returns:
      features     – list of feature dicts with px and mm coordinates
      scale_px_mm  – pixels per mm
      img_w_mm     – scanned image width  in mm
      img_h_mm     – scanned image height in mm
    """
    if not HAS_CV: return [], dpi / 25.4, 0, 0

    scale_px_mm = dpi / 25.4          # pixels per mm
    h_px, w_px  = img_shape[:2]
    img_w_mm    = round(w_px / scale_px_mm, 2)
    img_h_mm    = round(h_px / scale_px_mm, 2)

    features = []

    # --- contour bounding boxes via minAreaRect ---
    for i, cnt in enumerate(simplified_contours[:60]):
        area_px = cv2.contourArea(cnt)
        if area_px < 200: continue
        rect     = cv2.minAreaRect(cnt)        # ((cx,cy),(w,h),angle)
        (cx, cy), (rw, rh), angle = rect
        # Convert to mm from pixel origin (0,0)
        features.append({
            "id":      f"feat_{i}",
            "type":    "contour_rect",
            "cx_px":   round(float(cx), 1),
            "cy_px":   round(float(cy), 1),
            "w_px":    round(float(rw), 1),
            "h_px":    round(float(rh), 1),
            "cx_mm":   round(float(cx)  / scale_px_mm, 3),
            "cy_mm":   round(float(cy)  / scale_px_mm, 3),
            "w_mm":    round(float(rw)  / scale_px_mm, 3),
            "h_mm":    round(float(rh)  / scale_px_mm, 3),
            "angle":   round(float(angle), 2),
            "area_px": round(float(area_px), 1),
        })

    # --- circle features (from HoughCircles) ---
    for j, c in enumerate(circles):
        features.append({
            "id":     f"circ_{j}",
            "type":   "circle",
            "cx_px":  float(c["cx"]),
            "cy_px":  float(c["cy"]),
            "r_px":   float(c["r"]),
            "cx_mm":  round(float(c["cx"]) / scale_px_mm, 3),
            "cy_mm":  round(float(c["cy"]) / scale_px_mm, 3),
            "r_mm":   round(float(c["r"])  / scale_px_mm, 3),
            "d_mm":   round(float(c["r"]) * 2 / scale_px_mm, 3),
        })

    # --- inter-feature distances (nearest neighbours, first 10 features) ---
    for a in range(min(len(features), 10)):
        for b in range(a + 1, min(len(features), 10)):
            fa, fb = features[a], features[b]
            dx = fa["cx_px"] - fb["cx_px"]
            dy = fa["cy_px"] - fb["cy_px"]
            dist_px = math.hypot(dx, dy)
            dist_mm = round(dist_px / scale_px_mm, 3)
            # attach to the first feature for reference
            features[a].setdefault("distances", []).append({
                "to": fb["id"], "px": round(dist_px, 1), "mm": dist_mm
            })

    return features, scale_px_mm, img_w_mm, img_h_mm


# ════════════════════════════════════════════════════════════════════════════════
# OCR + Calibration
# ════════════════════════════════════════════════════════════════════════════════

def ocr_dimensions(img):
    """Tesseract OCR — extract dimension tokens and numeric values."""
    result = {"tokens": [], "dims": {}, "raw": ""}
    if not HAS_OCR or not HAS_PIL or img is None: return result
    try:
        h, w   = img.shape[:2]
        up_sc  = max(1.0, 2400 / max(w, h))
        up     = cv2.resize(img, None, fx=up_sc, fy=up_sc, interpolation=cv2.INTER_CUBIC) if up_sc > 1.05 else img
        pil    = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        data   = pytesseract.image_to_data(pil, config="--psm 11 --oem 3",
                                            output_type=pytesseract.Output.DICT)
        tokens = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            if not text or int(data["conf"][i]) < 20: continue
            tokens.append({
                "text": text, "conf": int(data["conf"][i]),
                "x": int(data["left"][i]  / up_sc), "y": int(data["top"][i]   / up_sc),
                "w": int(data["width"][i] / up_sc), "h": int(data["height"][i] / up_sc),
            })
        result["tokens"] = tokens
        combined = " ".join(t["text"] for t in tokens)
        result["raw"] = combined
        dims = {}
        for m in re.finditer(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)", combined):
            dims["ocr_width"]  = float(m.group(1))
            dims["ocr_height"] = float(m.group(2))
        diameters = [float(m.group(1)) for m in re.finditer(r"[ØøO∅]\s*(\d+\.?\d*)", combined)]
        if diameters:
            dims["ocr_diameters"] = sorted(set(diameters))
            dims["ocr_hole_dia"]  = max(diameters)
        all_mm = [float(m.group(1)) for m in re.finditer(r"(\d{2,4}(?:\.\d+)?)\s*(?:mm)?", combined, re.I)
                  if 10 < float(m.group(1)) < 5000]
        if all_mm and "ocr_width" not in dims:
            sm = sorted(set(all_mm), reverse=True)
            if len(sm) >= 2: dims["ocr_width"], dims["ocr_height"] = sm[0], sm[1]
            elif sm: dims["ocr_width"] = sm[0]
        result["dims"] = dims
    except Exception as e:
        result["error"] = str(e)
    return result


def calibrate(features, ocr_dims, dpi, img_w_mm, img_h_mm):
    """
    Reconcile DPI-based scale with OCR-read dimensions.
    Prefer OCR overall dimensions when plausible.
    Returns (final_w_mm, final_h_mm).
    """
    w_mm = ocr_dims.get("ocr_width",  img_w_mm) or img_w_mm
    h_mm = ocr_dims.get("ocr_height", img_h_mm) or img_h_mm
    w_mm = w_mm if w_mm and w_mm > 10 else img_w_mm
    h_mm = h_mm if h_mm and h_mm > 10 else img_h_mm
    return round(float(w_mm), 2), round(float(h_mm), 2)


# ════════════════════════════════════════════════════════════════════════════════
# Gemini Vision  (unchanged external API)
# ════════════════════════════════════════════════════════════════════════════════

def gemini_vision_analysis(image_path, cv_ctx):
    if not HAS_AI: return {}
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return {}
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        with open(image_path, "rb") as f: img_bytes = f.read()
        ext  = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "bmp": "image/bmp", "tiff": "image/tiff", "pdf": "application/pdf"}.get(ext, "image/jpeg")
        circles_hint = json.dumps(cv_ctx.get("circles", [])[:10])
        lines_hint   = json.dumps([{"angle": l.get("angle"), "is_h": l.get("is_horizontal")}
                                    for l in cv_ctx.get("lines", [])[:12]])
        ocr_hint     = json.dumps(cv_ctx.get("ocr_dims", {}))
        prompt = f"""You are an expert mechanical / sheet-metal CAD engineer.
Analyse this engineering sketch and extract ALL feature dimensions precisely.

OpenCV pre-analysis context:
  Image: {cv_ctx.get("img_w",0)}×{cv_ctx.get("img_h",0)}px @ {cv_ctx.get("dpi",96):.0f}dpi
  Estimated size: {cv_ctx.get("width_mm",0):.1f}×{cv_ctx.get("height_mm",0):.1f}mm
  CV circles: {circles_hint}
  CV lines: {lines_hint}
  OCR dimensions: {ocr_hint}

Rules:
- widthMM / heightMM must be the OUTER boundary of the part.
- For each circle include exact cx_mm, cy_mm measured from top-left corner (pixel origin 0,0).
- If OCR dimensions conflict with your visual estimate, prefer OCR when plausible.
- Report confidence 0.0–1.0 honestly.

Return ONLY valid JSON — no markdown:
{{
  "profileType": "sheet metal",
  "widthMM": <number>, "heightMM": <number>, "thicknessMM": <number or null>,
  "estimatedMaterial": "aluminum|steel|stainless|brass|copper|titanium|unknown",
  "toleranceClass": "fine (±0.05mm)|medium (±0.1mm)|coarse (±0.5mm)|general (±1mm)",
  "confidence": <0.0-1.0>,
  "engineeringNotes": "<brief>",
  "circles": [{{"label":"","cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>,"type":"large_hole|small_hole|cutout"}}],
  "smallHoles": [{{"cx_mm":<x>,"cy_mm":<y>,"diameter_mm":<d>}}],
  "bendLines": <int>, "slots": <int>
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

claude_vision_analysis = gemini_vision_analysis   # alias kept for server.js


def ai_dxf_interaction(instruction, current_analysis, image_path=None):
    """Natural-language DXF correction via Gemini. Unchanged API surface."""
    if not HAS_AI: return current_analysis, "AI not available"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return current_analysis, "No GEMINI_API_KEY set"
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        clean  = {k: v for k, v in current_analysis.items() if not k.startswith("_")}
        prompt = f"""You are a CAD engineer modifying a DXF design.
Current analysis (JSON):
{json.dumps(clean, indent=2)}

User instruction: "{instruction}"

Return ONLY a JSON object with ONLY the fields that need to change:
{{
  "width": <new_value_or_omit>, "height": <new_value_or_omit>,
  "holes": <new_value_or_omit>, "holesDiameter": <new_value_or_omit>,
  "bendLines": <new_value_or_omit>, "material": "<new_value_or_omit>",
  "thickness": <new_value_or_omit>, "tolerance": "<new_value_or_omit>",
  "_ai_circles": [<new_circles_or_omit>],
  "explanation": "<brief explanation>"
}}"""
        parts = [genai_types.Part.from_text(text=prompt)]
        if image_path and os.path.exists(str(image_path)):
            ext  = Path(image_path).suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                parts.insert(0, genai_types.Part.from_bytes(data=f.read(), mime_type=mime))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[genai_types.Content(parts=parts, role="user")],
        )
        text = response.text or ""
        text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        changes     = json.loads(text)
        explanation = changes.pop("explanation", "Changes applied")
        updated     = {**current_analysis, **{k: v for k, v in changes.items() if v is not None}}
        return updated, explanation
    except Exception as e:
        return current_analysis, f"Error: {str(e)}"


# ════════════════════════════════════════════════════════════════════════════════
# Merge analysis
# ════════════════════════════════════════════════════════════════════════════════

def merge_analysis(ai_data, ocr_data, w_mm, h_mm, circles, all_lines, features, opts):
    def pick(key, fallback):
        v = ai_data.get(key)
        return v if v not in (None, 0, "", []) else fallback

    ai_circles = ai_data.get("circles",   [])
    sm_holes   = ai_data.get("smallHoles", [])
    final_w    = float(pick("widthMM",  w_mm) or w_mm or 200)
    final_h    = float(pick("heightMM", h_mm) or h_mm or 150)

    n_holes  = len(ai_circles) if ai_circles else len(circles)
    hole_dia = 0.0
    if ai_circles:
        dias = [c.get("diameter_mm", 0) for c in ai_circles if c.get("diameter_mm", 0) > 0]
        hole_dia = round(float(sum(dias) / len(dias)) if dias else 6.0, 2)
    if not hole_dia:
        hole_dia = ocr_data.get("dims", {}).get("ocr_hole_dia", 0.0)

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
        "profileType":   str(pick("profileType", "sheet metal")),
        "tolerance":     str(pick("toleranceClass", "±0.1mm")),
        "material":      str(pick("estimatedMaterial", "unknown")),
        "confidence":    round(float(pick("confidence", 0.80)) * 100, 1),
        "notes":         str(ai_data.get("engineeringNotes", "")),
        "rawText":       json.dumps(ocr_data.get("dims", {})),
        "linesDetected": len(all_lines),
        "_ai_circles":   ai_circles,
        "_sm_holes":     sm_holes,
        "_cv_circles":   circles,
        "_features":     features,
    }


# ════════════════════════════════════════════════════════════════════════════════
# DXF export  (ezdxf)
# ════════════════════════════════════════════════════════════════════════════════

def build_dxf(analysis, dpi):
    """Build a precise DXF R2018 from merged analysis data using ezdxf."""
    if not HAS_DXF: return None, 0
    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"]    = 4   # mm
    doc.header["$MEASUREMENT"] = 1
    doc.header["$DIMSCALE"]    = 1.0
    doc.header["$LUNITS"]      = 4
    msp = doc.modelspace()
    W   = float(analysis.get("width",  200))
    H   = float(analysis.get("height", 150))

    def add_layer(name, color):
        if name not in doc.layers: doc.layers.add(name, color=color, linetype="Continuous")

    for ln, col in [("OUTLINE", 1), ("HOLES", 4), ("SMALL_HOLES", 3),
                    ("BEND_LINES", 2), ("CENTRE_LINES", 6), ("DIMENSIONS", 7),
                    ("TITLE_BLOCK", 7), ("NOTES", 8)]:
        add_layer(ln, col)
    for lt, pat in [("DASHED", "A,.5,-.25"), ("CENTER", "A,1.25,-.25,.25,-.25")]:
        if lt not in doc.linetypes:
            try: doc.linetypes.add(lt, pattern=pat)
            except Exception: pass

    count = 0

    # Outline
    for p1, p2 in [((0,0),(W,0)),((W,0),(W,H)),((W,H),(0,H)),((0,H),(0,0))]:
        msp.add_line(p1, p2, dxfattribs={"layer": "OUTLINE", "color": 1, "lineweight": 50})
        count += 1

    def px2mm(px): return round(px * 25.4 / dpi, 3)

    ai_circles  = analysis.get("_ai_circles",  [])
    cv_circles  = analysis.get("_cv_circles",  [])
    placed      = []

    if ai_circles:
        for c in ai_circles:
            cx = float(c.get("cx_mm", 0))
            cy = H - float(c.get("cy_mm", 0))   # flip Y for DXF origin at bottom-left
            d  = float(c.get("diameter_mm", analysis.get("holesDiameter", 6) or 6))
            r  = d / 2.0
            layer = "HOLES" if d >= 20 else "SMALL_HOLES"
            msp.add_circle((cx, cy), r, dxfattribs={"layer": layer, "color": 4 if layer == "HOLES" else 3})
            cross = r * 1.6
            msp.add_line((cx-cross, cy), (cx+cross, cy), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            msp.add_line((cx, cy-cross), (cx, cy+cross), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            count += 3
            placed.append((cx, cy, r))
    elif cv_circles:
        for c in cv_circles:
            cx_mm = px2mm(c["cx"])
            cy    = H - px2mm(c["cy"])
            r_mm  = float(c.get("confirmed_r_mm", px2mm(c["r"])))
            msp.add_circle((cx_mm, cy), r_mm, dxfattribs={"layer": "HOLES", "color": 4})
            cross = r_mm * 1.6
            msp.add_line((cx_mm-cross, cy), (cx_mm+cross, cy), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            msp.add_line((cx_mm, cy-cross), (cx_mm, cy+cross), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
            count += 3
            placed.append((cx_mm, cy, r_mm))
    else:
        n = analysis.get("holes", 0)
        r = (analysis.get("holesDiameter", 6) or 6) / 2.0
        if n > 0:
            sp = W / (n + 1)
            for i in range(n):
                cx, cy = sp * (i + 1), H / 2.0
                msp.add_circle((cx, cy), r, dxfattribs={"layer": "HOLES", "color": 4})
                cross = r * 1.6
                msp.add_line((cx-cross, cy), (cx+cross, cy), dxfattribs={"layer": "CENTRE_LINES", "color": 6})
                msp.add_line((cx, cy-cross), (cx, cy+cross), dxfattribs={"layer": "CENTRE_LINES", "color": 6})
                count += 3
                placed.append((cx, cy, r))

    # Small holes
    for sh in analysis.get("_sm_holes", []):
        cx = float(sh.get("cx_mm", 0))
        cy = H - float(sh.get("cy_mm", 0))
        r  = float(sh.get("diameter_mm", 10) or 10) / 2.0
        msp.add_circle((cx, cy), r, dxfattribs={"layer": "SMALL_HOLES", "color": 3})
        cross = r * 1.8
        msp.add_line((cx-cross, cy), (cx+cross, cy), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
        msp.add_line((cx, cy-cross), (cx, cy+cross), dxfattribs={"layer": "CENTRE_LINES", "color": 6, "linetype": "CENTER"})
        count += 3

    # Bend lines
    n_bends = analysis.get("bendLines", 0)
    for i in range(n_bends):
        yp = H * (i + 1) / (n_bends + 1)
        msp.add_line((0, yp), (W, yp), dxfattribs={"layer": "BEND_LINES", "color": 2, "linetype": "DASHED"})
        count += 1

    # Dimensions
    try:
        dw = msp.add_linear_dim(base=(W/2,-18), p1=(0,0), p2=(W,0), angle=0,
                                 dimstyle="Standard", override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dw.set_text(f"{W:.1f}mm"); dw.render(); count += 1
    except Exception: pass
    try:
        dh = msp.add_linear_dim(base=(-18,H/2), p1=(0,0), p2=(0,H), angle=90,
                                 dimstyle="Standard", override={"dimtxt":4,"dimasz":3,"dimexe":2,"dimexo":1.5})
        dh.set_text(f"{H:.1f}mm"); dh.render(); count += 1
    except Exception: pass

    # Diameter labels
    for (cx, cy, r) in placed:
        try:
            msp.add_text(f"Ø{r*2:.1f}",
                         dxfattribs={"layer":"DIMENSIONS","height":3.5,"insert":(cx, cy-r-8),"halign":1,"valign":0})
            count += 1
        except Exception: pass

    # Title block
    hole_d = analysis.get("holesDiameter", 6) or 6
    tb_x, tb_y = W + 25, 0
    for tx, ty, text, ht in [
        (tb_x, tb_y+60, f"PART: {analysis.get('profileType','PART').upper()}", 5.0),
        (tb_x, tb_y+50, f"W × H: {W:.1f} × {H:.1f} mm", 3.5),
        (tb_x, tb_y+42, f"THICKNESS: {analysis.get('thickness',2.0):.1f} mm", 3.5),
        (tb_x, tb_y+34, f"CIRCLES: {len(ai_circles or cv_circles)} × Ø{hole_d:.1f}", 3.5),
        (tb_x, tb_y+26, f"MATERIAL: {analysis.get('material','—')}", 3.5),
        (tb_x, tb_y+18, f"TOLERANCE: {analysis.get('tolerance','±0.1mm')}", 3.5),
        (tb_x, tb_y+10, f"CONFIDENCE: {analysis.get('confidence',0)}%", 3.0),
        (tb_x, tb_y+ 2, "SheetForge v6.0 — OpenCV DXF Pipeline", 2.5),
    ]:
        msp.add_text(text, dxfattribs={"layer": "TITLE_BLOCK", "height": ht, "insert": (tx, ty)})
        count += 1

    if analysis.get("notes"):
        msp.add_text(f"NOTE: {str(analysis['notes'])[:120]}",
                     dxfattribs={"layer": "NOTES", "height": 2.5, "insert": (0, -30)})
        count += 1

    return doc, count


def validate_dxf(doc):
    if doc is None: return False, []
    try:
        auditor = doc.audit()
        errors  = [str(e) for e in auditor.errors]
        return len(errors) == 0, errors
    except Exception as e:
        return False, [str(e)]


# ════════════════════════════════════════════════════════════════════════════════
# PDF export  (reportlab)  — displayed in <div id="dwg_main_viewer">
# ════════════════════════════════════════════════════════════════════════════════

def export_pdf(analysis, output_path, dpi=96):
    """
    Export a technical drawing PDF using reportlab.
    This PDF is what gets displayed inside <div id="dwg_main_viewer"> via an
    <iframe> / <embed> injected by the frontend after conversion.
    Renders: outline, circles with centre-lines, bend lines,
             dimension annotations, title block.
    """
    if not HAS_RL: return False
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor

        W = float(analysis.get("width",  200))
        H = float(analysis.get("height", 150))

        margin   = 20 * mm
        tb_width = 60 * mm
        page_w   = W * mm + 2 * margin + tb_width
        page_h   = H * mm + 2 * margin + 15 * mm

        c = rl_canvas.Canvas(str(output_path), pagesize=(page_w, page_h))
        orig_x = margin
        orig_y = margin + 12 * mm

        def dx(x_mm): return orig_x + x_mm * mm
        def dy(y_mm): return orig_y + y_mm * mm

        # Background
        c.setFillColor(HexColor("#0d1117"))
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        # Grid dots
        c.setFillColor(HexColor("#1e2d40"))
        gx = orig_x
        while gx <= orig_x + W * mm:
            gy = orig_y
            while gy <= orig_y + H * mm:
                c.circle(gx, gy, 0.7 * mm, fill=1, stroke=0)
                gy += 10 * mm
            gx += 10 * mm

        # Outline
        c.setStrokeColor(HexColor("#3d7eff"))
        c.setLineWidth(2)
        c.rect(dx(0), dy(0), W * mm, H * mm, fill=0)

        ai_circles = analysis.get("_ai_circles", [])
        cv_circles = analysis.get("_cv_circles", [])
        sm_holes   = analysis.get("_sm_holes",   [])

        def draw_circle_pdf(cx_mm, cy_mm, r_mm, color_hex, label=None):
            cx   = dx(cx_mm); cy = dy(cy_mm)
            r    = max(r_mm * mm, 1.5 * mm)
            cross = r * 1.55
            c.setStrokeColor(HexColor(color_hex)); c.setLineWidth(1.5)
            c.circle(cx, cy, r, fill=0)
            c.setStrokeColor(HexColor("#a855f7")); c.setLineWidth(0.6); c.setDash(3, 2)
            c.line(cx - cross, cy, cx + cross, cy)
            c.line(cx, cy - cross, cx, cy + cross)
            c.setDash()
            if label:
                c.setFont("Helvetica", 6)
                c.setFillColor(HexColor("#6b7a9b"))
                c.drawCentredString(cx, cy - r - 4 * mm, label)
                c.setFillColor(HexColor("#ffffff"))

        if ai_circles:
            for ci in ai_circles:
                d = float(ci.get("diameter_mm", 0) or 0)
                draw_circle_pdf(float(ci.get("cx_mm", 0)), float(ci.get("cy_mm", 0)),
                                d / 2, "#00d4a0" if d >= 20 else "#22c55e", f"Ø{d:.0f}")
        elif cv_circles:
            scale = dpi / 25.4
            for ci in cv_circles:
                cx_mm = float(ci["cx"]) / scale
                cy_mm = float(ci["cy"]) / scale
                r_mm  = float(ci.get("confirmed_r_mm", float(ci["r"]) / scale))
                draw_circle_pdf(cx_mm, cy_mm, r_mm, "#00d4a0", f"Ø{r_mm*2:.0f}")
        for sh in sm_holes:
            r = float(sh.get("diameter_mm", 10) or 10) / 2.0
            draw_circle_pdf(float(sh.get("cx_mm", 0)), float(sh.get("cy_mm", 0)),
                            r, "#22c55e", f"Ø{r*2:.0f}")

        # Bend lines
        n_bends = analysis.get("bendLines", 0)
        c.setStrokeColor(HexColor("#f59e0b")); c.setLineWidth(1.2); c.setDash(6, 3)
        for i in range(n_bends):
            yp = H * (i + 1) / (n_bends + 1)
            c.line(dx(0), dy(yp), dx(W), dy(yp))
        c.setDash()

        # Width dimension
        c.setStrokeColor(HexColor("#6b7a9b")); c.setLineWidth(0.7)
        dim_y = dy(0) - 9 * mm
        c.line(dx(0), dim_y, dx(W), dim_y)
        c.setFont("Helvetica", 7); c.setFillColor(HexColor("#6b7a9b"))
        c.drawCentredString(dx(W / 2), dim_y - 4 * mm, f"{W:.1f} mm")

        # Height dimension
        dim_x = dx(W) + 8 * mm
        c.line(dim_x, dy(0), dim_x, dy(H))
        c.saveState(); c.translate(dim_x + 4 * mm, dy(H / 2)); c.rotate(90)
        c.drawCentredString(0, 0, f"{H:.1f} mm"); c.restoreState()

        # Title block
        tb_x = orig_x + W * mm + 8 * mm
        tb_y = dy(H) - 5 * mm
        c.setFillColor(HexColor("#111827"))
        c.rect(tb_x - 3*mm, dy(0) - 2*mm, tb_width, H*mm + 4*mm, fill=1, stroke=0)
        c.setStrokeColor(HexColor("#1e3a5f")); c.setLineWidth(0.5)
        c.rect(tb_x - 3*mm, dy(0) - 2*mm, tb_width, H*mm + 4*mm, fill=0)
        hole_d = analysis.get("holesDiameter", 6) or 6
        lines_tb = [
            ("Helvetica-Bold", 8, "#3d7eff",  f"{analysis.get('profileType','PART').upper()}"),
            ("Helvetica",      7, "#e2e8f0",  f"W × H: {W:.1f} × {H:.1f} mm"),
            ("Helvetica",      7, "#e2e8f0",  f"Thickness: {analysis.get('thickness',2.0):.1f} mm"),
            ("Helvetica",      7, "#e2e8f0",  f"Holes: {len(ai_circles or cv_circles)} × Ø{hole_d:.1f}"),
            ("Helvetica",      7, "#e2e8f0",  f"Material: {analysis.get('material','—')}"),
            ("Helvetica",      7, "#e2e8f0",  f"Tolerance: {analysis.get('tolerance','±0.1mm')}"),
            ("Helvetica",      7, "#94a3b8",  f"Confidence: {analysis.get('confidence',0):.0f}%"),
            ("Helvetica",      6, "#475569",  "SheetForge v6.0"),
        ]
        ty = tb_y
        for font, size, color, text in lines_tb:
            c.setFont(font, size); c.setFillColor(HexColor(color))
            c.drawString(tb_x, ty, text)
            ty -= (size + 3) * mm

        c.save()
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════════
# G-Code generation  (unchanged from v5.0)
# ════════════════════════════════════════════════════════════════════════════════

def generate_gcode(analysis, options=None):
    opts        = options or {}
    W           = float(analysis.get("width",   200))
    H           = float(analysis.get("height",  150))
    feed_rate   = float(opts.get("feedRate",    1000))
    plunge_rate = float(opts.get("plungeRate",  300))
    spindle_rpm = int(opts.get("spindleRpm",    12000))
    cut_depth   = float(opts.get("cutDepth",    3.0))
    pass_depth  = float(opts.get("passDepth",   1.0))
    safe_z      = float(opts.get("safeZ",       5.0))
    tool_dia    = float(opts.get("toolDiameter",3.0))
    operation   = opts.get("operation",         "cut")
    ai_circles  = analysis.get("_ai_circles",   [])
    sm_holes    = analysis.get("_sm_holes",     [])
    ts          = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"; SheetForge v6.0 — G-Code Export",
        f"; Generated: {ts}",
        f"; Part: {analysis.get('profileType','PART').upper()} | Material: {analysis.get('material','unknown')}",
        f"; Dimensions: {W:.2f} × {H:.2f} mm | Thickness: {analysis.get('thickness',2.0):.2f}mm",
        f"; Tolerance: {analysis.get('tolerance','±0.1mm')} | Operation: {operation}",
        f"; Tool Diameter: {tool_dia}mm | Feed: {feed_rate}mm/min | Spindle: {spindle_rpm}rpm",
        "; =====================================================",
        "", "G21","G17","G90","G94","G40","G49","",
        "T01 M6","G43 H01",f"S{spindle_rpm} M3","G4 P2000","",
        f"G00 Z{safe_z:.2f}","G00 X0.000 Y0.000","",
    ]
    if operation in ("cut", "laser"):
        lines += ["; === OUTLINE ===","G00 X0.000 Y0.000",f"G00 Z{safe_z:.2f}"]
        for p in range(1, math.ceil(cut_depth / pass_depth) + 1):
            z = min(-p * pass_depth, -cut_depth)
            lines += [
                f"", f"G00 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}",
                f"G01 Z{z:.3f} F{plunge_rate:.0f}",
                f"G01 X{W+tool_dia/2:.3f} Y0.000 F{feed_rate:.0f}",
                f"G01 X{W+tool_dia/2:.3f} Y{H+tool_dia/2:.3f}",
                f"G01 X-{tool_dia/2:.3f} Y{H+tool_dia/2:.3f}",
                f"G01 X-{tool_dia/2:.3f} Y-{tool_dia/2:.3f}",
                f"G00 Z{safe_z:.2f}",
            ]
    if operation in ("cut", "drill"):
        for ci in ai_circles:
            cx = float(ci.get("cx_mm", 0)); cy = H - float(ci.get("cy_mm", 0))
            d  = float(ci.get("diameter_mm", 6)); r = d / 2
            lines += [f"", f"G00 X{cx:.3f} Y{cy:.3f}", f"G00 Z{safe_z:.2f}"]
            if d <= tool_dia * 1.1:
                lines += [f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}", "G80"]
            elif r - tool_dia / 2 > 0:
                ar = r - tool_dia / 2
                for p in range(1, math.ceil(cut_depth / pass_depth) + 1):
                    z = min(-p * pass_depth, -cut_depth)
                    lines += [f"G00 X{cx+ar:.3f} Y{cy:.3f}",
                              f"G01 Z{z:.3f} F{plunge_rate:.0f}",
                              f"G02 X{cx+ar:.3f} Y{cy:.3f} I-{ar:.3f} J0.000 F{feed_rate:.0f}",
                              f"G00 Z{safe_z:.2f}"]
        for sh in sm_holes:
            cx = float(sh.get("cx_mm", 0)); cy = H - float(sh.get("cy_mm", 0))
            lines += [f"G81 X{cx:.3f} Y{cy:.3f} Z-{cut_depth:.3f} R{safe_z:.3f} F{plunge_rate:.0f}", "G80"]
    n_bends = analysis.get("bendLines", 0)
    if n_bends:
        lines.append("; === BEND LINES ===")
        for i in range(n_bends):
            yp = H * (i + 1) / (n_bends + 1)
            lines += [f"G00 X0.000 Y{yp:.3f}",f"G01 Z-0.300 F{plunge_rate//2:.0f}",
                      f"G01 X{W:.3f} Y{yp:.3f} F{feed_rate//2:.0f}",f"G00 Z{safe_z:.2f}"]
    lines += ["", f"G00 Z{safe_z:.2f}", "G00 X0.000 Y0.000", "M5", "M30"]
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

    # ── AI interact mode (unchanged signature) ───────────────────────────────
    if opts.get("mode") == "ai_interact":
        instruction      = opts.get("instruction", "")
        current_analysis = opts.get("analysis", {})
        updated, explanation = ai_dxf_interaction(instruction, current_analysis, image_path)
        doc, entity_count = build_dxf(updated, float(opts.get("dpi", 96)))
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
            "mode": "ai_interact",
            "analysis": {k: v for k, v in updated.items() if not k.startswith("_")},
            "explanation": explanation,
            "dwg": {"entities": entity_count, "fileSize": file_size},
            "gcode": generate_gcode(updated, opts.get("gcodeOptions")),
        }, ensure_ascii=False))
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
    steps = []
    dpi   = float(opts.get("dpi", 96))

    # PRE-1: Load
    t0 = now_ms()
    img, detected_dpi = load_image(image_path)
    if detected_dpi > 1: dpi = detected_dpi
    img_h, img_w = (img.shape[:2] if img is not None else (0, 0))
    steps.append(step_record("PRE-1: Image Load", f"{img_w}×{img_h}px @ {dpi:.0f}dpi", t0))

    # PRE-2: Grayscale
    t0 = now_ms()
    gray = to_grayscale(img)
    steps.append(step_record("PRE-2: Grayscale (cvtColor)", "BGR → GRAY", t0))

    # PRE-3: Binarise
    t0 = now_ms()
    binary = binarise(gray)
    nonzero = int(np.count_nonzero(binary)) if (binary is not None and HAS_CV) else 0
    steps.append(step_record(
        "PRE-3: Binarise (adaptiveThreshold + THRESH_BINARY_INV)",
        f"{nonzero} ink pixels", t0
    ))

    # PRE-4: Smudge removal
    t0 = now_ms()
    binary = remove_smudges(binary)
    steps.append(step_record("PRE-4: Smudge Removal (morphologyEx OPEN+CLOSE)", "Cleaned binary", t0))

    # CV-1: Edges
    t0 = now_ms()
    edges = detect_edges(gray)
    n_edge = int(np.count_nonzero(edges)) if (edges is not None and HAS_CV) else 0
    steps.append(step_record("CV-1: Edge Detection (Canny + dilate)", f"{n_edge} edge pixels", t0))

    # CV-2: Lines
    t0 = now_ms()
    all_lines = detect_lines(edges)
    steps.append(step_record("CV-2: Line Detection (HoughLines)", f"{len(all_lines)} lines", t0))

    # CV-3: Circles
    t0 = now_ms()
    circles = detect_circles(gray)
    steps.append(step_record("CV-3: Circle Detection (HoughCircles ×2)", f"{len(circles)} circles", t0))

    # CV-4: Shapes
    t0 = now_ms()
    simplified_contours, shapes = extract_shapes(binary)
    circ_count = sum(1 for s in shapes if s["type"] in ("circle", "ellipse"))
    steps.append(step_record(
        "CV-4: Shape Extraction (findContours + approxPolyDP)",
        f"{len(shapes)} shapes, {circ_count} circular", t0
    ))

    # CV-5: Symbol matching
    t0 = now_ms()
    symbols = match_symbols(binary)
    steps.append(step_record(
        "CV-5: Symbol Matching (matchTemplate)",
        f"{len(symbols)} symbols detected", t0
    ))

    # CV-6: Dimensions
    t0 = now_ms()
    features, scale_px_mm, img_w_mm, img_h_mm = calculate_dimensions(
        simplified_contours, circles, dpi, img.shape if img is not None else (0, 0, 0)
    )
    steps.append(step_record(
        "CV-6: Dimension Calculation (minAreaRect + px→mm)",
        f"Scale={scale_px_mm:.3f}px/mm  Image={img_w_mm:.1f}×{img_h_mm:.1f}mm", t0
    ))

    # OCR
    t0 = now_ms()
    ocr_result = ocr_dimensions(img)
    steps.append(step_record("OCR: Tesseract Dimension Extraction",
                              f"{len(ocr_result.get('tokens',[]))} tokens", t0))

    # Calibration
    t0 = now_ms()
    w_mm, h_mm = calibrate(features, ocr_result.get("dims", {}), dpi, img_w_mm, img_h_mm)
    steps.append(step_record("CAL: Pixel→mm Calibration",
                              f"W={w_mm:.1f}mm H={h_mm:.1f}mm", t0))

    # Gemini Vision
    t0 = now_ms()
    cv_ctx = {
        "circles": circles[:12], "lines": all_lines[:20],
        "img_w": img_w, "img_h": img_h,
        "width_mm": w_mm, "height_mm": h_mm, "dpi": dpi,
        "ocr_dims": ocr_result.get("dims", {}),
    }
    ai_data = {}
    if image_path and os.path.exists(image_path):
        ai_data = gemini_vision_analysis(image_path, cv_ctx)
    steps.append(step_record("AI: Gemini Vision Analysis",
                              f"conf={ai_data.get('confidence',0):.2f}", t0))

    # Merge
    t0 = now_ms()
    analysis = merge_analysis(ai_data, ocr_result, w_mm, h_mm, circles, all_lines, features, opts)
    analysis["_dpi"] = dpi
    steps.append(step_record("MERGE: Data Fusion",
                              f"W={analysis['width']}mm H={analysis['height']}mm holes={analysis['holes']}", t0))

    # ── Output paths ─────────────────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        orig_dir = Path(image_path).parent
    else:
        orig_dir = Path(__file__).parent.parent / "uploads" / "output"

    server_out_dir = Path(__file__).parent.parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    ts_str   = int(time.time())
    dxf_name = f"design_{ts_str}.dxf"
    pdf_name = f"design_{ts_str}.pdf"
    gc_name  = f"design_{ts_str}.gcode"

    dxf_path = server_out_dir / dxf_name
    pdf_path = server_out_dir / pdf_name
    gc_path  = server_out_dir / gc_name

    # DXF Build
    t0 = now_ms()
    doc, entity_count = build_dxf(analysis, dpi)
    steps.append(step_record("DXF: Build (ezdxf R2018)", f"{entity_count} entities", t0))

    t0 = now_ms()
    valid, errors = validate_dxf(doc)
    steps.append(step_record("DXF: Validation", "Valid" if valid else f"Warnings: {len(errors)}", t0))

    # Save DXF
    dxf_str = ""; file_size = 0
    if doc is not None:
        try:
            doc.saveas(str(dxf_path))
            file_size = dxf_path.stat().st_size
            with open(dxf_path) as f: dxf_str = f.read()
            import shutil
            if str(orig_dir / dxf_name) != str(dxf_path):
                shutil.copy2(str(dxf_path), str(orig_dir / dxf_name))
        except Exception as e:
            dxf_str = f"; ERROR: {e}"

    # PDF Export
    t0 = now_ms()
    pdf_ok = export_pdf(analysis, pdf_path, dpi)
    steps.append(step_record(
        "PDF: Export (reportlab → dwg_main_viewer)",
        "OK" if pdf_ok else "FAILED", t0
    ))

    # G-Code
    t0 = now_ms()
    gcode_str = generate_gcode(analysis, opts.get("gcodeOptions"))
    with open(gc_path, "w") as f: f.write(gcode_str)
    steps.append(step_record("GCODE: G-Code Generation", f"{len(gcode_str.splitlines())} lines", t0))

    public_analysis = {k: v for k, v in analysis.items() if not k.startswith("_")}

    print(json.dumps({
        "steps":    steps,
        "analysis": public_analysis,
        "dwg": {
            "entities":      entity_count,
            "fileSize":      file_size,
            "filename":      dxf_name if file_size else "",
            "pdfFilename":   pdf_name if pdf_ok else "",
            "gcodeFilename": gc_name,
            "localPath":     str(orig_dir / dxf_name),
        },
        "dxfContent":     dxf_str[:8000] if dxf_str else "",   # first 8 KB for debugging
        "dxfAvailable":   file_size > 0,
        "pdfAvailable":   pdf_ok,
        "gcodeAvailable": True,
        "gcode":          gcode_str,
        "symbols":        symbols[:20],
        "shapes":         shapes[:20],
        "features":       features[:30],
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
            "dwg":       {"entities": 0, "fileSize": 0},
            "gcode":     "",
        }))
        sys.exit(1)
