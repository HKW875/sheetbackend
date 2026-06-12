#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v7.0
================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

Pipeline (precision contour → CAD):
  1. Load image              (cv2.imread)
  2. Median Blur             (cv2.medianBlur — salt-and-pepper noise reduction)
  3. Adaptive Threshold      (cv2.adaptiveThreshold — binarisation, lines=WHITE)
  4. Morph Clean             (cv2.morphologyEx MORPH_OPEN — speckle removal)
  5. Canny Edge Detection    (cv2.Canny — thin precise edges)
  6. Contour Extraction      (cv2.findContours → cv2.approxPolyDP simplification)
  7. DXF Export              (ezdxf — every contour as LWPOLYLINE in CAD space)
                              1 pixel of Canny image = 1 drawing unit
                              Origin (0,0) = bottom-left corner of Canny image
  8. PDF Export              (reportlab — edge image rendered to PDF page)
  9. PNG Preview             (dark-bg canvas with white edges — saved as .png)
"""

import sys, os, json, time, traceback
from pathlib import Path

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
    """
    Load image with cv2.imread.
    Returns (bgr, gray, dpi, img_w, img_h).
    DPI is read from EXIF via Pillow if available; defaults to 96.
    """
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
    """
    cv2.medianBlur — removes salt-and-pepper noise while preserving edges.
    ksize must be odd and positive (default 5).
    Returns blurred grayscale image.
    """
    if ksize % 2 == 0:
        ksize += 1          # enforce odd kernel
    blurred = cv2.medianBlur(gray, ksize)
    return blurred


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — ADAPTIVE THRESHOLD + BINARISATION + INVERSION
# ════════════════════════════════════════════════════════════════════════════════

def adaptive_threshold_binarize(blurred):
    """
    cv2.adaptiveThreshold — locally adaptive binarisation coping with uneven
    lighting across a hand-drawn sketch.

    Strategy:
      • ADAPTIVE_THRESH_GAUSSIAN_C: weights pixels by Gaussian kernel for smoother
        threshold map.
      • THRESH_BINARY_INV: dark ink lines on light paper  →  white lines on black
        background (the correct format for Canny and contour extraction).
      • blockSize=15 (odd), C=4: tuned for typical sketch scan DPI 96–300.

    Returns binary image where lines are WHITE (255) and background is BLACK (0).
    """
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # lines → white, background → black
        blockSize=15,
        C=4,
    )
    return binary


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — MORPHOLOGICAL OPEN (SPECKLE REMOVAL)
# ════════════════════════════════════════════════════════════════════════════════

def morph_clean(binary):
    """
    cv2.morphologyEx with MORPH_OPEN (erosion then dilation).
    Removes isolated white speckles (noise pixels smaller than the structuring
    element) while preserving actual line strokes.
    Kernel: 3×3 cross (MORPH_CROSS) — tight enough to eliminate small speckles
    without eroding fine stroke detail.
    Returns cleaned binary image.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — CANNY EDGE DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def canny_edges(cleaned, low_threshold=30, high_threshold=100):
    """
    cv2.Canny — double-threshold hysteresis edge detector applied to the
    cleaned binary image.

    Because input is already binarised (0/255), Canny mostly thins the white
    line regions down to their 1-pixel-wide centre-line edges, producing a
    clean, crisp edge map ideal for CAD export.

    low_threshold  : weak-edge lower bound  (default 30)
    high_threshold : strong-edge upper bound (default 100)
    Returns binary edge map (0 = no edge, 255 = edge).
    """
    edges = cv2.Canny(cleaned, low_threshold, high_threshold)
    return edges


# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — CONTOUR EXTRACTION + approxPolyDP SIMPLIFICATION
# ════════════════════════════════════════════════════════════════════════════════

def extract_simplified_contours(edges, epsilon_factor=0.5):
    """
    cv2.findContours  — extracts sequences of edge pixels as contours.
    cv2.approxPolyDP  — Douglas-Peucker polygon approximation reduces noise
                        and point count while preserving geometry shape.

    epsilon = epsilon_factor × contour arc length (default 0.5px tolerance).
    Returns list of simplified contour arrays.
    """
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,          # all contours — no hierarchy filtering
        cv2.CHAIN_APPROX_NONE,  # collect every edge pixel (dense chain)
    )

    simplified = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        arc      = cv2.arcLength(cnt, closed=False)
        epsilon  = epsilon_factor * arc / max(len(cnt), 1)
        epsilon  = max(epsilon, 0.3)   # floor: at least 0.3px tolerance
        approx   = cv2.approxPolyDP(cnt, epsilon, closed=False)
        if len(approx) >= 2:
            simplified.append(approx)

    return simplified


# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — DXF EXPORT
# ════════════════════════════════════════════════════════════════════════════════
#
# Coordinate system:
#   OpenCV pixel (px_x, px_y):  origin = top-left,  Y grows downward
#   CAD drawing unit (cad_x, cad_y): origin = bottom-left, Y grows upward
#
#   Mapping (1 pixel = 1 drawing unit):
#     cad_x = px_x
#     cad_y = (image_height - 1) - px_y          ← flip Y axis
#
# The DXF is set to INSUNITS=0 (unitless / generic drawing units) so FreeCAD
# will open it at exact 1:1 scale without any unit-conversion distortion.
# If you want millimetres, change $INSUNITS to 4 and add a DPI-based scale factor.
# ════════════════════════════════════════════════════════════════════════════════

def px_to_cad(px_x, px_y, img_h):
    """
    Convert OpenCV raster pixel (px_x, px_y) to CAD coordinates.
    Origin (0, 0) is placed at the bottom-left corner of the Canny image.
    1 pixel = 1 drawing unit (unitless — FreeCAD interprets as-is).
    Y axis is flipped so CAD Y increases upward.
    """
    cad_x = float(px_x)
    cad_y = float((img_h - 1) - px_y)
    return cad_x, cad_y


def build_and_save_dxf(simplified_contours, img_w, img_h, out_path):
    """
    Convert simplified contours to a DXF file using ezdxf.

    Each contour → one LWPOLYLINE entity on layer "EDGES".
    Coordinates are CAD units (1 pixel = 1 drawing unit, Y flipped).
    DXF version R2018, INSUNITS=0 (unitless) for direct FreeCAD import.

    Returns (doc, entity_count, file_size_bytes).
    """
    if not HAS_DXF:
        return None, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    # INSUNITS=0 → unitless; FreeCAD accepts this for 1:1 pixel-coordinate DXF.
    # Use INSUNITS=4 (mm) only when coordinates are already in millimetres.
    doc.header["$INSUNITS"] = 0
    # Set drawing extents so FreeCAD can auto-zoom to fit
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (float(img_w), float(img_h), 0.0)
    doc.header["$LIMMIN"] = (0.0, 0.0)
    doc.header["$LIMMAX"] = (float(img_w), float(img_h))

    msp = doc.modelspace()

    # Ensure layer exists with a visible colour
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
            # Two-point contours → LINE entity (most precise for short segments)
            msp.add_line(
                start=pts[0],
                end=pts[1],
                dxfattribs={"layer": "EDGES", "color": 256},   # 256 = by-layer
            )
        else:
            # Multi-point contours → LWPOLYLINE
            msp.add_lwpolyline(
                pts,
                format="xy",
                dxfattribs={"layer": "EDGES", "color": 256},
            )
        entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — PDF EXPORT  (edge image → PDF page via reportlab)
# ════════════════════════════════════════════════════════════════════════════════

def export_pdf(edges, out_path, orig_bgr=None):
    """
    Render the Canny edge map (and optionally the original image side-by-side)
    to a PDF page using reportlab.

    Layout (A4 landscape):
      Left half  — original image (if available)
      Right half — Canny edge map

    Returns True on success, False on failure.
    """
    if not HAS_RL:
        return False

    import tempfile, os as _os
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    try:
        page_w, page_h = landscape(A4)   # 841.89 × 595.28 pt
        c = rl_canvas.Canvas(str(out_path), pagesize=(page_w, page_h))

        margin    = 30
        col_w     = (page_w - margin * 3) / 2
        col_h     = page_h - margin * 2 - 40
        top_y     = page_h - margin - 30

        # ── Title bar ─────────────────────────────────────────────────────────
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

        # ── Left panel: original image ─────────────────────────────────────────
        left_x = margin
        if orig_bgr is not None:
            reader, tname = _arr_to_reader(orig_bgr)
            tmp_files.append(tname)
            _draw_panel(c, reader, left_x, margin, col_w, col_h, "Original Image")
        else:
            c.setFillColorRGB(0.08, 0.1, 0.13)
            c.roundRect(left_x, margin, col_w, col_h, 6, fill=1, stroke=0)
            c.setFillColorRGB(0.35, 0.4, 0.45)
            c.setFont("Helvetica", 10)
            c.drawCentredString(left_x + col_w / 2, margin + col_h / 2, "No original image")

        # ── Right panel: edge map ──────────────────────────────────────────────
        right_x = margin * 2 + col_w
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        reader, tname = _arr_to_reader(edges_bgr)
        tmp_files.append(tname)
        _draw_panel(c, reader, right_x, margin, col_w, col_h, "Canny Edge Detection")

        # ── Footer ────────────────────────────────────────────────────────────
        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 12,
                            "SheetForge v7.0  •  medianBlur → adaptiveThreshold → MORPH_OPEN → Canny  •  ezdxf R2018")

        c.save()

        for f in tmp_files:
            try: _os.unlink(f)
            except Exception: pass

        return True

    except Exception as e:
        sys.stderr.write(f"PDF export error: {e}\n{traceback.format_exc()}\n")
        return False


def _draw_panel(c, img_reader, x, y, w, h, title):
    """Draw a labelled image panel onto the reportlab canvas."""
    title_h = 22
    img_y   = y + title_h
    img_h   = h - title_h

    c.setFillColorRGB(0.08, 0.1, 0.13)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)

    c.setFillColorRGB(0.12, 0.15, 0.2)
    c.roundRect(x, y + img_h, w, title_h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.55, 0.65, 0.85)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + w / 2, y + img_h + 7, title)

    pad  = 8
    iw_  = w - pad * 2
    ih_  = img_h - pad * 2
    c.drawImage(img_reader, x + pad, img_y + pad,
                width=iw_, height=ih_, preserveAspectRatio=True, anchor="c",
                mask="auto")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse args ────────────────────────────────────────────────────────────
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    opts       = {}
    if len(sys.argv) > 2:
        try: opts = json.loads(sys.argv[2])
        except Exception: pass

    # Pipeline tuning via opts (all have sensible defaults)
    blur_ksize      = int(opts.get("blurKsize",    5))
    canny_low       = int(opts.get("cannyLow",    30))
    canny_high      = int(opts.get("cannyHigh",  100))
    epsilon_factor  = float(opts.get("epsilonFactor", 0.5))

    steps = []

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record(
        "CV-1: Load Image (cv2.imread)",
        f"{img_w}×{img_h}px  DPI={dpi:.0f}",
        t0
    ))

    # ── STEP 2: Median Blur ───────────────────────────────────────────────────
    t0 = now_ms()
    blurred = median_blur(gray, ksize=blur_ksize)
    steps.append(step_record(
        f"CV-2: Median Blur (ksize={blur_ksize})",
        "Salt-and-pepper noise reduced, edges preserved",
        t0
    ))

    # ── STEP 3: Adaptive Threshold + Binarise + Invert ───────────────────────
    t0 = now_ms()
    binary = adaptive_threshold_binarize(blurred)
    white_px = int(np.count_nonzero(binary))
    steps.append(step_record(
        "CV-3: Adaptive Threshold + Binarise (THRESH_BINARY_INV)",
        f"Lines → white on black  |  {white_px} white pixels",
        t0
    ))

    # ── STEP 4: Morphological Open (speckle removal) ─────────────────────────
    t0 = now_ms()
    cleaned = morph_clean(binary)
    cleaned_px = int(np.count_nonzero(cleaned))
    removed_px = white_px - cleaned_px
    steps.append(step_record(
        "CV-4: morphologyEx MORPH_OPEN (speckle removal)",
        f"{removed_px} speckle pixels removed  |  {cleaned_px} clean pixels remain",
        t0
    ))

    # ── STEP 5: Canny Edge Detection ──────────────────────────────────────────
    t0 = now_ms()
    edges = canny_edges(cleaned, canny_low, canny_high)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(
        f"CV-5: Canny Edge Detection (low={canny_low}, high={canny_high})",
        f"{edge_px} edge pixels  |  1 px = 1 drawing unit  |  origin = bottom-left",
        t0
    ))

    # ── STEP 6: Contour Extraction + approxPolyDP ─────────────────────────────
    t0 = now_ms()
    simplified_contours = extract_simplified_contours(edges, epsilon_factor)
    total_pts = sum(len(c) for c in simplified_contours)
    steps.append(step_record(
        f"CV-6: findContours + approxPolyDP (ε={epsilon_factor})",
        f"{len(simplified_contours)} contours  |  {total_pts} simplified vertices",
        t0
    ))

    # ── Output directories ────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent.parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str    = int(time.time())
    dxf_name  = f"design_{ts_str}.dxf"
    pdf_name  = f"design_{ts_str}.pdf"
    png_name  = f"preview_{ts_str}.png"        # .png extension enforced
    dxf_path  = server_out_dir / dxf_name
    pdf_path  = server_out_dir / pdf_name
    png_path  = server_out_dir / png_name      # always .png — Cloudinary compatible

    # ── STEP 7: DXF Export ────────────────────────────────────────────────────
    t0 = now_ms()
    doc, entity_count, file_size = build_and_save_dxf(
        simplified_contours, img_w, img_h, dxf_path
    )
    dxf_str = ""
    if file_size > 0:
        try:
            with open(dxf_path, encoding="utf-8") as f:
                dxf_str = f.read()
        except Exception:
            pass
    steps.append(step_record(
        "DXF: Export contours → LWPOLYLINE/LINE entities (ezdxf R2018)",
        (f"{entity_count} entities  |  {file_size // 1024 if file_size else 0} KB  |"
         f"  units=unitless  |  origin=bottom-left  |  1px=1du"),
        t0
    ))

    # ── STEP 8: PDF Export ────────────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, pdf_path, orig_bgr=bgr)
    steps.append(step_record(
        "PDF: Export edge preview (reportlab A4 landscape)",
        "OK" if pdf_ok else "FAILED — check reportlab install",
        t0
    ))

    # ── STEP 9: PNG Preview Export (.png — for Cloudinary and viewer) ─────────
    # White edges on dark (#0a0c0f) background, saved as PNG format.
    # This is the image that gets uploaded to Cloudinary for storage and displayed
    # in the viewer panel via /preview-inline.
    t0 = now_ms()
    png_ok   = False
    png_size = 0
    try:
        preview_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        preview_canvas[:] = (15, 12, 10)              # BGR equiv of #0a0c0f dark bg
        edge_mask = edges > 0
        preview_canvas[edge_mask] = (255, 255, 255)   # white edges
        # cv2.imwrite with .png path → lossless PNG format
        success = cv2.imwrite(str(png_path), preview_canvas)
        if success and png_path.exists():
            png_ok   = True
            png_size = png_path.stat().st_size
        else:
            sys.stderr.write(f"PNG write returned False for path: {png_path}\n")
    except Exception as png_err:
        sys.stderr.write(f"PNG preview export error: {png_err}\n")
    steps.append(step_record(
        "PNG: Save Canny edge preview (.png) for Cloudinary + viewer",
        f"OK — {png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED",
        t0
    ))

    # ── Analysis summary ──────────────────────────────────────────────────────
    # Drawing dimensions in raw pixel/drawing-unit space (1px = 1du)
    analysis = {
        "width"         : float(img_w),      # drawing units = pixels
        "height"        : float(img_h),      # drawing units = pixels
        "dpi"           : dpi,
        "edgePixels"    : edge_px,
        "edges"         : entity_count,      # DXF entity count (matches schema field)
        "contours"      : len(simplified_contours),
        "totalVertices" : total_pts,
        "blurKsize"     : blur_ksize,
        "cannyLow"      : canny_low,
        "cannyHigh"     : canny_high,
        "epsilonFactor" : epsilon_factor,
        "imgW"          : img_w,
        "imgH"          : img_h,
        "coordSystem"   : "origin=bottom-left, 1px=1du, Y-up (CAD convention)",
    }

    # ── Output ────────────────────────────────────────────────────────────────
    print(json.dumps({
        "steps"        : steps,
        "analysis"     : analysis,
        "dwg": {
            "entities"        : entity_count,
            "fileSize"        : file_size,
            "filename"        : dxf_name if file_size else "",
            "pdfFilename"     : pdf_name if pdf_ok else "",
            "edgePngFilename" : png_name if png_ok else "",
            "edgePngPath"     : str(png_path) if png_ok else "",
        },
        "dxfContent"   : dxf_str[:50000] if dxf_str else "",   # cap at 50 KB
        "dxfAvailable" : file_size > 0,
        "pdfAvailable" : pdf_ok,
        "pngAvailable" : png_ok,
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
