#!/usr/bin/env python3
"""
SheetForge — CV Pipeline  v6.0
================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, analysis, dwg, dxfContent, pdfAvailable }

Pipeline (lean, focused):
  1. Load image            (cv2.imread)
  2. Gaussian Blur         (cv2.GaussianBlur  — noise reduction)
  3. Canny Edge Detection  (cv2.Canny)
  4. DXF Export            (ezdxf — edges as LINE entities)
  5. PDF Export            (reportlab — edge image rendered to PDF page)
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
# STEP 2 — GAUSSIAN BLUR
# ════════════════════════════════════════════════════════════════════════════════

def gaussian_blur(gray, ksize=5, sigma=0):
    """
    cv2.GaussianBlur — reduces high-frequency noise before edge detection.
    ksize must be odd and positive (default 5×5).
    sigma=0 lets OpenCV auto-calculate from ksize.
    Returns blurred grayscale image.
    """
    if ksize % 2 == 0:
        ksize += 1          # enforce odd kernel
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    return blurred


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — CANNY EDGE DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def canny_edges(blurred, low_threshold=50, high_threshold=150):
    """
    cv2.Canny — double-threshold hysteresis edge detector.
    low_threshold  : weak-edge lower bound  (default 50)
    high_threshold : strong-edge upper bound (default 150)
    Returns binary edge map (0 = no edge, 255 = edge).
    """
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — DXF EXPORT  (edge pixels → LINE entities)
# ════════════════════════════════════════════════════════════════════════════════

def px_to_mm(px, dpi):
    """Convert pixel coordinate to millimetres."""
    return round(px * 25.4 / dpi, 4)


def build_and_save_dxf(edges, dpi, out_path):
    """
    Convert Canny edge map to a DXF file using ezdxf.

    Strategy:
      • Run cv2.findContours on the edge map to get connected edge chains.
      • Each contour is written as a LWPOLYLINE (lightweight polyline) in the
        DXF OBJECTS layer — one entity per contour, pixel coords → mm.
      • Falls back to individual LINE entities if ezdxf is unavailable.

    Returns (doc, entity_count, file_size_bytes).
    """
    if not HAS_DXF:
        return None, 0, 0

    doc = ezdxf.new(dxfversion="R2018")
    doc.header["$INSUNITS"] = 4   # millimetres
    msp = doc.modelspace()

    # Extract contours from the edge map for compact polyline representation
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    entity_count = 0
    h = edges.shape[0]   # image height — used to flip Y so DXF origin = bottom-left

    for cnt in contours:
        if len(cnt) < 2:
            continue
        # Build mm-coordinate list
        pts = []
        for pt in cnt:
            px_x = int(pt[0][0])
            px_y = int(pt[0][1])
            mm_x =  px_to_mm(px_x, dpi)
            mm_y =  px_to_mm(h - px_y, dpi)   # flip Y axis
            pts.append((mm_x, mm_y))

        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "EDGES", "color": 7})
            entity_count += 1

    doc.saveas(str(out_path))
    file_size = out_path.stat().st_size
    return doc, entity_count, file_size


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — PDF EXPORT  (edge image → PDF page via reportlab)
# ════════════════════════════════════════════════════════════════════════════════

def export_pdf(edges, dpi, out_path, orig_bgr=None):
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
        col_w     = (page_w - margin * 3) / 2    # two columns
        col_h     = page_h - margin * 2 - 40     # leave room for title bar
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

        # Helper: save numpy array as temp PNG and return an ImageReader
        def _arr_to_reader(arr):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, arr)
            return ImageReader(tmp.name), tmp.name

        tmp_files = []

        # ── Left panel: original image (if provided) ──────────────────────────
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
        # Convert single-channel edge map to 3-channel for saving as PNG
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        reader, tname = _arr_to_reader(edges_bgr)
        tmp_files.append(tname)
        _draw_panel(c, reader, right_x, margin, col_w, col_h, "Canny Edge Detection")

        # ── Footer ────────────────────────────────────────────────────────────
        c.setFillColorRGB(0.3, 0.35, 0.4)
        c.setFont("Helvetica", 8)
        c.drawCentredString(page_w / 2, 12,
                            "SheetForge v6.0  •  OpenCV Canny  •  ezdxf R2018")

        c.save()

        # Clean up temp PNGs
        for f in tmp_files:
            try: _os.unlink(f)
            except Exception: pass

        return True

    except Exception as e:
        sys.stderr.write(f"PDF export error: {e}\n{traceback.format_exc()}\n")
        return False


def _draw_panel(c, img_reader, x, y, w, h, title):
    """Draw a labelled image panel onto the reportlab canvas."""
    from reportlab.lib.utils import ImageReader

    title_h = 22
    img_y   = y + title_h
    img_h   = h - title_h

    # Panel background
    c.setFillColorRGB(0.08, 0.1, 0.13)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)

    # Title strip
    c.setFillColorRGB(0.12, 0.15, 0.2)
    c.roundRect(x, y + img_h, w, title_h, 6, fill=1, stroke=0)
    c.setFillColorRGB(0.55, 0.65, 0.85)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + w / 2, y + img_h + 7, title)

    # Draw image scaled to fit panel with 8 px padding
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
    blur_ksize      = int(opts.get("blurKsize",      5))
    canny_low       = int(opts.get("cannyLow",       50))
    canny_high      = int(opts.get("cannyHigh",     150))

    steps = []

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    t0 = now_ms()
    bgr, gray, dpi, img_w, img_h = load_image(image_path)
    steps.append(step_record(
        "CV-1: Load Image (cv2.imread)",
        f"{img_w}×{img_h}px  DPI={dpi:.0f}",
        t0
    ))

    # ── STEP 2: Gaussian Blur ─────────────────────────────────────────────────
    t0 = now_ms()
    blurred = gaussian_blur(gray, ksize=blur_ksize)
    steps.append(step_record(
        f"CV-2: Gaussian Blur (kernel {blur_ksize}×{blur_ksize})",
        f"Noise reduced — σ auto from ksize",
        t0
    ))

    # ── STEP 3: Canny Edge Detection ──────────────────────────────────────────
    t0 = now_ms()
    edges = canny_edges(blurred, canny_low, canny_high)
    edge_px = int(np.count_nonzero(edges))
    steps.append(step_record(
        f"CV-3: Canny Edge Detection (low={canny_low}, high={canny_high})",
        f"{edge_px} edge pixels detected",
        t0
    ))

    # ── Output directories ────────────────────────────────────────────────────
    server_out_dir = Path(__file__).parent.parent / "uploads" / "output"
    server_out_dir.mkdir(parents=True, exist_ok=True)

    ts_str    = int(time.time())
    dxf_name  = f"design_{ts_str}.dxf"
    pdf_name  = f"design_{ts_str}.pdf"
    png_name  = f"preview_{ts_str}.png"
    dxf_path  = server_out_dir / dxf_name
    pdf_path  = server_out_dir / pdf_name
    png_path  = server_out_dir / png_name

    # ── STEP 4: DXF Export ────────────────────────────────────────────────────
    t0 = now_ms()
    doc, entity_count, file_size = build_and_save_dxf(edges, dpi, dxf_path)
    dxf_str = ""
    if file_size > 0:
        try:
            with open(dxf_path) as f: dxf_str = f.read()
        except Exception: pass
    steps.append(step_record(
        "DXF: Export edge contours (ezdxf R2018 LWPOLYLINE)",
        f"{entity_count} polyline entities  {file_size // 1024 if file_size else 0} KB",
        t0
    ))

    # ── STEP 5: PDF Export ────────────────────────────────────────────────────
    t0 = now_ms()
    pdf_ok = export_pdf(edges, dpi, pdf_path, orig_bgr=bgr)
    steps.append(step_record(
        "PDF: Export edge preview (reportlab A4 landscape)",
        "OK" if pdf_ok else "FAILED — check reportlab install",
        t0
    ))

    # ── STEP 6: PNG Preview Export ────────────────────────────────────────────
    # Save the Canny edge map as a PNG so the frontend can display it directly
    # in #dwg-main-viewer via the /preview-inline endpoint.
    # We write white edges on a dark (#0a0c0f) background to match the UI theme.
    t0 = now_ms()
    png_ok   = False
    png_size = 0
    try:
        # Create a dark-background RGB canvas and paint edges white on it
        preview_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        preview_canvas[:] = (15, 12, 10)          # BGR equivalent of #0a0c0f
        edge_mask = edges > 0
        preview_canvas[edge_mask] = (255, 255, 255)  # white edges
        cv2.imwrite(str(png_path), preview_canvas)
        if png_path.exists():
            png_ok   = True
            png_size = png_path.stat().st_size
    except Exception as png_err:
        sys.stderr.write(f"PNG preview export error: {png_err}\n")
    steps.append(step_record(
        "PNG: Save Canny edge preview for viewer",
        f"OK — {png_size // 1024 if png_size else 0} KB" if png_ok else "FAILED",
        t0
    ))

    # ── Analysis summary ──────────────────────────────────────────────────────
    scale_px_mm  = dpi / 25.4                          # px per mm
    width_mm     = round(img_w / scale_px_mm, 2)
    height_mm    = round(img_h / scale_px_mm, 2)

    analysis = {
        "width"       : width_mm,
        "height"      : height_mm,
        "dpi"         : dpi,
        "edgePixels"  : edge_px,
        "edges"       : entity_count,   # matches server.js analysis.edges field
        "blurKsize"   : blur_ksize,
        "cannyLow"    : canny_low,
        "cannyHigh"   : canny_high,
        "imgW"        : img_w,
        "imgH"        : img_h,
    }

    # ── Output ────────────────────────────────────────────────────────────────
    print(json.dumps({
        "steps"        : steps,
        "analysis"     : analysis,
        "dwg": {
            "entities"      : entity_count,
            "fileSize"      : file_size,
            "filename"      : dxf_name if file_size else "",
            "pdfFilename"   : pdf_name if pdf_ok else "",
            "edgePngFilename": png_name if png_ok else "",
            "edgePngPath"   : str(png_path) if png_ok else "",
        },
        "dxfContent"   : dxf_str[:50000] if dxf_str else "",   # cap at 50 KB for stdout
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
