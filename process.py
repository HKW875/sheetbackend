"""
process.py - Sketch Processing Microservice
Handles: OpenCV preprocessing, color detection, shape/line/handwriting detection,
DXF generation, PDF generation, and GCode conversion via Gemini.
"""

import cv2
import numpy as np
import ezdxf
from ezdxf import colors as dxf_colors
from ezdxf.enums import TextEntityAlignment
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from PIL import Image
import io
import base64
import json
import math
import re
import os
import uuid
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from datetime import datetime
import tempfile

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Config ───────────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = mongo_client["cnc_sketch_db"]
scans_collection = db["scans"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"

OUTPUT_DIR = tempfile.mkdtemp()

# ─── Color detection ranges in HSV ───────────────────────────────────────────
COLOR_RANGES = {
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 50]),
        "rgb": (0, 0, 0),
        "dxf_color": 7,
    },
    "blue": {
        "lower": np.array([100, 50, 50]),
        "upper": np.array([130, 255, 255]),
        "rgb": (0, 0, 255),
        "dxf_color": 5,
    },
    "red_low": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255]),
        "rgb": (255, 0, 0),
        "dxf_color": 1,
    },
    "red_high": {
        "lower": np.array([160, 100, 100]),
        "upper": np.array([180, 255, 255]),
        "rgb": (255, 0, 0),
        "dxf_color": 1,
    },
    "green": {
        "lower": np.array([40, 50, 50]),
        "upper": np.array([90, 255, 255]),
        "rgb": (0, 180, 0),
        "dxf_color": 3,
    },
}


# ─── Image Preprocessing ─────────────────────────────────────────────────────
def preprocess_image(img_array):
    """
    Full preprocessing pipeline:
    - Denoise while preserving color
    - Adaptive contrast enhancement
    - Create masks per color channel
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # CLAHE on L channel only (preserve colors)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


# ─── Color Mask Extraction ────────────────────────────────────────────────────
def extract_color_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = {}

    # Black
    mask_black = cv2.inRange(hsv, COLOR_RANGES["black"]["lower"], COLOR_RANGES["black"]["upper"])
    masks["black"] = mask_black

    # Blue
    mask_blue = cv2.inRange(hsv, COLOR_RANGES["blue"]["lower"], COLOR_RANGES["blue"]["upper"])
    masks["blue"] = mask_blue

    # Red (wraps around 180)
    mask_red1 = cv2.inRange(hsv, COLOR_RANGES["red_low"]["lower"], COLOR_RANGES["red_low"]["upper"])
    mask_red2 = cv2.inRange(hsv, COLOR_RANGES["red_high"]["lower"], COLOR_RANGES["red_high"]["upper"])
    masks["red"] = cv2.bitwise_or(mask_red1, mask_red2)

    # Green
    mask_green = cv2.inRange(hsv, COLOR_RANGES["green"]["lower"], COLOR_RANGES["green"]["upper"])
    masks["green"] = mask_green

    # Clean up masks
    kernel = np.ones((3, 3), np.uint8)
    for key in masks:
        masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_CLOSE, kernel)
        masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_OPEN, kernel)

    return masks


# ─── Shape Detection ──────────────────────────────────────────────────────────
def detect_shapes(mask, color_name, img_h, img_w):
    """
    Detect lines, circles, rectangles, arcs from a color mask.
    Returns list of shape dicts.
    """
    shapes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # Circle detection
        if circularity > 0.75 and len(approx) > 6:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                shapes.append({
                    "type": "circle",
                    "color": color_name,
                    "cx": float(cx),
                    "cy": float(cy),
                    "radius": float(radius),
                    "area": float(area),
                })
            continue

        # Rectangle / polygon detection
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            shapes.append({
                "type": "rectangle",
                "color": color_name,
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "area": float(area),
            })
            continue

        # Line detection using HoughLinesP on the mask
    lines_raw = cv2.HoughLinesP(
        mask, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=15
    )
    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            shapes.append({
                "type": "line",
                "color": color_name,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "length": float(length),
                "angle": float(angle),
            })

    return shapes


# ─── Dimension Line Detection ─────────────────────────────────────────────────
def detect_dimension_lines(shapes):
    """
    Identify dimension lines: typically short arrows or lines with text nearby.
    Heuristic: short horizontal/vertical lines with angle ~0 or ~90.
    """
    dimension_lines = []
    for shape in shapes:
        if shape["type"] == "line":
            angle = abs(shape.get("angle", 45))
            length = shape.get("length", 0)
            # Near-horizontal or near-vertical shorter lines are likely dimension lines
            if (angle < 15 or angle > 165 or (75 < angle < 105)) and length < 200:
                dim = shape.copy()
                dim["type"] = "dimension_line"
                dim["dimension_value"] = round(length, 1)  # px value; user can override
                dimension_lines.append(dim)
    return dimension_lines


# ─── Handwriting / Text Region Detection ─────────────────────────────────────
def detect_handwriting_regions(img, mask_black):
    """
    Use MSER to find text-like regions on the black mask.
    Returns bounding boxes of text candidates.
    """
    text_regions = []
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Only look at regions where black ink exists
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_black)
    _, bboxes = mser.detectRegions(gray_masked)
    for bbox in bboxes:
        x, y, w, h = bbox
        aspect = w / max(h, 1)
        if 0.1 < aspect < 15 and 8 < h < 80 and w > 5:
            text_regions.append({
                "type": "text_region",
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "color": "black",
            })
    # Deduplicate overlapping boxes
    text_regions = _dedupe_boxes(text_regions)
    return text_regions


def _dedupe_boxes(regions, iou_thresh=0.5):
    if not regions:
        return regions
    boxes = [(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]) for r in regions]
    keep = []
    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        keep.append(regions[i])
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            iou = _compute_iou(boxes[i], boxes[j])
            if iou > iou_thresh:
                used.add(j)
    return keep


def _compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1)


# ─── Full Processing Pipeline ─────────────────────────────────────────────────
def process_sketch(img_array):
    h, w = img_array.shape[:2]
    preprocessed = preprocess_image(img_array)
    masks = extract_color_masks(preprocessed)

    all_shapes = []
    for color_name, mask in masks.items():
        detected = detect_shapes(mask, color_name, h, w)
        all_shapes.extend(detected)

    dimension_lines = detect_dimension_lines(all_shapes)
    # Replace generic lines that are dimension lines
    dim_line_keys = set()
    for dl in dimension_lines:
        key = (dl.get("x1"), dl.get("y1"), dl.get("x2"), dl.get("y2"))
        dim_line_keys.add(key)

    filtered_shapes = [
        s for s in all_shapes
        if not (s["type"] == "line" and (s.get("x1"), s.get("y1"), s.get("x2"), s.get("y2")) in dim_line_keys)
    ]
    filtered_shapes.extend(dimension_lines)

    text_regions = detect_handwriting_regions(preprocessed, masks.get("black", np.zeros((h, w), np.uint8)))

    return {
        "image_width": w,
        "image_height": h,
        "shapes": filtered_shapes,
        "text_regions": text_regions,
        "color_summary": {color: int(np.count_nonzero(m)) for color, m in masks.items()},
    }


# ─── DXF Generation ───────────────────────────────────────────────────────────
def generate_dxf(shapes, image_width, image_height, scale=1.0):
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()

    # Layers
    for layer_name, color_code in [("BLACK", 7), ("BLUE", 5), ("RED", 1), ("GREEN", 3), ("DIMENSIONS", 2)]:
        doc.layers.add(name=layer_name, color=color_code)

    def layer_for(color):
        return color.upper() if color.upper() in ["BLACK", "BLUE", "RED", "GREEN"] else "BLACK"

    def flip_y(y):
        return (image_height - y) * scale

    for shape in shapes:
        color = shape.get("color", "black")
        layer = layer_for(color)

        if shape["type"] == "line":
            x1 = shape["x1"] * scale
            y1 = flip_y(shape["y1"])
            x2 = shape["x2"] * scale
            y2 = flip_y(shape["y2"])
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer})

        elif shape["type"] == "circle":
            cx = shape["cx"] * scale
            cy = flip_y(shape["cy"])
            r = shape["radius"] * scale
            msp.add_circle((cx, cy), r, dxfattribs={"layer": layer})

        elif shape["type"] == "rectangle":
            x = shape["x"] * scale
            y = flip_y(shape["y"])
            rw = shape["width"] * scale
            rh = shape["height"] * scale
            pts = [(x, y), (x + rw, y), (x + rw, y - rh), (x, y - rh), (x, y)]
            msp.add_lwpolyline(pts, dxfattribs={"layer": layer})

        elif shape["type"] == "dimension_line":
            x1 = shape["x1"] * scale
            y1 = flip_y(shape["y1"])
            x2 = shape["x2"] * scale
            y2 = flip_y(shape["y2"])
            dim_val = shape.get("dimension_value", shape.get("length", 0)) * scale
            msp.add_linear_dim(
                base=(x1, y1 - 10),
                p1=(x1, y1),
                p2=(x2, y2),
                dxfattribs={"layer": "DIMENSIONS"},
            ).render()

    filepath = os.path.join(OUTPUT_DIR, f"sketch_{uuid.uuid4().hex}.dxf")
    doc.saveas(filepath)
    return filepath


# ─── PDF Generation ───────────────────────────────────────────────────────────
def generate_pdf(shapes, image_width, image_height, original_img_array=None):
    filepath = os.path.join(OUTPUT_DIR, f"sketch_{uuid.uuid4().hex}.pdf")
    page_w, page_h = A4  # 595.27 x 841.89 pt

    scale_x = page_w / image_width
    scale_y = page_h / image_height
    scale = min(scale_x, scale_y) * 0.9

    c = pdf_canvas.Canvas(filepath, pagesize=A4)
    c.setTitle("CNC Sketch Drawing")

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20, page_h - 30, "CNC SKETCH — GENERATED DRAWING")
    c.setFont("Helvetica", 9)
    c.drawString(20, page_h - 45, f"Image size: {image_width}x{image_height}px  |  Scale: {scale:.4f}")

    color_map = {
        "black": rl_colors.black,
        "blue": rl_colors.blue,
        "red": rl_colors.red,
        "green": rl_colors.green,
    }

    def to_pdf_y(y):
        return page_h - 60 - (y * scale)

    c.saveState()
    c.translate(20, 0)

    for shape in shapes:
        color = shape.get("color", "black")
        c.setStrokeColor(color_map.get(color, rl_colors.black))
        c.setFillColor(color_map.get(color, rl_colors.black))
        c.setLineWidth(1.2 if shape["type"] == "dimension_line" else 1.5)

        if shape["type"] in ("line", "dimension_line"):
            x1 = shape["x1"] * scale
            y1 = to_pdf_y(shape["y1"])
            x2 = shape["x2"] * scale
            y2 = to_pdf_y(shape["y2"])
            c.line(x1, y1, x2, y2)
            if shape["type"] == "dimension_line":
                c.setStrokeColor(rl_colors.HexColor("#FF6600"))
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                c.setFont("Helvetica", 7)
                c.setFillColor(rl_colors.HexColor("#FF6600"))
                dim_val = shape.get("dimension_value", shape.get("length", 0))
                c.drawString(mid_x, mid_y + 3, f"{dim_val:.1f}")

        elif shape["type"] == "circle":
            cx = shape["cx"] * scale
            cy = to_pdf_y(shape["cy"])
            r = shape["radius"] * scale
            c.circle(cx, cy, r, stroke=1, fill=0)
            c.setFont("Helvetica", 7)
            c.setFillColor(rl_colors.HexColor("#0066CC"))
            c.drawString(cx + r + 2, cy, f"⌀{shape['radius'] * 2:.1f}")

        elif shape["type"] == "rectangle":
            x = shape["x"] * scale
            y = to_pdf_y(shape["y"])
            rw = shape["width"] * scale
            rh = shape["height"] * scale
            c.rect(x, y - rh, rw, rh, stroke=1, fill=0)

    c.restoreState()

    # Legend
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20, 50, "Legend:")
    legend_items = [("Black lines", rl_colors.black), ("Blue lines", rl_colors.blue),
                    ("Red lines", rl_colors.red), ("Green lines", rl_colors.green)]
    for i, (label, clr) in enumerate(legend_items):
        c.setFillColor(clr)
        c.rect(100 + i * 100, 48, 8, 8, fill=1)
        c.setFillColor(rl_colors.black)
        c.setFont("Helvetica", 8)
        c.drawString(112 + i * 100, 50, label)

    c.save()
    return filepath


# ─── GCode via Gemini ─────────────────────────────────────────────────────────
def generate_gcode_from_dxf(dxf_filepath, material="mild steel", thickness_mm=1.0):
    with open(dxf_filepath, "r") as f:
        dxf_content = f.read()[:8000]  # limit tokens

    prompt = f"""
You are a CNC GCode expert for sheet metal cutting (laser/plasma).
Convert the following DXF drawing data into optimized GCode for a CNC sheet metal cutting machine.
Material: {material}, Thickness: {thickness_mm}mm.
Include: tool paths, lead-in/lead-out, feed rates, safe Z height, start/end sequences.
Return ONLY raw GCode, no explanations.

DXF DATA:
{dxf_content}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }
    resp = requests.post(GEMINI_URL, json=payload, timeout=60)
    if resp.status_code == 200:
        result = resp.json()
        gcode = result["candidates"][0]["content"]["parts"][0]["text"]
        return gcode
    else:
        return f"; GCode generation failed: {resp.text}"


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "sketch-processor"})


@app.route("/process", methods=["POST"])
def process():
    """Main endpoint: receive image, process, return shapes + upload to Cloudinary."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Upload original to Cloudinary
    cloudinary_result = cloudinary.uploader.upload(
        img_bytes,
        folder="cnc_sketches",
        public_id=f"sketch_{uuid.uuid4().hex}",
        resource_type="image",
    )
    cloudinary_url = cloudinary_result.get("secure_url", "")
    public_id = cloudinary_result.get("public_id", "")

    # Process
    result = process_sketch(img)

    # Save to MongoDB
    doc = {
        "cloudinary_url": cloudinary_url,
        "cloudinary_public_id": public_id,
        "created_at": datetime.utcnow(),
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "shape_count": len(result["shapes"]),
        "color_summary": result["color_summary"],
    }
    inserted = scans_collection.insert_one(doc)

    return jsonify({
        "scan_id": str(inserted.inserted_id),
        "cloudinary_url": cloudinary_url,
        "shapes": result["shapes"],
        "text_regions": result["text_regions"],
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "color_summary": result["color_summary"],
    })


@app.route("/generate-dxf", methods=["POST"])
def api_generate_dxf():
    data = request.json
    shapes = data.get("shapes", [])
    w = data.get("image_width", 800)
    h = data.get("image_height", 600)
    scale = float(data.get("scale", 0.1))  # px to mm

    filepath = generate_dxf(shapes, w, h, scale)
    return send_file(filepath, as_attachment=True, download_name="drawing.dxf", mimetype="application/dxf")


@app.route("/generate-pdf", methods=["POST"])
def api_generate_pdf():
    data = request.json
    shapes = data.get("shapes", [])
    w = data.get("image_width", 800)
    h = data.get("image_height", 600)

    filepath = generate_pdf(shapes, w, h)
    return send_file(filepath, as_attachment=True, download_name="drawing.pdf", mimetype="application/pdf")


@app.route("/generate-gcode", methods=["POST"])
def api_generate_gcode():
    data = request.json
    shapes = data.get("shapes", [])
    w = data.get("image_width", 800)
    h = data.get("image_height", 600)
    material = data.get("material", "mild steel")
    thickness = float(data.get("thickness", 1.0))

    # Generate DXF first, then send to Gemini
    dxf_path = generate_dxf(shapes, w, h, scale=0.1)
    gcode = generate_gcode_from_dxf(dxf_path, material, thickness)

    gcode_path = os.path.join(OUTPUT_DIR, f"gcode_{uuid.uuid4().hex}.nc")
    with open(gcode_path, "w") as f:
        f.write(gcode)

    return send_file(gcode_path, as_attachment=True, download_name="output.nc", mimetype="text/plain")


@app.route("/scans", methods=["GET"])
def get_scans():
    scans = list(scans_collection.find({}, {"_id": 0}).sort("created_at", -1).limit(20))
    return jsonify(scans)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
