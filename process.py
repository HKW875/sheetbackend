#!/usr/bin/env python3
"""
process.py — Lean OpenCV Shape Extraction Pipeline
====================================================
Receives: image_path, options_json (from node child_process)
Outputs:  JSON on stdout  { steps, circles, lines, curves }

Pipeline:
  1. Load color image (BGR, preserved)
  2. Channel separation for color-aware masking
  3. Gaussian blur + denoise
  4. Adaptive threshold + Canny edge detection
  5. Morphological cleanup
  6. HoughCircles  → circles
  7. HoughLinesP   → straight lines
  8. Contour approx + curvature filter → curves
"""

import sys, os, json, math, time, traceback
from pathlib import Path

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False

def now_ms():
    return int(time.time() * 1000)

def step_record(name, details, t0):
    return {"name": name, "status": "done", "duration": now_ms() - t0, "details": details}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load color image
# ══════════════════════════════════════════════════════════════════════════════
def load_image(image_path):
    """Read full-color BGR image; also derive grayscale for edge ops."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"Image has zero pixels: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Channel separation (color-aware mask)
# ══════════════════════════════════════════════════════════════════════════════
def separate_channels(img):
    """
    Split BGR channels and produce dominant-color binary masks.
    Returns: blue_bin, red_bin, green_bin — each isolates that color.
    """
    b, g, r = cv2.split(img)

    red_mask   = cv2.subtract(r, cv2.max(b, g))
    _, red_bin = cv2.threshold(red_mask, 40, 255, cv2.THRESH_BINARY)

    blue_mask   = cv2.subtract(b, cv2.max(r, g))
    _, blue_bin = cv2.threshold(blue_mask, 30, 255, cv2.THRESH_BINARY)

    green_mask   = cv2.subtract(g, cv2.max(r, b))
    _, green_bin = cv2.threshold(green_mask, 30, 255, cv2.THRESH_BINARY)

    return blue_bin, red_bin, green_bin


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Gaussian blur + fast denoise
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(img, gray):
    """
    Light denoising and blur to reduce noise before detection.
    Keeps a strong-blur variant for HoughCircles (needs smoother input).
    """
    # Bilateral filter: edge-preserving denoise
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

    # Standard blur for edges
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Stronger blur for circle detection
    blurred_strong = cv2.GaussianBlur(gray, (9, 9), 2)

    return denoised, blurred_gray, blurred_strong


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Adaptive threshold + Canny edges
# ══════════════════════════════════════════════════════════════════════════════
def compute_edges(blurred_gray):
    """
    Combine adaptive thresholding with multi-scale Canny for robust edges.
    """
    # Adaptive Gaussian threshold
    adapt = cv2.adaptiveThreshold(
        blurred_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Two Canny scales; merge for completeness
    canny_lo = cv2.Canny(blurred_gray, 30, 90)
    canny_hi = cv2.Canny(blurred_gray, 60, 150)
    canny    = cv2.bitwise_or(canny_lo, canny_hi)

    # Merge adaptive + canny
    edges = cv2.bitwise_or(adapt, canny)
    return edges, adapt, canny


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Morphological cleanup
# ══════════════════════════════════════════════════════════════════════════════
def morph_clean(binary):
    """
    Close small gaps in lines, remove tiny noise blobs.
    Returns the cleaned binary mask.
    """
    k3 = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  k3)
    return opened


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Circle detection  (HoughCircles, two passes)
# ══════════════════════════════════════════════════════════════════════════════
def detect_circles(blurred_strong):
    """
    Two-pass HoughCircles: standard gradient then ALT gradient.
    Deduplicates overlapping results.
    Returns list of {cx, cy, r}.
    """
    raw = []

    # Pass 1 — HOUGH_GRADIENT
    r1 = cv2.HoughCircles(
        blurred_strong, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=25,
        param1=120, param2=35,
        minRadius=5, maxRadius=500
    )
    if r1 is not None:
        for c in np.uint16(np.around(r1[0])):
            raw.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])})

    # Pass 2 — HOUGH_GRADIENT_ALT (more sensitive)
    try:
        r2 = cv2.HoughCircles(
            blurred_strong, cv2.HOUGH_GRADIENT_ALT,
            dp=1.5, minDist=20,
            param1=250, param2=0.80,
            minRadius=5, maxRadius=500
        )
        if r2 is not None:
            for c in np.uint16(np.around(r2[0])):
                raw.append({"cx": int(c[0]), "cy": int(c[1]), "r": int(c[2])})
    except Exception:
        pass  # HOUGH_GRADIENT_ALT not always available in older builds

    # Deduplicate: merge circles closer than 20px with similar radius
    merged = []
    for c in raw:
        dup = False
        for m in merged:
            if (math.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"]) < 20
                    and abs(c["r"] - m["r"]) < 15):
                dup = True
                if c["r"] > m["r"]:   # keep larger
                    m.update(c)
                break
        if not dup:
            merged.append({"cx": c["cx"], "cy": c["cy"], "r": c["r"]})

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Straight line detection  (HoughLinesP)
# ══════════════════════════════════════════════════════════════════════════════
def detect_lines(edges):
    """
    Probabilistic Hough lines; filters to meaningful length (>= 20px).
    Returns list of {x1, y1, x2, y2, length, angle, is_horizontal, is_vertical}.
    """
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=math.pi / 180,
        threshold=60, minLineLength=20, maxLineGap=8
    )
    lines = []
    if raw is None:
        return lines
    for seg in raw:
        x1, y1, x2, y2 = seg[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle  = math.degrees(math.atan2(y2 - y1, x2 - x1))
        lines.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "length":        round(length, 2),
            "angle":         round(angle, 2),
            "is_horizontal": abs(angle) < 8 or abs(angle - 180) < 8 or abs(angle + 180) < 8,
            "is_vertical":   abs(abs(angle) - 90) < 8,
        })
    # Sort longest first
    lines.sort(key=lambda l: l["length"], reverse=True)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Curve detection  (contour curvature filter)
# ══════════════════════════════════════════════════════════════════════════════
def detect_curves(cleaned_binary, min_area=200, straightness_threshold=0.75):
    """
    Find contours, then keep only those whose shape is genuinely curved.

    Straightness score = chord_length / arc_length  (per contour segment).
    Contours with low average straightness (< threshold) are curved.

    Returns list of {points, area, perimeter, circularity, is_closed}.
    """
    contours, _ = cv2.findContours(
        cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    curves = []
    for cnt in contours:
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < min_area or perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        # Skip anything that is clearly a circle (handled by HoughCircles)
        # or clearly a polygon (high circularity + low vertex count after approx)
        epsilon = 0.02 * perimeter
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        n_verts = len(approx)

        # Skip closed polygons with few vertices (triangles, rectangles, etc.)
        if n_verts <= 5 and circularity > 0.7:
            continue

        # Measure curvature: compare chord vs arc for segments
        pts       = cnt[:, 0, :]          # shape (N, 2)
        n         = len(pts)
        seg_size  = max(1, n // 8)        # sample ~8 chord/arc pairs
        straight_scores = []
        for i in range(0, n - seg_size, seg_size):
            p0  = pts[i].astype(float)
            p1  = pts[min(i + seg_size, n - 1)].astype(float)
            chord = float(np.linalg.norm(p1 - p0))
            arc   = sum(
                float(np.linalg.norm(
                    pts[j + 1].astype(float) - pts[j].astype(float)
                ))
                for j in range(i, min(i + seg_size, n - 1))
            )
            if arc > 0:
                straight_scores.append(chord / arc)

        if not straight_scores:
            continue

        avg_straightness = sum(straight_scores) / len(straight_scores)

        # Keep as "curve" if NOT mostly straight
        if avg_straightness >= straightness_threshold:
            continue

        # Simplify points for output
        simplified = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx]

        curves.append({
            "points":       simplified,
            "area":         round(float(area), 2),
            "perimeter":    round(float(perimeter), 2),
            "circularity":  round(circularity, 4),
            "n_vertices":   n_verts,
            "straightness": round(avg_straightness, 4),
            "is_closed":    bool(cv2.isContourConvex(approx) or circularity > 0.4),
        })

    # Sort by area descending
    curves.sort(key=lambda c: c["area"], reverse=True)
    return curves


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not HAS_CV:
        print(json.dumps({"error": "opencv-python not installed", "steps": []}))
        sys.exit(1)

    args = sys.argv[1:]
    if not args:
        print(json.dumps({"error": "Usage: process.py <image_path> [options_json]", "steps": []}))
        sys.exit(1)

    image_path = args[0]
    opts = json.loads(args[1]) if len(args) > 1 else {}

    steps = []

    # 1 — Load
    t0 = now_ms()
    img, gray = load_image(image_path)
    h, w = img.shape[:2]
    steps.append(step_record("1: Load image", f"{w}×{h}px  color=BGR", t0))

    # 2 — Channel separation
    t0 = now_ms()
    blue_bin, red_bin, green_bin = separate_channels(img)
    steps.append(step_record("2: Channel separation", "blue / red / green masks", t0))

    # 3 — Preprocess
    t0 = now_ms()
    denoised, blurred_gray, blurred_strong = preprocess(img, gray)
    steps.append(step_record("3: Gaussian blur + denoise", "bilateral + 5×5 / 9×9 blur", t0))

    # 4 — Edges
    t0 = now_ms()
    edges, adapt_bin, canny = compute_edges(blurred_gray)
    steps.append(step_record("4: Adaptive threshold + Canny", f"{int(cv2.countNonZero(edges))} edge pixels", t0))

    # 5 — Morphology
    t0 = now_ms()
    cleaned = morph_clean(edges)
    steps.append(step_record("5: Morphological cleanup", "close → open on edges", t0))

    # 6 — Circles
    t0 = now_ms()
    circles = detect_circles(blurred_strong)
    steps.append(step_record("6: HoughCircles (2-pass)", f"{len(circles)} circles detected", t0))

    # 7 — Lines
    t0 = now_ms()
    lines = detect_lines(cleaned)
    h_count = sum(1 for l in lines if l["is_horizontal"])
    v_count = sum(1 for l in lines if l["is_vertical"])
    steps.append(step_record(
        "7: HoughLinesP",
        f"{len(lines)} lines  ({h_count} horizontal, {v_count} vertical)",
        t0
    ))

    # 8 — Curves
    t0 = now_ms()
    curves = detect_curves(cleaned)
    steps.append(step_record("8: Contour curve extraction", f"{len(curves)} curves", t0))

    result = {
        "steps":   steps,
        "image":   {"width": w, "height": h, "path": str(image_path)},
        "circles": circles,
        "lines":   lines[:200],    # cap output size
        "curves":  curves[:100],
        "counts":  {
            "circles": len(circles),
            "lines":   len(lines),
            "curves":  len(curves),
        },
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
        }))
        sys.exit(1)
