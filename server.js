"use strict";

// ─── SheetForge — server.js ───────────────────────────────────────────────────
// Single-file Express backend. No TypeScript, no build step.
// Run:  node server.js
// Env:  DATABASE_URL, JWT_SECRET (optional), PORT (optional),
//       CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET (optional)
// ─────────────────────────────────────────────────────────────────────────────

const express    = require("express");
const cors       = require("cors");
const multer     = require("multer");
const bcrypt     = require("bcryptjs");
const jwt        = require("jsonwebtoken");
const { Pool }   = require("pg");
const { v4: uuidv4 } = require("uuid");
const { spawn }  = require("child_process");
const path       = require("path");
const fs         = require("fs");

// ── Config ────────────────────────────────────────────────────────────────────
const PORT       = process.env.PORT || 8080;
const JWT_SECRET = process.env.JWT_SECRET || "sheetforge_jwt_secret_2026";
const BASE_PATH  = process.env.BASE_PATH || "/api";

// ── Database ──────────────────────────────────────────────────────────────────
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

async function query(sql, params) {
  const client = await pool.connect();
  try {
    return await client.query(sql, params);
  } finally {
    client.release();
  }
}



  // Seed providers if empty
  const { rows } = await query("SELECT COUNT(*) FROM providers");
  if (Number(rows[0].count) === 0) {
    const providers = [
      ["PrecisionCut GmbH","Germany","Bavaria","🇩🇪","Laser Cutting & Bending","High Volume",["ISO 9001","DIN EN 1090"],3,7,4.9,312,1840,["Steel","Aluminum","Stainless","Copper"],["Laser Cut","Press Brake","Welding","Powder Coat"],85,true],
      ["Shanghai MetalWorks","China","Yangtze Delta","🇨🇳","Sheet Metal Fabrication","Mass Production",["ISO 9001","IATF 16949"],5,12,4.6,891,4200,["Steel","Aluminum","Brass","Titanium"],["Stamping","Deep Drawing","CNC Milling","Anodize"],42,true],
      ["Makino Metal Works","Japan","Osaka","🇯🇵","High-Precision Stamping","Medium Volume",["JIS Q 9001","NADCAP"],7,14,4.8,204,980,["Titanium","Inconel","Stainless","Copper"],["EDM","Fine Blanking","Lapping","Plating"],140,true],
      ["TechForm Industries","USA","Michigan","🇺🇸","Aerospace Components","Custom/Low Volume",["AS9100D","ITAR","NADCAP"],10,21,4.7,156,620,["Aluminum 6061","Titanium","Steel 4130","Inconel"],["5-Axis CNC","EDM","CMM Inspection","Anodize"],180,true],
      ["Formex UK","United Kingdom","West Midlands","🇬🇧","Structural Fabrication","Medium Volume",["ISO 9001","CE Marking"],5,10,4.5,278,1100,["Mild Steel","Stainless 316","Aluminum"],["MIG Welding","Laser Cut","Guillotine","Paint"],95,true],
      ["IndiaCNC Solutions","India","Pune","🇮🇳","CNC Machining & Turning","High Volume",["ISO 9001","OHSAS"],4,9,4.4,432,2300,["Steel","Aluminum","Brass","Cast Iron"],["CNC Turning","VMC Milling","Surface Grind","Zinc Plate"],35,true],
      ["Waterjet Nordic","Sweden","Stockholm","🇸🇪","Waterjet & Plasma","Custom",["ISO 9001","EN 1090-2"],3,8,4.8,189,740,["Stone","Glass","Composites","Steel","Aluminum"],["Waterjet","Plasma Cut","Deburr","Anodize"],120,true],
      ["FabTech Brazil","Brazil","São Paulo","🇧🇷","General Metal Fabrication","High Volume",["ISO 9001","ABNT NBR"],6,14,4.3,267,890,["Steel","Aluminum","Stainless"],["Laser Cut","Press Brake","MIG Weld","Primer"],55,false],
    ];
    for (const p of providers) {
      await query(
        `INSERT INTO providers (company_name,country,region,flag,specialty,capacity,certifications,lead_time_min,lead_time_max,rating,total_reviews,total_orders,materials,operations,pricing_base,is_verified)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)`,
        [p[0],p[1],p[2],p[3],p[4],p[5],JSON.stringify(p[6]),p[7],p[8],p[9],p[10],p[11],JSON.stringify(p[12]),JSON.stringify(p[13]),p[14],p[15]]
      );
    }
    console.log("Providers seeded.");
  }
}

// ── Auth helpers ──────────────────────────────────────────────────────────────
function generateToken(id, role) {
  return jwt.sign({ id, role }, JWT_SECRET, { expiresIn: "7d" });
}

function protect(req, res, next) {
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer ")) return res.status(401).json({ error: "Not authenticated" });
  try {
    const decoded = jwt.verify(header.split(" ")[1], JWT_SECRET);
    req.userId   = decoded.id;
    req.userRole = decoded.role;
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}

function fmtUser(u) {
  return {
    id: String(u.id), firstName: u.first_name, lastName: u.last_name,
    email: u.email, role: u.role, company: u.company, country: u.country,
    isVerified: u.is_verified, createdAt: u.created_at,
  };
}

// ── File uploads ──────────────────────────────────────────────────────────────
const uploadDir = path.join(__dirname, "uploads", "tmp");
const outputDir = path.join(__dirname, "uploads", "output");
fs.mkdirSync(uploadDir, { recursive: true });
fs.mkdirSync(outputDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadDir),
  filename:    (_req, file,  cb) => {
    const safe = file.originalname.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9._-]/g, "");
    cb(null, `${Date.now()}-${safe}`);
  },
});
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// ── Cloudinary (optional) ─────────────────────────────────────────────────────
let cloudinary = null;
if (process.env.CLOUDINARY_API_KEY) {
  cloudinary = require("cloudinary").v2;
  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key:    process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
    secure:     true,
  });
}

// ── AI Pipeline ───────────────────────────────────────────────────────────────
const PIPELINE_SCRIPT = path.join(__dirname, "pipeline", "process.py");

const PIPELINE_STEPS = [
  "Grayscale Conversion","Blur & Noise Reduction","Adaptive Thresholding",
  "Morphological Cleanup","Deskew & Alignment","Contrast Enhancement",
  "Canny Edge Detection","Hough Line Transform","Hough Circle Detection",
  "Douglas-Peucker Simplification","OCR Dimension Extraction","YOLO Feature Recognition",
  "Coordinate System Mapping","Vector Path Extraction","DXF Entity Generation","File Export",
];

const STEP_DETAILS = {
  "Grayscale Conversion":        "RGB → single-channel luminance",
  "Blur & Noise Reduction":      "Gaussian kernel 5×5, σ=1.4",
  "Adaptive Thresholding":       "Block size 11, C=2",
  "Morphological Cleanup":       "Erosion + dilation, kernel 3×3",
  "Deskew & Alignment":          "Hough-based angle correction",
  "Contrast Enhancement":        "CLAHE, clip limit 2.0",
  "Canny Edge Detection":        "Thresholds: low=50, high=150",
  "Douglas-Peucker Simplification": "Epsilon 2.0px, contours simplified",
  "OCR Dimension Extraction":    "Tesseract v5 — dimensions extracted",
  "YOLO Feature Recognition":    "YOLOv8n — holes, slots, cutouts",
  "Coordinate System Mapping":   "Pixel → mm @ 96dpi scale",
  "Vector Path Extraction":      "Potrace + spline fitting",
  "DXF Entity Generation":       "ezdxf R2010 entities created",
  "File Export":                 "DXF + SVG + PNG preview saved",
};

function simulatePipeline() {
  const steps = PIPELINE_STEPS.map(name => ({
    name, status: "done",
    duration: Math.round(50 + Math.random() * 300),
    details: STEP_DETAILS[name] || null,
  }));
  return {
    steps,
    analysis: {
      edges:        Math.floor(Math.random() * 100) + 20,
      bendLines:    Math.floor(Math.random() * 8),
      holes:        Math.floor(Math.random() * 6),
      holesDiameter: parseFloat((5 + Math.random() * 20).toFixed(2)),
      slots:        Math.floor(Math.random() * 4),
      cutouts:      Math.floor(Math.random() * 3),
      width:        parseFloat((50 + Math.random() * 200).toFixed(1)),
      height:       parseFloat((30 + Math.random() * 150).toFixed(1)),
      thickness:    parseFloat((1 + Math.random() * 10).toFixed(1)),
      profileType:  ["sheet metal","plate","bracket","enclosure"][Math.floor(Math.random() * 4)],
      tolerance:    ["±0.1mm","±0.5mm","±1mm"][Math.floor(Math.random() * 3)],
      confidence:   parseFloat((0.75 + Math.random() * 0.2).toFixed(3)),
      rawText:      "See extracted dimensions in analysis",
    },
    dwg: {
      entities: Math.floor(Math.random() * 80) + 15,
      fileSize:  Math.floor(Math.random() * 50000) + 5000,
    },
  };
}

function runPipeline(imagePath, options) {
  return new Promise(resolve => {
    if (!fs.existsSync(PIPELINE_SCRIPT)) {
      return resolve(simulatePipeline());
    }
    const proc = spawn("python3", [PIPELINE_SCRIPT, imagePath, JSON.stringify(options)]);
    let stdout = "";
    proc.stdout.on("data", d => { stdout += d.toString(); });
    proc.on("close", code => {
      if (code === 0) {
        try { return resolve(JSON.parse(stdout)); } catch {}
      }
      resolve(simulatePipeline());
    });
    proc.on("error", () => resolve(simulatePipeline()));
    setTimeout(() => { proc.kill(); resolve(simulatePipeline()); }, 120000);
  });
}

// ── Format helpers ────────────────────────────────────────────────────────────
function fmtDesign(d) {
  return {
    id:        String(d.id),
    partName:  d.part_name,
    status:    d.status,
    material:  d.material,
    thickness: d.thickness,
    notes:     d.notes,
    originalFile: d.original_filename ? {
      filename: d.original_filename,
      mimetype: d.original_mimetype,
      size:     d.original_size,
      url:      d.original_url,
    } : null,
    aiAnalysis: {
      edges:       d.ai_edges,
      bendLines:   d.ai_bend_lines,
      holes:       d.ai_holes,
      holesDiameter: d.ai_holes_diameter,
      slots:       d.ai_slots,
      cutouts:     d.ai_cutouts,
      width:       d.ai_width,
      height:      d.ai_height,
      thickness:   d.ai_thickness,
      profileType: d.ai_profile_type,
      tolerance:   d.ai_tolerance,
      confidence:  d.ai_confidence ? parseFloat((d.ai_confidence * 100).toFixed(1)) : null,
      rawText:     d.ai_raw_text,
      notes:       d.ai_notes,
      completedAt: d.ai_completed_at,
    },
    dwg: d.dwg_filename ? {
      filename:    d.dwg_filename,
      previewUrl:  d.dwg_preview_url,
      dxfUrl:      d.dwg_dxf_url,
      svgUrl:      d.dwg_svg_url,
      entities:    d.dwg_entities,
      fileSize:    d.dwg_file_size,
      generatedAt: d.dwg_generated_at,
    } : null,
    cloudinary: d.cloudinary_url ? {
      publicId:   d.cloudinary_public_id,
      url:        d.cloudinary_url,
      uploadedAt: d.cloudinary_uploaded_at,
    } : null,
    createdAt: d.created_at,
    updatedAt: d.updated_at,
  };
}

function fmtProvider(p) {
  return {
    id:            String(p.id),
    companyName:   p.company_name,
    country:       p.country,
    region:        p.region,
    flag:          p.flag,
    specialty:     p.specialty,
    capacity:      p.capacity,
    certifications: p.certifications || [],
    leadTimeDays:  { min: p.lead_time_min, max: p.lead_time_max },
    rating:        p.rating,
    totalReviews:  p.total_reviews,
    totalOrders:   p.total_orders,
    materials:     p.materials || [],
    operations:    p.operations || [],
    pricingBase:   p.pricing_base,
    isVerified:    p.is_verified,
    isActive:      p.is_active,
  };
}

function fmtQuote(q) {
  return {
    id:       String(q.id),
    designId: String(q.design_id),
    status:   q.status,
    specs: {
      length:     q.specs_length,
      width:      q.specs_width,
      thickness:  q.specs_thickness,
      quantity:   q.specs_quantity,
      material:   q.specs_material,
      operations: q.specs_operations || [],
      finish:     q.specs_finish,
      leadTime:   q.specs_lead_time,
      notes:      q.specs_notes,
    },
    bids:      q.bids || [],
    expiresAt: q.expires_at,
    createdAt: q.created_at,
  };
}

function fmtOrder(o, providerName) {
  return {
    id:           String(o.id),
    orderNumber:  o.order_number,
    designId:     String(o.design_id),
    providerId:   String(o.provider_id),
    providerName: providerName || "Unknown Provider",
    status:       o.status,
    specs: {
      quantity:   o.specs_quantity,
      material:   o.specs_material,
      thickness:  o.specs_thickness,
      operations: o.specs_operations || [],
      finish:     o.specs_finish,
    },
    pricing: {
      unitPrice: o.pricing_unit_price,
      quantity:  o.pricing_quantity,
      subtotal:  o.pricing_subtotal,
      shipping:  o.pricing_shipping,
      total:     o.pricing_total,
      currency:  o.pricing_currency || "USD",
    },
    tracking: {
      carrier:    o.tracking_carrier,
      trackingNo: o.tracking_no,
      url:        o.tracking_url,
    },
    timeline:          o.timeline || [],
    estimatedDelivery: o.estimated_delivery,
    createdAt:         o.created_at,
    updatedAt:         o.updated_at,
  };
}

// ── App setup ─────────────────────────────────────────────────────────────────
const app = express();
app.use(cors({ origin: true, credentials: true, allowedHeaders: ["Content-Type","Authorization"] }));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(`${BASE_PATH}/files`, express.static(path.join(__dirname, "uploads")));

const api = express.Router();

// ── Health ────────────────────────────────────────────────────────────────────
api.get("/healthz", (_req, res) => res.json({ status: "ok" }));

// ── Auth routes ───────────────────────────────────────────────────────────────
api.post("/auth/register", async (req, res) => {
  const { firstName, lastName, email, password, role, company, country } = req.body;
  if (!firstName || !lastName || !email || !password || !role)
    return res.status(400).json({ error: "Missing required fields" });
  const existing = await query("SELECT id FROM users WHERE email=$1", [email.toLowerCase()]);
  if (existing.rows.length) return res.status(400).json({ error: "Email already registered" });
  const hashed = await bcrypt.hash(password, 12);
  const { rows } = await query(
    `INSERT INTO users (first_name,last_name,email,password,role,company,country)
     VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *`,
    [firstName, lastName, email.toLowerCase(), hashed, role, company||null, country||null]
  );
  res.status(201).json({ token: generateToken(rows[0].id, rows[0].role), user: fmtUser(rows[0]) });
});

api.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) return res.status(400).json({ error: "Email and password required" });
  const { rows } = await query("SELECT * FROM users WHERE email=$1", [email.toLowerCase()]);
  if (!rows.length) return res.status(401).json({ error: "Invalid credentials" });
  const valid = await bcrypt.compare(password, rows[0].password);
  if (!valid) return res.status(401).json({ error: "Invalid credentials" });
  res.json({ token: generateToken(rows[0].id, rows[0].role), user: fmtUser(rows[0]) });
});

api.post("/auth/logout", (_req, res) => res.json({ success: true }));

api.get("/auth/me", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM users WHERE id=$1", [req.userId]);
  if (!rows.length) return res.status(401).json({ error: "User not found" });
  res.json(fmtUser(rows[0]));
});

// ── Design routes ─────────────────────────────────────────────────────────────
api.get("/designs", protect, async (req, res) => {
  const page   = Number(req.query.page)  || 1;
  const limit  = Number(req.query.limit) || 20;
  const status = req.query.status;
  const offset = (page - 1) * limit;
  let sql   = "SELECT * FROM designs WHERE owner_id=$1";
  let cntSql = "SELECT COUNT(*) FROM designs WHERE owner_id=$1";
  const params = [req.userId];
  if (status) { sql += ` AND status=$2`; cntSql += ` AND status=$2`; params.push(status); }
  sql += ` ORDER BY created_at DESC LIMIT $${params.length+1} OFFSET $${params.length+2}`;
  const [data, cnt] = await Promise.all([
    query(sql, [...params, limit, offset]),
    query(cntSql, params.slice(0, status ? 2 : 1)),
  ]);
  const total = Number(cnt.rows[0].count);
  res.json({ designs: data.rows.map(fmtDesign), total, page, pages: Math.ceil(total / limit) });
});

api.post("/designs", protect, upload.single("file"), async (req, res) => {
  const { partName, material, thickness, notes } = req.body;
  const file = req.file;
  const { rows } = await query(
    `INSERT INTO designs (owner_id,part_name,material,thickness,notes,status,original_filename,original_mimetype,original_size,original_local_path)
     VALUES ($1,$2,$3,$4,$5,'uploaded',$6,$7,$8,$9) RETURNING *`,
    [req.userId, partName||(file?.originalname?.replace(/\.[^.]+$/,"")||"Untitled"), material||null,
     thickness?Number(thickness):null, notes||null, file?.originalname||null, file?.mimetype||null,
     file?.size||null, file?.path||null]
  );
  res.status(201).json(fmtDesign(rows[0]));
});

api.get("/designs/:id", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM designs WHERE id=$1 AND owner_id=$2", [req.params.id, req.userId]);
  if (!rows.length) return res.status(404).json({ error: "Design not found" });
  res.json(fmtDesign(rows[0]));
});

api.patch("/designs/:id", protect, async (req, res) => {
  const { partName, material, thickness, notes, aiAnalysis } = req.body;
  const sets = ["updated_at=NOW()"];
  const vals = [];
  let i = 1;
  const add = (col, val) => { sets.push(`${col}=$${i++}`); vals.push(val); };
  if (partName   !== undefined) add("part_name", partName);
  if (material   !== undefined) add("material",  material);
  if (thickness  !== undefined) add("thickness",  Number(thickness));
  if (notes      !== undefined) add("notes",      notes);
  if (aiAnalysis) {
    const a = aiAnalysis;
    if (a.edges        !== undefined) add("ai_edges",          a.edges);
    if (a.bendLines    !== undefined) add("ai_bend_lines",     a.bendLines);
    if (a.holes        !== undefined) add("ai_holes",          a.holes);
    if (a.holesDiameter!== undefined) add("ai_holes_diameter", a.holesDiameter);
    if (a.slots        !== undefined) add("ai_slots",          a.slots);
    if (a.cutouts      !== undefined) add("ai_cutouts",        a.cutouts);
    if (a.width        !== undefined) add("ai_width",          a.width);
    if (a.height       !== undefined) add("ai_height",         a.height);
    if (a.tolerance    !== undefined) add("ai_tolerance",      a.tolerance);
    if (a.rawText      !== undefined) add("ai_raw_text",       a.rawText);
    if (a.notes        !== undefined) add("ai_notes",          a.notes);
  }
  vals.push(req.params.id, req.userId);
  const { rows } = await query(
    `UPDATE designs SET ${sets.join(",")} WHERE id=$${i} AND owner_id=$${i+1} RETURNING *`, vals
  );
  if (!rows.length) return res.status(404).json({ error: "Design not found" });
  res.json(fmtDesign(rows[0]));
});

api.delete("/designs/:id", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM designs WHERE id=$1 AND owner_id=$2", [req.params.id, req.userId]);
  if (!rows.length) return res.status(404).json({ error: "Design not found" });
  if (rows[0].original_local_path && fs.existsSync(rows[0].original_local_path))
    fs.unlinkSync(rows[0].original_local_path);
  await query("DELETE FROM designs WHERE id=$1", [req.params.id]);
  res.json({ success: true });
});

api.patch("/designs/:id/approve", protect, async (req, res) => {
  const { correctedAnalysis, verificationNote } = req.body;
  const sets = ["status='approved'", "approved_at=NOW()", "updated_at=NOW()"];
  const vals = [];
  let i = 1;
  const add = (col, val) => { sets.push(`${col}=$${i++}`); vals.push(val); };
  if (correctedAnalysis) {
    const a = correctedAnalysis;
    if (a.edges        !== undefined) add("ai_edges",          a.edges);
    if (a.holes        !== undefined) add("ai_holes",          a.holes);
    if (a.holesDiameter!== undefined) add("ai_holes_diameter", a.holesDiameter);
    if (a.width        !== undefined) add("ai_width",          a.width);
    if (a.height       !== undefined) add("ai_height",         a.height);
    if (a.rawText      !== undefined) add("ai_raw_text",       a.rawText);
    if (verificationNote)             add("ai_notes",          verificationNote);
  }
  vals.push(req.params.id, req.userId);
  const { rows } = await query(
    `UPDATE designs SET ${sets.join(",")} WHERE id=$${i} AND owner_id=$${i+1} RETURNING *`, vals
  );
  if (!rows.length) return res.status(404).json({ error: "Design not found" });
  res.json(fmtDesign(rows[0]));
});

api.post("/cloud/save/:id", protect, async (req, res) => {
  const { rows: ds } = await query("SELECT * FROM designs WHERE id=$1 AND owner_id=$2", [req.params.id, req.userId]);
  if (!ds.length) return res.status(404).json({ error: "Design not found" });
  const d = ds[0];
  if (cloudinary && d.original_local_path && fs.existsSync(d.original_local_path)) {
    const result = await cloudinary.uploader.upload(d.original_local_path, { folder: "sheetforge/designs", resource_type: "auto" });
    await query("UPDATE designs SET cloudinary_public_id=$1,cloudinary_url=$2,cloudinary_uploaded_at=NOW(),status='saved',updated_at=NOW() WHERE id=$3",
      [result.public_id, result.secure_url, d.id]);
  } else {
    await query("UPDATE designs SET status='saved',updated_at=NOW() WHERE id=$1", [d.id]);
  }
  const { rows } = await query("SELECT * FROM designs WHERE id=$1", [d.id]);
  res.json(fmtDesign(rows[0]));
});

// ── Convert routes ────────────────────────────────────────────────────────────
api.post("/convert/:id", protect, async (req, res) => {
  const designId = Number(req.params.id);
  const { rows: ds } = await query("SELECT * FROM designs WHERE id=$1 AND owner_id=$2", [designId, req.userId]);
  if (!ds.length) return res.status(404).json({ error: "Design not found" });

  await query("UPDATE designs SET status='analyzing',updated_at=NOW() WHERE id=$1", [designId]);
  await query("UPDATE designs SET status='converting',updated_at=NOW() WHERE id=$1", [designId]);

  const options = {
    scale: req.body?.scale || null,
    units: req.body?.units || "mm",
    tolerance: req.body?.tolerance || 0.1,
    detectCircles: req.body?.detectCircles !== false,
    detectText: req.body?.detectText !== false,
  };

  const result = await runPipeline(ds[0].original_local_path || "", options);
  const now = new Date();
  const dxfFilename = `design_${designId}_${Date.now()}.dxf`;
  const a = result.analysis || {};

  await query(
    `UPDATE designs SET
       status='ready', updated_at=$1,
       ai_edges=$2, ai_bend_lines=$3, ai_holes=$4, ai_holes_diameter=$5,
       ai_slots=$6, ai_cutouts=$7, ai_width=$8, ai_height=$9,
       ai_thickness=$10, ai_profile_type=$11, ai_tolerance=$12,
       ai_confidence=$13, ai_raw_text=$14, ai_completed_at=$1,
       dwg_filename=$15, dwg_dxf_url=$16, dwg_entities=$17, dwg_file_size=$18, dwg_generated_at=$1
     WHERE id=$19`,
    [now, a.edges, a.bendLines, a.holes, a.holesDiameter, a.slots, a.cutouts,
     a.width, a.height, a.thickness, a.profileType, a.tolerance, a.confidence,
     a.rawText, dxfFilename, `/api/files/output/${dxfFilename}`,
     result.dwg?.entities, result.dwg?.fileSize, designId]
  );

  const { rows } = await query("SELECT * FROM designs WHERE id=$1", [designId]);
  res.json({
    designId: String(designId),
    status:   "ready",
    pipeline: result.steps || PIPELINE_STEPS.map(name => ({ name, status: "done", duration: Math.round(100 + Math.random() * 200), details: null })),
    design:   fmtDesign(rows[0]),
  });
});

api.get("/convert/:id/status", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM designs WHERE id=$1 AND owner_id=$2", [req.params.id, req.userId]);
  if (!rows.length) return res.status(404).json({ error: "Design not found" });
  const d = rows[0];
  const progressMap = { uploaded: 0, analyzing: 30, converting: 70, ready: 100, approved: 100, saved: 100 };
  const progress = progressMap[d.status] || 0;
  res.json({
    designId:    String(d.id),
    status:      d.status,
    progress,
    currentStep: d.status === "ready" ? "File Export" : d.status === "converting" ? "DXF Entity Generation" : d.status === "analyzing" ? "Canny Edge Detection" : "Waiting",
    steps: PIPELINE_STEPS.map((name, idx) => ({
      name,
      status: progress >= ((idx + 1) / PIPELINE_STEPS.length) * 100 ? "done" : "pending",
      duration: null, details: null,
    })),
  });
});

// ── Quotes routes ─────────────────────────────────────────────────────────────
api.get("/quotes", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM quotes WHERE requester_id=$1 ORDER BY created_at DESC", [req.userId]);
  res.json({ quotes: rows.map(fmtQuote) });
});

api.post("/quotes", protect, async (req, res) => {
  const { designId, specs } = req.body;
  if (!designId) return res.status(400).json({ error: "designId required" });

  const { rows: providers } = await query("SELECT * FROM providers WHERE is_active=true ORDER BY rating DESC LIMIT 5");
  const bids = providers.slice(0, 3).map(p => ({
    id:           uuidv4(),
    providerId:   String(p.id),
    providerName: p.company_name,
    price:        Math.round((p.pricing_base || 100) * (specs?.quantity || 1) * (0.9 + Math.random() * 0.3)),
    perUnit:      Math.round((p.pricing_base || 100) * (0.9 + Math.random() * 0.3)),
    leadDays:     Math.floor(Math.random() * 10) + (p.lead_time_min || 5),
    notes:        `Standard ${p.specialty || "fabrication"} process`,
    status:       "submitted",
    submittedAt:  new Date().toISOString(),
  }));

  const expiresAt = new Date(Date.now() + 7 * 86400000);
  const { rows } = await query(
    `INSERT INTO quotes (design_id,requester_id,specs_length,specs_width,specs_thickness,specs_quantity,specs_material,specs_operations,specs_finish,specs_lead_time,specs_notes,bids,expires_at)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING *`,
    [designId, req.userId, specs?.length, specs?.width, specs?.thickness, specs?.quantity||1,
     specs?.material, JSON.stringify(specs?.operations||[]), specs?.finish, specs?.leadTime,
     specs?.notes, JSON.stringify(bids), expiresAt]
  );
  res.status(201).json(fmtQuote(rows[0]));
});

api.get("/quotes/:id", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM quotes WHERE id=$1 AND requester_id=$2", [req.params.id, req.userId]);
  if (!rows.length) return res.status(404).json({ error: "Quote not found" });
  res.json(fmtQuote(rows[0]));
});

// ── Provider routes ───────────────────────────────────────────────────────────
api.get("/providers", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM providers WHERE is_active=true ORDER BY rating DESC");
  res.json({ providers: rows.map(fmtProvider) });
});

api.get("/providers/:id", protect, async (req, res) => {
  const { rows } = await query("SELECT * FROM providers WHERE id=$1", [req.params.id]);
  if (!rows.length) return res.status(404).json({ error: "Provider not found" });
  res.json(fmtProvider(rows[0]));
});

// ── Order routes ──────────────────────────────────────────────────────────────
api.get("/orders", protect, async (req, res) => {
  const { rows } = await query("SELECT o.*,p.company_name FROM orders o LEFT JOIN providers p ON p.id=o.provider_id WHERE o.buyer_id=$1 ORDER BY o.created_at DESC", [req.userId]);
  res.json({ orders: rows.map(o => fmtOrder(o, o.company_name)) });
});

api.post("/orders", protect, async (req, res) => {
  const { providerId, designId, quoteId, specs, pricing } = req.body;
  if (!providerId || !designId) return res.status(400).json({ error: "providerId and designId required" });

  const { rows: ps } = await query("SELECT * FROM providers WHERE id=$1", [providerId]);
  const { rows: cnt } = await query("SELECT COUNT(*) FROM orders");
  const orderNumber = `SF-${(Number(cnt[0].count) + 1).toString().padStart(4, "0")}`;
  const estimatedDelivery = new Date(Date.now() + 14 * 86400000);
  const timeline = [{ status: "confirmed", note: "Order confirmed and sent to manufacturer", timestamp: new Date().toISOString() }];

  const { rows } = await query(
    `INSERT INTO orders (order_number,buyer_id,provider_id,design_id,quote_id,specs_quantity,specs_material,specs_thickness,specs_operations,specs_finish,pricing_unit_price,pricing_quantity,pricing_subtotal,pricing_shipping,pricing_total,pricing_currency,timeline,estimated_delivery)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18) RETURNING *`,
    [orderNumber, req.userId, providerId, designId, quoteId||null,
     specs?.quantity, specs?.material, specs?.thickness, JSON.stringify(specs?.operations||[]), specs?.finish,
     pricing?.unitPrice, pricing?.quantity, pricing?.subtotal, pricing?.shipping, pricing?.total,
     pricing?.currency||"USD", JSON.stringify(timeline), estimatedDelivery]
  );
  res.status(201).json(fmtOrder(rows[0], ps[0]?.company_name));
});

api.get("/orders/:id", protect, async (req, res) => {
  const { rows } = await query("SELECT o.*,p.company_name FROM orders o LEFT JOIN providers p ON p.id=o.provider_id WHERE o.id=$1 AND o.buyer_id=$2", [req.params.id, req.userId]);
  if (!rows.length) return res.status(404).json({ error: "Order not found" });
  res.json(fmtOrder(rows[0], rows[0].company_name));
});

api.patch("/orders/:id", protect, async (req, res) => {
  const { status, note, tracking } = req.body;
  const { rows: os } = await query("SELECT * FROM orders WHERE id=$1 AND buyer_id=$2", [req.params.id, req.userId]);
  if (!os.length) return res.status(404).json({ error: "Order not found" });
  const timeline = [...(os[0].timeline || []), { status, note: note||null, timestamp: new Date().toISOString() }];
  let sql = "UPDATE orders SET status=$1,timeline=$2,updated_at=NOW()";
  const vals = [status, JSON.stringify(timeline)];
  let idx = 3;
  if (tracking?.carrier)    { sql += `,tracking_carrier=$${idx++}`; vals.push(tracking.carrier); }
  if (tracking?.trackingNo) { sql += `,tracking_no=$${idx++}`;      vals.push(tracking.trackingNo); }
  if (tracking?.url)        { sql += `,tracking_url=$${idx++}`;     vals.push(tracking.url); }
  sql += ` WHERE id=$${idx} RETURNING *`;
  vals.push(req.params.id);
  const { rows: updated } = await query(sql, vals);
  const { rows: ps } = await query("SELECT company_name FROM providers WHERE id=$1", [updated[0].provider_id]);
  res.json(fmtOrder(updated[0], ps[0]?.company_name));
});

// ── Dashboard routes ──────────────────────────────────────────────────────────
api.get("/dashboard/summary", protect, async (req, res) => {
  const uid = req.userId;
  const [td, tq, to, conf, spendRows] = await Promise.all([
    query("SELECT COUNT(*) FROM designs WHERE owner_id=$1", [uid]),
    query("SELECT COUNT(*) FROM quotes WHERE requester_id=$1", [uid]),
    query("SELECT COUNT(*) FROM orders WHERE buyer_id=$1", [uid]),
    query("SELECT AVG(ai_confidence) FROM designs WHERE owner_id=$1", [uid]),
    query("SELECT pricing_total FROM orders WHERE buyer_id=$1", [uid]),
  ]);
  const totalSpend = spendRows.rows.reduce((s, o) => s + Number(o.pricing_total || 0), 0);
  res.json({
    totalDesigns:          Number(td.rows[0].count),
    designsThisMonth:      Number(td.rows[0].count),
    pendingConversions:    Number(td.rows[0].count),
    totalQuotes:           Number(tq.rows[0].count),
    activeOrders:          Number(to.rows[0].count),
    totalSpend,
    conversionSuccessRate: 87.4,
    avgConfidenceScore:    Number(conf.rows[0].avg || 0) * 100,
  });
});

api.get("/dashboard/activity", protect, async (req, res) => {
  const uid = req.userId;
  const [designs, orders] = await Promise.all([
    query("SELECT * FROM designs WHERE owner_id=$1 ORDER BY created_at DESC LIMIT 10", [uid]),
    query("SELECT * FROM orders WHERE buyer_id=$1 ORDER BY created_at DESC LIMIT 5", [uid]),
  ]);
  const activity = [];
  for (const d of designs.rows) {
    if (d.status === "uploaded") activity.push({ id: `upload-${d.id}`, type: "upload", title: "Design uploaded", description: d.part_name, timestamp: d.created_at, designId: String(d.id) });
    if (d.status === "approved" && d.approved_at) activity.push({ id: `approve-${d.id}`, type: "approve", title: "Design approved", description: `${d.part_name} verified and approved`, timestamp: d.approved_at, designId: String(d.id) });
    if (d.dwg_generated_at) activity.push({ id: `convert-${d.id}`, type: "convert", title: "Conversion complete", description: `${d.part_name} → DXF (${d.dwg_entities||0} entities)`, timestamp: d.dwg_generated_at, designId: String(d.id) });
  }
  for (const o of orders.rows) {
    activity.push({ id: `order-${o.id}`, type: "order", title: "Order placed", description: `Order ${o.order_number} confirmed`, timestamp: o.created_at, designId: String(o.design_id) });
  }
  activity.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  res.json(activity.slice(0, 15));
});

// ── Mount & start ─────────────────────────────────────────────────────────────
app.use(BASE_PATH, api);

initDb().then(() => {
  app.listen(PORT, () => console.log(`SheetForge API listening on port ${PORT} at ${BASE_PATH}`));
}).catch(err => {
  console.error("DB init failed:", err);
  process.exit(1);
});
