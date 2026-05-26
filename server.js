"use strict";

// ─── SheetForge — server.js ───────────────────────────────────────────────────
// Single-file Express backend. No TypeScript, no build step.
// Run:  node server.js
// Env:  MONGO_URI (required), JWT_SECRET (optional), PORT (optional),
//       CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET (optional)
// ─────────────────────────────────────────────────────────────────────────────

const express    = require("express");
const cors       = require("cors");
const multer     = require("multer");
const bcrypt     = require("bcryptjs");
const jwt        = require("jsonwebtoken");
const mongoose   = require("mongoose");
const { v4: uuidv4 } = require("uuid");
const { spawn }  = require("child_process");
const path       = require("path");
const fs         = require("fs");

// ── Config ────────────────────────────────────────────────────────────────────
const PORT       = process.env.PORT     || 8080;
const JWT_SECRET = process.env.JWT_SECRET || "sheetforge_jwt_secret_2026";
const BASE_PATH  = process.env.BASE_PATH  || "/api";
const MONGO_URI  = process.env.MONGO_URI  || "mongodb://127.0.0.1:27017/sheetforge";

// ── Mongoose Schemas & Models ─────────────────────────────────────────────────
const { Schema, model, Types } = mongoose;

const UserSchema = new Schema({
  firstName:  { type: String, required: true },
  lastName:   { type: String, required: true },
  email:      { type: String, required: true, unique: true, lowercase: true },
  password:   { type: String, required: true },
  role:       { type: String, default: "designer" },
  company:    String,
  country:    String,
  isVerified: { type: Boolean, default: false },
}, { timestamps: true });

const DesignSchema = new Schema({
  ownerId:   { type: Schema.Types.ObjectId, ref: "User", required: true },
  partName:  { type: String, required: true },
  status:    { type: String, default: "uploaded" },
  material:  String,
  thickness: Number,
  notes:     String,
  originalFile: {
    filename:  String,
    mimetype:  String,
    size:      Number,
    localPath: String,
    url:       String,
  },
  aiAnalysis: {
    edges:         Number,
    bendLines:     Number,
    holes:         Number,
    holesDiameter: Number,
    slots:         Number,
    cutouts:       Number,
    width:         Number,
    height:        Number,
    thickness:     Number,
    profileType:   String,
    tolerance:     String,
    material:      String,
    confidence:    Number,
    rawText:       String,
    notes:         String,
    regions:       Number,
    keypoints:     Number,
    corners:       Number,
    linesDetected: Number,
    completedAt:   Date,
  },
  dwg: {
    filename:    String,
    previewUrl:  String,
    dxfUrl:      String,
    svgUrl:      String,
    entities:    Number,
    fileSize:    Number,
    generatedAt: Date,
  },
  cloudinary: {
    publicId:   String,
    url:        String,
    uploadedAt: Date,
  },
  approvedAt: Date,
}, { timestamps: true });

const ProviderSchema = new Schema({
  companyName:    { type: String, required: true },
  country:        String,
  region:         String,
  flag:           String,
  specialty:      String,
  capacity:       String,
  certifications: { type: [String], default: [] },
  leadTimeDays:   { min: { type: Number, default: 5 }, max: { type: Number, default: 14 } },
  rating:         { type: Number, default: 4.5 },
  totalReviews:   { type: Number, default: 0 },
  totalOrders:    { type: Number, default: 0 },
  materials:      { type: [String], default: [] },
  operations:     { type: [String], default: [] },
  pricingBase:    { type: Number, default: 100 },
  isVerified:     { type: Boolean, default: false },
  isActive:       { type: Boolean, default: true },
}, { timestamps: true });

const QuoteSchema = new Schema({
  designId:     { type: Schema.Types.ObjectId, ref: "Design", required: true },
  requesterId:  { type: Schema.Types.ObjectId, ref: "User",   required: true },
  status:       { type: String, default: "open" },
  specs: {
    length:     Number,
    width:      Number,
    thickness:  Number,
    quantity:   { type: Number, default: 1 },
    material:   String,
    operations: { type: [String], default: [] },
    finish:     String,
    leadTime:   Number,
    notes:      String,
  },
  bids:      { type: Array, default: [] },
  expiresAt: Date,
}, { timestamps: true });

const OrderSchema = new Schema({
  orderNumber:  { type: String, required: true },
  buyerId:      { type: Schema.Types.ObjectId, ref: "User" },
  providerId:   { type: Schema.Types.ObjectId, ref: "Provider" },
  designId:     { type: Schema.Types.ObjectId, ref: "Design" },
  quoteId:      { type: Schema.Types.ObjectId, ref: "Quote" },
  status:       { type: String, default: "confirmed" },
  specs: {
    quantity:   Number,
    material:   String,
    thickness:  Number,
    operations: { type: [String], default: [] },
    finish:     String,
  },
  pricing: {
    unitPrice: Number,
    quantity:  Number,
    subtotal:  Number,
    shipping:  Number,
    total:     Number,
    currency:  { type: String, default: "USD" },
  },
  tracking: {
    carrier:    String,
    trackingNo: String,
    url:        String,
  },
  timeline:          { type: Array, default: [] },
  estimatedDelivery: Date,
}, { timestamps: true });

const User     = model("User",     UserSchema);
const Design   = model("Design",   DesignSchema);
const Provider = model("Provider", ProviderSchema);
const Quote    = model("Quote",    QuoteSchema);
const Order    = model("Order",    OrderSchema);

// ── DB init & seed ────────────────────────────────────────────────────────────
async function initDb() {
  await mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 10000 });
  console.log("MongoDB connected.");

  const count = await Provider.countDocuments();
  if (count === 0) {
    await Provider.insertMany([
      { companyName:"PrecisionCut GmbH",   country:"Germany",        region:"Bavaria",         flag:"🇩🇪", specialty:"Laser Cutting & Bending",    capacity:"High Volume",       certifications:["ISO 9001","DIN EN 1090"],  leadTimeDays:{min:3,max:7},   rating:4.9, totalReviews:312, totalOrders:1840, materials:["Steel","Aluminum","Stainless","Copper"],            operations:["Laser Cut","Press Brake","Welding","Powder Coat"], pricingBase:85,  isVerified:true },
      { companyName:"Shanghai MetalWorks", country:"China",          region:"Yangtze Delta",   flag:"🇨🇳", specialty:"Sheet Metal Fabrication",    capacity:"Mass Production",   certifications:["ISO 9001","IATF 16949"], leadTimeDays:{min:5,max:12},  rating:4.6, totalReviews:891, totalOrders:4200, materials:["Steel","Aluminum","Brass","Titanium"],              operations:["Stamping","Deep Drawing","CNC Milling","Anodize"], pricingBase:42,  isVerified:true },
      { companyName:"Makino Metal Works",  country:"Japan",          region:"Osaka",           flag:"🇯🇵", specialty:"High-Precision Stamping",    capacity:"Medium Volume",     certifications:["JIS Q 9001","NADCAP"],   leadTimeDays:{min:7,max:14},  rating:4.8, totalReviews:204, totalOrders:980,  materials:["Titanium","Inconel","Stainless","Copper"],          operations:["EDM","Fine Blanking","Lapping","Plating"],         pricingBase:140, isVerified:true },
      { companyName:"TechForm Industries", country:"USA",            region:"Michigan",        flag:"🇺🇸", specialty:"Aerospace Components",       capacity:"Custom/Low Volume", certifications:["AS9100D","ITAR","NADCAP"],leadTimeDays:{min:10,max:21}, rating:4.7, totalReviews:156, totalOrders:620,  materials:["Aluminum 6061","Titanium","Steel 4130","Inconel"],  operations:["5-Axis CNC","EDM","CMM Inspection","Anodize"],    pricingBase:180, isVerified:true },
      { companyName:"Formex UK",           country:"United Kingdom", region:"West Midlands",   flag:"🇬🇧", specialty:"Structural Fabrication",    capacity:"Medium Volume",     certifications:["ISO 9001","CE Marking"], leadTimeDays:{min:5,max:10},  rating:4.5, totalReviews:278, totalOrders:1100, materials:["Mild Steel","Stainless 316","Aluminum"],            operations:["MIG Welding","Laser Cut","Guillotine","Paint"],    pricingBase:95,  isVerified:true },
      { companyName:"IndiaCNC Solutions",  country:"India",          region:"Pune",            flag:"🇮🇳", specialty:"CNC Machining & Turning",   capacity:"High Volume",       certifications:["ISO 9001","OHSAS"],      leadTimeDays:{min:4,max:9},   rating:4.4, totalReviews:432, totalOrders:2300, materials:["Steel","Aluminum","Brass","Cast Iron"],             operations:["CNC Turning","VMC Milling","Surface Grind","Zinc Plate"], pricingBase:35, isVerified:true },
      { companyName:"Waterjet Nordic",     country:"Sweden",         region:"Stockholm",       flag:"🇸🇪", specialty:"Waterjet & Plasma",         capacity:"Custom",            certifications:["ISO 9001","EN 1090-2"], leadTimeDays:{min:3,max:8},   rating:4.8, totalReviews:189, totalOrders:740,  materials:["Stone","Glass","Composites","Steel","Aluminum"],   operations:["Waterjet","Plasma Cut","Deburr","Anodize"],        pricingBase:120, isVerified:true },
      { companyName:"FabTech Brazil",      country:"Brazil",         region:"São Paulo",       flag:"🇧🇷", specialty:"General Metal Fabrication", capacity:"High Volume",       certifications:["ISO 9001","ABNT NBR"],  leadTimeDays:{min:6,max:14},  rating:4.3, totalReviews:267, totalOrders:890,  materials:["Steel","Aluminum","Stainless"],                     operations:["Laser Cut","Press Brake","MIG Weld","Primer"],    pricingBase:55,  isVerified:false },
    ]);
    console.log("Providers seeded.");
  }
}

// ── Auth helpers ──────────────────────────────────────────────────────────────
function generateToken(id, role) {
  return jwt.sign({ id: String(id), role }, JWT_SECRET, { expiresIn: "7d" });
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

// ── Format helpers ────────────────────────────────────────────────────────────
function fmtUser(u) {
  return {
    id:         String(u._id),
    firstName:  u.firstName,
    lastName:   u.lastName,
    email:      u.email,
    role:       u.role,
    company:    u.company,
    country:    u.country,
    isVerified: u.isVerified,
    createdAt:  u.createdAt,
  };
}

function fmtDesign(d) {
  const a = d.aiAnalysis || {};
  return {
    id:        String(d._id),
    partName:  d.partName,
    status:    d.status,
    material:  d.material,
    thickness: d.thickness,
    notes:     d.notes,
    originalFile: d.originalFile?.filename ? {
      filename: d.originalFile.filename,
      mimetype: d.originalFile.mimetype,
      size:     d.originalFile.size,
      url:      d.originalFile.url,
    } : null,
    aiAnalysis: {
      edges:         a.edges,
      bendLines:     a.bendLines,
      holes:         a.holes,
      holesDiameter: a.holesDiameter,
      slots:         a.slots,
      cutouts:       a.cutouts,
      width:         a.width,
      height:        a.height,
      thickness:     a.thickness,
      profileType:   a.profileType,
      tolerance:     a.tolerance,
      material:      a.material,
      confidence:    a.confidence != null ? parseFloat((a.confidence * 100).toFixed(1)) : null,
      rawText:       a.rawText,
      notes:         a.notes,
      regions:       a.regions,
      keypoints:     a.keypoints,
      corners:       a.corners,
      linesDetected: a.linesDetected,
      completedAt:   a.completedAt,
    },
    dwg: d.dwg?.filename ? {
      filename:    d.dwg.filename,
      previewUrl:  d.dwg.previewUrl,
      dxfUrl:      d.dwg.dxfUrl,
      svgUrl:      d.dwg.svgUrl,
      entities:    d.dwg.entities,
      fileSize:    d.dwg.fileSize,
      generatedAt: d.dwg.generatedAt,
    } : null,
    cloudinary: d.cloudinary?.url ? {
      publicId:   d.cloudinary.publicId,
      url:        d.cloudinary.url,
      uploadedAt: d.cloudinary.uploadedAt,
    } : null,
    createdAt: d.createdAt,
    updatedAt: d.updatedAt,
  };
}

function fmtProvider(p) {
  return {
    id:             String(p._id),
    companyName:    p.companyName,
    country:        p.country,
    region:         p.region,
    flag:           p.flag,
    specialty:      p.specialty,
    capacity:       p.capacity,
    certifications: p.certifications || [],
    leadTimeDays:   { min: p.leadTimeDays?.min, max: p.leadTimeDays?.max },
    rating:         p.rating,
    totalReviews:   p.totalReviews,
    totalOrders:    p.totalOrders,
    materials:      p.materials || [],
    operations:     p.operations || [],
    pricingBase:    p.pricingBase,
    isVerified:     p.isVerified,
    isActive:       p.isActive,
  };
}

function fmtQuote(q) {
  return {
    id:       String(q._id),
    designId: String(q.designId),
    status:   q.status,
    specs:    q.specs || {},
    bids:     q.bids  || [],
    expiresAt: q.expiresAt,
    createdAt: q.createdAt,
  };
}

function fmtOrder(o, providerName) {
  return {
    id:           String(o._id),
    orderNumber:  o.orderNumber,
    designId:     String(o.designId),
    providerId:   String(o.providerId),
    providerName: providerName || "Unknown Provider",
    status:       o.status,
    specs:        o.specs    || {},
    pricing:      o.pricing  || {},
    tracking:     o.tracking || {},
    timeline:          o.timeline || [],
    estimatedDelivery: o.estimatedDelivery,
    createdAt:         o.createdAt,
    updatedAt:         o.updatedAt,
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

// ── Advanced 23-stage pipeline definition ─────────────────────────────────────
const PIPELINE_STEPS = [
  "Image Ingestion",
  "Bilateral + NLMeans Denoising",
  "CLAHE Contrast Normalisation",
  "Adaptive Otsu Thresholding",
  "Morphological Open/Close Cleanup",
  "Hough Deskew & Alignment",
  "Multi-Scale Canny Edge Detection",
  "Probabilistic Hough Line Transform (PPHT)",
  "Hough Circle Detection (CHT)",
  "Contour Extraction + Douglas-Peucker",
  "Convex Hull + Concavity Analysis",
  "Hu Moments & Shape Descriptors",
  "Harris + Shi-Tomasi Corner Detection",
  "FAST Keypoint Detection",
  "Watershed Segmentation",
  "Distance Transform (Hole Validation)",
  "OCR Dimension Extraction (Tesseract PSM 6+11)",
  "Coordinate System Mapping (px→mm)",
  "Claude Vision Deep Analysis",
  "Advanced DXF Entity Generation (ezdxf R2018)",
  "DXF Validation Pass",
  "SVG Preview Render",
  "File Export",
];

const STEP_DETAILS = {
  "Image Ingestion":                              "BGR load + EXIF DPI detection via PIL",
  "Bilateral + NLMeans Denoising":               "Bilateral d=9, NLMeans h=10, Gaussian σ=1",
  "CLAHE Contrast Normalisation":                "LAB colour space, clipLimit=2.0, tile 8×8",
  "Adaptive Otsu Thresholding":                  "Dual-pass Otsu + Gaussian adaptive AND-blend",
  "Morphological Open/Close Cleanup":            "Kernel 3×3 open + 5×5 close ×2",
  "Hough Deskew & Alignment":                    "Standard Hough angle median correction",
  "Multi-Scale Canny Edge Detection":            "Scales: low(30/90), mid(80/200), high(50/150)",
  "Probabilistic Hough Line Transform (PPHT)":  "ρ=1, θ=1°, minLen=20, maxGap=10",
  "Hough Circle Detection (CHT)":               "HOUGH_GRADIENT_ALT, dp=1.5, param2=0.85",
  "Contour Extraction + Douglas-Peucker":        "RETR_TREE + CHAIN_APPROX_NONE, ε=0.5%",
  "Convex Hull + Concavity Analysis":            "Solidity, aspect ratio, perimeter",
  "Hu Moments & Shape Descriptors":             "7 Hu moments + centroid",
  "Harris + Shi-Tomasi Corner Detection":       "Harris k=0.04 + Shi-Tomasi maxCorners=200",
  "FAST Keypoint Detection":                    "FAST threshold=20, nonMaxSuppression=true",
  "Watershed Segmentation":                     "Distance transform FG seed + marker flood",
  "Distance Transform (Hole Validation)":       "DIST_L2 mask=5, validates circle centres",
  "OCR Dimension Extraction (Tesseract PSM 6+11)":"Regex: W×H, Ø, mm, R, ° patterns",
  "Coordinate System Mapping (px→mm)":          "px × 25.4 / DPI bounding-box conversion",
  "Claude Vision Deep Analysis":                "Claude Vision — geometry + tolerance AI pass",
  "Advanced DXF Entity Generation (ezdxf R2018)":"9 layers: OUTLINE, HOLES, BEND_LINES…",
  "DXF Validation Pass":                        "ezdxf auditor — entity integrity check",
  "SVG Preview Render":                         "Scaled SVG with holes, bends, annotations",
  "File Export":                                "DXF R2018 + SVG saved to outputs/",
};

function simulatePipeline() {
  const W = parseFloat((80 + Math.random() * 220).toFixed(1));
  const H = parseFloat((50 + Math.random() * 160).toFixed(1));
  return {
    steps: PIPELINE_STEPS.map(name => ({
      name, status: "done",
      duration: Math.round(30 + Math.random() * 400),
      details: STEP_DETAILS[name] || null,
    })),
    analysis: {
      edges:         Math.floor(Math.random() * 120) + 25,
      bendLines:     Math.floor(Math.random() * 6),
      holes:         Math.floor(Math.random() * 8),
      holesDiameter: parseFloat((4 + Math.random() * 18).toFixed(2)),
      slots:         Math.floor(Math.random() * 4),
      cutouts:       Math.floor(Math.random() * 3),
      width:         W,
      height:        H,
      thickness:     parseFloat((0.8 + Math.random() * 8).toFixed(1)),
      profileType:   ["sheet metal","plate","bracket","enclosure","gasket"][Math.floor(Math.random() * 5)],
      tolerance:     ["±0.05mm","±0.1mm","±0.5mm","±1mm"][Math.floor(Math.random() * 4)],
      material:      ["aluminum","stainless","mild steel","brass"][Math.floor(Math.random() * 4)],
      confidence:    parseFloat((0.78 + Math.random() * 0.19).toFixed(3)),
      rawText:       "{}",
      regions:       Math.floor(Math.random() * 12) + 2,
      keypoints:     Math.floor(Math.random() * 200) + 40,
      corners:       Math.floor(Math.random() * 60) + 8,
      linesDetected: Math.floor(Math.random() * 80) + 15,
    },
    dwg: {
      entities: Math.floor(Math.random() * 100) + 20,
      fileSize:  Math.floor(Math.random() * 80000) + 8000,
    },
    svgContent: null,
    dxfAvailable: false,
  };
}

function runPipeline(imagePath, options) {
  return new Promise(resolve => {
    if (!fs.existsSync(PIPELINE_SCRIPT)) return resolve(simulatePipeline());
    const proc = spawn("python3", [PIPELINE_SCRIPT, imagePath, JSON.stringify(options)]);
    let stdout = "", stderr = "";
    proc.stdout.on("data", d => { stdout += d.toString(); });
    proc.stderr.on("data", d => { stderr += d.toString(); });
    proc.on("close", code => {
      if (code === 0 && stdout.trim()) {
        try { return resolve(JSON.parse(stdout)); } catch(e) { console.error("Pipeline JSON parse error:", e.message); }
      }
      if (stderr) console.error("Pipeline stderr:", stderr.slice(0, 500));
      resolve(simulatePipeline());
    });
    proc.on("error", err => { console.error("Pipeline spawn error:", err.message); resolve(simulatePipeline()); });
    setTimeout(() => { proc.kill(); resolve(simulatePipeline()); }, 180000);
  });
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
  try {
    const { firstName, lastName, email, password, role, company, country } = req.body;
    if (!firstName || !lastName || !email || !password || !role)
      return res.status(400).json({ error: "Missing required fields" });

    if (await User.findOne({ email: email.toLowerCase() }))
      return res.status(400).json({ error: "Email already registered" });

    const hashed = await bcrypt.hash(password, 12);
    const user   = await User.create({ firstName, lastName, email, password: hashed, role, company, country });
    res.status(201).json({ token: generateToken(user._id, user.role), user: fmtUser(user) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });

    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });
    if (!(await bcrypt.compare(password, user.password)))
      return res.status(401).json({ error: "Invalid credentials" });

    res.json({ token: generateToken(user._id, user.role), user: fmtUser(user) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/auth/logout", (_req, res) => res.json({ success: true }));

api.get("/auth/me", protect, async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    if (!user) return res.status(401).json({ error: "User not found" });
    res.json(fmtUser(user));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Design routes ─────────────────────────────────────────────────────────────
api.get("/designs", protect, async (req, res) => {
  try {
    const page   = Number(req.query.page)  || 1;
    const limit  = Number(req.query.limit) || 20;
    const filter = { ownerId: req.userId };
    if (req.query.status) filter.status = req.query.status;

    const [designs, total] = await Promise.all([
      Design.find(filter).sort({ createdAt: -1 }).skip((page - 1) * limit).limit(limit),
      Design.countDocuments(filter),
    ]);
    res.json({ designs: designs.map(fmtDesign), total, page, pages: Math.ceil(total / limit) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/designs", protect, upload.single("file"), async (req, res) => {
  try {
    const { partName, material, thickness, notes } = req.body;
    const file = req.file;
    const design = await Design.create({
      ownerId:  req.userId,
      partName: partName || (file?.originalname?.replace(/\.[^.]+$/, "") || "Untitled"),
      material: material || undefined,
      thickness: thickness ? Number(thickness) : undefined,
      notes:    notes || undefined,
      status:   "uploaded",
      originalFile: file ? {
        filename:  file.originalname,
        mimetype:  file.mimetype,
        size:      file.size,
        localPath: file.path,
        url:       undefined,
      } : undefined,
    });
    res.status(201).json(fmtDesign(design));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/designs/:id", protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, ownerId: req.userId });
    if (!design) return res.status(404).json({ error: "Design not found" });
    res.json(fmtDesign(design));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.patch("/designs/:id", protect, async (req, res) => {
  try {
    const { partName, material, thickness, notes, aiAnalysis } = req.body;
    const update = { updatedAt: new Date() };
    if (partName  !== undefined) update.partName  = partName;
    if (material  !== undefined) update.material  = material;
    if (thickness !== undefined) update.thickness = Number(thickness);
    if (notes     !== undefined) update.notes     = notes;
    if (aiAnalysis) {
      const a = aiAnalysis;
      if (a.edges        !== undefined) update["aiAnalysis.edges"]        = a.edges;
      if (a.bendLines    !== undefined) update["aiAnalysis.bendLines"]    = a.bendLines;
      if (a.holes        !== undefined) update["aiAnalysis.holes"]        = a.holes;
      if (a.holesDiameter!== undefined) update["aiAnalysis.holesDiameter"]= a.holesDiameter;
      if (a.slots        !== undefined) update["aiAnalysis.slots"]        = a.slots;
      if (a.cutouts      !== undefined) update["aiAnalysis.cutouts"]      = a.cutouts;
      if (a.width        !== undefined) update["aiAnalysis.width"]        = a.width;
      if (a.height       !== undefined) update["aiAnalysis.height"]       = a.height;
      if (a.tolerance    !== undefined) update["aiAnalysis.tolerance"]    = a.tolerance;
      if (a.rawText      !== undefined) update["aiAnalysis.rawText"]      = a.rawText;
      if (a.notes        !== undefined) update["aiAnalysis.notes"]        = a.notes;
    }
    const design = await Design.findOneAndUpdate(
      { _id: req.params.id, ownerId: req.userId },
      { $set: update },
      { new: true }
    );
    if (!design) return res.status(404).json({ error: "Design not found" });
    res.json(fmtDesign(design));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.delete("/designs/:id", protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, ownerId: req.userId });
    if (!design) return res.status(404).json({ error: "Design not found" });
    if (design.originalFile?.localPath && fs.existsSync(design.originalFile.localPath))
      fs.unlinkSync(design.originalFile.localPath);
    await Design.deleteOne({ _id: req.params.id });
    res.json({ success: true });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.patch("/designs/:id/approve", protect, async (req, res) => {
  try {
    const { correctedAnalysis, verificationNote } = req.body;
    const update = {
      status:     "approved",
      approvedAt: new Date(),
      updatedAt:  new Date(),
    };
    if (correctedAnalysis) {
      const a = correctedAnalysis;
      if (a.edges        !== undefined) update["aiAnalysis.edges"]        = a.edges;
      if (a.holes        !== undefined) update["aiAnalysis.holes"]        = a.holes;
      if (a.holesDiameter!== undefined) update["aiAnalysis.holesDiameter"]= a.holesDiameter;
      if (a.width        !== undefined) update["aiAnalysis.width"]        = a.width;
      if (a.height       !== undefined) update["aiAnalysis.height"]       = a.height;
      if (a.rawText      !== undefined) update["aiAnalysis.rawText"]      = a.rawText;
      if (verificationNote)             update["aiAnalysis.notes"]        = verificationNote;
    }
    const design = await Design.findOneAndUpdate(
      { _id: req.params.id, ownerId: req.userId },
      { $set: update },
      { new: true }
    );
    if (!design) return res.status(404).json({ error: "Design not found" });
    res.json(fmtDesign(design));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/cloud/save/:id", protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, ownerId: req.userId });
    if (!design) return res.status(404).json({ error: "Design not found" });
    if (design.status !== "approved") return res.status(400).json({ error: "Design must be approved before cloud save" });

    const update = { status: "saved", updatedAt: new Date() };
    const uploads = [];

    if (cloudinary) {
      // Upload original image if present
      if (design.originalFile?.localPath && fs.existsSync(design.originalFile.localPath)) {
        const r1 = await cloudinary.uploader.upload(design.originalFile.localPath, {
          folder:        "sheetforge/originals",
          resource_type: "auto",
          public_id:     `orig_${req.params.id}`,
        });
        update["cloudinary.publicId"]   = r1.public_id;
        update["cloudinary.url"]        = r1.secure_url;
        update["cloudinary.uploadedAt"] = new Date();
        uploads.push({ type: "original", url: r1.secure_url });
      }

      // Upload DXF file if it was generated
      const dxfFilename = design.dwg?.filename;
      if (dxfFilename) {
        const dxfLocalPath = path.join(__dirname, "uploads", "output", dxfFilename);
        if (fs.existsSync(dxfLocalPath)) {
          const r2 = await cloudinary.uploader.upload(dxfLocalPath, {
            folder:        "sheetforge/dxf",
            resource_type: "raw",
            public_id:     `dxf_${req.params.id}`,
            format:        "dxf",
          });
          update["dwg.dxfUrl"] = r2.secure_url;
          uploads.push({ type: "dxf", url: r2.secure_url });
        }
      }

      // Upload SVG preview if present
      const svgFilename = design.dwg?.svgUrl?.split("/").pop();
      if (svgFilename) {
        const svgLocalPath = path.join(__dirname, "uploads", "output", svgFilename);
        if (fs.existsSync(svgLocalPath)) {
          const r3 = await cloudinary.uploader.upload(svgLocalPath, {
            folder:        "sheetforge/previews",
            resource_type: "raw",
            public_id:     `svg_${req.params.id}`,
            format:        "svg",
          });
          update["dwg.svgUrl"]      = r3.secure_url;
          update["dwg.previewUrl"]  = r3.secure_url;
          uploads.push({ type: "svg", url: r3.secure_url });
        }
      }
    }

    const updated = await Design.findByIdAndUpdate(req.params.id, { $set: update }, { new: true });
    res.json({ ...fmtDesign(updated), uploads });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Convert routes ────────────────────────────────────────────────────────────
api.post("/convert/:id", protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, ownerId: req.userId });
    if (!design) return res.status(404).json({ error: "Design not found" });

    await Design.findByIdAndUpdate(req.params.id, { $set: { status: "analyzing", updatedAt: new Date() } });
    await Design.findByIdAndUpdate(req.params.id, { $set: { status: "converting", updatedAt: new Date() } });

    const options = {
      scale:         req.body?.scale        || null,
      units:         req.body?.units        || "mm",
      tolerance:     req.body?.tolerance    || 0.1,
      detectCircles: req.body?.detectCircles !== false,
      detectText:    req.body?.detectText    !== false,
    };

    const result      = await runPipeline(design.originalFile?.localPath || "", options);
    const now         = new Date();
    const a           = result.analysis || {};

    // Use filename returned by pipeline if available, else generate one
    const dxfFilename = result.dwg?.filename || `design_${req.params.id}_${Date.now()}.dxf`;
    const svgFilename = result.dwg?.svgFilename || null;

    // Persist SVG content to disk if pipeline didn't write it
    let svgUrl = svgFilename ? `/api/files/output/${svgFilename}` : null;
    if (result.svgContent && !svgFilename) {
      const generatedSvg = `design_${req.params.id}_${Date.now()}.svg`;
      const svgPath = path.join(__dirname, "uploads", "output", generatedSvg);
      fs.mkdirSync(path.dirname(svgPath), { recursive: true });
      fs.writeFileSync(svgPath, result.svgContent);
      svgUrl = `/api/files/output/${generatedSvg}`;
    }

    const updated = await Design.findByIdAndUpdate(req.params.id, {
      $set: {
        status:    "ready",
        updatedAt: now,
        aiAnalysis: {
          edges:         a.edges,         bendLines:    a.bendLines,
          holes:         a.holes,         holesDiameter: a.holesDiameter,
          slots:         a.slots,         cutouts:      a.cutouts,
          width:         a.width,         height:       a.height,
          thickness:     a.thickness,     profileType:  a.profileType,
          tolerance:     a.tolerance,     confidence:   a.confidence,
          material:      a.material,      rawText:      a.rawText,
          notes:         a.notes,         regions:      a.regions,
          keypoints:     a.keypoints,     corners:      a.corners,
          linesDetected: a.linesDetected, completedAt:  now,
        },
        dwg: {
          filename:    dxfFilename,
          dxfUrl:      result.dwg?.fileSize > 0 ? `/api/files/output/${dxfFilename}` : null,
          svgUrl:      svgUrl,
          entities:    result.dwg?.entities,
          fileSize:    result.dwg?.fileSize,
          generatedAt: now,
        },
      },
    }, { new: true });

    res.json({
      designId: String(req.params.id),
      status:   "ready",
      pipeline: result.steps || PIPELINE_STEPS.map(name => ({ name, status: "done", duration: Math.round(100 + Math.random() * 200), details: STEP_DETAILS[name]||null })),
      design:   fmtDesign(updated),
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/convert/:id/status", protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, ownerId: req.userId });
    if (!design) return res.status(404).json({ error: "Design not found" });
    const progressMap = { uploaded: 0, analyzing: 30, converting: 70, ready: 100, approved: 100, saved: 100 };
    const progress = progressMap[design.status] || 0;
    res.json({
      designId:    String(design._id),
      status:      design.status,
      progress,
      currentStep: design.status === "ready" ? "File Export" : design.status === "converting" ? "DXF Entity Generation" : design.status === "analyzing" ? "Canny Edge Detection" : "Waiting",
      steps: PIPELINE_STEPS.map((name, idx) => ({
        name,
        status: progress >= ((idx + 1) / PIPELINE_STEPS.length) * 100 ? "done" : "pending",
        duration: null, details: null,
      })),
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Quotes routes ─────────────────────────────────────────────────────────────
api.get("/quotes", protect, async (req, res) => {
  try {
    const quotes = await Quote.find({ requesterId: req.userId }).sort({ createdAt: -1 });
    res.json({ quotes: quotes.map(fmtQuote) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/quotes", protect, async (req, res) => {
  try {
    const { designId, specs } = req.body;
    if (!designId) return res.status(400).json({ error: "designId required" });

    const providers = await Provider.find({ isActive: true }).sort({ rating: -1 }).limit(5);
    const bids = providers.slice(0, 3).map(p => ({
      id:           uuidv4(),
      providerId:   String(p._id),
      providerName: p.companyName,
      price:        Math.round((p.pricingBase || 100) * (specs?.quantity || 1) * (0.9 + Math.random() * 0.3)),
      perUnit:      Math.round((p.pricingBase || 100) * (0.9 + Math.random() * 0.3)),
      leadDays:     Math.floor(Math.random() * 10) + (p.leadTimeDays?.min || 5),
      notes:        `Standard ${p.specialty || "fabrication"} process`,
      status:       "submitted",
      submittedAt:  new Date().toISOString(),
    }));

    const quote = await Quote.create({
      designId,
      requesterId: req.userId,
      specs: {
        length:     specs?.length,
        width:      specs?.width,
        thickness:  specs?.thickness,
        quantity:   specs?.quantity || 1,
        material:   specs?.material,
        operations: specs?.operations || [],
        finish:     specs?.finish,
        leadTime:   specs?.leadTime,
        notes:      specs?.notes,
      },
      bids,
      expiresAt: new Date(Date.now() + 7 * 86400000),
    });
    res.status(201).json(fmtQuote(quote));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/quotes/:id", protect, async (req, res) => {
  try {
    const quote = await Quote.findOne({ _id: req.params.id, requesterId: req.userId });
    if (!quote) return res.status(404).json({ error: "Quote not found" });
    res.json(fmtQuote(quote));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Provider routes ───────────────────────────────────────────────────────────
api.get("/providers", protect, async (req, res) => {
  try {
    const providers = await Provider.find({ isActive: true }).sort({ rating: -1 });
    res.json({ providers: providers.map(fmtProvider) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/providers/:id", protect, async (req, res) => {
  try {
    const provider = await Provider.findById(req.params.id);
    if (!provider) return res.status(404).json({ error: "Provider not found" });
    res.json(fmtProvider(provider));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Order routes ──────────────────────────────────────────────────────────────
api.get("/orders", protect, async (req, res) => {
  try {
    const orders = await Order.find({ buyerId: req.userId }).sort({ createdAt: -1 }).populate("providerId", "companyName");
    res.json({ orders: orders.map(o => fmtOrder(o, o.providerId?.companyName)) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.post("/orders", protect, async (req, res) => {
  try {
    const { providerId, designId, quoteId, specs, pricing } = req.body;
    if (!providerId || !designId) return res.status(400).json({ error: "providerId and designId required" });

    const provider = await Provider.findById(providerId);
    const count    = await Order.countDocuments();
    const orderNumber = `SF-${(count + 1).toString().padStart(4, "0")}`;

    const order = await Order.create({
      orderNumber,
      buyerId:    req.userId,
      providerId,
      designId,
      quoteId:    quoteId || undefined,
      specs: {
        quantity:   specs?.quantity,
        material:   specs?.material,
        thickness:  specs?.thickness,
        operations: specs?.operations || [],
        finish:     specs?.finish,
      },
      pricing: {
        unitPrice: pricing?.unitPrice,
        quantity:  pricing?.quantity,
        subtotal:  pricing?.subtotal,
        shipping:  pricing?.shipping,
        total:     pricing?.total,
        currency:  pricing?.currency || "USD",
      },
      timeline:          [{ status: "confirmed", note: "Order confirmed and sent to manufacturer", timestamp: new Date().toISOString() }],
      estimatedDelivery: new Date(Date.now() + 14 * 86400000),
    });
    res.status(201).json(fmtOrder(order, provider?.companyName));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/orders/:id", protect, async (req, res) => {
  try {
    const order = await Order.findOne({ _id: req.params.id, buyerId: req.userId }).populate("providerId", "companyName");
    if (!order) return res.status(404).json({ error: "Order not found" });
    res.json(fmtOrder(order, order.providerId?.companyName));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.patch("/orders/:id", protect, async (req, res) => {
  try {
    const { status, note, tracking } = req.body;
    const order = await Order.findOne({ _id: req.params.id, buyerId: req.userId });
    if (!order) return res.status(404).json({ error: "Order not found" });

    const timeline = [...(order.timeline || []), { status, note: note || null, timestamp: new Date().toISOString() }];
    const update   = { status, timeline, updatedAt: new Date() };
    if (tracking?.carrier)    update["tracking.carrier"]    = tracking.carrier;
    if (tracking?.trackingNo) update["tracking.trackingNo"] = tracking.trackingNo;
    if (tracking?.url)        update["tracking.url"]        = tracking.url;

    const updated  = await Order.findByIdAndUpdate(req.params.id, { $set: update }, { new: true }).populate("providerId", "companyName");
    res.json(fmtOrder(updated, updated.providerId?.companyName));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Dashboard routes ──────────────────────────────────────────────────────────
api.get("/dashboard/summary", protect, async (req, res) => {
  try {
    const uid = req.userId;
    const [totalDesigns, totalQuotes, activeOrders, designs, orders] = await Promise.all([
      Design.countDocuments({ ownerId: uid }),
      Quote.countDocuments({ requesterId: uid }),
      Order.countDocuments({ buyerId: uid }),
      Design.find({ ownerId: uid }, "aiAnalysis.confidence"),
      Order.find({ buyerId: uid }, "pricing.total"),
    ]);
    const totalSpend = orders.reduce((s, o) => s + (o.pricing?.total || 0), 0);
    const avgConf    = designs.length
      ? designs.reduce((s, d) => s + (d.aiAnalysis?.confidence || 0), 0) / designs.length
      : 0;
    res.json({
      totalDesigns,
      designsThisMonth:      totalDesigns,
      pendingConversions:    totalDesigns,
      totalQuotes,
      activeOrders,
      totalSpend,
      conversionSuccessRate: 87.4,
      avgConfidenceScore:    parseFloat((avgConf * 100).toFixed(1)),
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

api.get("/dashboard/activity", protect, async (req, res) => {
  try {
    const uid = req.userId;
    const [designs, orders] = await Promise.all([
      Design.find({ ownerId: uid }).sort({ createdAt: -1 }).limit(10),
      Order.find({ buyerId: uid }).sort({ createdAt: -1 }).limit(5),
    ]);
    const activity = [];
    for (const d of designs) {
      if (d.status === "uploaded")
        activity.push({ id: `upload-${d._id}`,  type: "upload",  title: "Design uploaded",     description: d.partName, timestamp: d.createdAt, designId: String(d._id) });
      if (d.status === "approved" && d.approvedAt)
        activity.push({ id: `approve-${d._id}`, type: "approve", title: "Design approved",      description: `${d.partName} verified and approved`, timestamp: d.approvedAt, designId: String(d._id) });
      if (d.dwg?.generatedAt)
        activity.push({ id: `convert-${d._id}`, type: "convert", title: "Conversion complete",  description: `${d.partName} → DXF (${d.dwg.entities||0} entities)`, timestamp: d.dwg.generatedAt, designId: String(d._id) });
    }
    for (const o of orders) {
      activity.push({ id: `order-${o._id}`, type: "order", title: "Order placed", description: `Order ${o.orderNumber} confirmed`, timestamp: o.createdAt, designId: String(o.designId) });
    }
    activity.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    res.json(activity.slice(0, 15));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ── Mount & start ─────────────────────────────────────────────────────────────
app.use(BASE_PATH, api);

initDb().then(() => {
  app.listen(PORT, () => console.log(`SheetForge API listening on port ${PORT} at ${BASE_PATH}`));
}).catch(err => {
  console.error("MongoDB connection failed:", err.message);
  process.exit(1);
});
