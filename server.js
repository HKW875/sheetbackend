/**
 * SheetForge — Backend Server
 * Node.js + Express + MongoDB + Cloudinary
 * =========================================
 * Routes:
 *   Auth       POST /api/auth/register, /api/auth/login, /api/auth/logout
 *   Designs    POST /api/designs/upload, GET /api/designs, GET /api/designs/:id
 *              PATCH /api/designs/:id/approve, DELETE /api/designs/:id
 *   Convert    POST /api/convert/:id  (triggers AI-assisted DWG generation)
 *   Cloud      POST /api/cloud/save/:id  (Cloudinary upload + MongoDB link)
 *   Quotes     POST /api/quotes, GET /api/quotes, GET /api/quotes/:id
 *   Providers  GET /api/providers, GET /api/providers/:id
 *   Orders     POST /api/orders, GET /api/orders, PATCH /api/orders/:id
 */

require('dotenv').config();

const express       = require('express');
const mongoose      = require('mongoose');
const cors          = require('cors');
const helmet        = require('helmet');
const morgan        = require('morgan');
const multer        = require('multer');
const cloudinary    = require('cloudinary').v2;
const jwt           = require('jsonwebtoken');
const bcrypt        = require('bcryptjs');
const sharp         = require('sharp');
const path          = require('path');
const fs            = require('fs');
const { promisify } = require('util');
// CV + CAD + GCode pipeline (see HELPERS section below)
// @anthropic-ai/sdk   — Claude Vision for geometry extraction
// @tarikjabiri/dxf    — real DXF R12/R2000 file writer
// All installed via package.json

const app  = express();
const PORT = process.env.PORT || 5000;

// Debug logging for all requests
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// ================================================================
// MIDDLEWARE
// ================================================================
app.use(helmet({ contentSecurityPolicy: false }));

// CORS — supports comma-separated CLIENT_URL env var for multiple origins
const allowedOrigins = (process.env.CLIENT_URL || '')
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

app.use(cors({
  origin: function(origin, callback) {
    // Allow requests with no origin (same-origin, Postman, curl)
    if (!origin) return callback(null, true);
    // Allow any origin if none configured (dev mode)
    if (allowedOrigins.length === 0) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    callback(new Error(`CORS: Origin ${origin} not allowed`));
  },
  credentials: true,
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(morgan('dev'));

// Serve index.html from disk if it exists alongside server.js
app.get('/', (req, res) => {
  const htmlPath = path.join(__dirname, 'index.html');
  if (fs.existsSync(htmlPath)) {
    res.sendFile(htmlPath);
  } else {
    res.json({ status: 'SheetForge API running', version: '1.0.0' });
  }
});

// ================================================================
// DATABASE
// ================================================================
mongoose.connect(process.env.MONGO_URI)
.then(async () => {
  console.log('✅ MongoDB connected');
  await seedProviders();
})
.catch(err => {
  console.error('❌ MongoDB connection failed:', err.message);
  console.error('→ If using local MongoDB, ensure mongod is running: sudo systemctl start mongod');
  console.error('→ If using Atlas, check MONGO_URI in your .env file');
  // Exit so container restarts and retries rather than silently failing
  process.exit(1);
});

// ================================================================
// CLOUDINARY CONFIG
// ================================================================
cloudinary.config({
  cloud_name : process.env.CLOUDINARY_CLOUD_NAME || 'cloud-sheetforge-demo',
  api_key    : process.env.CLOUDINARY_API_KEY,
  api_secret : process.env.CLOUDINARY_API_SECRET,
  secure     : true,
});

// ================================================================
// MULTER — File uploads to /uploads/tmp
// ================================================================
const uploadDir = path.join(__dirname, 'uploads', 'tmp');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename   : (req, file, cb) => cb(null, `${Date.now()}-${file.originalname.replace(/\s+/g, '_')}`),
});

const fileFilter = (req, file, cb) => {
  const allowed = [
    'image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml',
    'image/tiff', 'image/bmp', 'application/pdf',
    'image/x-dwg', 'application/acad', 'application/x-dwg',
  ];
  const ext = path.extname(file.originalname).toLowerCase();
  const allowedExts = ['.png','.jpg','.jpeg','.pdf','.svg','.tiff','.bmp','.dwg','.dxf'];
  if (allowed.includes(file.mimetype) || allowedExts.includes(ext)) cb(null, true);
  else cb(new Error(`File type not allowed: ${file.mimetype}`), false);
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
});

// ================================================================
// MONGOOSE SCHEMAS
// ================================================================

// --- User ---
const userSchema = new mongoose.Schema({
  firstName   : { type: String, required: true, trim: true },
  lastName    : { type: String, required: true, trim: true },
  email       : { type: String, required: true, unique: true, lowercase: true },
  password    : { type: String, required: true },
  role        : { type: String, enum: ['designer', 'provider', 'admin'], default: 'designer' },
  company     : String,
  country     : String,
  phone       : String,
  avatar      : String,
  isVerified  : { type: Boolean, default: false },
  createdAt   : { type: Date, default: Date.now },
});
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});
userSchema.methods.comparePassword = function(plain) {
  return bcrypt.compare(plain, this.password);
};
const User = mongoose.model('User', userSchema);

// --- Design ---
const designSchema = new mongoose.Schema({
  owner       : { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  partName    : { type: String, default: 'Untitled Part' },
  originalFile: {
    filename  : String,
    mimetype  : String,
    size      : Number,
    path      : String,
  },
  material    : String,
  thickness   : Number,
  notes       : String,
  status      : {
    type    : String,
    enum    : ['uploaded', 'analyzing', 'converting', 'ready', 'approved', 'saved'],
    default : 'uploaded',
  },
  aiAnalysis  : {
    edges        : Number,
    bendLines    : Number,
    holes        : Number,
    holesDiameter: Number,
    slots        : Number,
    cutouts      : Number,
    width        : Number,
    height       : Number,
    thickness    : Number,
    profileType  : String,
    tolerance    : String,
    scaleDetected: Boolean,
    partOutline  : [{ x: Number, y: Number }],
    holePositions: [{ x: Number, y: Number, d: Number }],
    bendPositions: [{ x1: Number, y1: Number, x2: Number, y2: Number }],
    rawText      : String,
    confidence   : Number,
    notes        : String,
    completedAt  : Date,
  },
  dwg         : {
    filename   : String,
    path       : String,       // preview PNG path
    dxfPath    : String,       // real .DXF file path
    gcodeFiles : {             // map of machine type → .nc file path
      laser    : String,
      plasma   : String,
      waterjet : String,
      mill     : String,
      router   : String,
      lathe    : String,
      edm      : String,
      oxyfuel  : String,
    },
    entities   : Number,
    fileSize   : Number,
    generatedAt: Date,
  },
  cloudinary  : {
    publicId   : String,
    url        : String,
    secureUrl  : String,
    width      : Number,
    height     : Number,
    bytes      : Number,
    format     : String,
    uploadedAt : Date,
  },
  mongodb     : {
    savedAt    : Date,
    collection : { type: String, default: 'designs' },
  },
  approvedAt  : Date,
  createdAt   : { type: Date, default: Date.now },
  updatedAt   : { type: Date, default: Date.now },
});
designSchema.pre('save', function(next) { this.updatedAt = Date.now(); next(); });
const Design = mongoose.model('Design', designSchema);

// --- Quote ---
const quoteSchema = new mongoose.Schema({
  design      : { type: mongoose.Schema.Types.ObjectId, ref: 'Design', required: true },
  requester   : { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  specs       : {
    length       : Number,
    width        : Number,
    thickness    : Number,
    quantity     : Number,
    material     : String,
    operations   : [String],
    finish       : String,
    leadTime     : String,
    notes        : String,
  },
  bids        : [{
    provider   : { type: mongoose.Schema.Types.ObjectId, ref: 'Provider' },
    price      : Number,
    perUnit    : Number,
    leadDays   : Number,
    notes      : String,
    submittedAt: Date,
    status     : { type: String, enum: ['pending', 'submitted', 'accepted', 'rejected'], default: 'pending' },
  }],
  status      : { type: String, enum: ['open', 'closed', 'ordered'], default: 'open' },
  expiresAt   : { type: Date, default: () => new Date(Date.now() + 7 * 86400000) },
  createdAt   : { type: Date, default: Date.now },
});
const Quote = mongoose.model('Quote', quoteSchema);

// --- Provider ---
const providerSchema = new mongoose.Schema({
  user        : { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  companyName : { type: String, required: true },
  country     : String,
  region      : String,
  flag        : String,
  specialty   : String,
  capacity    : String,
  certifications : [String],
  leadTimeDays : { min: Number, max: Number },
  rating      : { type: Number, default: 0 },
  totalReviews: { type: Number, default: 0 },
  totalOrders : { type: Number, default: 0 },
  materials   : [String],
  operations  : [String],
  pricingBase : Number,
  isVerified  : { type: Boolean, default: false },
  isActive    : { type: Boolean, default: true },
  createdAt   : { type: Date, default: Date.now },
});
const Provider = mongoose.model('Provider', providerSchema);

// --- Order ---
const orderSchema = new mongoose.Schema({
  orderNumber : { type: String, unique: true },
  buyer       : { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  provider    : { type: mongoose.Schema.Types.ObjectId, ref: 'Provider', required: true },
  design      : { type: mongoose.Schema.Types.ObjectId, ref: 'Design', required: true },
  quote       : { type: mongoose.Schema.Types.ObjectId, ref: 'Quote' },
  specs       : {
    quantity     : Number,
    material     : String,
    thickness    : Number,
    operations   : [String],
    finish       : String,
  },
  pricing     : {
    unitPrice    : Number,
    quantity     : Number,
    subtotal     : Number,
    shipping     : Number,
    total        : Number,
    currency     : { type: String, default: 'USD' },
  },
  status      : {
    type    : String,
    enum    : ['confirmed', 'in_production', 'qc', 'shipped', 'delivered', 'cancelled'],
    default : 'confirmed',
  },
  tracking    : {
    carrier    : String,
    trackingNo : String,
    url        : String,
  },
  timeline    : [{
    status     : String,
    note       : String,
    timestamp  : { type: Date, default: Date.now },
  }],
  estimatedDelivery : Date,
  createdAt   : { type: Date, default: Date.now },
  updatedAt   : { type: Date, default: Date.now },
});
orderSchema.pre('save', async function(next) {
  if (!this.orderNumber) {
    const count = await Order.countDocuments();
    this.orderNumber = `SF-${String(count + 1).padStart(4, '0')}`;
  }
  this.updatedAt = Date.now();
  next();
});
const Order = mongoose.model('Order', orderSchema);

// ================================================================
// AUTH MIDDLEWARE
// ================================================================
const protect = async (req, res, next) => {
  try {
    const token = req.headers.authorization?.startsWith('Bearer ')
      ? req.headers.authorization.split(' ')[1]
      : null;
    if (!token) return res.status(401).json({ error: 'Not authenticated' });
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'sheetforge_jwt_secret_2026');
    req.user = await User.findById(decoded.id).select('-password');
    if (!req.user) return res.status(401).json({ error: 'User not found' });
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

const restrictTo = (...roles) => (req, res, next) => {
  if (!roles.includes(req.user.role))
    return res.status(403).json({ error: 'Permission denied' });
  next();
};

const signToken = (id) =>
  jwt.sign({ id }, process.env.JWT_SECRET || 'sheetforge_jwt_secret_2026', {
    expiresIn: process.env.JWT_EXPIRES_IN || '30d',
  });

// ================================================================
// HELPERS
// ================================================================
const cleanupFile = async (filePath) => {
  try { if (filePath && fs.existsSync(filePath)) await promisify(fs.unlink)(filePath); }
  catch (_) {}
};

// Simulates AI-driven DWG analysis (replace with real AI/CAD API call)
// ================================================================
// REAL CV + AI PIPELINE — imports
// ================================================================
const Anthropic   = require('@anthropic-ai/sdk');
const DxfWriter   = require('@tarikjabiri/dxf').DxfWriter;  // real DXF writer

const anthropicClient = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ----------------------------------------------------------------
// STEP 1: OpenCV-style pre-processing with sharp
//         Converts any input (PNG/JPEG/PDF/TIFF/BMP) to a
//         clean, high-contrast grayscale PNG ready for AI.
// ----------------------------------------------------------------
const preprocessImageForCV = async (filePath, mimetype) => {
  const ts   = Date.now();
  const outP = path.join(uploadDir, `preprocessed_${ts}.png`);

  // If it is a PDF, rasterise the first page at 300 DPI via sharp's built-in libvips
  // (sharp uses libvips which supports PDF when built with poppler/libheif)
  let inputBuffer;
  try {
    if (mimetype === 'application/pdf' || filePath.endsWith('.pdf')) {
      inputBuffer = await sharp(filePath, { page: 0, density: 300 })
        .flatten({ background: '#ffffff' })
        .toBuffer();
    } else {
      inputBuffer = fs.readFileSync(filePath);   // ← fs.readFileSync as requested
    }
  } catch (_) {
    inputBuffer = fs.readFileSync(filePath);
  }

  // Apply engineering-drawing pre-processing chain:
  //   1. Flatten alpha to white
  //   2. Greyscale
  //   3. Normalise brightness / contrast
  //   4. Sharpen edges (unsharp mask)
  //   5. Threshold → crisp black lines on white (simulates adaptive threshold)
  await sharp(inputBuffer)
    .flatten({ background: '#ffffff' })
    .grayscale()
    .normalise()
    .sharpen({ sigma: 1.2, m1: 1.5, m2: 0.7 })
    .threshold(160)                              // binarise for edge clarity
    .png({ compressionLevel: 6 })
    .toFile(outP);

  return outP;
};

// ----------------------------------------------------------------
// STEP 2: Anthropic Vision — structured geometry extraction
//         Sends the pre-processed image to Claude's vision model
//         and asks for a strict JSON geometry report.
// ----------------------------------------------------------------
const runAIAnalysis = async (filePath, mimetype) => {
  // Pre-process: normalise & binarise the image
  let processedPath = filePath;
  try {
    processedPath = await preprocessImageForCV(filePath, mimetype);
  } catch (cvErr) {
    console.warn('⚠️  Pre-processing failed, using raw file:', cvErr.message);
    processedPath = filePath;
  }

  // Encode image to base64 for Anthropic Vision
  const imgBuffer   = fs.readFileSync(processedPath);          // ← readFileSync as requested
  const base64Image = imgBuffer.toString('base64');
  const imgMime     = 'image/png';

  // Build a detailed engineering extraction prompt
  const systemPrompt = `You are an expert CAD/CAM engineer and computer-vision system.
Analyse the engineering drawing or scanned part sketch and extract ALL geometric features.
Respond ONLY with a single valid JSON object — no markdown, no commentary.
JSON schema:
{
  "width":          <number mm>,
  "height":         <number mm>,
  "thickness":      <number mm or null>,
  "edges":          <integer — total line/arc entity count>,
  "bendLines":      <integer>,
  "holes":          <integer>,
  "holesDiameter":  <number mm average or null>,
  "slots":          <integer>,
  "cutouts":        <integer>,
  "material":       <string detected or "Unknown">,
  "profileType":    <"rectangular"|"L-bracket"|"U-channel"|"custom">,
  "tolerance":      <string e.g. "ISO 2768-m" or null>,
  "scaleDetected":  <boolean>,
  "partOutline":    [{"x":<number>,"y":<number>}, ...],   // up to 20 key vertices in mm
  "holePositions":  [{"x":<number>,"y":<number>,"d":<number>}, ...],
  "bendPositions":  [{"x1":<number>,"y1":<number>,"x2":<number>,"y2":<number>}, ...],
  "rawText":        <string — any text/dimensions found in the drawing>,
  "confidence":     <number 0-100>,
  "notes":          <string — any important observations>
}`;

  let analysisResult;
  try {
    const response = await anthropicClient.messages.create({
      model      : 'claude-opus-4-5',
      max_tokens : 1500,
      system     : systemPrompt,
      messages   : [{
        role   : 'user',
        content: [{
          type  : 'image',
          source: { type: 'base64', media_type: imgMime, data: base64Image },
        }, {
          type: 'text',
          text: 'Extract all geometric features from this engineering drawing. Return only JSON.',
        }],
      }],
    });

    const rawJson = response.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('')
      .replace(/```json|```/g, '')
      .trim();

    analysisResult = JSON.parse(rawJson);
  } catch (aiErr) {
    console.error('❌  Anthropic Vision error:', aiErr.message);
    // Fallback: extract what we can from image metadata via sharp
    const meta = await sharp(processedPath).metadata();
    const mmPerPx = 0.264583; // 96 DPI default
    analysisResult = {
      width     : Math.round((meta.width  || 400) * mmPerPx),
      height    : Math.round((meta.height || 300) * mmPerPx),
      edges     : 24,
      bendLines : 2,
      holes     : 4,
      holesDiameter: 8,
      slots: 0, cutouts: 0,
      profileType: 'rectangular',
      confidence  : 40,
      rawText     : 'AI extraction failed — using image-dimension fallback.',
      notes       : aiErr.message,
    };
  }

  // Cleanup pre-processed temp file
  cleanupFile(processedPath).catch(() => {});

  return {
    ...analysisResult,
    completedAt: new Date(),
  };
};

// ----------------------------------------------------------------
// STEP 3: Generate real .DXF file from the analysis geometry
//         Uses @tarikjabiri/dxf — a full AutoCAD DXF R12/R2000 writer
// ----------------------------------------------------------------
const generateDXFFile = (analysis, material, thickness) => {
  const dxf = new DxfWriter();

  // Add layers
  dxf.addLayer('OUTLINE',    7,  'CONTINUOUS');  // white = black in CAD
  dxf.addLayer('BEND_LINES', 2,  'DASHED');      // yellow
  dxf.addLayer('HOLES',      4,  'CONTINUOUS');  // cyan
  dxf.addLayer('SLOTS',      5,  'CONTINUOUS');  // blue
  dxf.addLayer('DIMENSIONS', 3,  'CONTINUOUS');  // green
  dxf.addLayer('ANNOTATION', 7,  'CONTINUOUS');

  const w = analysis.width  || 200;
  const h = analysis.height || 150;

  // --- OUTLINE: use detected vertices or fall back to bounding rect ---
  const vertices = analysis.partOutline && analysis.partOutline.length >= 3
    ? analysis.partOutline
    : [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }, { x: 0, y: h }];

  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    dxf.addLine({ x: a.x, y: a.y, z: 0 }, { x: b.x, y: b.y, z: 0 }, 'OUTLINE');
  }

  // --- BEND LINES ---
  if (analysis.bendPositions && analysis.bendPositions.length) {
    analysis.bendPositions.forEach(bl => {
      dxf.addLine(
        { x: bl.x1, y: bl.y1, z: 0 },
        { x: bl.x2, y: bl.y2, z: 0 },
        'BEND_LINES'
      );
    });
  } else {
    // Fallback: evenly spaced horizontal bend lines
    const bendCount = analysis.bendLines || 2;
    for (let i = 1; i <= bendCount; i++) {
      const y = (h / (bendCount + 1)) * i;
      dxf.addLine({ x: 0, y, z: 0 }, { x: w, y, z: 0 }, 'BEND_LINES');
    }
  }

  // --- HOLES ---
  const holeDia = analysis.holesDiameter || 8;
  const holeR   = holeDia / 2;
  if (analysis.holePositions && analysis.holePositions.length) {
    analysis.holePositions.forEach(hp => {
      const r = (hp.d || holeDia) / 2;
      dxf.addCircle({ x: hp.x, y: hp.y, z: 0 }, r, 'HOLES');
    });
  } else {
    const holeCount = analysis.holes || 4;
    const colCount  = Math.ceil(holeCount / 2);
    for (let i = 0; i < holeCount; i++) {
      const cx = 20 + (i % colCount) * ((w - 40) / Math.max(colCount - 1, 1));
      const cy = i < colCount ? 20 : h - 20;
      dxf.addCircle({ x: cx, y: cy, z: 0 }, holeR, 'HOLES');
    }
  }

  // --- DIMENSION — width ---
  dxf.addLine({ x: 0,  y: -15, z: 0 }, { x: w, y: -15, z: 0 }, 'DIMENSIONS');
  dxf.addLine({ x: 0,  y: 0,   z: 0 }, { x: 0, y: -18, z: 0 }, 'DIMENSIONS');
  dxf.addLine({ x: w,  y: 0,   z: 0 }, { x: w, y: -18, z: 0 }, 'DIMENSIONS');
  dxf.addText(`${w}mm`, { x: w / 2, y: -22, z: 0 }, 3, 0, 'ANNOTATION');

  // --- DIMENSION — height ---
  dxf.addLine({ x: -15, y: 0, z: 0 }, { x: -15, y: h, z: 0 }, 'DIMENSIONS');
  dxf.addLine({ x: 0,   y: 0, z: 0 }, { x: -18, y: 0, z: 0 }, 'DIMENSIONS');
  dxf.addLine({ x: 0,   y: h, z: 0 }, { x: -18, y: h, z: 0 }, 'DIMENSIONS');
  dxf.addText(`${h}mm`, { x: -25, y: h / 2, z: 0 }, 3, 90, 'ANNOTATION');

  // --- TITLE BLOCK text ---
  const t = thickness || 2;
  dxf.addText(`SHEETFORGE EXPORT`, { x: 0, y: -40, z: 0 }, 4, 0, 'ANNOTATION');
  dxf.addText(`Material: ${material || 'Mild Steel'} | Thickness: ${t}mm | ISO 2768-m`, { x: 0, y: -48, z: 0 }, 2.5, 0, 'ANNOTATION');
  dxf.addText(`Generated: ${new Date().toISOString()} | Entities: ${analysis.edges}`, { x: 0, y: -54, z: 0 }, 2, 0, 'ANNOTATION');
  if (analysis.rawText) {
    dxf.addText(String(analysis.rawText).slice(0, 80), { x: 0, y: -60, z: 0 }, 2, 0, 'ANNOTATION');
  }

  return dxf.stringify();  // returns the DXF string
};

// ----------------------------------------------------------------
// STEP 4: Generate G-Code from the analysis
//         Covers: laser cutting, plasma, waterjet, milling, lathe,
//         router, EDM, oxyfuel — with machine-specific preambles.
// ----------------------------------------------------------------
const MACHINE_PROFILES = {
  laser:    { name: 'Laser Cutter (CO₂/Fiber)',  feedCut: 3000, feedRapid: 8000, spindleOn: 'M04 S1000', spindleOff: 'M05', coolant: '', unitCode: 'G21', retract: 2 },
  plasma:   { name: 'Plasma Cutter',             feedCut: 4500, feedRapid: 9000, spindleOn: 'M03 S100',  spindleOff: 'M05', coolant: '', unitCode: 'G21', retract: 5 },
  waterjet: { name: 'Waterjet',                  feedCut: 600,  feedRapid: 3000, spindleOn: 'M03 S800',  spindleOff: 'M05', coolant: 'M08', unitCode: 'G21', retract: 3 },
  mill:     { name: 'CNC Milling (Fanuc/HAAS)',  feedCut: 800,  feedRapid: 5000, spindleOn: 'M03 S3000', spindleOff: 'M05', coolant: 'M08', unitCode: 'G21', retract: 5 },
  router:   { name: 'CNC Router',               feedCut: 1200, feedRapid: 6000, spindleOn: 'M03 S18000',spindleOff: 'M05', coolant: '',    unitCode: 'G21', retract: 5 },
  lathe:    { name: 'CNC Lathe (Fanuc)',         feedCut: 200,  feedRapid: 2000, spindleOn: 'M03 S1200', spindleOff: 'M05', coolant: 'M08', unitCode: 'G21', retract: 2 },
  edm:      { name: 'Wire EDM',                  feedCut: 50,   feedRapid: 500,  spindleOn: 'M03 S0',    spindleOff: 'M05', coolant: 'M08', unitCode: 'G21', retract: 1 },
  oxyfuel:  { name: 'Oxyfuel / Flame Cutter',    feedCut: 300,  feedRapid: 1000, spindleOn: 'M03 S100',  spindleOff: 'M05', coolant: '',    unitCode: 'G21', retract: 5 },
  grinder:  { name: 'CNC Surface Grinder',       feedCut: 100,  feedRapid: 1000, spindleOn: 'M03 S3500', spindleOff: 'M05', coolant: 'M08', unitCode: 'G21', retract: 2 },
};

const generateGCode = (analysis, material, thickness, machineType = 'laser') => {
  const m     = MACHINE_PROFILES[machineType] || MACHINE_PROFILES.laser;
  const w     = analysis.width  || 200;
  const h     = analysis.height || 150;
  const t     = thickness || 2;
  const z0    = 0;
  const zSafe = m.retract;
  const lines = [];
  const ts    = new Date().toISOString();

  // ---- PREAMBLE ----
  lines.push(`; ============================================================`);
  lines.push(`; SheetForge G-Code Export`);
  lines.push(`; Machine   : ${m.name}`);
  lines.push(`; Material  : ${material || 'Mild Steel'} — ${t}mm`);
  lines.push(`; Part size : ${w} x ${h} mm`);
  lines.push(`; Entities  : ${analysis.edges}`);
  lines.push(`; Generated : ${ts}`);
  lines.push(`; ============================================================`);
  lines.push(`%`);
  lines.push(`O0001 (SHEETFORGE_PART)`);
  lines.push(m.unitCode);          // G21 = metric
  lines.push(`G17`);               // XY plane
  lines.push(`G40`);               // cancel tool radius comp
  lines.push(`G49`);               // cancel tool length comp
  lines.push(`G80`);               // cancel canned cycles
  lines.push(`G90`);               // absolute positioning
  lines.push(`G94`);               // feed per minute
  if (m.coolant) lines.push(m.coolant);
  lines.push(m.spindleOn);
  lines.push(`G04 P1.0`);          // dwell 1s for spindle/laser/plasma to stabilise
  lines.push(``);

  // Helper: rapid move
  const rapid = (x, y, z) => lines.push(`G00 X${x.toFixed(3)} Y${y.toFixed(3)} Z${z.toFixed(3)} F${m.feedRapid}`);
  const cut   = (x, y, z) => lines.push(`G01 X${x.toFixed(3)} Y${y.toFixed(3)} Z${z.toFixed(3)} F${m.feedCut}`);
  const arc   = (x, y, i, j, dir = 2) => lines.push(`G0${dir} X${x.toFixed(3)} Y${y.toFixed(3)} I${i.toFixed(3)} J${j.toFixed(3)} F${m.feedCut}`);

  // ---- OUTLINE / PROFILE ----
  lines.push(`; === PROFILE CUT ===`);
  const vertices = analysis.partOutline && analysis.partOutline.length >= 3
    ? analysis.partOutline
    : [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }, { x: 0, y: h }];

  rapid(vertices[0].x, vertices[0].y, zSafe);
  cut(vertices[0].x, vertices[0].y, z0);
  for (let i = 1; i < vertices.length; i++) {
    cut(vertices[i].x, vertices[i].y, z0);
  }
  cut(vertices[0].x, vertices[0].y, z0); // close profile
  rapid(vertices[0].x, vertices[0].y, zSafe);
  lines.push(``);

  // ---- HOLES (using canned drilling cycle G83 for mills, or circle arcs for laser/plasma) ----
  lines.push(`; === HOLES ===`);
  const holeDia = analysis.holesDiameter || 8;
  const holeR   = holeDia / 2;

  const holePositions = analysis.holePositions && analysis.holePositions.length
    ? analysis.holePositions
    : Array.from({ length: analysis.holes || 4 }, (_, i) => {
        const cols = Math.ceil((analysis.holes || 4) / 2);
        return {
          x: 20 + (i % cols) * ((w - 40) / Math.max(cols - 1, 1)),
          y: i < cols ? 20 : h - 20,
          d: holeDia,
        };
      });

  if (['mill', 'lathe', 'grinder', 'edm'].includes(machineType)) {
    // Use G83 peck drilling canned cycle
    lines.push(`G83 Z${(-t - 1).toFixed(3)} Q${(t / 3).toFixed(3)} R2.0 F${(m.feedCut / 5).toFixed(0)}`);
    holePositions.forEach(hp => lines.push(`X${hp.x.toFixed(3)} Y${hp.y.toFixed(3)}`));
    lines.push(`G80`);
  } else {
    // Laser/plasma/waterjet: cut circles as arcs
    holePositions.forEach(hp => {
      const r = (hp.d || holeDia) / 2;
      rapid(hp.x + r, hp.y, zSafe);
      cut(hp.x + r, hp.y, z0);
      // Full circle: I = centre_x - start_x, J = centre_y - start_y
      arc(hp.x + r, hp.y, -r, 0, 2); // G02 = CW
      rapid(hp.x + r, hp.y, zSafe);
    });
  }
  lines.push(``);

  // ---- BEND LINES (score pass at reduced power/feed for laser/plasma) ----
  if (analysis.bendLines > 0) {
    lines.push(`; === BEND LINES (score pass) ===`);
    const bendFeed   = Math.round(m.feedCut * 1.6);  // faster = shallower score
    const bendPositions = analysis.bendPositions && analysis.bendPositions.length
      ? analysis.bendPositions
      : Array.from({ length: analysis.bendLines || 2 }, (_, i) => ({
          x1: 0, y1: (h / (analysis.bendLines + 1)) * (i + 1),
          x2: w, y2: (h / (analysis.bendLines + 1)) * (i + 1),
        }));

    bendPositions.forEach(bl => {
      lines.push(`; Bend score`);
      rapid(bl.x1, bl.y1, zSafe);
      cut(bl.x1,  bl.y1,  z0);
      lines.push(`G01 X${bl.x2.toFixed(3)} Y${bl.y2.toFixed(3)} F${bendFeed}`);
      rapid(bl.x1, bl.y1, zSafe);
    });
    lines.push(``);
  }

  // ---- END ----
  lines.push(`; === END ===`);
  rapid(0, 0, zSafe);
  if (m.coolant) lines.push(`M09`);
  lines.push(m.spindleOff);
  lines.push(`G28`);    // return to machine home
  lines.push(`M30`);    // end of program
  lines.push(`%`);

  return lines.join('\n');
};

// ----------------------------------------------------------------
// STEP 5: Generate DWG preview PNG (accurate to analysis geometry)
// ----------------------------------------------------------------
const generateDWGPreview = async (analysis, material, thickness) => {
  const w = 1200, h = 800;
  const pad = 180;

  const pw     = analysis.width  || 200;
  const ph     = analysis.height || 150;
  const scaleX = (w - pad * 2) / pw;
  const scaleY = (h - pad * 2 - 80) / ph;   // 80 for title block
  const scale  = Math.min(scaleX, scaleY);
  const ox     = pad + (w - pad * 2 - pw * scale) / 2;
  const oy     = 50  + (h - 130 - ph * scale) / 2;

  const tx = (x) => ox + x * scale;
  const ty = (y) => oy + (ph - y) * scale;   // flip Y for SVG

  const holeDia = analysis.holesDiameter || 8;

  // Build vertices
  const verts = analysis.partOutline && analysis.partOutline.length >= 3
    ? analysis.partOutline
    : [{ x: 0, y: 0 }, { x: pw, y: 0 }, { x: pw, y: ph }, { x: 0, y: ph }];

  const outlinePts = verts.map(v => `${tx(v.x).toFixed(1)},${ty(v.y).toFixed(1)}`).join(' ');

  const bendSVG = (() => {
    const bends = analysis.bendPositions && analysis.bendPositions.length
      ? analysis.bendPositions
      : Array.from({ length: analysis.bendLines || 2 }, (_, i) => ({
          x1: 0, y1: (ph / ((analysis.bendLines || 2) + 1)) * (i + 1),
          x2: pw, y2: (ph / ((analysis.bendLines || 2) + 1)) * (i + 1),
        }));
    return bends.map(bl =>
      `<line x1="${tx(bl.x1).toFixed(1)}" y1="${ty(bl.y1).toFixed(1)}"
             x2="${tx(bl.x2).toFixed(1)}" y2="${ty(bl.y2).toFixed(1)}"
             stroke="#e67700" stroke-width="1.5" stroke-dasharray="10,5"/>`
    ).join('\n    ');
  })();

  const holeSVG = (() => {
    const hps = analysis.holePositions && analysis.holePositions.length
      ? analysis.holePositions
      : Array.from({ length: analysis.holes || 4 }, (_, i) => {
          const cols = Math.ceil((analysis.holes || 4) / 2);
          return {
            x: 20 + (i % cols) * ((pw - 40) / Math.max(cols - 1, 1)),
            y: i < cols ? 20 : ph - 20,
            d: holeDia,
          };
        });
    return hps.map(hp => {
      const r = (hp.d || holeDia) / 2 * scale;
      return `<circle cx="${tx(hp.x).toFixed(1)}" cy="${ty(hp.y).toFixed(1)}"
        r="${Math.max(r, 4).toFixed(1)}" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
      <line x1="${(tx(hp.x) - r * 1.4).toFixed(1)}" y1="${ty(hp.y).toFixed(1)}"
            x2="${(tx(hp.x) + r * 1.4).toFixed(1)}" y2="${ty(hp.y).toFixed(1)}"
            stroke="#94a3b8" stroke-width="0.6"/>
      <line x1="${tx(hp.x).toFixed(1)}" y1="${(ty(hp.y) - r * 1.4).toFixed(1)}"
            x2="${tx(hp.x).toFixed(1)}" y2="${(ty(hp.y) + r * 1.4).toFixed(1)}"
            stroke="#94a3b8" stroke-width="0.6"/>`;
    }).join('\n    ');
  })();

  const gridLines = [
    ...Array.from({ length: 41 }, (_, i) => `<line x1="${i * 30}" y1="0" x2="${i * 30}" y2="${h}" stroke="#e8edf2" stroke-width="0.4"/>`),
    ...Array.from({ length: 28 }, (_, i) => `<line x1="0" y1="${i * 30}" x2="${w}" y2="${i * 30}" stroke="#e8edf2" stroke-width="0.4"/>`),
  ].join('');

  const svg = `<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg" font-family="'Courier New',monospace">
  <rect width="${w}" height="${h}" fill="white"/>
  ${gridLines}
  <!-- Drawing border -->
  <rect x="10" y="10" width="${w - 20}" height="${h - 20}" fill="none" stroke="#94a3b8" stroke-width="1"/>
  <!-- Part outline -->
  <polygon points="${outlinePts}" fill="#f0f7ff" stroke="#1e3a5f" stroke-width="2.5" stroke-linejoin="round"/>
  <!-- Bend lines -->
  ${bendSVG}
  <!-- Holes -->
  ${holeSVG}
  <!-- Width dimension -->
  <line x1="${tx(0).toFixed(1)}" y1="${(ty(0) + 28).toFixed(1)}" x2="${tx(pw).toFixed(1)}" y2="${(ty(0) + 28).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <line x1="${tx(0).toFixed(1)}" y1="${(ty(0) + 20).toFixed(1)}" x2="${tx(0).toFixed(1)}" y2="${(ty(0) + 36).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <line x1="${tx(pw).toFixed(1)}" y1="${(ty(0) + 20).toFixed(1)}" x2="${tx(pw).toFixed(1)}" y2="${(ty(0) + 36).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <text x="${((tx(0) + tx(pw)) / 2).toFixed(1)}" y="${(ty(0) + 46).toFixed(1)}" fill="#1e40af" font-size="11" text-anchor="middle">${pw}mm</text>
  <!-- Height dimension -->
  <line x1="${(tx(0) - 28).toFixed(1)}" y1="${ty(0).toFixed(1)}" x2="${(tx(0) - 28).toFixed(1)}" y2="${ty(ph).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <line x1="${(tx(0) - 20).toFixed(1)}" y1="${ty(0).toFixed(1)}" x2="${(tx(0) - 36).toFixed(1)}" y2="${ty(0).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <line x1="${(tx(0) - 20).toFixed(1)}" y1="${ty(ph).toFixed(1)}" x2="${(tx(0) - 36).toFixed(1)}" y2="${ty(ph).toFixed(1)}" stroke="#2563eb" stroke-width="1"/>
  <text x="${(tx(0) - 44).toFixed(1)}" y="${((ty(0) + ty(ph)) / 2).toFixed(1)}" fill="#1e40af" font-size="11" text-anchor="middle" transform="rotate(-90,${(tx(0) - 44).toFixed(1)},${((ty(0) + ty(ph)) / 2).toFixed(1)})">${ph}mm</text>
  <!-- Title block -->
  <rect x="0" y="${h - 65}" width="${w}" height="65" fill="#f8fafc" stroke="#c0c8d4" stroke-width="1"/>
  <line x1="0" y1="${h - 65}" x2="${w}" y2="${h - 65}" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="20" y="${h - 44}" fill="#1e3a5f" font-size="13" font-weight="bold">SHEETFORGE AUTO-DXF</text>
  <text x="240" y="${h - 44}" fill="#475569" font-size="11">Material: ${material || 'Mild Steel'} ${thickness || 2}mm  |  ${analysis.profileType || 'Profile'}  |  ISO 2768-m  |  Scale 1:1</text>
  <text x="20" y="${h - 26}" fill="#64748b" font-size="10">Entities: ${analysis.edges}  |  Bend lines: ${analysis.bendLines || 0}  |  Holes: ${analysis.holes || 0}  |  Confidence: ${analysis.confidence || 0}%</text>
  <text x="20" y="${h - 10}" fill="#94a3b8" font-size="9">${new Date().toISOString()}  |  ${analysis.rawText ? String(analysis.rawText).slice(0, 100) : ''}</text>
  <!-- Legend -->
  <line x1="${w - 200}" y1="${h - 50}" x2="${w - 170}" y2="${h - 50}" stroke="#1e3a5f" stroke-width="2.5"/>
  <text x="${w - 165}" y="${h - 46}" fill="#475569" font-size="9">Profile</text>
  <line x1="${w - 200}" y1="${h - 36}" x2="${w - 170}" y2="${h - 36}" stroke="#e67700" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="${w - 165}" y="${h - 32}" fill="#475569" font-size="9">Bend line</text>
  <circle cx="${w - 185}" cy="${h - 20}" r="6" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="${w - 165}" y="${h - 17}" fill="#475569" font-size="9">Hole</text>
</svg>`;

  const outputPath = path.join(uploadDir, `dwg_preview_${Date.now()}.png`);
  await sharp(Buffer.from(svg)).png({ quality: 95 }).toFile(outputPath);
  return outputPath;
};

// ================================================================
// ROUTES — AUTH
// ================================================================
app.post('/api/auth/register', async (req, res) => {
  try {
    const { firstName, lastName, email, password, role, company, country } = req.body;
    if (!firstName || !lastName || !email || !password)
      return res.status(400).json({ error: 'All fields are required' });

    const exists = await User.findOne({ email });
    if (exists) return res.status(409).json({ error: 'Email already registered' });

    const user = await User.create({ firstName, lastName, email, password, role, company, country });
    const token = signToken(user._id);
    res.status(201).json({
      token,
      user: { id: user._id, firstName, lastName, email, role, company },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });

    const user = await User.findOne({ email });
    if (!user || !(await user.comparePassword(password)))
      return res.status(401).json({ error: 'Invalid credentials' });

    const token = signToken(user._id);
    res.json({
      token,
      user: {
        id        : user._id,
        firstName : user.firstName,
        lastName  : user.lastName,
        email     : user.email,
        role      : user.role,
        company   : user.company,
        country   : user.country,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/auth/me', protect, (req, res) => res.json({ user: req.user }));

// Logout (client should discard the token; this endpoint is for completeness)
app.post('/api/auth/logout', (req, res) => res.json({ message: 'Logged out successfully' }));

// ================================================================
// ROUTES — DESIGNS
// ================================================================

// Upload a rough design file
app.post('/api/designs/upload', protect, upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  try {
    const { partName, material, thickness, notes } = req.body;
    const design = await Design.create({
      owner       : req.user._id,
      partName    : partName || req.file.originalname.replace(/\.[^.]+$/, ''),
      originalFile: {
        filename : req.file.originalname,
        mimetype : req.file.mimetype,
        size     : req.file.size,
        path     : req.file.path,
      },
      material    : material || null,
      thickness   : thickness ? parseFloat(thickness) : null,
      notes,
      status      : 'uploaded',
    });
    res.status(201).json({ design });
  } catch (err) {
    await cleanupFile(req.file?.path);
    res.status(500).json({ error: err.message });
  }
});

// Get all designs for current user
app.get('/api/designs', protect, async (req, res) => {
  try {
    const { status, search, sort = '-createdAt', limit = 20, page = 1 } = req.query;
    const query = { owner: req.user._id };
    if (status) query.status = status;
    if (search) query.partName = { $regex: search, $options: 'i' };

    const designs = await Design.find(query)
      .sort(sort)
      .limit(parseInt(limit))
      .skip((parseInt(page) - 1) * parseInt(limit))
      .lean();
    const total = await Design.countDocuments(query);
    res.json({ designs, total, page: parseInt(page), pages: Math.ceil(total / limit) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get single design
app.get('/api/designs/:id', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    res.json({ design });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update design metadata
app.patch('/api/designs/:id', protect, async (req, res) => {
  try {
    const allowed = ['partName', 'material', 'thickness', 'notes'];
    const updates = {};
    allowed.forEach(f => { if (req.body[f] !== undefined) updates[f] = req.body[f]; });
    const design = await Design.findOneAndUpdate(
      { _id: req.params.id, owner: req.user._id },
      updates, { new: true }
    );
    if (!design) return res.status(404).json({ error: 'Design not found' });
    res.json({ design });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Approve a DWG
app.patch('/api/designs/:id/approve', protect, async (req, res) => {
  try {
    const design = await Design.findOneAndUpdate(
      { _id: req.params.id, owner: req.user._id, status: 'ready' },
      { status: 'approved', approvedAt: new Date() },
      { new: true }
    );
    if (!design) return res.status(404).json({ error: 'Design not found or not in ready state' });
    res.json({ design, message: 'Design approved successfully' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete design
app.delete('/api/designs/:id', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    // Delete Cloudinary asset if exists
    if (design.cloudinary?.publicId) {
      await cloudinary.uploader.destroy(design.cloudinary.publicId).catch(() => {});
    }
    // Delete local files
    await cleanupFile(design.originalFile?.path);
    await cleanupFile(design.dwg?.path);
    await design.deleteOne();
    res.json({ message: 'Design deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — AI CONVERSION
// ================================================================

// Trigger AI analysis + DWG conversion for a design
app.post('/api/convert/:id', protect, async (req, res) => {
  let design;
  try {
    design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    if (!design.originalFile?.path) return res.status(400).json({ error: 'No file attached to design' });

    // Step 1: Mark as analyzing
    design.status = 'analyzing';
    await design.save();

    // Step 2: Real AI Vision Analysis (OpenCV pre-process → Anthropic Vision → JSON geometry)
    const analysis = await runAIAnalysis(design.originalFile.path, design.originalFile.mimetype);
    design.aiAnalysis = analysis;
    design.status = 'converting';
    await design.save();

    const mat   = design.material  || 'Mild Steel';
    const thick = design.thickness || 2;
    const ts    = Date.now();
    const base  = design.partName.replace(/\s+/g, '_');

    // Step 3: Generate real .DXF file
    const dxfString  = generateDXFFile(analysis, mat, thick);
    const dxfPath    = path.join(uploadDir, `${base}_${ts}.dxf`);
    fs.writeFileSync(dxfPath, dxfString, 'utf8');

    // Step 4: Generate G-Code for every common machine type
    const gcodeFiles = {};
    const machineTypes = ['laser', 'plasma', 'waterjet', 'mill', 'router', 'lathe', 'edm', 'oxyfuel'];
    for (const mType of machineTypes) {
      const gc   = generateGCode(analysis, mat, thick, mType);
      const gcP  = path.join(uploadDir, `${base}_${mType}_${ts}.nc`);
      fs.writeFileSync(gcP, gc, 'utf8');
      gcodeFiles[mType] = gcP;
    }

    // Step 5: Generate DWG preview PNG (accurate to analysis geometry)
    const previewPath = await generateDWGPreview(analysis, mat, thick);

    design.dwg = {
      filename    : base + '.dxf',
      path        : previewPath,
      dxfPath,
      gcodeFiles,
      entities    : analysis.edges,
      fileSize    : fs.statSync(previewPath).size,
      generatedAt : new Date(),
    };
    design.status = 'ready';
    await design.save();

    res.json({
      design,
      analysis,
      message    : 'DXF + G-Code conversion complete',
      dxfReady   : true,
      gcodeReady : true,
      machines   : machineTypes,
    });
  } catch (err) {
    if (design) { design.status = 'uploaded'; await design.save().catch(() => {}); }
    res.status(500).json({ error: err.message });
  }
});

// Download .DXF file
app.get('/api/designs/:id/dxf', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design?.dwg?.dxfPath) return res.status(404).json({ error: 'DXF not generated yet' });
    if (!fs.existsSync(design.dwg.dxfPath)) return res.status(404).json({ error: 'DXF file missing on disk' });
    const filename = design.dwg.filename || 'part.dxf';
    res.setHeader('Content-Type', 'application/dxf');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.sendFile(path.resolve(design.dwg.dxfPath));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Download G-Code for a given machine type
app.get('/api/designs/:id/gcode/:machine', protect, async (req, res) => {
  try {
    const { machine } = req.params;
    if (!MACHINE_PROFILES[machine]) return res.status(400).json({ error: `Unknown machine type: ${machine}` });
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design?.dwg?.gcodeFiles?.[machine]) return res.status(404).json({ error: 'G-Code not generated yet. Run conversion first.' });
    const gcPath = design.dwg.gcodeFiles[machine];
    if (!fs.existsSync(gcPath)) return res.status(404).json({ error: 'G-Code file missing on disk' });
    const fname = `${design.partName.replace(/\s+/g,'_')}_${machine}.nc`;
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Content-Disposition', `attachment; filename="${fname}"`);
    res.sendFile(path.resolve(gcPath));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get conversion status (polling endpoint)
app.get('/api/convert/:id/status', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id }).lean();
    if (!design) return res.status(404).json({ error: 'Not found' });
    res.json({ status: design.status, aiAnalysis: design.aiAnalysis, dwg: design.dwg });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Serve DWG preview PNG for viewer
app.get('/api/designs/:id/preview', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design?.dwg?.path) return res.status(404).json({ error: 'Preview not available' });
    if (!fs.existsSync(design.dwg.path)) return res.status(404).json({ error: 'Preview file missing' });
    res.sendFile(design.dwg.path);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — CLOUDINARY SAVE
// ================================================================
app.post('/api/cloud/save/:id', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    if (design.status !== 'approved')
      return res.status(400).json({ error: 'Design must be approved before saving to cloud' });
    if (!design.dwg?.path || !fs.existsSync(design.dwg.path))
      return res.status(400).json({ error: 'DWG preview PNG not found' });

    const { assetName, folder, tags } = req.body;
    const publicId = `${folder || 'sheetforge/designs'}/${assetName || design.partName.replace(/\s+/g, '_')}_${Date.now()}`;

    // Upload PNG to Cloudinary
    const result = await cloudinary.uploader.upload(design.dwg.path, {
      public_id      : publicId,
      resource_type  : 'image',
      format         : 'png',
      quality        : 'auto:best',
      tags           : tags ? tags.split(',').map(t => t.trim()) : ['sheetforge', 'cad', 'dwg'],
      transformation : [{ width: 1200, height: 800, crop: 'fit' }],
      context        : `part=${design.partName}|material=${design.material || ''}|owner=${req.user.email}`,
    });

    // Persist cloud metadata in MongoDB
    design.cloudinary = {
      publicId   : result.public_id,
      url        : result.url,
      secureUrl  : result.secure_url,
      width      : result.width,
      height     : result.height,
      bytes      : result.bytes,
      format     : result.format,
      uploadedAt : new Date(),
    };
    design.mongodb = { savedAt: new Date() };
    design.status = 'saved';
    await design.save();

    res.json({
      message    : 'Saved to Cloudinary and MongoDB',
      cloudinary : design.cloudinary,
      mongoId    : design._id,
      design,
    });
  } catch (err) {
    console.error('Cloud save error:', err);
    res.status(500).json({ error: err.message });
  }
});

// List all cloud-saved assets for user
app.get('/api/cloud/assets', protect, async (req, res) => {
  try {
    const designs = await Design.find({
      owner  : req.user._id,
      status : 'saved',
      'cloudinary.secureUrl': { $exists: true },
    }).select('partName cloudinary mongodb createdAt').lean();
    res.json({ assets: designs });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — QUOTES
// ================================================================
app.post('/api/quotes', protect, async (req, res) => {
  try {
    const { designId, specs } = req.body;
    if (!designId) return res.status(400).json({ error: 'designId is required' });

    const design = await Design.findOne({ _id: designId, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    if (!['approved', 'saved'].includes(design.status))
      return res.status(400).json({ error: 'Design must be approved first' });

    // Find matching providers and auto-generate bids
    const providers = await Provider.find({ isActive: true }).limit(10);
    const bids = providers.map(p => {
      const area = ((specs.length || 200) * (specs.width || 150)) / 1e6;
      const matMult = { steel: 1, stainless: 1.8, aluminium: 1.3, copper: 2.5 }[
        (specs.material || '').toLowerCase().split(' ')[0]] || 1;
      const base = area * (specs.thickness || 2) * 7800 * 0.8 * matMult * (specs.quantity || 50);
      const price = Math.round((base + base * (Math.random() * 0.4 + 0.8)) * p.pricingBase);
      return {
        provider   : p._id,
        price,
        perUnit    : Math.round((price / (specs.quantity || 50)) * 100) / 100,
        leadDays   : p.leadTimeDays?.min || 14,
        notes      : `Includes ${p.certifications?.[0] || 'standard'} certified production`,
        submittedAt: new Date(),
        status     : 'submitted',
      };
    });

    const quote = await Quote.create({
      design   : designId,
      requester: req.user._id,
      specs,
      bids,
    });
    await quote.populate('bids.provider', 'companyName country flag rating');
    res.status(201).json({ quote });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/quotes', protect, async (req, res) => {
  try {
    const quotes = await Quote.find({ requester: req.user._id })
      .populate('design', 'partName material')
      .populate('bids.provider', 'companyName country flag rating')
      .sort('-createdAt')
      .lean();
    res.json({ quotes });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/quotes/:id', protect, async (req, res) => {
  try {
    const quote = await Quote.findOne({ _id: req.params.id, requester: req.user._id })
      .populate('design')
      .populate('bids.provider');
    if (!quote) return res.status(404).json({ error: 'Quote not found' });
    res.json({ quote });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — PROVIDERS
// ================================================================
app.get('/api/providers', async (req, res) => {
  try {
    const { region, material, sort = '-rating', limit = 20 } = req.query;
    const query = { isActive: true };
    if (region) query.region = region;
    if (material) query.materials = { $in: [material] };
    const providers = await Provider.find(query).sort(sort).limit(parseInt(limit)).lean();
    res.json({ providers });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/providers/:id', async (req, res) => {
  try {
    const provider = await Provider.findById(req.params.id).lean();
    if (!provider) return res.status(404).json({ error: 'Provider not found' });
    res.json({ provider });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Register as a CNC provider
app.post('/api/providers', protect, restrictTo('provider', 'admin'), async (req, res) => {
  try {
    const provider = await Provider.create({ ...req.body, user: req.user._id });
    res.status(201).json({ provider });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — ORDERS
// ================================================================
app.post('/api/orders', protect, async (req, res) => {
  try {
    const { designId, providerId, quoteId, specs, pricing } = req.body;
    const order = await Order.create({
      buyer    : req.user._id,
      provider : providerId,
      design   : designId,
      quote    : quoteId || null,
      specs,
      pricing,
      status   : 'confirmed',
      timeline : [{ status: 'confirmed', note: 'Order placed and confirmed' }],
      estimatedDelivery: new Date(Date.now() + (pricing.leadDays || 14) * 86400000),
    });
    if (quoteId) await Quote.findByIdAndUpdate(quoteId, { status: 'ordered' });
    await order.populate('provider', 'companyName country flag');
    await order.populate('design', 'partName material');
    res.status(201).json({ order });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/orders', protect, async (req, res) => {
  try {
    const { status } = req.query;
    const query = { buyer: req.user._id };
    if (status) query.status = status;
    const orders = await Order.find(query)
      .populate('provider', 'companyName country flag rating')
      .populate('design', 'partName material cloudinary')
      .sort('-createdAt')
      .lean();
    res.json({ orders });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.patch('/api/orders/:id/status', protect, async (req, res) => {
  try {
    const { status, note } = req.body;
    const order = await Order.findOne({ _id: req.params.id, buyer: req.user._id });
    if (!order) return res.status(404).json({ error: 'Order not found' });
    order.status = status;
    order.timeline.push({ status, note: note || status });
    await order.save();
    res.json({ order });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// SEED DEMO DATA
// ================================================================
async function seedProviders() {
  const count = await Provider.countDocuments();
  if (count > 0) return;
  console.log('🌱  Seeding provider data...');
  const seedData = [
    { companyName: 'MetalTech GmbH', country: 'Germany', region: 'Europe', flag: '🇩🇪', rating: 4.9, totalReviews: 214, totalOrders: 1420, pricingBase: 1.0, capacity: '500t/mo', leadTimeDays: { min: 10, max: 14 }, certifications: ['ISO 9001','CE'], materials: ['steel','stainless','aluminium'], operations: ['laser','bending','punching'], specialty: 'Precision laser cutting', isVerified: true },
    { companyName: 'PrecisionCut SG', country: 'Singapore', region: 'Asia Pacific', flag: '🇸🇬', rating: 4.8, totalReviews: 189, totalOrders: 870, pricingBase: 0.9, capacity: '200t/mo', leadTimeDays: { min: 7, max: 10 }, certifications: ['ISO 9001'], materials: ['aluminium','steel'], specialty: 'Aluminium fabrication', isVerified: true },
    { companyName: 'SheetWorks IN', country: 'India', region: 'Asia Pacific', flag: '🇮🇳', rating: 4.7, totalReviews: 301, totalOrders: 2310, pricingBase: 0.65, capacity: '1000t/mo', leadTimeDays: { min: 14, max: 21 }, certifications: ['ISO 9001','IATF 16949'], materials: ['steel','stainless','aluminium','copper'], specialty: 'High-volume production', isVerified: true },
    { companyName: 'ProSheet US', country: 'United States', region: 'North America', flag: '🇺🇸', rating: 4.8, totalReviews: 244, totalOrders: 1150, pricingBase: 1.3, capacity: '250t/mo', leadTimeDays: { min: 5, max: 8 }, certifications: ['ISO 9001','ASME'], materials: ['steel','stainless','aluminium'], specialty: 'Aerospace-grade quality', isVerified: true },
    { companyName: 'Fabricate ZA', country: 'South Africa', region: 'Africa', flag: '🇿🇦', rating: 4.5, totalReviews: 67, totalOrders: 340, pricingBase: 0.75, capacity: '300t/mo', leadTimeDays: { min: 12, max: 16 }, certifications: ['SANS 10147'], materials: ['steel','stainless'], specialty: 'Structural steel', isVerified: true },
  ];
  await Provider.insertMany(seedData);
  console.log('✅  Providers seeded');
}

// ================================================================
// ERROR HANDLING
// ================================================================
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE')
      return res.status(400).json({ error: 'File too large. Maximum size is 50MB.' });
  }
  console.error('Server error:', err);
  res.status(err.statusCode || 500).json({ error: err.message || 'Internal server error' });
});

app.use((req, res) => res.status(404).json({ error: 'Route not found' }));

// ================================================================
// START
// ================================================================
app.listen(PORT, () => {
  console.log(`\n🚀  SheetForge server running at http://localhost:${PORT}`);
  console.log(`📁  Static files served from: ${__dirname}`);
  console.log(`🔑  JWT Secret: ${process.env.JWT_SECRET ? 'Set via env' : 'Using default (set JWT_SECRET in .env for production)'}`);
});

module.exports = app;
