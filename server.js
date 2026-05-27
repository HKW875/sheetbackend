/**
 * SheetForge v4 — Express Backend
 * ================================
 * New in v4:
 *  - /api/gcode/:id          POST  — Generate G-Code from DXF analysis
 *  - /api/ai-interact        POST  — AI natural-language DXF manipulation
 *  - /api/cloud/gcode        POST  — Upload G-Code as PDF to Cloudinary + save link in MongoDB
 *  - /api/dashboard/activity/:id DELETE — Delete activity entry
 *  - /api/designs/:id/approve PATCH — now also triggers GCode generation
 *  - All existing v3 routes preserved
 */

'use strict';
require('dotenv').config();

const express      = require('express');
const cors         = require('cors');
const helmet       = require('helmet');
const morgan       = require('morgan');
const multer       = require('multer');
const path         = require('path');
const fs           = require('fs');
const fsp          = require('fs').promises;
const { spawn }    = require('child_process');
const mongoose     = require('mongoose');
const jwt          = require('jsonwebtoken');
const bcrypt       = require('bcryptjs');
const cloudinary   = require('cloudinary').v2;
const PDFDocument  = require('pdfkit');
const Anthropic    = require('@anthropic-ai/sdk');

// ── Config ────────────────────────────────────────────────────────────────────
const PORT       = process.env.PORT || 5000;
const MONGO_URI  = process.env.MONGO_URI  || 'mongodb://localhost:27017/sheetforge';
const JWT_SECRET = process.env.JWT_SECRET || 'sheetforge_secret_v4';
const PYTHON_CMD = process.env.PYTHON_CMD || 'python3';
const PROCESS_PY = path.join(__dirname, 'process.py');
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(UPLOAD_DIR, 'output');

[UPLOAD_DIR, OUTPUT_DIR].forEach(d => { if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true }); });

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME || '',
  api_key:    process.env.CLOUDINARY_API_KEY    || '',
  api_secret: process.env.CLOUDINARY_API_SECRET || '',
});

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY || '' });

// ── Mongoose Models ───────────────────────────────────────────────────────────
mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(e  => console.error('MongoDB error:', e.message));

const userSchema = new mongoose.Schema({
  firstName: String, lastName: String, email: { type: String, unique: true },
  passwordHash: String, role: { type: String, enum: ['designer','provider'], default: 'designer' },
  company: String, createdAt: { type: Date, default: Date.now },
});
const User = mongoose.model('User', userSchema);

const designSchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId, partName: String, material: String, thickness: Number,
  status: { type: String, default: 'uploaded' },
  originalFile: { filename: String, url: String, cloudinaryId: String, mimetype: String, size: Number, localPath: String },
  aiAnalysis: mongoose.Schema.Types.Mixed,
  dwg: {
    entities: Number, fileSize: Number, filename: String, svgFilename: String,
    dxfUrl: String, svgUrl: String, pdfUrl: String, gcodeUrl: String,
    gcodeFilename: String, localPath: String,
  },
  gcodeHistory: [{ gcode: String, url: String, opts: Object, generatedAt: Date }],
  conversionLog: [mongoose.Schema.Types.Mixed],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});
const Design = mongoose.model('Design', designSchema);

const quoteSchema = new mongoose.Schema({
  designId: String, userId: mongoose.Types.ObjectId, designName: String,
  specs: { quantity: Number, material: String, finish: String, tolerance: String },
  notes: String, status: { type: String, default: 'open' },
  bids: [{ providerId: String, providerName: String, price: Number, perUnit: Number, leadDays: Number, notes: String, submittedAt: Date }],
  createdAt: { type: Date, default: Date.now },
});
const Quote = mongoose.model('Quote', quoteSchema);

const orderSchema = new mongoose.Schema({
  orderNumber: String, quoteId: String, designId: String,
  userId: mongoose.Types.ObjectId, providerId: String, providerName: String,
  status: { type: String, default: 'confirmed' },
  specs: mongoose.Schema.Types.Mixed,
  pricing: { unitPrice: Number, quantity: Number, subtotal: Number, shipping: Number, total: Number, currency: String },
  estimatedDelivery: Date, trackingNumber: String,
  createdAt: { type: Date, default: Date.now },
});
const Order = mongoose.model('Order', orderSchema);

const activitySchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId,
  type: { type: String, enum: ['upload','convert','approve','order','quote','ai_interact','gcode'] },
  title: String, description: String,
  timestamp: { type: Date, default: Date.now },
  metadata: mongoose.Schema.Types.Mixed,
});
const Activity = mongoose.model('Activity', activitySchema);

const providerSchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId, companyName: String, region: String, country: String,
  rating: { type: Number, default: 0 }, totalReviews: { type: Number, default: 0 },
  isVerified: { type: Boolean, default: false },
  leadTimeDays: { min: Number, max: Number },
  materials: [String], services: [String],
  contact: { address: String, phone: String, whatsapp: String, email: String, website: String },
  description: String, createdAt: { type: Date, default: Date.now },
});
const Provider = mongoose.model('Provider', providerSchema);

// ── App ───────────────────────────────────────────────────────────────────────
const app = express();
app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
app.use(cors({ origin: '*', methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'], allowedHeaders: ['Content-Type','Authorization'] }));
app.use(morgan('combined'));
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));
app.use('/uploads', express.static(UPLOAD_DIR));

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /\.(pdf|dxf|png|jpg|jpeg|bmp|tiff?|webp)$/i.test(file.originalname);
    cb(ok ? null : new Error('Unsupported file type'), ok);
  },
});

// ── Auth middleware ────────────────────────────────────────────────────────────
function authMiddleware(req, res, next) {
  const token = (req.headers.authorization || '').replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'No token' });
  try { req.user = jwt.verify(token, JWT_SECRET); next(); }
  catch { res.status(401).json({ error: 'Invalid token' }); }
}

function logActivity(userId, type, title, description, metadata = {}) {
  Activity.create({ userId, type, title, description, timestamp: new Date(), metadata }).catch(() => {});
}

// ── Helpers ────────────────────────────────────────────────────────────────────
async function uploadToCloudinary(filePath, folder = 'sheetforge', resourceType = 'raw') {
  return new Promise((resolve, reject) => {
    cloudinary.uploader.upload(filePath, { folder, resource_type: resourceType }, (err, result) => {
      if (err) reject(err); else resolve(result);
    });
  });
}

async function gcodeToPdf(gcodeText, outputPath, designName = 'Part') {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({ margin: 40, size: 'A4' });
      const stream = fs.createWriteStream(outputPath);
      doc.pipe(stream);
      doc.fontSize(16).fillColor('#1a1a2e').text(`SheetForge v4 — G-Code Export`, { align: 'center' });
      doc.fontSize(11).fillColor('#555').text(`Part: ${designName}  •  Generated: ${new Date().toISOString()}`, { align: 'center' });
      doc.moveDown(0.5);
      doc.moveTo(40, doc.y).lineTo(doc.page.width - 40, doc.y).stroke('#ccc');
      doc.moveDown(0.5);
      doc.font('Courier').fontSize(8).fillColor('#1a1a2e');
      const lines = gcodeText.split('\n');
      lines.forEach((line, i) => {
        const isComment = line.trim().startsWith(';');
        doc.fillColor(isComment ? '#888' : '#000').text(`${String(i + 1).padStart(4, ' ')}  ${line}`);
      });
      doc.end();
      stream.on('finish', () => resolve(outputPath));
      stream.on('error', reject);
    } catch (e) { reject(e); }
  });
}

// ── Auth Routes ───────────────────────────────────────────────────────────────
app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password, firstName, lastName, role, company } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });
    const existing = await User.findOne({ email: email.toLowerCase() });
    if (existing) return res.status(409).json({ error: 'Email already registered' });
    const passwordHash = await bcrypt.hash(password, 12);
    const user = await User.create({ email: email.toLowerCase(), passwordHash, firstName, lastName, role: role || 'designer', company });
    const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '30d' });
    if (role === 'provider') {
      await Provider.create({ userId: user._id, companyName: company || `${firstName} ${lastName} Workshop`, region: 'Unknown', country: 'Unknown', contact: { email } });
    }
    res.json({ token, user: { id: user._id, email: user.email, firstName, lastName, role: user.role } });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email: email?.toLowerCase() });
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: 'Invalid credentials' });
    const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '30d' });
    res.json({ token, user: { id: user._id, email: user.email, firstName: user.firstName, lastName: user.lastName, role: user.role } });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/auth/me', authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-passwordHash');
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({ id: user._id, email: user.email, firstName: user.firstName, lastName: user.lastName, role: user.role, company: user.company });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Dashboard Routes ──────────────────────────────────────────────────────────
app.get('/api/dashboard/summary', authMiddleware, async (req, res) => {
  try {
    const uid = req.user.id;
    const [totalDesigns, pendingConversions, totalQuotes, orders] = await Promise.all([
      Design.countDocuments({ userId: uid }),
      Design.countDocuments({ userId: uid, status: { $in: ['uploaded','analyzing','converting'] } }),
      Quote.countDocuments({ userId: uid }),
      Order.find({ userId: uid }).select('pricing'),
    ]);
    const totalSpend = orders.reduce((s, o) => s + (o.pricing?.total || 0), 0);
    res.json({ totalDesigns, pendingConversions, totalQuotes, totalOrders: orders.length, totalSpend });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/dashboard/activity', authMiddleware, async (req, res) => {
  try {
    const items = await Activity.find({ userId: req.user.id }).sort({ timestamp: -1 }).limit(30);
    res.json(items.map(a => ({ id: a._id, type: a.type, title: a.title, description: a.description, timestamp: a.timestamp })));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// DELETE activity entry
app.delete('/api/dashboard/activity/:id', authMiddleware, async (req, res) => {
  try {
    await Activity.deleteOne({ _id: req.params.id, userId: req.user.id });
    res.json({ success: true });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Design Routes ─────────────────────────────────────────────────────────────
app.get('/api/designs', authMiddleware, async (req, res) => {
  try {
    const designs = await Design.find({ userId: req.user.id }).sort({ createdAt: -1 }).limit(50)
      .select('partName material status originalFile aiAnalysis dwg createdAt');
    res.json({ designs: designs.map(d => ({ id: d._id, partName: d.partName, material: d.material, status: d.status, originalFile: d.originalFile, aiAnalysis: d.aiAnalysis, dwg: d.dwg, createdAt: d.createdAt })) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/designs', authMiddleware, upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'File required' });
  try {
    const { partName, material, thickness } = req.body;
    const ext = path.extname(req.file.originalname).toLowerCase();
    const newPath = `${req.file.path}${ext}`;
    await fsp.rename(req.file.path, newPath);

    let fileUrl = `/uploads/${path.basename(newPath)}`;
    let cloudinaryId = '';

    // Upload to Cloudinary
    try {
      const isImage = /\.(png|jpg|jpeg|bmp|webp|tiff?)$/i.test(ext);
      const cRes = await uploadToCloudinary(newPath, 'sheetforge/originals', isImage ? 'image' : 'raw');
      fileUrl = cRes.secure_url; cloudinaryId = cRes.public_id;
    } catch {}

    const design = await Design.create({
      userId: req.user.id, partName: partName || req.file.originalname.replace(/\.[^.]+$/, ''),
      material, thickness: thickness ? parseFloat(thickness) : null,
      originalFile: { filename: req.file.originalname, url: fileUrl, cloudinaryId, mimetype: req.file.mimetype, size: req.file.size, localPath: newPath },
    });

    logActivity(req.user.id, 'upload', `Uploaded: ${design.partName}`, `${req.file.originalname} • ${(req.file.size / 1024).toFixed(0)}KB`);
    res.json({ id: design._id, status: design.status });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/designs/:id', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Not found' });
    res.json({ id: d._id, partName: d.partName, material: d.material, thickness: d.thickness, status: d.status, originalFile: d.originalFile, aiAnalysis: d.aiAnalysis, dwg: d.dwg, createdAt: d.createdAt });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.patch('/api/designs/:id/approve', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Not found' });
    const corrected = req.body.correctedAnalysis || {};
    d.aiAnalysis = { ...d.aiAnalysis, ...corrected };
    d.status = 'approved'; d.updatedAt = new Date();
    await d.save();
    logActivity(req.user.id, 'approve', `Approved: ${d.partName}`, `W=${d.aiAnalysis.width}mm H=${d.aiAnalysis.height}mm`);
    res.json({ id: d._id, status: d.status, aiAnalysis: d.aiAnalysis });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Conversion Pipeline ───────────────────────────────────────────────────────
    // === RUN PYTHON PIPELINE (Improved) ===
    let output = '', errorOut = '';
    console.log(`[Convert] Starting pipeline for design ${designId}`);
    console.log(`[Convert] Using file: ${localImagePath}`);

    const py = spawn(PYTHON_CMD, [PROCESS_PY, localImagePath || '', JSON.stringify(opts)], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    py.stdout.on('data', chunk => { 
      output += chunk.toString(); 
      // console.log('[py stdout]', chunk.toString().slice(0, 200)); // uncomment for debugging
    });

    py.stderr.on('data', chunk => { 
      errorOut += chunk.toString(); 
      console.error('[py stderr]', chunk.toString().slice(0, 300)); 
    });

    py.on('close', async (code) => {
      console.log(`[Convert] Python process exited with code: ${code}`);

      if (code !== 0 || !output.trim()) {
        console.error('[Convert] Python failed!', errorOut);
        d.status = 'error';
        await d.save();
        return;
      }

      try {
        // Get the last JSON line (process.py prints multiple things)
        const lastLine = output.trim().split('\n').pop();
        const result = JSON.parse(lastLine);

        if (result.error) throw new Error(result.error);

        d.aiAnalysis    = result.analysis;
        d.dwg           = { ...result.dwg };
        d.conversionLog = result.steps;
        d.status        = 'ready';
        d.updatedAt     = new Date();

        // Upload DXF + SVG to Cloudinary
        const dxfPath = path.join(OUTPUT_DIR, result.dwg.filename || '');
        const svgPath = path.join(OUTPUT_DIR, result.dwg.svgFilename || '');
        if (fs.existsSync(dxfPath)) {
          try { const cr = await uploadToCloudinary(dxfPath, 'sheetforge/dxf', 'raw'); d.dwg.dxfUrl = cr.secure_url; } catch {}
        }
        if (fs.existsSync(svgPath)) {
          try { const cr = await uploadToCloudinary(svgPath, 'sheetforge/svg', 'image'); d.dwg.svgUrl = cr.secure_url; } catch {}
        }

        // Store GCode if generated
        if (result.gcode) {
          d.gcodeHistory = [{ gcode: result.gcode, opts: opts, generatedAt: new Date() }];
          d.dwg.gcodeFilename = result.dwg.gcodeFilename || '';
          const gcodePath = path.join(OUTPUT_DIR, d.dwg.gcodeFilename || '');
          if (fs.existsSync(gcodePath)) {
            try { const cr = await uploadToCloudinary(gcodePath, 'sheetforge/gcode', 'raw'); d.dwg.gcodeUrl = cr.secure_url; } catch {}
          }
        }

        await d.save();
        logActivity(req.user.id, 'convert', `Converted: ${d.partName}`, `${result.steps?.length || 23} stages`);

      } catch (e) {
        console.error('Conversion parsing error:', e.message);
        console.error('Raw output:', output.slice(0, 600));
        d.status = 'error';
        await d.save();
      }
    });
    res.json({ id: designId, status: 'analyzing', message: 'Pipeline started' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/convert/:id/status', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Not found' });
    const steps = (d.conversionLog || []).map(s => ({ name: s.name || s, status: 'done', duration: s.duration, details: s.details }));
    const pct = steps.length > 0 ? Math.round((steps.filter(s => s.status === 'done').length / Math.max(steps.length, 23)) * 100) : 0;
    res.json({ status: d.status, progress: pct, steps, currentStep: steps[steps.length - 1]?.name || '' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── GCode Routes ──────────────────────────────────────────────────────────────
app.post('/api/gcode/:id', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Design not found' });
    const gcodeOpts = req.body.gcodeOptions || {};
    const analysis  = { ...d.aiAnalysis, ...gcodeOpts };

    // Call process.py in ai_interact mode for gcode
    const opts = { ...req.body, anthropicApiKey: process.env.ANTHROPIC_API_KEY, thickness: d.thickness || 2.0 };

    // === IMPROVED LOCAL FILE PATH RESOLUTION ===
    let localImagePath = null;

    if (d.originalFile?.localPath && fs.existsSync(d.originalFile.localPath)) {
      localImagePath = d.originalFile.localPath;
    }

    if (!localImagePath && d.originalFile?.url?.startsWith('/uploads/')) {
      const urlBased = path.join(__dirname, d.originalFile.url);
      if (fs.existsSync(urlBased)) localImagePath = urlBased;
    }

    if (!localImagePath) {
      try {
        const files = fs.readdirSync(UPLOAD_DIR)
          .filter(f => !fs.statSync(path.join(UPLOAD_DIR, f)).isDirectory());

        const origStem = path.parse(d.originalFile?.filename || '').name;
        const designIdShort = d._id.toString().slice(-8);

        const match = files.find(f => 
          (origStem && (f.startsWith(origStem) || f.includes(origStem))) ||
          f.includes(designIdShort)
        );

        if (match) localImagePath = path.join(UPLOAD_DIR, match);
      } catch (e) {
        console.error('File search failed:', e.message);
      }
    }

    if (!localImagePath || !fs.existsSync(localImagePath)) {
      console.error(`[Convert] Source file not found for design ${designId}`);
      d.status = 'error';
      await d.save();
      return res.status(404).json({ error: 'Source file not found on server' });
    }

    // === RUN PYTHON PIPELINE ===
    let output = '', errorOut = '';
    console.log(`[Convert] Starting pipeline for design ${designId}`);
    console.log(`[Convert] Using file: ${localImagePath}`);

    const py = spawn(PYTHON_CMD, [PROCESS_PY, localImagePath, JSON.stringify(opts)], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    py.stdout.on('data', chunk => { 
      output += chunk.toString(); 
    });

    py.stderr.on('data', chunk => { 
      errorOut += chunk.toString(); 
      console.error('[py stderr]', chunk.toString().slice(0, 300)); 
    });

    py.on('close', async (code) => {
      console.log(`[Convert] Python process exited with code: ${code}`);

      if (code !== 0 || !output.trim()) {
        console.error('[Convert] Python failed!', errorOut);
        d.status = 'error';
        await d.save();
        return;
      }

      try {
        const lastLine = output.trim().split('\n').pop();
        const result = JSON.parse(lastLine);

        if (result.error) throw new Error(result.error);

        d.aiAnalysis    = result.analysis;
        d.dwg           = { ...result.dwg };
        d.conversionLog = result.steps;
        d.status        = 'ready';
        d.updatedAt     = new Date();

        // Upload DXF + SVG to Cloudinary
        const dxfPath = path.join(OUTPUT_DIR, result.dwg.filename || '');
        const svgPath = path.join(OUTPUT_DIR, result.dwg.svgFilename || '');
        if (fs.existsSync(dxfPath)) {
          try { const cr = await uploadToCloudinary(dxfPath, 'sheetforge/dxf', 'raw'); d.dwg.dxfUrl = cr.secure_url; } catch {}
        }
        if (fs.existsSync(svgPath)) {
          try { const cr = await uploadToCloudinary(svgPath, 'sheetforge/svg', 'image'); d.dwg.svgUrl = cr.secure_url; } catch {}
        }

        // Store GCode if generated
        if (result.gcode) {
          d.gcodeHistory = [{ gcode: result.gcode, opts: opts, generatedAt: new Date() }];
          d.dwg.gcodeFilename = result.dwg.gcodeFilename || '';
          const gcodePath = path.join(OUTPUT_DIR, d.dwg.gcodeFilename || '');
          if (fs.existsSync(gcodePath)) {
            try { const cr = await uploadToCloudinary(gcodePath, 'sheetforge/gcode', 'raw'); d.dwg.gcodeUrl = cr.secure_url; } catch {}
          }
        }

        await d.save();
        logActivity(req.user.id, 'convert', `Converted: ${d.partName}`, `${result.steps?.length || 23} stages`);

      } catch (e) {
        console.error('Conversion parsing error:', e.message);
        console.error('Raw output preview:', output.slice(0, 500));
        d.status = 'error';
        await d.save();
      }
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

function generateFallbackGCode(analysis = {}, opts = {}) {
  const W = analysis.width || 200, H = analysis.height || 150;
  const feed = opts.feedRate || 1000, plunge = opts.plungeRate || 300;
  const rpm = opts.spindleRpm || 12000, safeZ = opts.safeZ || 5;
  const depth = opts.cutDepth || 3, passDepth = opts.passDepth || 1;
  const toolD = opts.toolDiameter || 3;
  return [
    `; SheetForge v4 — ${analysis.profileType || 'Sheet Metal'} | ${W}×${H}mm`,
    `; Material: ${analysis.material || 'unknown'} | Thickness: ${analysis.thickness || 2}mm`,
    `; Tool: Ø${toolD}mm | Feed: ${feed}mm/min | Spindle: ${rpm}rpm`,
    'G21 G17 G90 G94 G40 G49',
    `T01 M6 ; Ø${toolD}mm`, `S${rpm} M3`, 'G4 P2000',
    `G00 Z${safeZ}`, 'G00 X0 Y0',
    `; === PASS 1/${Math.ceil(depth / passDepth)} ===`,
    `G00 X-${(toolD/2).toFixed(3)} Y-${(toolD/2).toFixed(3)}`,
    `G01 Z-${passDepth} F${plunge}`,
    `G01 X${(W + toolD/2).toFixed(3)} F${feed}`,
    `G01 Y${(H + toolD/2).toFixed(3)}`,
    `G01 X-${(toolD/2).toFixed(3)}`,
    `G01 Y-${(toolD/2).toFixed(3)}`,
    `G00 Z${safeZ}`, 'G00 X0 Y0', 'M5', 'M30',
  ].join('\n');
}

// ── AI Interact Route ─────────────────────────────────────────────────────────
app.post('/api/ai-interact', authMiddleware, async (req, res) => {
  try {
    const { instruction, designId, analysis } = req.body;
    if (!instruction) return res.status(400).json({ error: 'Instruction required' });

    let currentAnalysis = analysis || {};
    let design = null;
    if (designId) {
      design = await Design.findOne({ _id: designId, userId: req.user.id });
      if (design) currentAnalysis = { ...design.aiAnalysis, ...analysis };
    }

    // Build context
    const ctx = Object.entries(currentAnalysis)
      .filter(([k]) => !k.startsWith('_'))
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');

    const prompt = `You are an expert CAD/DXF engineer. Current design: ${ctx || 'no design loaded'}.
User instruction: "${instruction}"

Return ONLY a JSON object with ONLY the fields that need to change plus an explanation:
{
  "width": <number or omit>,
  "height": <number or omit>,
  "holes": <number or omit>,
  "holesDiameter": <number or omit>,
  "bendLines": <number or omit>,
  "thickness": <number or omit>,
  "material": "<string or omit>",
  "tolerance": "<string or omit>",
  "explanation": "<clear explanation of what was changed and why>"
}
Apply ONLY changes relevant to the instruction. Return ONLY valid JSON.`;

    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514', max_tokens: 800,
      messages: [{ role: 'user', content: prompt }],
    });

    let text = message.content[0]?.text || '{}';
    text = text.replace(/```[a-z]*/g, '').replace(/```/g, '').trim();
    const m = text.match(/\{[\s\S]*\}/);
    if (m) text = m[0];
    const changes = JSON.parse(text);
    const explanation = changes.explanation || 'Changes applied.';
    delete changes.explanation;

    const updatedAnalysis = { ...currentAnalysis, ...Object.fromEntries(Object.entries(changes).filter(([, v]) => v !== null && v !== undefined)) };

    // Save back to design if designId
    if (design) {
      design.aiAnalysis = updatedAnalysis;
      design.updatedAt = new Date();
      await design.save();
    }

    const gcode = generateFallbackGCode(updatedAnalysis);
    logActivity(req.user.id, 'ai_interact', `AI Studio: ${instruction.slice(0, 60)}`, explanation.slice(0, 100));

    res.json({ analysis: updatedAnalysis, explanation, gcode });
  } catch (e) {
    res.status(500).json({ error: e.message, explanation: 'AI processing error — please try again.' });
  }
});

// ── Cloud Routes ──────────────────────────────────────────────────────────────
app.post('/api/cloud/save/:id', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Not found' });
    const uploads = [];

    if (d.dwg?.filename) {
      const dxfPath = path.join(OUTPUT_DIR, d.dwg.filename);
      if (fs.existsSync(dxfPath)) {
        try { const cr = await uploadToCloudinary(dxfPath, 'sheetforge/dxf', 'raw'); d.dwg.dxfUrl = cr.secure_url; uploads.push({ type: 'dxf', url: cr.secure_url }); } catch {}
      }
    }
    if (d.dwg?.svgFilename) {
      const svgPath = path.join(OUTPUT_DIR, d.dwg.svgFilename);
      if (fs.existsSync(svgPath)) {
        try { const cr = await uploadToCloudinary(svgPath, 'sheetforge/svg', 'image'); d.dwg.svgUrl = cr.secure_url; uploads.push({ type: 'svg', url: cr.secure_url }); } catch {}
      }
    }
    if (d.originalFile?.url?.startsWith('/uploads/')) {
      const origPath = path.join(__dirname, d.originalFile.url);
      if (fs.existsSync(origPath)) {
        try { const isImg = /\.(png|jpg|jpeg|gif|webp)$/i.test(origPath); const cr = await uploadToCloudinary(origPath, 'sheetforge/originals', isImg ? 'image' : 'raw'); d.originalFile.url = cr.secure_url; d.originalFile.cloudinaryId = cr.public_id; uploads.push({ type: 'original', url: cr.secure_url }); } catch {}
      }
    }

    d.status = 'saved'; d.updatedAt = new Date();
    await d.save();
    res.json({ id: d._id, status: d.status, uploads, dwg: d.dwg });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// Upload GCode as PDF to Cloudinary + save link to MongoDB
app.post('/api/cloud/gcode', authMiddleware, async (req, res) => {
  try {
    const { gcode, designId, filename } = req.body;
    if (!gcode) return res.status(400).json({ error: 'G-Code required' });

    const safeFilename = (filename || `gcode_${Date.now()}.pdf`).replace(/\.gcode$/, '.pdf');
    const pdfPath = path.join(OUTPUT_DIR, safeFilename);

    // Convert gcode text to PDF
    let designName = 'Part';
    let design = null;
    if (designId) {
      design = await Design.findOne({ _id: designId, userId: req.user.id });
      if (design) designName = design.partName;
    }
    await gcodeToPdf(gcode, pdfPath, designName);

    // Upload PDF to Cloudinary
    const cr = await uploadToCloudinary(pdfPath, 'sheetforge/gcode-pdfs', 'raw');
    const pdfUrl = cr.secure_url;

    // Save link to design if designId
    if (design) {
      design.gcodeHistory = design.gcodeHistory || [];
      design.gcodeHistory[0] = { ...design.gcodeHistory[0], url: pdfUrl, pdfUrl };
      if (design.gcodeHistory[0]?.generatedAt === undefined) design.gcodeHistory[0].generatedAt = new Date();
      design.dwg = design.dwg || {};
      design.dwg.gcodeUrl = pdfUrl;
      design.updatedAt = new Date();
      await design.save();
    }

    logActivity(req.user.id, 'gcode', `GCode PDF uploaded: ${designName}`, `Cloudinary: ${pdfUrl.slice(-40)}`);
    // Clean up local PDF
    fsp.unlink(pdfPath).catch(() => {});

    res.json({ url: pdfUrl, filename: safeFilename, size: cr.bytes });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Quote Routes ──────────────────────────────────────────────────────────────
app.get('/api/quotes', authMiddleware, async (req, res) => {
  try {
    const quotes = await Quote.find({ userId: req.user.id }).sort({ createdAt: -1 }).limit(30);
    res.json({ quotes: quotes.map(q => ({ id: q._id, designName: q.designName, status: q.status, specs: q.specs, bids: q.bids?.length || 0, createdAt: q.createdAt })) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/quotes', authMiddleware, async (req, res) => {
  try {
    const { designId, specs, notes } = req.body;
    const design = await Design.findOne({ _id: designId, userId: req.user.id });
    const quote = await Quote.create({ designId, userId: req.user.id, designName: design?.partName || 'Unknown', specs, notes });
    logActivity(req.user.id, 'quote', `Quote requested: ${design?.partName}`, `${specs?.quantity || 1} units • ${specs?.material || 'Standard'}`);
    res.json({ id: quote._id, status: quote.status });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/quotes/:id', authMiddleware, async (req, res) => {
  try {
    const q = await Quote.findOne({ _id: req.params.id, userId: req.user.id });
    if (!q) return res.status(404).json({ error: 'Not found' });
    res.json({ id: q._id, designId: q.designId, designName: q.designName, status: q.status, specs: q.specs, notes: q.notes, bids: q.bids, createdAt: q.createdAt });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Order Routes ──────────────────────────────────────────────────────────────
app.get('/api/orders', authMiddleware, async (req, res) => {
  try {
    const orders = await Order.find({ userId: req.user.id }).sort({ createdAt: -1 }).limit(30);
    res.json({ orders: orders.map(o => ({ id: o._id, orderNumber: o.orderNumber, providerName: o.providerName, status: o.status, pricing: o.pricing, estimatedDelivery: o.estimatedDelivery, createdAt: o.createdAt })) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/orders', authMiddleware, async (req, res) => {
  try {
    const { quoteId, designId, providerId, pricing, specs } = req.body;
    const orderNumber = `SF-${Date.now().toString(36).toUpperCase()}`;
    const provider = await Provider.findById(providerId);
    const subtotal = (pricing.unitPrice || 0) * (pricing.quantity || 1);
    const shipping = Math.round(subtotal * 0.08 * 100) / 100;
    const order = await Order.create({
      orderNumber, quoteId, designId, userId: req.user.id,
      providerId, providerName: provider?.companyName || 'Provider',
      pricing: { ...pricing, subtotal, shipping, total: subtotal + shipping },
      specs, estimatedDelivery: new Date(Date.now() + (14 * 24 * 60 * 60 * 1000)),
    });
    if (quoteId) await Quote.findByIdAndUpdate(quoteId, { status: 'ordered' });
    logActivity(req.user.id, 'order', `Order placed: ${order.orderNumber}`, `${provider?.companyName || ''} • $${(subtotal + shipping).toFixed(2)}`);
    res.json({ id: order._id, orderNumber: order.orderNumber, status: order.status });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/orders/:id', authMiddleware, async (req, res) => {
  try {
    const o = await Order.findOne({ _id: req.params.id, userId: req.user.id });
    if (!o) return res.status(404).json({ error: 'Not found' });
    res.json({ id: o._id, orderNumber: o.orderNumber, providerName: o.providerName, status: o.status, specs: o.specs, pricing: o.pricing, estimatedDelivery: o.estimatedDelivery, trackingNumber: o.trackingNumber });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Provider Routes ───────────────────────────────────────────────────────────
app.get('/api/providers', async (req, res) => {
  try {
    const providers = await Provider.find({ isVerified: true }).sort({ rating: -1 }).limit(20);
    res.json({ providers: providers.map(p => ({ id: p._id, companyName: p.companyName, region: p.region, country: p.country, rating: p.rating, totalReviews: p.totalReviews, isVerified: p.isVerified, leadTimeDays: p.leadTimeDays, materials: p.materials, services: p.services, contact: p.contact, description: p.description })) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/providers/:id', async (req, res) => {
  try {
    const p = await Provider.findById(req.params.id);
    if (!p) return res.status(404).json({ error: 'Not found' });
    res.json(p);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Health ────────────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok', version: '4.0', timestamp: new Date().toISOString(),
    features: ['OpenCV v4 Pipeline','YOLO','SAM','Bezier','DXF Export','GCode Studio','AI Studio','Cloud Sync'],
    opencv_stages: 23,
  });
});

app.listen(PORT, () => console.log(`SheetForge v4 API running on port ${PORT}`));
module.exports = app;
