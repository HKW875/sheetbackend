/**
 * SheetForge v4 — Express Backend
 * ================================
 * Fixed: Syntax error + Conversion Pipeline
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

// ── Config ────────────────────────────────────────────────────────────────────
const PORT       = process.env.PORT || 5000;
const MONGO_URI  = process.env.MONGO_URI  || 'mongodb://localhost:27017/sheetforge';
const JWT_SECRET = process.env.JWT_SECRET || 'sheetforge_secret_v4';
const PYTHON_CMD = process.env.PYTHON_CMD || 'python3';
const PROCESS_PY = path.join(__dirname, 'process.py');
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(UPLOAD_DIR, 'output');

[UPLOAD_DIR, OUTPUT_DIR].forEach(d => { 
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true }); 
});

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME || '',
  api_key:    process.env.CLOUDINARY_API_KEY    || '',
  api_secret: process.env.CLOUDINARY_API_SECRET || '',
});

// ── Mongoose Models ───────────────────────────────────────────────────────────
mongoose.connect(MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(e => console.error('MongoDB error:', e.message));

const userSchema = new mongoose.Schema({
  firstName: String, lastName: String, email: { type: String, unique: true },
  passwordHash: String, role: { type: String, enum: ['designer','provider'], default: 'designer' },
  company: String, createdAt: { type: Date, default: Date.now },
});
const User = mongoose.model('User', userSchema);

const designSchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId, 
  partName: String, 
  material: String, 
  thickness: Number,
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
  specs: { quantity: Number, material: String, finish: String },
  notes: String, status: { type: String, default: 'open' },
  bids: [{ providerId: String, providerName: String, price: Number, perUnit: Number, leadDays: Number }],
  createdAt: { type: Date, default: Date.now },
});
const Quote = mongoose.model('Quote', quoteSchema);

const orderSchema = new mongoose.Schema({
  orderNumber: String, quoteId: String, designId: String,
  userId: mongoose.Types.ObjectId, providerId: String, providerName: String,
  status: { type: String, default: 'confirmed' },
  specs: mongoose.Schema.Types.Mixed,
  pricing: { unitPrice: Number, quantity: Number, subtotal: Number, shipping: Number, total: Number },
  estimatedDelivery: Date,
  createdAt: { type: Date, default: Date.now },
});
const Order = mongoose.model('Order', orderSchema);

const activitySchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId,
  type: String,
  title: String, 
  description: String,
  timestamp: { type: Date, default: Date.now },
});
const Activity = mongoose.model('Activity', activitySchema);

const providerSchema = new mongoose.Schema({
  userId: mongoose.Types.ObjectId, 
  companyName: String, 
  region: String, 
  country: String,
  rating: { type: Number, default: 0 },
  isVerified: { type: Boolean, default: false },
  contact: { email: String },
  createdAt: { type: Date, default: Date.now },
});
const Provider = mongoose.model('Provider', providerSchema);

// ── App ───────────────────────────────────────────────────────────────────────
const app = express();
app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
app.use(cors({ origin: '*', methods: ['*'], allowedHeaders: ['*'] }));
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

// ── Auth Middleware ───────────────────────────────────────────────────────────
function authMiddleware(req, res, next) {
  const token = (req.headers.authorization || '').replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'No token' });
  try { 
    req.user = jwt.verify(token, JWT_SECRET); 
    next(); 
  } catch { 
    res.status(401).json({ error: 'Invalid token' }); 
  }
}

function logActivity(userId, type, title, description) {
  Activity.create({ userId, type, title, description }).catch(() => {});
}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function uploadToCloudinary(filePath, folder = 'sheetforge', resourceType = 'raw') {
  return new Promise((resolve, reject) => {
    cloudinary.uploader.upload(filePath, { folder, resource_type: resourceType }, (err, result) => {
      if (err) reject(err); else resolve(result);
    });
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
    res.json(user);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Conversion Pipeline (Fixed) ───────────────────────────────────────────────
app.post('/api/convert/:id', authMiddleware, async (req, res) => {
  try {
    const designId = req.params.id;
    const d = await Design.findOne({ _id: designId, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Design not found' });

    let localImagePath = null;
    if (d.originalFile?.localPath && fs.existsSync(d.originalFile.localPath)) {
      localImagePath = d.originalFile.localPath;
    } else if (d.originalFile?.url?.startsWith('/uploads/')) {
      const urlBased = path.join(__dirname, d.originalFile.url);
      if (fs.existsSync(urlBased)) localImagePath = urlBased;
    }

    if (!localImagePath) {
      try {
        const files = fs.readdirSync(UPLOAD_DIR);
        const designIdShort = d._id.toString().slice(-8);
        const match = files.find(f => f.includes(designIdShort));
        if (match) localImagePath = path.join(UPLOAD_DIR, match);
      } catch (e) {}
    }

    if (!localImagePath || !fs.existsSync(localImagePath)) {
      return res.status(404).json({ error: 'Source file not found' });
    }

    d.status = 'analyzing';
    await d.save();

    const opts = { thickness: d.thickness || 2.0 };

    const py = spawn(PYTHON_CMD, [PROCESS_PY, localImagePath, JSON.stringify(opts)], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    let output = '';
    py.stdout.on('data', chunk => output += chunk.toString());
    py.stderr.on('data', chunk => console.error('[Python]', chunk.toString().slice(0, 300)));

    py.on('close', async (code) => {
      if (code !== 0) {
        d.status = 'error';
        await d.save();
        return;
      }
      try {
        const lastLine = output.trim().split('\n').pop();
        const result = JSON.parse(lastLine);

        d.aiAnalysis = result.analysis;
        d.dwg = result.dwg || {};
        d.conversionLog = result.steps || [];
        d.status = 'ready';
        d.updatedAt = new Date();
        await d.save();

        logActivity(req.user.id, 'convert', `Converted: ${d.partName}`, 'Pipeline completed successfully');
      } catch (e) {
        console.error('Parse error:', e);
        d.status = 'error';
        await d.save();
      }
    });

    res.json({ id: designId, status: 'analyzing', message: 'Pipeline started' });
  } catch (e) {
    console.error('Convert error:', e);
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/convert/:id/status', authMiddleware, async (req, res) => {
  try {
    const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
    if (!d) return res.status(404).json({ error: 'Not found' });

    const steps = (d.conversionLog || []).map(s => ({ name: s.name || s, status: 'done' }));
    const pct = Math.round((steps.length / 23) * 100);

    res.json({ 
      status: d.status, 
      progress: pct, 
      steps, 
      currentStep: steps[steps.length - 1]?.name || 'Processing...' 
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Other Routes (Dashboard, Designs, etc.) ───────────────────────────────────
app.get('/api/designs', authMiddleware, async (req, res) => {
  const designs = await Design.find({ userId: req.user.id }).sort({ createdAt: -1 });
  res.json({ designs });
});

app.post('/api/designs', authMiddleware, upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'File required' });
  try {
    const { partName, material, thickness } = req.body;
    const newPath = `${req.file.path}${path.extname(req.file.originalname)}`;
    await fsp.rename(req.file.path, newPath);

    const design = await Design.create({
      userId: req.user.id,
      partName: partName || req.file.originalname.replace(/\.[^.]+$/, ''),
      material,
      thickness: thickness ? parseFloat(thickness) : null,
      originalFile: {
        filename: req.file.originalname,
        url: `/uploads/${path.basename(newPath)}`,
        localPath: newPath,
        mimetype: req.file.mimetype,
        size: req.file.size
      }
    });

    logActivity(req.user.id, 'upload', `Uploaded: ${design.partName}`, req.file.originalname);
    res.json({ id: design._id, status: design.status });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/designs/:id', authMiddleware, async (req, res) => {
  const d = await Design.findOne({ _id: req.params.id, userId: req.user.id });
  if (!d) return res.status(404).json({ error: 'Not found' });
  res.json(d);
});

// Add other routes as needed...

app.get('/api/health', (req, res) => res.json({ status: 'ok', version: '4.0' }));

app.listen(PORT, () => console.log(`✅ SheetForge v4 API running on port ${PORT}`));
