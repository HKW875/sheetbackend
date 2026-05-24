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

const app  = express();
const PORT = process.env.PORT || 5000;

// ================================================================
// MIDDLEWARE
// ================================================================
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: process.env.CLIENT_URL || 'https://sheetfg.hkw875.workers.dev/', credentials: true }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(morgan('dev'));

// Serve the main HTML SPA
app.get('/', (req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(INDEX_HTML);
});

// ================================================================
// DATABASE
// ================================================================
mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/sheetforge', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(async () => {
  console.log('✅  MongoDB connected');
  await seedProviders();
}).catch(err => console.error('❌  MongoDB error:', err.message));

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
    width        : Number,
    height       : Number,
    rawText      : String,
    confidence   : Number,
    completedAt  : Date,
  },
  dwg         : {
    filename   : String,
    path       : String,
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
const runAIAnalysis = async (filePath, mimetype) => {
  // In production: call OpenCV / Anthropic Vision API / ODA DWG SDK
  await new Promise(r => setTimeout(r, 1200));
  return {
    edges       : Math.floor(Math.random() * 60) + 20,
    bendLines   : Math.floor(Math.random() * 6) + 2,
    holes       : Math.floor(Math.random() * 10) + 2,
    width       : Math.round((Math.random() * 400 + 100) * 10) / 10,
    height      : Math.round((Math.random() * 300 + 80) * 10) / 10,
    rawText     : 'Rectangular profile with flanges detected. Edge count: 48 entities.',
    confidence  : Math.round((Math.random() * 15 + 84) * 10) / 10,
    completedAt : new Date(),
  };
};

// Generates a minimal valid DWG-like export (PNG render of the geometry)
const generateDWGPreview = async (analysis, material, thickness) => {
  // In production: use ODA Drawings SDK, LibreDWG, or DXFLib
  // Here we generate a PNG placeholder that represents the DWG
  const w = 1200, h = 800;
  // Sharp creates a white background PNG
  const svg = `
  <svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg" font-family="monospace">
    <rect width="${w}" height="${h}" fill="white"/>
    <!-- Grid -->
    ${Array.from({ length: 40 }, (_, i) =>
      `<line x1="${i * 30}" y1="0" x2="${i * 30}" y2="${h}" stroke="#e0e8f0" stroke-width="0.5"/>`
    ).join('')}
    ${Array.from({ length: 27 }, (_, i) =>
      `<line x1="0" y1="${i * 30}" x2="${w}" y2="${i * 30}" stroke="#e0e8f0" stroke-width="0.5"/>`
    ).join('')}
    <!-- Part outline -->
    <rect x="200" y="150" width="${analysis.width * 1.8}" height="${analysis.height * 1.8}"
      fill="none" stroke="#1a2a3a" stroke-width="2"/>
    <!-- Bend lines -->
    <line x1="200" y1="${150 + analysis.height * 0.6}" x2="${200 + analysis.width * 1.8}"
      y2="${150 + analysis.height * 0.6}" stroke="#e67700" stroke-width="1.2"
      stroke-dasharray="8,5"/>
    <!-- Holes -->
    ${Array.from({ length: Math.min(analysis.holes, 6) }, (_, i) =>
      `<circle cx="${250 + i * 80}" cy="${150 + analysis.height * 0.9}"
        r="12" fill="none" stroke="#1a2a3a" stroke-width="1.5"/>`
    ).join('')}
    <!-- Dimension lines -->
    <line x1="200" y1="${150 + analysis.height * 1.8 + 30}"
          x2="${200 + analysis.width * 1.8}" y2="${150 + analysis.height * 1.8 + 30}"
          stroke="#4a6480" stroke-width="1"/>
    <text x="${200 + analysis.width * 0.9}" y="${150 + analysis.height * 1.8 + 48}"
          fill="#2a4060" font-size="14" text-anchor="middle">${analysis.width}mm</text>
    <!-- Title block -->
    <rect x="0" y="${h - 60}" width="${w}" height="60" fill="#f8fafc" stroke="#c0c8d4" stroke-width="1"/>
    <text x="20" y="${h - 35}" fill="#2a4060" font-size="13" font-weight="bold">
      SHEETFORGE AUTO-DWG  |  ${material || 'MILD STEEL'} ${thickness || 2}mm  |  ISO 2768-m  |  1:1
    </text>
    <text x="20" y="${h - 15}" fill="#6a8090" font-size="11">
      Generated: ${new Date().toISOString()}  |  Confidence: ${analysis.confidence}%  |  Entities: ${analysis.edges}
    </text>
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

    // Step 2: AI Analysis
    const analysis = await runAIAnalysis(design.originalFile.path, design.originalFile.mimetype);
    design.aiAnalysis = analysis;
    design.status = 'converting';
    await design.save();

    // Step 3: Generate DWG preview PNG
    const previewPath = await generateDWGPreview(analysis, design.material, design.thickness);
    design.dwg = {
      filename    : design.partName.replace(/\s+/g, '_') + '.dwg',
      path        : previewPath,
      entities    : analysis.edges,
      fileSize    : fs.statSync(previewPath).size,
      generatedAt : new Date(),
    };
    design.status = 'ready';
    await design.save();

    res.json({
      design,
      analysis,
      message: 'DWG conversion complete',
    });
  } catch (err) {
    if (design) { design.status = 'uploaded'; await design.save().catch(() => {}); }
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
