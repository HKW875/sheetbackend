/**
 * SheetForge — Backend Server
 * Node.js + Express + MongoDB + Cloudinary
 * =========================================
 * Routes:
 *   Auth       POST /api/auth/register, /api/auth/login, /api/auth/logout
 *   Designs    POST /api/designs/upload, GET /api/designs, GET /api/designs/:id
 *              PATCH /api/designs/:id/approve, DELETE /api/designs/:id
 *   Convert    POST /api/convert/:id  (triggers AI-assisted DWG generation via process.py)
 *   Cloud      POST /api/cloud/save/:id  (Cloudinary upload + MongoDB link)
 *   Quotes     POST /api/quotes, GET /api/quotes, GET /api/quotes/:id
 *   Providers  GET /api/providers, GET /api/providers/:id
 *   Orders     POST /api/orders, GET /api/orders, PATCH /api/orders/:id
 *   Progress   GET /api/convert/:id/progress  (SSE real-time progress stream)
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
const path          = require('path');
const fs            = require('fs');
const { promisify } = require('util');
const { spawn, exec } = require('child_process');

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
    if (!origin) return callback(null, true);
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
const outputDir = path.join(__dirname, 'uploads', 'output');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

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
    path       : String,       // preview PNG/SVG path
    dxfPath    : String,       // real .DXF file path
    gcodeFile  : String,       // G-code path from process.py
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
// Inside server.js -> const protect = async (req, res, next) => { ... }
const protect = async (req, res, next) => {
  let token;

  // 1. Check standard Authorization Header
  if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
    token = req.headers.authorization.split(' ')[1];
  } 
  // 2. FALLBACK FIX: Check query parameters for EventSource (SSE) compatibility
  else if (req.query && req.query.token) {
    token = req.query.token;
  }

  if (!token) {
    return res.status(401).json({ error: 'Not authorized, token missing' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-fallback-secret');
    req.user = await User.findById(decoded.id).select('-password');
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Not authorized, token invalid' });
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

// ================================================================
// PYTHON PIPELINE — calls process.py via child_process
// ================================================================

/**
 * Run process.py via child_process.spawn, stream stdout for progress,
 * and return the final parsed JSON result.
 * @param {string} imagePath   - path to the uploaded file
 * @param {object} opts        - JSON options forwarded to process.py as argv[2]
 * @param {Function} onStep    - optional callback(stepName, detail) for real-time progress
 */
function runProcessPy(imagePath, opts = {}, onStep = null) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'process.py');
    if (!fs.existsSync(scriptPath)) {
      return reject(new Error('process.py not found next to server.js'));
    }

    const args    = [scriptPath, imagePath, JSON.stringify(opts)];
    const py      = spawn('python', args, {
      env: {
        ...process.env,
        GEMINI_API_KEY: process.env.GEMINI_API_KEY || '',
        // Keep PATH so python can find system libs
        PATH: process.env.PATH,
      },
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      const line = chunk.toString().trim();
      stderr += line + '\n';
      // Surface step names from stderr to progress callback
      if (onStep && line) {
        onStep('progress', line);
      }
      console.log('[process.py stderr]', line);
    });

    py.on('close', (code) => {
      if (code !== 0 && !stdout.trim()) {
        return reject(new Error(`process.py exited ${code}: ${stderr.slice(0, 500)}`));
      }
      try {
        // Find the last complete JSON object in stdout
        const lastBrace = stdout.lastIndexOf('}');
        const firstBrace = stdout.indexOf('{');
        if (firstBrace === -1 || lastBrace === -1) {
          return reject(new Error('process.py produced no JSON output. stderr: ' + stderr.slice(0, 300)));
        }
        const jsonStr = stdout.slice(firstBrace, lastBrace + 1);
        const result  = JSON.parse(jsonStr);
        resolve(result);
      } catch (parseErr) {
        reject(new Error(`process.py JSON parse error: ${parseErr.message}. stdout: ${stdout.slice(0, 200)}`));
      }
    });

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn python3: ${err.message}`));
    });
  });
}

/**
 * Run a quick exec-based call to process.py (for non-streaming tasks like
 * AI DXF interaction or small utility calls).
 */
function execProcessPy(imagePath, opts = {}) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'process.py');
    const optsJson   = JSON.stringify(opts).replace(/'/g, "'\\''");
    const cmd        = `python "${scriptPath}" "${imagePath}" '${optsJson}'`;

    exec(cmd, {
      maxBuffer: 50 * 1024 * 1024,  // 50 MB stdout buffer
      env: {
        ...process.env,
        GEMINI_API_KEY: process.env.GEMINI_API_KEY || '',
        PATH: process.env.PATH,
      },
    }, (err, stdout, stderr) => {
      if (err && !stdout.trim()) {
        return reject(new Error(`exec process.py failed: ${err.message}. stderr: ${stderr.slice(0, 300)}`));
      }
      try {
        const lastBrace  = stdout.lastIndexOf('}');
        const firstBrace = stdout.indexOf('{');
        if (firstBrace === -1 || lastBrace === -1) {
          return reject(new Error('process.py produced no JSON. stderr: ' + stderr.slice(0, 300)));
        }
        resolve(JSON.parse(stdout.slice(firstBrace, lastBrace + 1)));
      } catch (e) {
        reject(new Error(`process.py JSON parse error: ${e.message}`));
      }
    });
  });
}

// ================================================================
// SSE PROGRESS — in-memory store for active conversion jobs
// ================================================================

// ================================================================
// SSE PROGRESS STREAM — MATCHING FRONTEND DEMAND
// ================================================================
app.get('/api/convert/:id/progress', async (req, res) => {
  const designId = req.params.id;

  // Establish SSE Header configs
  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
  });
  res.write('\n');

  // Register this response handle into your global progress registry
  if (!global.progressClients) {
    global.progressClients = {};
  }
  if (!global.progressClients[designId]) {
    global.progressClients[designId] = [];
  }
  
  global.progressClients[designId].push(res);
  console.log(`[SSE Connected] Progress listener attached for design: ${designId}`);

  // Keep connection alive with a periodic heartbeat ping
  const heartbeat = setInterval(() => {
    res.write(':\n\n');
  }, 15000);

  // Clean up if user closes page or disconnects browser tab
  req.on('close', () => {
    clearInterval(heartbeat);
    if (global.progressClients[designId]) {
      global.progressClients[designId] = global.progressClients[designId].filter(client => client !== res);
    }
    console.log(`[SSE Disconnected] Progress listener removed for design: ${designId}`);
  });
});

const activeJobs = new Map(); // designId → { clients: Set<res>, steps: Array }

function pushProgress(designId, event, data) {
  const job = activeJobs.get(designId);
  if (!job) return;
  const payload = `data: ${JSON.stringify({ event, ...data })}\n\n`;
  for (const client of job.clients) {
    try { client.write(payload); } catch (_) {}
  }
  job.steps.push({ event, ...data, ts: Date.now() });
}


// Add this route right near your other conversion endpoints in server.js
app.get('/api/convert/:id/status', protect, async (req, res) => {
  try {
    const design = await Design.findById(req.params.id);
    if (!design) return res.status(404).json({ error: 'Design not found' });
    
    // Return a standard JSON payload that Cloudflare handles perfectly
    res.json({
      status: design.status, // e.g., 'analyzing', 'completed', 'failed'
      progress: global.currentProgressMessage?.[req.params.id] || "Processing layout..."
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

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

    const user  = await User.create({ firstName, lastName, email, password, role, company, country });
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
app.post('/api/auth/logout', (req, res) => res.json({ message: 'Logged out successfully' }));

// ================================================================
// ROUTES — DESIGNS
// ================================================================

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

app.get('/api/designs/:id', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    res.json({ design });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

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

app.delete('/api/designs/:id', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    if (design.cloudinary?.publicId) {
      await cloudinary.uploader.destroy(design.cloudinary.publicId).catch(() => {});
    }
    await cleanupFile(design.originalFile?.path);
    await cleanupFile(design.dwg?.path);
    await design.deleteOne();
    res.json({ message: 'Design deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ================================================================
// ROUTES — SSE PROGRESS STREAM
// ================================================================

/**
 * GET /api/convert/:id/progress
 * Server-Sent Events stream for real-time conversion progress.
 * The frontend EventSource connects here before calling POST /api/convert/:id.
 */
app.get('/api/convert/:id/progress', protect, (req, res) => {
  const designId = req.params.id;

  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');  // disable nginx buffering
  res.flushHeaders();

  // Register client
  if (!activeJobs.has(designId)) {
    activeJobs.set(designId, { clients: new Set(), steps: [] });
  }
  const job = activeJobs.get(designId);
  job.clients.add(res);

  // Send any buffered steps already recorded
  for (const step of job.steps) {
    res.write(`data: ${JSON.stringify(step)}\n\n`);
  }

  // Heartbeat every 15s so the connection stays alive through firewalls
  const heartbeat = setInterval(() => {
    try { res.write(': heartbeat\n\n'); } catch (_) {}
  }, 15000);

  req.on('close', () => {
    clearInterval(heartbeat);
    job.clients.delete(res);
    if (job.clients.size === 0) {
      // Keep steps buffer for late pollers but remove after 5 min
      setTimeout(() => activeJobs.delete(designId), 5 * 60 * 1000);
    }
  });
});

// ================================================================
// ROUTES — AI CONVERSION (orchestrates process.py)
// ================================================================

/**
 * POST /api/convert/:id
 * Full pipeline:
 *   1. Mark design as analyzing
 *   2. Call process.py via child_process.spawn (streams progress via SSE)
 *   3. Parse JSON output — analysis, DXF path, G-code
 *   4. Save results to MongoDB
 */
// ================================================================
// ROUTES — AI CONVERSION (orchestrates process.py)
// ================================================================
app.post('/api/convert/:id', protect, async (req, res) => {
  let design;
  try {
    design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });
    if (!design.originalFile?.path) return res.status(400).json({ error: 'No file attached to design' });

    // ─── FIX A: SANITIZE AND FORCE FULL ABSOLUTE RESOLUTION ───
    const filePath = path.resolve(design.originalFile.path); 
    
    // Debug log to terminal to verify the disk layout
    console.log(`[SheetForge Debug] Passing absolute image path to Python: ${filePath}`);

    if (!fs.existsSync(filePath)) {
      return res.status(400).json({ error: `Uploaded file not found on disk at: ${filePath}` });
    }

    // Step 1: Mark as analyzing
    design.status = 'analyzing';
    await design.save();
    pushProgress(design._id.toString(), 'status', { message: 'Starting OpenCV pipeline…', phase: 'preprocessing' });

    const mat   = design.material  || 'Mild Steel';
    const thick = design.thickness || 2;

    const opts = {
      material  : mat,
      thickness : thick,
      dpi       : 96,
      gcodeOptions: {
        feedRate    : 1000,
        plungeRate  : 300,
        spindleRpm  : 12000,
        cutDepth    : thick,
        passDepth   : thick / 2,
        safeZ       : 5,
        toolDiameter: 3,
        operation   : 'cut',
      },
    };

    // Step 2: Call process.py via spawn — stream steps to SSE clients
    // FIXED: Using the fully resolved absolute 'filePath' here safely
    const pyResult = await runProcessPy(filePath, opts, (event, message) => {
      pushProgress(design._id.toString(), event, { message });
    });

    if (pyResult.error) {
      throw new Error(`process.py reported error: ${pyResult.error}`);
    }

    // Step 3: Parse process.py output
    const analysis     = pyResult.analysis  || {};
    const dwgInfo      = pyResult.dwg       || {};
    const gcodeStr     = pyResult.gcode     || '';
    const svgContent   = pyResult.svgContent || '';

    design.status = 'converting';
    pushProgress(design._id.toString(), 'status', { message: 'Saving DXF and G-code outputs…', phase: 'saving' });
    await design.save();

    // Resolve DXF path — use what process.py wrote
    const dxfPath = dwgInfo.localPath || path.join(outputDir, dwgInfo.filename || '');
    const dxfExists = dxfPath && fs.existsSync(dxfPath);

    // Write G-code if process.py returned it as a string
    let gcodeFilePath = '';
    if (gcodeStr) {
      const gcFname    = `${design.partName.replace(/\s+/g,'_')}_${Date.now()}.nc`;
      gcodeFilePath    = path.join(outputDir, gcFname);
      fs.writeFileSync(gcodeFilePath, gcodeStr, 'utf8');
    } else if (dwgInfo.gcodeFilename) {
      gcodeFilePath = path.join(outputDir, dwgInfo.gcodeFilename);
    }

    // Write SVG preview if returned
    let previewPath = '';
    if (svgContent) {
      const svgFname = `preview_${Date.now()}.svg`;
      previewPath    = path.join(outputDir, svgFname);
      fs.writeFileSync(previewPath, svgContent, 'utf8');
    }

    // Step 4: Persist results
    design.aiAnalysis = {
      edges        : analysis.edges        || 0,
      bendLines    : analysis.bendLines    || 0,
      holes        : analysis.holes        || 0,
      holesDiameter: analysis.holesDiameter|| 0,
      slots        : analysis.slots        || 0,
      cutouts      : analysis.cutouts      || 0,
      width        : analysis.width        || 0,
      height       : analysis.height       || 0,
      thickness    : analysis.thickness    || thick,
      profileType  : analysis.profileType  || 'sheet metal',
      tolerance    : analysis.tolerance    || '±0.1mm',
      confidence   : analysis.confidence   || 0,
      rawText      : analysis.rawText      || '',
      notes        : analysis.notes        || '',
      completedAt  : new Date(),
    };

    design.dwg = {
      filename    : dwgInfo.filename   || '',
      path        : previewPath        || dwgInfo.svgFilename || '',
      dxfPath     : dxfExists ? dxfPath : '',
      gcodeFile   : gcodeFilePath,
      entities    : dwgInfo.entities   || 0,
      fileSize    : dwgInfo.fileSize   || 0,
      generatedAt : new Date(),
    };

    design.status = 'ready';
    await design.save();

    pushProgress(design._id.toString(), 'complete', {
      message    : 'Conversion complete',
      dxfReady   : dxfExists,
      gcodeReady : !!gcodeFilePath,
    });

    res.json({
      design,
      analysis  : design.aiAnalysis,
      message   : 'DXF + G-Code conversion complete via OpenCV/Gemini pipeline',
      dxfReady  : dxfExists,
      gcodeReady: !!gcodeFilePath,
      steps     : pyResult.steps || [],
    });

  } catch (err) {
    console.error('❌ Conversion error:', err.message);
    if (design) {
      design.status = 'uploaded';
      await design.save().catch(() => {});
      pushProgress(design._id.toString(), 'error', { message: err.message });
    }
    res.status(500).json({ error: err.message });
  }
});

// ----------------------------------------------------------------
// AI DXF Interaction — delegate entirely to process.py
// ----------------------------------------------------------------
app.post('/api/convert/:id/interact', protect, async (req, res) => {
  try {
    const { instruction } = req.body;
    if (!instruction) return res.status(400).json({ error: 'instruction is required' });

    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design) return res.status(404).json({ error: 'Design not found' });

    const imagePath = design.originalFile?.path || '';

    const opts = {
      mode        : 'ai_interact',
      instruction,
      analysis    : design.aiAnalysis || {},
      dpi         : 96,
    };

    // Use exec for single-shot interactions
    const result = await execProcessPy(imagePath, opts);

    if (result.analysis) {
      design.aiAnalysis = { ...design.aiAnalysis.toObject?.() || design.aiAnalysis, ...result.analysis };
      await design.save();
    }

    res.json({
      explanation : result.explanation || 'Changes applied',
      analysis    : result.analysis,
      design,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ----------------------------------------------------------------
// Get conversion status (polling fallback)
// ----------------------------------------------------------------
app.get('/api/convert/:id/status', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id }).lean();
    if (!design) return res.status(404).json({ error: 'Not found' });
    // Also return buffered SSE steps if still in activeJobs
    const job   = activeJobs.get(req.params.id);
    const steps = job ? job.steps : [];
    res.json({ status: design.status, aiAnalysis: design.aiAnalysis, dwg: design.dwg, steps });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ----------------------------------------------------------------
// Download .DXF file
// ----------------------------------------------------------------
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

// ----------------------------------------------------------------
// Download G-Code
// ----------------------------------------------------------------
app.get('/api/designs/:id/gcode', protect, async (req, res) => {
  try {
    const design = await Design.findOne({ _id: req.params.id, owner: req.user._id });
    if (!design?.dwg?.gcodeFile) return res.status(404).json({ error: 'G-Code not generated yet. Run conversion first.' });
    const gcPath = design.dwg.gcodeFile;
    if (!fs.existsSync(gcPath)) return res.status(404).json({ error: 'G-Code file missing on disk' });
    const fname = `${design.partName.replace(/\s+/g,'_')}.nc`;
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Content-Disposition', `attachment; filename="${fname}"`);
    res.sendFile(path.resolve(gcPath));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Serve DWG/SVG preview
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

    // Use DXF path or SVG preview — process.py handles both
    const assetPath = design.dwg?.dxfPath || design.dwg?.path;
    if (!assetPath || !fs.existsSync(assetPath))
      return res.status(400).json({ error: 'Processed design file not found' });

    const { assetName, folder, tags } = req.body;
    const publicId = `${folder || 'sheetforge/designs'}/${assetName || design.partName.replace(/\s+/g, '_')}_${Date.now()}`;

    const result = await cloudinary.uploader.upload(assetPath, {
      public_id      : publicId,
      resource_type  : 'raw',   // DXF/SVG are raw assets
      tags           : tags ? tags.split(',').map(t => t.trim()) : ['sheetforge', 'cad', 'dxf'],
      context        : `part=${design.partName}|material=${design.material || ''}|owner=${req.user.email}`,
    });

    design.cloudinary = {
      publicId   : result.public_id,
      url        : result.url,
      secureUrl  : result.secure_url,
      width      : result.width  || 0,
      height     : result.height || 0,
      bytes      : result.bytes,
      format     : result.format,
      uploadedAt : new Date(),
    };
    design.mongodb = { savedAt: new Date() };
    design.status  = 'saved';
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
  if (err instanceof multer.MulterError || err?.code?.startsWith('LIMIT_')) {
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
  console.log(`🤖  GEMINI_API_KEY: ${process.env.GEMINI_API_KEY ? 'Set ✓' : 'NOT SET — set GEMINI_API_KEY in .env'}`);
  console.log(`🐍  Python pipeline: process.py (spawn/exec via child_process)`);
});

module.exports = app;
