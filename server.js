/**
 * server.js — CNC Sketch Platform Backend
 * Handles: REST API, WebSocket (chat + WebRTC signaling), CNC providers,
 * proxying to Python microservice.
 */

require("dotenv").config();
require('socket.io');
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const { MongoClient, ObjectId } = require("mongodb");
const FormData = require("form-data");
const fetch = (...args) => import("node-fetch").then(({ default: f }) => f(...args));

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] },
});

// ─── Config ────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE = process.env.PYTHON_SERVICE_URL || "http://localhost:5001";
const MONGO_URI = process.env.MONGO_URI;
mongoose.connect(MONGO_URI);
const DB_NAME = "cnc_sketch_db";

let db, providersCol, chatsCol, scansCol;

// ─── Middleware ─────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));
app.use("/index.html", express.static(path.join(__dirname, "index.html")));

const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    cb(null, allowed.includes(file.mimetype));
  },
});

// ─── MongoDB Init ──────────────────────────────────────────────────────────
async function initDB() {
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  db = client.db(DB_NAME);
  providersCol = db.collection("providers");
  chatsCol = db.collection("chats");
  scansCol = db.collection("scans");

  // Seed sample CNC providers if empty
  const count = await providersCol.countDocuments();
  if (count === 0) {
    await providersCol.insertMany(SEED_PROVIDERS);
    console.log("✅ Seeded CNC providers");
  }
  console.log("✅ MongoDB connected");
}

// ─── Seed Data ─────────────────────────────────────────────────────────────
const SEED_PROVIDERS = [
  {
    name: "PrecisionCut Technologies",
    country: "Germany",
    city: "Stuttgart",
    services: ["Laser Cutting", "Plasma Cutting", "Bending"],
    materials: ["Mild Steel", "Stainless Steel", "Aluminium"],
    email: "info@precisioncut.de",
    phone: "+49 711 123456",
    website: "https://precisioncut.de",
    rating: 4.8,
    reviewCount: 134,
    verified: true,
    online: true,
    avatar: "https://ui-avatars.com/api/?name=Precision+Cut&background=1a2744&color=00d4ff&size=64",
    bio: "ISO certified sheet metal CNC specialists since 1998. Tolerances down to ±0.05mm.",
    location: { lat: 48.7758, lng: 9.1829 },
  },
  {
    name: "MetalWorks USA",
    country: "United States",
    city: "Detroit, MI",
    services: ["Waterjet", "Laser Cutting", "Punching"],
    materials: ["Carbon Steel", "Copper", "Titanium"],
    email: "orders@metalworksusa.com",
    phone: "+1 313 555 0192",
    website: "https://metalworksusa.com",
    rating: 4.6,
    reviewCount: 89,
    verified: true,
    online: false,
    avatar: "https://ui-avatars.com/api/?name=MetalWorks+USA&background=1a2744&color=ff6b35&size=64",
    bio: "High-volume sheet metal fabrication for automotive and aerospace sectors.",
    location: { lat: 42.3314, lng: -83.0458 },
  },
  {
    name: "SteelCraft Asia",
    country: "Japan",
    city: "Osaka",
    services: ["Laser Cutting", "CNC Milling", "Welding"],
    materials: ["Mild Steel", "Stainless Steel", "Aluminium", "Brass"],
    email: "contact@steelcraft.jp",
    phone: "+81 6 1234 5678",
    website: "https://steelcraft.jp",
    rating: 4.9,
    reviewCount: 212,
    verified: true,
    online: true,
    avatar: "https://ui-avatars.com/api/?name=SteelCraft+Asia&background=1a2744&color=00ff9f&size=64",
    bio: "Precision sheet metal with Japanese quality standards. 48h turnaround.",
    location: { lat: 34.6937, lng: 135.5023 },
  },
  {
    name: "AfriCNC Solutions",
    country: "Kenya",
    city: "Nairobi",
    services: ["Plasma Cutting", "Laser Cutting", "Metal Fabrication"],
    materials: ["Mild Steel", "Aluminium", "Galvanized Steel"],
    email: "info@africnc.co.ke",
    phone: "+254 700 123456",
    website: "https://africnc.co.ke",
    rating: 4.5,
    reviewCount: 47,
    verified: true,
    online: true,
    avatar: "https://ui-avatars.com/api/?name=AfriCNC&background=1a2744&color=ffd700&size=64",
    bio: "Leading CNC sheet metal fabrication in East Africa. Fast local delivery.",
    location: { lat: -1.2921, lng: 36.8219 },
  },
  {
    name: "EuroCut Precision",
    country: "United Kingdom",
    city: "Birmingham",
    services: ["Laser Cutting", "Tube Laser", "Bending", "Welding"],
    materials: ["Mild Steel", "Stainless Steel", "Aluminium"],
    email: "hello@eurocut.co.uk",
    phone: "+44 121 456 7890",
    website: "https://eurocut.co.uk",
    rating: 4.7,
    reviewCount: 163,
    verified: true,
    online: false,
    avatar: "https://ui-avatars.com/api/?name=EuroCut&background=1a2744&color=e0e0ff&size=64",
    bio: "Full service sheet metal fabrication, prototyping to production runs.",
    location: { lat: 52.4862, lng: -1.8904 },
  },
  {
    name: "IndiaFab Pro",
    country: "India",
    city: "Pune",
    services: ["Laser Cutting", "Plasma Cutting", "Sheet Metal Forming"],
    materials: ["Mild Steel", "Stainless Steel", "Galvanized", "Aluminium"],
    email: "orders@indiafab.in",
    phone: "+91 20 1234 5678",
    website: "https://indiafab.in",
    rating: 4.4,
    reviewCount: 98,
    verified: false,
    online: true,
    avatar: "https://ui-avatars.com/api/?name=IndiaFab+Pro&background=1a2744&color=ff9933&size=64",
    bio: "Cost-effective precision sheet metal manufacturing. Export ready.",
    location: { lat: 18.5204, lng: 73.8567 },
  },
];

// ─── Sketch Processing Routes ──────────────────────────────────────────────
app.post("/api/process-sketch", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const form = new FormData();
    form.append("image", req.file.buffer, {
      filename: req.file.originalname || "sketch.png",
      contentType: req.file.mimetype,
    });

    const pyRes = await fetch(`${PYTHON_SERVICE}/process`, {
      method: "POST",
      body: form,
      headers: form.getHeaders(),
    });

    if (!pyRes.ok) {
      const err = await pyRes.text();
      return res.status(500).json({ error: "Processing failed", detail: err });
    }

    const data = await pyRes.json();
    res.json(data);
  } catch (err) {
    console.error("process-sketch error:", err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/generate-dxf", async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_SERVICE}/generate-dxf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.setHeader("Content-Type", "application/dxf");
    res.setHeader("Content-Disposition", "attachment; filename=drawing.dxf");
    pyRes.body.pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/generate-pdf", async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_SERVICE}/generate-pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", "inline; filename=drawing.pdf");
    pyRes.body.pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/generate-gcode", async (req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_SERVICE}/generate-gcode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.setHeader("Content-Type", "text/plain");
    res.setHeader("Content-Disposition", "attachment; filename=output.nc");
    pyRes.body.pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Provider Routes ───────────────────────────────────────────────────────
app.get("/api/providers", async (req, res) => {
  try {
    const { country, service, search } = req.query;
    const filter = {};
    if (country && country !== "all") filter.country = country;
    if (service) filter.services = { $in: [service] };
    if (search) {
      filter.$or = [
        { name: { $regex: search, $options: "i" } },
        { city: { $regex: search, $options: "i" } },
        { country: { $regex: search, $options: "i" } },
      ];
    }
    const providers = await providersCol.find(filter).toArray();
    res.json(providers.map((p) => ({ ...p, _id: p._id.toString() })));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/providers", async (req, res) => {
  try {
    const provider = {
      ...req.body,
      rating: 0,
      reviewCount: 0,
      verified: false,
      online: true,
      createdAt: new Date(),
      avatar: `https://ui-avatars.com/api/?name=${encodeURIComponent(req.body.name)}&background=1a2744&color=00d4ff&size=64`,
    };
    const result = await providersCol.insertOne(provider);
    res.json({ ...provider, _id: result.insertedId.toString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Scans History ─────────────────────────────────────────────────────────
app.get("/api/scans", async (req, res) => {
  try {
    const scans = await scansCol.find({}).sort({ created_at: -1 }).limit(20).toArray();
    res.json(scans.map((s) => ({ ...s, _id: s._id.toString() })));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Chat History ──────────────────────────────────────────────────────────
app.get("/api/chat/:roomId", async (req, res) => {
  try {
    const messages = await chatsCol
      .find({ roomId: req.params.roomId })
      .sort({ timestamp: 1 })
      .limit(100)
      .toArray();
    res.json(messages.map((m) => ({ ...m, _id: m._id.toString() })));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── WebSocket: Chat + WebRTC Signaling ───────────────────────────────────
const rooms = new Map(); // roomId -> Set of socket ids
const userNames = new Map(); // socketId -> name

io.on("connection", (socket) => {
  console.log("Socket connected:", socket.id);

  socket.on("set-username", (name) => {
    userNames.set(socket.id, name || `User_${socket.id.slice(0, 5)}`);
  });

  // ── Chat ──────────────────────────────────────────────────────────────
  socket.on("join-chat", async ({ roomId, username }) => {
    socket.join(roomId);
    userNames.set(socket.id, username || userNames.get(socket.id));

    if (!rooms.has(roomId)) rooms.set(roomId, new Set());
    rooms.get(roomId).add(socket.id);

    io.to(roomId).emit("user-joined", {
      username: userNames.get(socket.id),
      participants: rooms.get(roomId).size,
    });
  });

  socket.on("chat-message", async ({ roomId, message, username }) => {
    const msg = {
      roomId,
      username: username || userNames.get(socket.id) || "Anonymous",
      message,
      timestamp: new Date(),
    };
    try {
      await chatsCol.insertOne(msg);
    } catch (_) {}

    io.to(roomId).emit("chat-message", {
      ...msg,
      timestamp: msg.timestamp.toISOString(),
    });
  });

  // ── WebRTC Signaling ──────────────────────────────────────────────────
  socket.on("join-video-room", ({ roomId, username }) => {
    socket.join(`video:${roomId}`);
    socket.to(`video:${roomId}`).emit("peer-joined", {
      socketId: socket.id,
      username: username || userNames.get(socket.id),
    });
  });

  socket.on("webrtc-offer", ({ roomId, to, offer }) => {
    io.to(to).emit("webrtc-offer", { from: socket.id, offer });
  });

  socket.on("webrtc-answer", ({ to, answer }) => {
    io.to(to).emit("webrtc-answer", { from: socket.id, answer });
  });

  socket.on("webrtc-ice-candidate", ({ to, candidate }) => {
    io.to(to).emit("webrtc-ice-candidate", { from: socket.id, candidate });
  });

  socket.on("leave-video-room", ({ roomId }) => {
    socket.to(`video:${roomId}`).emit("peer-left", { socketId: socket.id });
    socket.leave(`video:${roomId}`);
  });

  // ── Cleanup ───────────────────────────────────────────────────────────
  socket.on("disconnecting", () => {
    for (const roomId of socket.rooms) {
      if (rooms.has(roomId)) {
        rooms.get(roomId).delete(socket.id);
        io.to(roomId).emit("user-left", {
          username: userNames.get(socket.id),
          participants: rooms.get(roomId).size,
        });
      }
      socket.to(`video:${roomId}`).emit("peer-left", { socketId: socket.id });
    }
    userNames.delete(socket.id);
  });
});

// ─── Default route ─────────────────────────────────────────────────────────
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ─── Start ─────────────────────────────────────────────────────────────────
initDB()
  .then(() => {
    server.listen(PORT, () => {
      console.log(`🚀 CNC Sketch Platform running at http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("❌ Failed to connect to MongoDB:", err.message);
    process.exit(1);
  });
