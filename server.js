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
app.use(cors({ origin: process.env.CLIENT_URL || 'http://localhost:3000', credentials: true }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(morgan('dev'));

// ================================================================
// EMBEDDED HTML (no separate index.html file needed)
// ================================================================
const INDEX_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SheetForge — CAD Conversion & CNC Quoting Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
  :root {
    --bg: #0a0c0f;
    --bg2: #0f1217;
    --bg3: #151a22;
    --bg4: #1c2330;
    --panel: #1a2132;
    --border: rgba(255,255,255,0.07);
    --border-med: rgba(255,255,255,0.12);
    --border-hi: rgba(255,255,255,0.2);
    --text: #e8eaed;
    --text-dim: #8892a4;
    --text-muted: #4a5568;
    --accent: #3d7eff;
    --accent-glow: rgba(61,126,255,0.15);
    --accent2: #00d4a0;
    --accent2-glow: rgba(0,212,160,0.12);
    --warn: #f59e0b;
    --danger: #ef4444;
    --success: #10b981;
    --purple: #8b5cf6;
    --font-main: 'Instrument Sans', sans-serif;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
    --r: 10px;
    --r-lg: 16px;
    --r-xl: 24px;
    --shadow: 0 4px 32px rgba(0,0,0,0.4);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-main);
    min-height: 100vh;
    overflow-x: hidden;
    font-size: 14px;
    line-height: 1.6;
  }

  /* SCROLLBAR */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-med); border-radius: 2px; }

  /* AUTH SCREEN */
  #auth-screen {
    position: fixed; inset: 0; z-index: 1000;
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
  }

  .auth-bg {
    position: absolute; inset: 0; overflow: hidden;
    pointer-events: none;
  }
  .auth-bg::before {
    content: '';
    position: absolute; top: -200px; left: 50%; transform: translateX(-50%);
    width: 800px; height: 800px;
    background: radial-gradient(ellipse, rgba(61,126,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .auth-bg::after {
    content: '';
    position: absolute; bottom: -200px; right: -100px;
    width: 600px; height: 600px;
    background: radial-gradient(ellipse, rgba(0,212,160,0.06) 0%, transparent 70%);
    border-radius: 50%;
  }

  .auth-logo {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 48px;
    font-family: var(--font-display);
    font-size: 28px; font-weight: 800;
    letter-spacing: -0.5px;
  }
  .auth-logo .logo-icon {
    width: 44px; height: 44px;
    background: var(--accent);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }

  .auth-box {
    background: var(--bg3);
    border: 1px solid var(--border-med);
    border-radius: var(--r-xl);
    padding: 40px;
    width: 420px;
    position: relative;
    z-index: 1;
  }
  .auth-tabs {
    display: flex; gap: 0;
    background: var(--bg);
    border-radius: var(--r);
    padding: 4px;
    margin-bottom: 28px;
  }
  .auth-tab {
    flex: 1; padding: 8px 16px;
    border: none; background: transparent;
    color: var(--text-dim); cursor: pointer;
    border-radius: 8px;
    font-family: var(--font-main);
    font-size: 13px; font-weight: 500;
    transition: all 0.2s;
  }
  .auth-tab.active {
    background: var(--accent);
    color: white;
  }
  .auth-field { margin-bottom: 16px; }
  .auth-field label {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--text-dim); margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .auth-field input, .auth-field select {
    width: 100%; padding: 10px 14px;
    background: var(--bg); border: 1px solid var(--border-med);
    border-radius: var(--r); color: var(--text);
    font-family: var(--font-main); font-size: 14px;
    transition: border-color 0.2s;
    outline: none;
  }
  .auth-field input:focus, .auth-field select:focus {
    border-color: var(--accent);
  }
  .auth-field select option { background: var(--bg); }
  .btn-primary {
    width: 100%; padding: 12px;
    background: var(--accent); border: none;
    color: white; border-radius: var(--r);
    font-family: var(--font-main);
    font-size: 14px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    letter-spacing: 0.3px;
  }
  .btn-primary:hover { background: #5590ff; transform: translateY(-1px); }
  .auth-divider {
    text-align: center; margin: 20px 0;
    color: var(--text-muted); font-size: 12px;
    position: relative;
  }
  .auth-divider::before, .auth-divider::after {
    content: ''; position: absolute; top: 50%;
    width: 42%; height: 1px;
    background: var(--border);
  }
  .auth-divider::before { left: 0; }
  .auth-divider::after { right: 0; }
  .role-select-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; margin-bottom: 16px;
  }
  .role-card {
    border: 1px solid var(--border-med);
    border-radius: var(--r); padding: 14px 12px;
    cursor: pointer; transition: all 0.2s;
    text-align: center;
  }
  .role-card:hover, .role-card.active {
    border-color: var(--accent);
    background: var(--accent-glow);
  }
  .role-card i { font-size: 22px; margin-bottom: 6px; display: block; color: var(--accent); }
  .role-card span { font-size: 12px; font-weight: 500; color: var(--text-dim); }

  /* MAIN APP */
  #app { display: none; height: 100vh; overflow: hidden; }
  #app.visible { display: flex; }

  /* SIDEBAR */
  .sidebar {
    width: 240px; min-width: 240px;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 20px 0;
    position: relative; z-index: 10;
  }
  .sidebar-logo {
    padding: 0 20px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-display);
    font-size: 18px; font-weight: 800;
  }
  .sidebar-logo .logo-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
  }
  .sidebar-section {
    padding: 8px 12px 4px;
    font-size: 10px; font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px;
  }
  .sidebar-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 20px;
    color: var(--text-dim); cursor: pointer;
    transition: all 0.15s;
    font-size: 13.5px; font-weight: 500;
    border-left: 2px solid transparent;
    position: relative;
  }
  .sidebar-item:hover { color: var(--text); background: rgba(255,255,255,0.03); }
  .sidebar-item.active {
    color: var(--accent);
    background: var(--accent-glow);
    border-left-color: var(--accent);
  }
  .sidebar-item i { width: 18px; text-align: center; font-size: 15px; }
  .sidebar-badge {
    margin-left: auto;
    background: var(--danger);
    color: white; font-size: 10px; font-weight: 700;
    padding: 2px 6px; border-radius: 10px;
    min-width: 18px; text-align: center;
  }
  .sidebar-badge.green { background: var(--success); }
  .sidebar-user {
    margin-top: auto;
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
  }
  .user-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    color: white;
  }
  .user-info { flex: 1; min-width: 0; }
  .user-name { font-size: 13px; font-weight: 600; }
  .user-role { font-size: 11px; color: var(--text-dim); }
  .user-logout { color: var(--text-muted); cursor: pointer; font-size: 14px; transition: color 0.2s; }
  .user-logout:hover { color: var(--danger); }

  /* MAIN CONTENT */
  .main-content {
    flex: 1; overflow-y: auto; overflow-x: hidden;
    display: flex; flex-direction: column;
  }

  /* TOP BAR */
  .topbar {
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
    background: var(--bg2);
    position: sticky; top: 0; z-index: 5;
  }
  .topbar-title {
    font-family: var(--font-display);
    font-size: 16px; font-weight: 700;
  }
  .topbar-sub { font-size: 12px; color: var(--text-dim); margin-top: 1px; }
  .topbar-actions { margin-left: auto; display: flex; gap: 10px; align-items: center; }

  .btn {
    padding: 8px 16px;
    border-radius: var(--r);
    border: 1px solid var(--border-med);
    background: transparent;
    color: var(--text); cursor: pointer;
    font-family: var(--font-main);
    font-size: 13px; font-weight: 500;
    transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .btn:hover { background: var(--bg3); border-color: var(--border-hi); }
  .btn-accent {
    background: var(--accent); border-color: var(--accent); color: white;
  }
  .btn-accent:hover { background: #5590ff; border-color: #5590ff; }
  .btn-success { background: var(--success); border-color: var(--success); color: white; }
  .btn-success:hover { background: #0ea672; }
  .btn-danger { background: transparent; border-color: var(--danger); color: var(--danger); }
  .btn-danger:hover { background: rgba(239,68,68,0.1); }
  .btn-sm { padding: 5px 10px; font-size: 12px; }
  .btn-icon { padding: 8px; width: 34px; height: 34px; justify-content: center; }

  /* PAGES */
  .page { display: none; flex: 1; padding: 28px; }
  .page.active { display: block; }

  /* CARDS */
  .card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
  }
  .card-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title {
    font-family: var(--font-display);
    font-size: 14px; font-weight: 700;
    display: flex; align-items: center; gap: 8px;
  }
  .card-body { padding: 20px; }

  /* GRID */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }

  /* STAT CARDS */
  .stat-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 20px;
    position: relative; overflow: hidden;
  }
  .stat-card::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle at 100% 0%, var(--accent-glow) 0%, transparent 70%);
  }
  .stat-label {
    font-size: 11px; font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 8px;
  }
  .stat-value {
    font-family: var(--font-display);
    font-size: 28px; font-weight: 800;
    letter-spacing: -1px;
  }
  .stat-change {
    font-size: 11px; margin-top: 4px;
    display: flex; align-items: center; gap: 4px;
  }
  .stat-change.up { color: var(--success); }
  .stat-change.down { color: var(--danger); }

  /* UPLOAD ZONE */
  .upload-zone {
    border: 2px dashed var(--border-med);
    border-radius: var(--r-lg);
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s;
    position: relative;
    background: var(--bg);
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: var(--accent-glow);
  }
  .upload-zone input[type=file] {
    position: absolute; inset: 0;
    opacity: 0; cursor: pointer;
    width: 100%; height: 100%;
  }
  .upload-icon {
    width: 60px; height: 60px;
    background: var(--bg3);
    border: 1px solid var(--border-med);
    border-radius: 16px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 24px; color: var(--accent);
    margin-bottom: 16px;
  }
  .upload-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
  .upload-sub { color: var(--text-dim); font-size: 13px; }
  .upload-formats {
    display: flex; gap: 8px; justify-content: center;
    margin-top: 16px; flex-wrap: wrap;
  }
  .format-tag {
    padding: 4px 10px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 11px; font-family: var(--font-mono);
    color: var(--accent);
  }

  /* FILE LIST */
  .file-list { list-style: none; }
  .file-item {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  .file-item:last-child { border-bottom: none; }
  .file-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
  }
  .file-icon.pdf { background: rgba(239,68,68,0.15); color: #f87171; }
  .file-icon.img { background: rgba(61,126,255,0.15); color: var(--accent); }
  .file-icon.dwg { background: rgba(0,212,160,0.15); color: var(--accent2); }
  .file-meta { flex: 1; min-width: 0; }
  .file-name { font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .file-size { font-size: 11px; color: var(--text-dim); }

  /* STATUS BADGES */
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.3px;
  }
  .badge::before { content: ''; width: 5px; height: 5px; border-radius: 50%; }
  .badge-pending { background: rgba(245,158,11,0.12); color: var(--warn); }
  .badge-pending::before { background: var(--warn); }
  .badge-processing { background: rgba(61,126,255,0.12); color: var(--accent); }
  .badge-processing::before { background: var(--accent); animation: pulse 1s infinite; }
  .badge-ready { background: rgba(16,185,129,0.12); color: var(--success); }
  .badge-ready::before { background: var(--success); }
  .badge-approved { background: rgba(139,92,246,0.12); color: var(--purple); }
  .badge-approved::before { background: var(--purple); }
  .badge-saved { background: rgba(0,212,160,0.12); color: var(--accent2); }
  .badge-saved::before { background: var(--accent2); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* DWG VIEWER */
  .dwg-viewer {
    background: #fff;
    border-radius: var(--r-lg);
    overflow: hidden;
    position: relative;
    min-height: 480px;
    display: flex; align-items: center; justify-content: center;
  }
  .dwg-canvas { width: 100%; height: 480px; }
  .dwg-toolbar {
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .dwg-tool {
    width: 32px; height: 32px;
    background: rgba(10,12,15,0.8);
    border: 1px solid var(--border-hi);
    border-radius: 6px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: var(--text);
    transition: all 0.2s;
  }
  .dwg-tool:hover { background: var(--bg3); }
  .dwg-info {
    position: absolute; bottom: 12px; left: 12px; right: 12px;
    background: rgba(10,12,15,0.85);
    border: 1px solid var(--border-med);
    border-radius: 8px; padding: 10px 14px;
    display: flex; gap: 20px; flex-wrap: wrap;
    backdrop-filter: blur(8px);
  }
  .dwg-info-item { font-size: 11px; font-family: var(--font-mono); }
  .dwg-info-item span { color: var(--accent2); font-weight: 500; }

  /* PROGRESS BAR */
  .progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px; overflow: hidden;
    margin: 8px 0;
  }
  .progress-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
    transition: width 0.4s ease;
  }
  .progress-fill.green { background: var(--success); }

  /* PROVIDER CARDS */
  .provider-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 20px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  .provider-card:hover { border-color: var(--border-hi); transform: translateY(-1px); }
  .provider-card.selected { border-color: var(--accent); background: var(--accent-glow); }
  .provider-card.best-value::before {
    content: 'BEST VALUE';
    position: absolute; top: -8px; right: 16px;
    background: var(--success);
    color: white; font-size: 9px; font-weight: 700;
    padding: 3px 8px; border-radius: 4px;
    letter-spacing: 0.5px;
  }
  .provider-head {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
  }
  .provider-avatar {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800;
    font-family: var(--font-display);
  }
  .provider-name { font-size: 15px; font-weight: 700; }
  .provider-location { font-size: 11px; color: var(--text-dim); display: flex; align-items: center; gap: 4px; }
  .provider-price {
    font-family: var(--font-display);
    font-size: 26px; font-weight: 800;
    color: var(--accent2);
    letter-spacing: -0.5px;
  }
  .provider-price span { font-size: 14px; color: var(--text-dim); font-weight: 400; }
  .provider-stats {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 10px; margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }
  .provider-stat { text-align: center; }
  .provider-stat-val { font-size: 14px; font-weight: 700; }
  .provider-stat-label { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
  .star-rating { color: var(--warn); font-size: 12px; }

  /* TABS */
  .tabs {
    display: flex; gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .tab {
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
    color: var(--text-dim); cursor: pointer;
    font-size: 13.5px; font-weight: 500;
    transition: all 0.15s;
    display: flex; align-items: center; gap: 7px;
    white-space: nowrap;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* FORM ELEMENTS */
  .form-group { margin-bottom: 16px; }
  .form-label {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--text-dim); margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .form-input, .form-select, .form-textarea {
    width: 100%; padding: 10px 14px;
    background: var(--bg); border: 1px solid var(--border-med);
    border-radius: var(--r); color: var(--text);
    font-family: var(--font-main); font-size: 14px;
    transition: border-color 0.2s; outline: none;
  }
  .form-input:focus, .form-select:focus, .form-textarea:focus {
    border-color: var(--accent);
  }
  .form-select option { background: var(--bg); }
  .form-textarea { resize: vertical; min-height: 80px; }
  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* TABLE */
  .data-table { width: 100%; border-collapse: collapse; }
  .data-table th {
    padding: 10px 14px; text-align: left;
    font-size: 11px; font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .data-table td {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    vertical-align: middle;
  }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background: rgba(255,255,255,0.02); }

  /* LOADING / SPINNER */
  .spinner {
    width: 20px; height: 20px;
    border: 2px solid var(--border-med);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* MODAL */
  .modal-overlay {
    position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.6);
    display: none; align-items: center; justify-content: center;
    backdrop-filter: blur(4px);
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--bg3);
    border: 1px solid var(--border-hi);
    border-radius: var(--r-xl);
    width: 600px; max-width: 95vw;
    max-height: 90vh;
    overflow-y: auto;
    animation: modal-in 0.25s ease;
  }
  @keyframes modal-in {
    from { transform: scale(0.96) translateY(10px); opacity: 0; }
    to { transform: scale(1) translateY(0); opacity: 1; }
  }
  .modal-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .modal-title { font-family: var(--font-display); font-size: 18px; font-weight: 700; }
  .modal-close {
    width: 30px; height: 30px;
    border-radius: 8px; border: 1px solid var(--border);
    background: transparent; color: var(--text-dim);
    cursor: pointer; font-size: 14px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .modal-close:hover { background: var(--bg4); color: var(--text); }
  .modal-body { padding: 24px; }
  .modal-footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex; justify-content: flex-end; gap: 10px;
  }

  /* NOTIFICATION */
  .notification-stack {
    position: fixed; top: 20px; right: 20px;
    z-index: 200; display: flex; flex-direction: column; gap: 8px;
    pointer-events: none;
  }
  .notif {
    background: var(--bg3);
    border: 1px solid var(--border-hi);
    border-radius: 10px;
    padding: 12px 16px;
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; min-width: 280px; max-width: 360px;
    pointer-events: auto;
    animation: notif-in 0.3s ease;
    box-shadow: var(--shadow);
  }
  @keyframes notif-in {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  .notif.out { animation: notif-out 0.3s ease forwards; }
  @keyframes notif-out {
    to { transform: translateX(110%); opacity: 0; }
  }
  .notif-icon { font-size: 16px; flex-shrink: 0; }
  .notif.success .notif-icon { color: var(--success); }
  .notif.error .notif-icon { color: var(--danger); }
  .notif.info .notif-icon { color: var(--accent); }
  .notif.warning .notif-icon { color: var(--warn); }
  .notif-text { flex: 1; }
  .notif-title { font-weight: 600; }
  .notif-sub { font-size: 11px; color: var(--text-dim); margin-top: 2px; }

  /* TOOLTIP */
  [data-tip] { position: relative; }
  [data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%);
    background: var(--bg4); color: var(--text);
    font-size: 11px; padding: 5px 10px;
    border-radius: 6px; white-space: nowrap;
    border: 1px solid var(--border-med);
    pointer-events: none; z-index: 50;
  }

  /* DIMENSION TABLE */
  .dim-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .dim-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 12px;
  }
  .dim-item-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .dim-item-val { font-family: var(--font-mono); font-size: 15px; font-weight: 500; color: var(--accent2); }

  /* PROCESS STEPS */
  .steps { display: flex; align-items: flex-start; gap: 0; margin-bottom: 28px; }
  .step {
    display: flex; flex-direction: column; align-items: center;
    flex: 1;
  }
  .step-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    border: 2px solid var(--border-med);
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    font-family: var(--font-mono);
    color: var(--text-dim);
    position: relative; z-index: 1;
    transition: all 0.3s;
  }
  .step-dot.done { background: var(--success); border-color: var(--success); color: white; }
  .step-dot.active { background: var(--accent); border-color: var(--accent); color: white; }
  .step-label { font-size: 11px; font-weight: 500; color: var(--text-dim); margin-top: 8px; text-align: center; line-height: 1.3; }
  .step-label.done { color: var(--success); }
  .step-label.active { color: var(--accent); }
  .step-line {
    flex: 1; height: 2px;
    background: var(--border);
    margin-top: 16px; margin-left: -10px; margin-right: -10px;
    position: relative; z-index: 0;
    transition: background 0.3s;
  }
  .step-line.done { background: var(--success); }

  /* AI ANALYSIS PANEL */
  .analysis-card {
    background: linear-gradient(135deg, rgba(61,126,255,0.08), rgba(0,212,160,0.05));
    border: 1px solid rgba(61,126,255,0.2);
    border-radius: var(--r-lg);
    padding: 20px;
  }
  .analysis-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px;
    font-size: 13px; font-weight: 600;
    color: var(--accent);
  }

  /* CLOUD SAVE ANIMATION */
  .cloud-save-btn {
    position: relative; overflow: hidden;
  }
  .cloud-save-btn::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transform: translateX(-100%);
    transition: transform 0.5s;
  }
  .cloud-save-btn:hover::after { transform: translateX(100%); }

  /* QUOTE FORM */
  .material-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 16px;
  }
  .material-card {
    border: 1px solid var(--border);
    border-radius: var(--r); padding: 12px 10px;
    cursor: pointer; transition: all 0.2s;
    text-align: center;
  }
  .material-card:hover, .material-card.active {
    border-color: var(--accent); background: var(--accent-glow);
  }
  .material-card i { font-size: 20px; display: block; margin-bottom: 4px; }
  .material-card span { font-size: 11px; font-weight: 500; color: var(--text-dim); }

  /* SEARCH & FILTER */
  .search-bar {
    display: flex; align-items: center; gap: 0;
    background: var(--bg);
    border: 1px solid var(--border-med);
    border-radius: var(--r);
    padding: 8px 14px;
    flex: 1;
  }
  .search-bar i { color: var(--text-muted); margin-right: 10px; }
  .search-bar input {
    border: none; background: transparent;
    color: var(--text); font-family: var(--font-main);
    font-size: 13px; outline: none; flex: 1;
  }

  /* MISC */
  .divider { height: 1px; background: var(--border); margin: 20px 0; }
  .text-accent { color: var(--accent); }
  .text-success { color: var(--success); }
  .text-warn { color: var(--warn); }
  .text-danger { color: var(--danger); }
  .text-dim { color: var(--text-dim); }
  .text-mono { font-family: var(--font-mono); }
  .text-sm { font-size: 12px; }
  .mb-0 { margin-bottom: 0 !important; }
  .mb-4 { margin-bottom: 4px; }
  .mb-8 { margin-bottom: 8px; }
  .mb-12 { margin-bottom: 12px; }
  .mb-16 { margin-bottom: 16px; }
  .mb-20 { margin-bottom: 20px; }
  .mb-24 { margin-bottom: 24px; }
  .flex { display: flex; }
  .flex-center { display: flex; align-items: center; }
  .gap-8 { gap: 8px; }
  .gap-12 { gap: 12px; }
  .gap-16 { gap: 16px; }
  .flex-1 { flex: 1; }
  .ml-auto { margin-left: auto; }
  .w-full { width: 100%; }

  /* CHECKBOX */
  .checkbox-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; cursor: pointer;
  }
  .checkbox {
    width: 16px; height: 16px;
    border: 1.5px solid var(--border-hi);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: all 0.15s;
    font-size: 10px; color: transparent;
  }
  .checkbox.checked {
    background: var(--accent); border-color: var(--accent);
    color: white;
  }

  /* RANGE SLIDER */
  input[type=range] {
    -webkit-appearance: none;
    width: 100%; height: 4px;
    background: var(--border);
    border-radius: 2px; outline: none;
    margin: 10px 0;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer; border: 2px solid var(--bg);
  }

  /* CLOUDINARY STATUS */
  .cloud-status {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r);
    font-size: 13px;
  }
  .cloud-status.uploading { border-color: var(--accent); }
  .cloud-status.done { border-color: var(--success); }
  .cloud-url {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent2);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    word-break: break-all;
    display: flex; align-items: flex-start; gap: 8px;
  }
  .copy-btn {
    flex-shrink: 0; padding: 4px 8px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 5px; cursor: pointer;
    color: var(--text-dim); font-size: 11px;
    transition: all 0.15s;
  }
  .copy-btn:hover { color: var(--text); background: var(--bg4); }

  /* RESPONSIVE */
  @media (max-width: 900px) {
    .grid-4 { grid-template-columns: 1fr 1fr; }
    .grid-3 { grid-template-columns: 1fr 1fr; }
    .grid-2 { grid-template-columns: 1fr; }
    .dim-grid { grid-template-columns: 1fr 1fr; }
    .material-grid { grid-template-columns: repeat(2, 1fr); }
    .sidebar { width: 60px; min-width: 60px; }
    .sidebar-logo span, .sidebar-item span, .user-info, .sidebar-section { display: none; }
    .sidebar-item { padding: 12px; justify-content: center; }
    .sidebar-badge { position: absolute; top: 4px; right: 4px; }
  }
</style>
</head>
<body>

<!-- AUTH SCREEN -->
<div id="auth-screen">
  <div class="auth-bg"></div>
  <div class="auth-logo">
    <div class="logo-icon"><i class="fas fa-layer-group"></i></div>
    SheetForge
  </div>
  <div class="auth-box">
    <div class="auth-tabs">
      <button class="auth-tab active" onclick="switchAuthTab('login')">Sign In</button>
      <button class="auth-tab" onclick="switchAuthTab('signup')">Create Account</button>
    </div>
    <!-- LOGIN -->
    <div id="login-form">
      <div class="auth-field">
        <label>Email Address</label>
        <input type="email" id="login-email" placeholder="you@company.com" value="demo@sheetforge.io">
      </div>
      <div class="auth-field">
        <label>Password</label>
        <input type="password" id="login-pass" placeholder="••••••••" value="demo123">
      </div>
      <button class="btn-primary" onclick="doLogin()">
        <i class="fas fa-arrow-right" style="margin-right:8px"></i> Sign In
      </button>
      <div class="auth-divider">or continue with demo</div>
      <div class="form-row" style="margin-top:0">
        <button class="btn" style="width:100%;justify-content:center" onclick="demoLogin('designer')">
          <i class="fas fa-drafting-compass" style="margin-right:6px;color:var(--accent)"></i> Designer
        </button>
        <button class="btn" style="width:100%;justify-content:center" onclick="demoLogin('provider')">
          <i class="fas fa-industry" style="margin-right:6px;color:var(--accent2)"></i> Provider
        </button>
      </div>
    </div>
    <!-- SIGNUP -->
    <div id="signup-form" style="display:none">
      <div class="role-select-row">
        <div class="role-card active" id="role-designer" onclick="selectRole('designer')">
          <i class="fas fa-drafting-compass"></i>
          <span>Designer / Buyer</span>
        </div>
        <div class="role-card" id="role-provider" onclick="selectRole('provider')">
          <i class="fas fa-industry"></i>
          <span>CNC Provider</span>
        </div>
      </div>
      <div class="form-row">
        <div class="auth-field">
          <label>First Name</label>
          <input type="text" placeholder="John">
        </div>
        <div class="auth-field">
          <label>Last Name</label>
          <input type="text" placeholder="Smith">
        </div>
      </div>
      <div class="auth-field">
        <label>Company</label>
        <input type="text" placeholder="Acme Engineering">
      </div>
      <div class="auth-field">
        <label>Email</label>
        <input type="email" placeholder="you@company.com">
      </div>
      <div class="auth-field">
        <label>Password</label>
        <input type="password" placeholder="Min. 8 characters">
      </div>
      <button class="btn-primary" onclick="doSignup()">Create Account</button>
    </div>
  </div>
</div>

<!-- NOTIFICATION STACK -->
<div class="notification-stack" id="notif-stack"></div>

<!-- MAIN APP -->
<div id="app">
  <!-- SIDEBAR -->
  <nav class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon"><i class="fas fa-layer-group"></i></div>
      <span>SheetForge</span>
    </div>

    <div class="sidebar-section">Workspace</div>
    <div class="sidebar-item active" onclick="showPage('dashboard')" id="nav-dashboard">
      <i class="fas fa-th-large"></i><span>Dashboard</span>
    </div>
    <div class="sidebar-item" onclick="showPage('upload')" id="nav-upload">
      <i class="fas fa-cloud-upload-alt"></i><span>Upload Design</span>
    </div>
    <div class="sidebar-item" onclick="showPage('drawings')" id="nav-drawings">
      <i class="fas fa-drafting-compass"></i><span>My Drawings</span>
      <span class="sidebar-badge" id="badge-drawings">3</span>
    </div>
    <div class="sidebar-item" onclick="showPage('viewer')" id="nav-viewer">
      <i class="fas fa-eye"></i><span>CAD Viewer</span>
    </div>

    <div class="sidebar-section">Procurement</div>
    <div class="sidebar-item" onclick="showPage('quote')" id="nav-quote">
      <i class="fas fa-calculator"></i><span>Get Quote</span>
    </div>
    <div class="sidebar-item" onclick="showPage('providers')" id="nav-providers">
      <i class="fas fa-globe"></i><span>Providers</span>
      <span class="sidebar-badge green" id="badge-providers">12</span>
    </div>
    <div class="sidebar-item" onclick="showPage('orders')" id="nav-orders">
      <i class="fas fa-box"></i><span>Orders</span>
    </div>

    <div class="sidebar-section">Storage</div>
    <div class="sidebar-item" onclick="showPage('cloudinary')" id="nav-cloudinary">
      <i class="fas fa-cloud"></i><span>Cloud Assets</span>
    </div>
    <div class="sidebar-item" onclick="showPage('settings')" id="nav-settings">
      <i class="fas fa-cog"></i><span>Settings</span>
    </div>

    <div class="sidebar-user">
      <div class="user-avatar" id="user-avatar-sidebar">DS</div>
      <div class="user-info">
        <div class="user-name" id="sidebar-user-name">Demo User</div>
        <div class="user-role" id="sidebar-user-role">Designer</div>
      </div>
      <div class="user-logout" onclick="doLogout()" data-tip="Sign out">
        <i class="fas fa-sign-out-alt"></i>
      </div>
    </div>
  </nav>

  <!-- MAIN CONTENT -->
  <div class="main-content">

    <!-- ===== DASHBOARD ===== -->
    <div class="page active" id="page-dashboard">
      <div class="topbar">
        <div>
          <div class="topbar-title">Dashboard</div>
          <div class="topbar-sub">Welcome back, <span id="dashboard-user-name">Designer</span></div>
        </div>
        <div class="topbar-actions">
          <button class="btn" onclick="showPage('upload')"><i class="fas fa-plus"></i> New Design</button>
          <button class="btn-accent btn" onclick="showPage('quote')"><i class="fas fa-calculator"></i> Quick Quote</button>
        </div>
      </div>
      <div style="padding:28px">
        <!-- Stats -->
        <div class="grid-4 mb-24">
          <div class="stat-card">
            <div class="stat-label">Total Designs</div>
            <div class="stat-value">24</div>
            <div class="stat-change up"><i class="fas fa-arrow-up"></i> 4 this month</div>
          </div>
          <div class="stat-card" style="--accent-glow:rgba(0,212,160,0.12)">
            <div class="stat-label">DWG Conversions</div>
            <div class="stat-value text-accent2" style="color:var(--accent2)">18</div>
            <div class="stat-change up"><i class="fas fa-arrow-up"></i> 75% success</div>
          </div>
          <div class="stat-card" style="--accent-glow:rgba(139,92,246,0.12)">
            <div class="stat-label">Active Quotes</div>
            <div class="stat-value" style="color:var(--purple)">6</div>
            <div class="stat-change up"><i class="fas fa-arrow-up"></i> 2 pending</div>
          </div>
          <div class="stat-card" style="--accent-glow:rgba(16,185,129,0.12)">
            <div class="stat-label">Orders Placed</div>
            <div class="stat-value" style="color:var(--success)">11</div>
            <div class="stat-change up"><i class="fas fa-check"></i> $14,280 total</div>
          </div>
        </div>

        <div class="grid-2 mb-24">
          <!-- Recent Designs -->
          <div class="card">
            <div class="card-header">
              <div class="card-title"><i class="fas fa-file-alt text-accent"></i> Recent Designs</div>
              <button class="btn btn-sm" onclick="showPage('drawings')">View All</button>
            </div>
            <div class="card-body" style="padding:0">
              <table class="data-table">
                <thead><tr>
                  <th>Design</th><th>Status</th><th>Date</th><th></th>
                </tr></thead>
                <tbody>
                  <tr>
                    <td><div class="file-name">bracket_v3.pdf</div><div class="file-size" style="font-size:11px;color:var(--text-dim)">PDF • 2.4 MB</div></td>
                    <td><span class="badge badge-approved">Approved</span></td>
                    <td class="text-dim text-sm">Today</td>
                    <td><button class="btn btn-sm" onclick="showPage('viewer')"><i class="fas fa-eye"></i></button></td>
                  </tr>
                  <tr>
                    <td><div class="file-name">enclosure_front.png</div><div class="file-size" style="font-size:11px;color:var(--text-dim)">PNG • 1.1 MB</div></td>
                    <td><span class="badge badge-ready">Ready</span></td>
                    <td class="text-dim text-sm">Yesterday</td>
                    <td><button class="btn btn-sm" onclick="showPage('viewer')"><i class="fas fa-eye"></i></button></td>
                  </tr>
                  <tr>
                    <td><div class="file-name">panel_chassis.jpg</div><div class="file-size" style="font-size:11px;color:var(--text-dim)">JPG • 840 KB</div></td>
                    <td><span class="badge badge-processing">Processing</span></td>
                    <td class="text-dim text-sm">2 days ago</td>
                    <td><button class="btn btn-sm"><i class="fas fa-spinner fa-spin"></i></button></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Recent Quotes -->
          <div class="card">
            <div class="card-header">
              <div class="card-title"><i class="fas fa-tags text-accent"></i> Quote Requests</div>
              <button class="btn btn-sm" onclick="showPage('providers')">View Providers</button>
            </div>
            <div class="card-body" style="padding:0">
              <table class="data-table">
                <thead><tr>
                  <th>Part</th><th>Best Price</th><th>Providers</th><th>Status</th>
                </tr></thead>
                <tbody>
                  <tr>
                    <td><div class="file-name">bracket_v3</div><div style="font-size:11px;color:var(--text-dim)">Qty: 50 × Steel 2mm</div></td>
                    <td class="text-success text-mono">$312</td>
                    <td>7 bids</td>
                    <td><span class="badge badge-ready">Active</span></td>
                  </tr>
                  <tr>
                    <td><div class="file-name">enclosure_front</div><div style="font-size:11px;color:var(--text-dim)">Qty: 10 × Aluminium 3mm</div></td>
                    <td class="text-success text-mono">$198</td>
                    <td>4 bids</td>
                    <td><span class="badge badge-pending">Pending</span></td>
                  </tr>
                  <tr>
                    <td><div class="file-name">panel_chassis</div><div style="font-size:11px;color:var(--text-dim)">Qty: 100 × Stainless 1.5mm</div></td>
                    <td class="text-dim text-mono">—</td>
                    <td>Awaiting</td>
                    <td><span class="badge badge-processing">Quoting</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Activity Feed -->
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fas fa-stream text-accent"></i> Activity Feed</div>
          </div>
          <div class="card-body">
            <div style="display:flex;flex-direction:column;gap:0">
              <div id="activity-feed">
                <!-- populated by JS -->
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== UPLOAD PAGE ===== -->
    <div class="page" id="page-upload">
      <div class="topbar">
        <div>
          <div class="topbar-title">Upload Design</div>
          <div class="topbar-sub">Upload rough sketches or CAD files for AI conversion</div>
        </div>
      </div>
      <div style="padding:28px">
        <!-- Process Steps -->
        <div class="steps mb-24">
          <div class="step">
            <div class="step-dot active" id="step1">1</div>
            <div class="step-label active">Upload File</div>
          </div>
          <div class="step-line" id="line1"></div>
          <div class="step">
            <div class="step-dot" id="step2">2</div>
            <div class="step-label">AI Analysis</div>
          </div>
          <div class="step-line" id="line2"></div>
          <div class="step">
            <div class="step-dot" id="step3">3</div>
            <div class="step-label">DWG Convert</div>
          </div>
          <div class="step-line" id="line3"></div>
          <div class="step">
            <div class="step-dot" id="step4">4</div>
            <div class="step-label">Review & Approve</div>
          </div>
          <div class="step-line" id="line4"></div>
          <div class="step">
            <div class="step-dot" id="step5">5</div>
            <div class="step-label">Save to Cloud</div>
          </div>
        </div>

        <div class="grid-2">
          <div>
            <!-- Upload Zone -->
            <div class="upload-zone mb-20" id="upload-zone">
              <input type="file" id="file-input" accept=".pdf,.png,.jpg,.jpeg,.svg,.dwg,.dxf,.tiff,.bmp" onchange="handleFileSelect(event)">
              <div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
              <div class="upload-title">Drop your design file here</div>
              <div class="upload-sub">or click to browse files</div>
              <div class="upload-formats">
                <span class="format-tag">.PDF</span>
                <span class="format-tag">.PNG</span>
                <span class="format-tag">.JPG</span>
                <span class="format-tag">.SVG</span>
                <span class="format-tag">.DXF</span>
                <span class="format-tag">.TIFF</span>
              </div>
            </div>

            <!-- File Info -->
            <div id="file-info-panel" style="display:none">
              <div class="card mb-16">
                <div class="card-header">
                  <div class="card-title"><i class="fas fa-file-alt text-accent"></i> Uploaded File</div>
                  <span class="badge badge-ready" id="upload-status-badge">Ready</span>
                </div>
                <div class="card-body">
                  <div class="flex-center gap-12 mb-12">
                    <div class="file-icon img" id="upload-file-icon"><i class="fas fa-file-image"></i></div>
                    <div>
                      <div class="file-name" id="upload-file-name">filename.png</div>
                      <div class="file-size" id="upload-file-size">1.2 MB • PNG</div>
                    </div>
                  </div>
                  <div id="upload-preview-area" style="margin-top:12px;border-radius:8px;overflow:hidden;max-height:200px;display:flex;align-items:center;justify-content:center;background:var(--bg);">
                    <img id="upload-preview-img" style="max-width:100%;max-height:200px;display:none;object-fit:contain" alt="Preview">
                    <canvas id="upload-preview-canvas" style="display:none;max-width:100%;max-height:200px"></canvas>
                    <div id="upload-preview-placeholder" style="padding:20px;text-align:center;color:var(--text-dim)">
                      <i class="fas fa-file-pdf" style="font-size:32px;margin-bottom:8px;display:block;color:#f87171"></i>
                      PDF file uploaded
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Design Details Form -->
            <div class="card" id="design-details-form" style="display:none">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-info-circle text-accent"></i> Design Details</div>
              </div>
              <div class="card-body">
                <div class="form-group">
                  <label class="form-label">Part Name</label>
                  <input class="form-input" type="text" id="part-name" placeholder="e.g. Mounting Bracket v3">
                </div>
                <div class="form-row">
                  <div class="form-group">
                    <label class="form-label">Material</label>
                    <select class="form-select" id="material-select">
                      <option value="">Select material</option>
                      <option>Mild Steel</option>
                      <option>Stainless Steel 304</option>
                      <option>Stainless Steel 316</option>
                      <option>Aluminium 5052</option>
                      <option>Aluminium 6061</option>
                      <option>Galvanized Steel</option>
                      <option>Copper</option>
                      <option>Brass</option>
                    </select>
                  </div>
                  <div class="form-group">
                    <label class="form-label">Thickness (mm)</label>
                    <input class="form-input" type="number" id="thickness-input" placeholder="e.g. 2.0" step="0.5">
                  </div>
                </div>
                <div class="form-group">
                  <label class="form-label">Special Notes</label>
                  <textarea class="form-textarea" id="design-notes" placeholder="Any special requirements, tolerances, surface finish, etc."></textarea>
                </div>
                <button class="btn btn-accent w-full" onclick="startConversion()" id="convert-btn">
                  <i class="fas fa-magic"></i> Convert to DWG with AI
                </button>
              </div>
            </div>
          </div>

          <div>
            <!-- AI Analysis Panel -->
            <div class="analysis-card mb-16" id="ai-analysis-panel" style="display:none">
              <div class="analysis-header">
                <i class="fas fa-robot"></i> AI Analysis
                <div class="spinner" id="analysis-spinner" style="margin-left:auto;width:16px;height:16px"></div>
              </div>
              <div id="ai-analysis-content">
                <div style="color:var(--text-dim);font-size:13px">Analyzing your design...</div>
              </div>
            </div>

            <!-- Conversion Progress -->
            <div class="card mb-16" id="conversion-progress-panel" style="display:none">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-cogs text-accent"></i> DWG Conversion</div>
                <span class="badge badge-processing" id="conv-status-badge">Processing</span>
              </div>
              <div class="card-body">
                <div id="conversion-steps-list"></div>
              </div>
            </div>

            <!-- DWG Result -->
            <div class="card" id="dwg-result-panel" style="display:none">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-check-circle text-success"></i> DWG Generated</div>
                <span class="badge badge-ready">Ready to Review</span>
              </div>
              <div class="card-body">
                <div class="flex-center gap-12 mb-16">
                  <div class="file-icon dwg"><i class="fas fa-drafting-compass"></i></div>
                  <div>
                    <div class="file-name" id="dwg-filename">part_converted.dwg</div>
                    <div class="file-size">AutoCAD DWG • 348 KB</div>
                  </div>
                </div>
                <div class="dim-grid mb-16">
                  <div class="dim-item"><div class="dim-item-label">Width</div><div class="dim-item-val" id="dim-w">240mm</div></div>
                  <div class="dim-item"><div class="dim-item-label">Height</div><div class="dim-item-val" id="dim-h">160mm</div></div>
                  <div class="dim-item"><div class="dim-item-label">Thickness</div><div class="dim-item-val" id="dim-t">2mm</div></div>
                  <div class="dim-item"><div class="dim-item-label">Bend Lines</div><div class="dim-item-val" id="dim-b">4</div></div>
                  <div class="dim-item"><div class="dim-item-label">Holes</div><div class="dim-item-val" id="dim-holes">6</div></div>
                  <div class="dim-item"><div class="dim-item-label">Cut Length</div><div class="dim-item-val">1,240mm</div></div>
                </div>
                <div class="flex gap-8">
                  <button class="btn flex-1" onclick="showPage('viewer')"><i class="fas fa-eye"></i> View DWG</button>
                  <button class="btn btn-success flex-1" onclick="approveDWG()" id="approve-btn"><i class="fas fa-check"></i> Approve</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== DRAWINGS PAGE ===== -->
    <div class="page" id="page-drawings">
      <div class="topbar">
        <div>
          <div class="topbar-title">My Drawings</div>
          <div class="topbar-sub">Manage all your uploaded and converted designs</div>
        </div>
        <div class="topbar-actions">
          <div class="search-bar" style="width:220px">
            <i class="fas fa-search"></i>
            <input type="text" placeholder="Search drawings..." oninput="filterDrawings(this.value)">
          </div>
          <select class="form-select" style="width:auto;padding:8px 28px 8px 12px">
            <option>All Status</option>
            <option>Approved</option>
            <option>Pending</option>
            <option>Processing</option>
          </select>
          <button class="btn btn-accent" onclick="showPage('upload')"><i class="fas fa-plus"></i> Upload</button>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-3" id="drawings-grid">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- ===== CAD VIEWER ===== -->
    <div class="page" id="page-viewer">
      <div class="topbar">
        <div>
          <div class="topbar-title">CAD Viewer</div>
          <div class="topbar-sub">View and approve your DWG drawings</div>
        </div>
        <div class="topbar-actions">
          <select class="form-select" style="width:auto;padding:8px 28px 8px 12px" id="viewer-file-select" onchange="loadDwg(this.value)">
            <option value="bracket">bracket_v3.dwg</option>
            <option value="enclosure">enclosure_front.dwg</option>
            <option value="panel">panel_chassis.dwg</option>
          </select>
          <button class="btn" onclick="downloadDWG()"><i class="fas fa-download"></i> DWG</button>
          <button class="btn" id="viewer-approve-btn" onclick="approveFromViewer()"><i class="fas fa-check"></i> Approve</button>
          <button class="btn btn-accent cloud-save-btn" id="viewer-save-btn" onclick="openCloudSaveModal()" style="display:none">
            <i class="fas fa-cloud-upload-alt"></i> Save to Cloud
          </button>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-2" style="gap:20px">
          <!-- Canvas Viewer -->
          <div style="grid-column:1/-1">
            <div class="dwg-viewer" id="dwg-main-viewer">
              <canvas id="dwg-canvas" class="dwg-canvas"></canvas>
              <div class="dwg-toolbar">
                <div class="dwg-tool" onclick="dwgZoom(1.2)" data-tip="Zoom In"><i class="fas fa-plus"></i></div>
                <div class="dwg-tool" onclick="dwgZoom(0.8)" data-tip="Zoom Out"><i class="fas fa-minus"></i></div>
                <div class="dwg-tool" onclick="dwgFitView()" data-tip="Fit View"><i class="fas fa-expand"></i></div>
                <div class="dwg-tool" onclick="dwgPan()" data-tip="Pan"><i class="fas fa-hand-paper"></i></div>
                <div class="dwg-tool" onclick="dwgMeasure()" data-tip="Measure"><i class="fas fa-ruler"></i></div>
                <div class="dwg-tool" onclick="dwgLayers()" data-tip="Layers"><i class="fas fa-layer-group"></i></div>
              </div>
              <div class="dwg-info" id="dwg-info">
                <div class="dwg-info-item">Scale: <span id="info-scale">1:1</span></div>
                <div class="dwg-info-item">Units: <span>mm</span></div>
                <div class="dwg-info-item">Width: <span id="info-w">240</span>mm</div>
                <div class="dwg-info-item">Height: <span id="info-h">160</span>mm</div>
                <div class="dwg-info-item">Layers: <span>5</span></div>
                <div class="dwg-info-item">Entities: <span>48</span></div>
              </div>
            </div>
          </div>

          <!-- Properties Panel -->
          <div class="card">
            <div class="card-header">
              <div class="card-title"><i class="fas fa-info-circle text-accent"></i> Properties</div>
            </div>
            <div class="card-body">
              <div class="dim-grid mb-16">
                <div class="dim-item"><div class="dim-item-label">Width</div><div class="dim-item-val">240mm</div></div>
                <div class="dim-item"><div class="dim-item-label">Height</div><div class="dim-item-val">160mm</div></div>
                <div class="dim-item"><div class="dim-item-label">Thickness</div><div class="dim-item-val">2mm</div></div>
                <div class="dim-item"><div class="dim-item-label">Bend Lines</div><div class="dim-item-val">4</div></div>
                <div class="dim-item"><div class="dim-item-label">Holes</div><div class="dim-item-val">6</div></div>
                <div class="dim-item"><div class="dim-item-label">Area</div><div class="dim-item-val">384cm²</div></div>
              </div>
              <table class="data-table">
                <tr><td class="text-dim">Material</td><td class="text-right"><strong>Mild Steel</strong></td></tr>
                <tr><td class="text-dim">Standard</td><td class="text-right">ISO 2768</td></tr>
                <tr><td class="text-dim">Created</td><td class="text-right">Today 14:32</td></tr>
                <tr><td class="text-dim">Version</td><td class="text-right">DWG R2018</td></tr>
                <tr><td class="text-dim">Status</td><td class="text-right"><span class="badge badge-ready" id="prop-status">Ready</span></td></tr>
              </table>
            </div>
          </div>

          <!-- Layers Panel -->
          <div class="card">
            <div class="card-header">
              <div class="card-title"><i class="fas fa-layer-group text-accent"></i> Layers</div>
            </div>
            <div class="card-body" style="padding:12px 20px">
              <div id="layer-list">
                <!-- populated by JS -->
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== QUOTE PAGE ===== -->
    <div class="page" id="page-quote">
      <div class="topbar">
        <div>
          <div class="topbar-title">Request Quote</div>
          <div class="topbar-sub">Get competitive pricing from global CNC providers</div>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-2">
          <div>
            <div class="card mb-20">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-cube text-accent"></i> Part Specifications</div>
              </div>
              <div class="card-body">
                <div class="form-group">
                  <label class="form-label">Select Design</label>
                  <select class="form-select" id="quote-design-select">
                    <option>bracket_v3.dwg — Approved ✓</option>
                    <option>enclosure_front.dwg — Approved ✓</option>
                    <option>panel_chassis.dwg — Processing...</option>
                  </select>
                </div>

                <div class="form-row">
                  <div class="form-group">
                    <label class="form-label">Length (mm)</label>
                    <input class="form-input" type="number" id="q-length" value="240" oninput="recalcQuote()">
                  </div>
                  <div class="form-group">
                    <label class="form-label">Width (mm)</label>
                    <input class="form-input" type="number" id="q-width" value="160" oninput="recalcQuote()">
                  </div>
                </div>
                <div class="form-row">
                  <div class="form-group">
                    <label class="form-label">Thickness (mm)</label>
                    <input class="form-input" type="number" id="q-thickness" value="2" step="0.5" oninput="recalcQuote()">
                  </div>
                  <div class="form-group">
                    <label class="form-label">Quantity</label>
                    <input class="form-input" type="number" id="q-qty" value="50" oninput="recalcQuote()">
                  </div>
                </div>

                <div class="form-group">
                  <label class="form-label">Material</label>
                  <div class="material-grid">
                    <div class="material-card active" id="mat-steel" onclick="selectMaterial('steel')">
                      <i class="fas fa-cube" style="color:#888"></i>
                      <span>Mild Steel</span>
                    </div>
                    <div class="material-card" id="mat-ss" onclick="selectMaterial('ss')">
                      <i class="fas fa-cube" style="color:#aaa"></i>
                      <span>Stainless</span>
                    </div>
                    <div class="material-card" id="mat-al" onclick="selectMaterial('al')">
                      <i class="fas fa-cube" style="color:#cce4ff"></i>
                      <span>Aluminium</span>
                    </div>
                    <div class="material-card" id="mat-cu" onclick="selectMaterial('cu')">
                      <i class="fas fa-cube" style="color:#f0a060"></i>
                      <span>Copper</span>
                    </div>
                  </div>
                </div>

                <div class="form-group">
                  <label class="form-label">Operations Required</label>
                  <div id="ops-checklist">
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox checked"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">Laser Cutting</span>
                    </div>
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox checked"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">CNC Bending</span>
                    </div>
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">Punching</span>
                    </div>
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">Powder Coating</span>
                    </div>
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">Deburring / Finishing</span>
                    </div>
                    <div class="checkbox-row" onclick="toggleCheck(this)">
                      <div class="checkbox"><i class="fas fa-check"></i></div>
                      <span style="font-size:13px">Thread Tapping</span>
                    </div>
                  </div>
                </div>

                <div class="form-group">
                  <label class="form-label">Lead Time Required</label>
                  <select class="form-select" id="q-leadtime" onchange="recalcQuote()">
                    <option value="30">Standard (3–4 weeks)</option>
                    <option value="20">Express (1–2 weeks)</option>
                    <option value="10">Rush (3–5 days +50%)</option>
                  </select>
                </div>

                <div class="form-group">
                  <label class="form-label">Surface Finish</label>
                  <select class="form-select">
                    <option>As cut (no finish)</option>
                    <option>Deburred</option>
                    <option>Powder coated</option>
                    <option>Anodized</option>
                    <option>Zinc plated</option>
                  </select>
                </div>

                <div class="form-group">
                  <label class="form-label">Additional Notes</label>
                  <textarea class="form-textarea" placeholder="Tolerances, certifications, packaging requirements..."></textarea>
                </div>

                <button class="btn btn-accent w-full" onclick="requestQuotes()">
                  <i class="fas fa-paper-plane"></i> Send to All Providers
                </button>
              </div>
            </div>
          </div>

          <!-- Price Estimate & Summary -->
          <div>
            <div class="card mb-20">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-calculator text-accent"></i> Live Price Estimate</div>
              </div>
              <div class="card-body">
                <div style="text-align:center;margin-bottom:20px">
                  <div style="font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Estimated Range</div>
                  <div style="font-family:var(--font-display);font-size:40px;font-weight:800;letter-spacing:-2px">
                    $<span id="price-low">280</span> – $<span id="price-high">420</span>
                  </div>
                  <div style="font-size:12px;color:var(--text-dim)">for <span id="price-qty">50</span> units • excl. shipping</div>
                </div>

                <div style="background:var(--bg);border-radius:var(--r);padding:16px;margin-bottom:16px">
                  <div style="font-size:12px;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px">Cost Breakdown</div>
                  <table style="width:100%;font-size:13px">
                    <tr style="margin-bottom:8px">
                      <td class="text-dim" style="padding:4px 0">Material cost</td>
                      <td style="text-align:right;font-family:var(--font-mono)">$<span id="bd-mat">112</span></td>
                    </tr>
                    <tr>
                      <td class="text-dim" style="padding:4px 0">Cutting</td>
                      <td style="text-align:right;font-family:var(--font-mono)">$<span id="bd-cut">84</span></td>
                    </tr>
                    <tr>
                      <td class="text-dim" style="padding:4px 0">Bending</td>
                      <td style="text-align:right;font-family:var(--font-mono)">$<span id="bd-bend">56</span></td>
                    </tr>
                    <tr>
                      <td class="text-dim" style="padding:4px 0">Setup</td>
                      <td style="text-align:right;font-family:var(--font-mono)">$28</td>
                    </tr>
                    <tr style="border-top:1px solid var(--border)">
                      <td style="padding:8px 0 0;font-weight:600">Total</td>
                      <td style="text-align:right;font-family:var(--font-mono);font-weight:600;padding-top:8px;color:var(--success)">$<span id="bd-total">280</span></td>
                    </tr>
                  </table>
                </div>

                <div style="margin-bottom:12px">
                  <div style="font-size:11px;color:var(--text-dim);margin-bottom:6px">Per-unit cost (estimated)</div>
                  <div style="font-family:var(--font-display);font-size:22px;font-weight:700;color:var(--accent2)">$<span id="per-unit">5.60</span> <span style="font-size:13px;color:var(--text-dim);font-weight:400">/ unit</span></div>
                </div>

                <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:var(--r);padding:12px;font-size:12px;color:var(--success)">
                  <i class="fas fa-lightbulb" style="margin-right:6px"></i>
                  <strong>Tip:</strong> Ordering 100+ units reduces material cost by ~18%
                </div>
              </div>
            </div>

            <!-- Recommended Providers Preview -->
            <div class="card">
              <div class="card-header">
                <div class="card-title"><i class="fas fa-star text-accent"></i> Top Matches</div>
                <button class="btn btn-sm" onclick="showPage('providers')">See All</button>
              </div>
              <div class="card-body" style="padding:12px 20px">
                <div id="top-providers-preview">
                  <!-- Populated by JS -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== PROVIDERS PAGE ===== -->
    <div class="page" id="page-providers">
      <div class="topbar">
        <div>
          <div class="topbar-title">Global CNC Providers</div>
          <div class="topbar-sub">Compare and select from verified sheet metal fabricators worldwide</div>
        </div>
        <div class="topbar-actions">
          <div class="search-bar" style="width:200px">
            <i class="fas fa-search"></i>
            <input type="text" placeholder="Search providers..." oninput="filterProviders(this.value)">
          </div>
          <select class="form-select" style="width:auto;padding:8px 28px 8px 12px" onchange="sortProviders(this.value)">
            <option value="rating">Sort: Rating</option>
            <option value="price">Sort: Price ↑</option>
            <option value="lead">Sort: Lead Time</option>
          </select>
          <select class="form-select" style="width:auto;padding:8px 28px 8px 12px" id="region-filter" onchange="filterByRegion(this.value)">
            <option value="">All Regions</option>
            <option>Asia Pacific</option>
            <option>Europe</option>
            <option>North America</option>
            <option>Middle East</option>
            <option>Africa</option>
          </select>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-3" id="providers-grid">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- ===== ORDERS PAGE ===== -->
    <div class="page" id="page-orders">
      <div class="topbar">
        <div>
          <div class="topbar-title">Orders</div>
          <div class="topbar-sub">Track your production orders</div>
        </div>
      </div>
      <div style="padding:28px">
        <div class="tabs">
          <div class="tab active" onclick="switchTab('tab-active')"><i class="fas fa-spinner"></i> Active (3)</div>
          <div class="tab" onclick="switchTab('tab-completed')"><i class="fas fa-check-circle"></i> Completed (8)</div>
          <div class="tab" onclick="switchTab('tab-cancelled')"><i class="fas fa-times-circle"></i> Cancelled (0)</div>
        </div>
        <div class="tab-content active" id="tab-active">
          <div class="card">
            <table class="data-table">
              <thead><tr>
                <th>Order</th><th>Design</th><th>Provider</th><th>Qty</th><th>Total</th><th>Lead Time</th><th>Status</th><th></th>
              </tr></thead>
              <tbody>
                <tr>
                  <td class="text-mono text-sm">#SF-0041</td>
                  <td><div class="file-name">bracket_v3</div><div style="font-size:11px;color:var(--text-dim)">Mild Steel 2mm</div></td>
                  <td><div style="font-size:13px;font-weight:500">MetalTech GmbH</div><div style="font-size:11px;color:var(--text-dim)"><i class="fas fa-map-marker-alt"></i> Germany</div></td>
                  <td>50</td>
                  <td class="text-success text-mono">$312</td>
                  <td><span style="font-size:12px">12 days left</span></td>
                  <td><span class="badge badge-processing">In Production</span></td>
                  <td><button class="btn btn-sm"><i class="fas fa-eye"></i></button></td>
                </tr>
                <tr>
                  <td class="text-mono text-sm">#SF-0039</td>
                  <td><div class="file-name">enclosure_front</div><div style="font-size:11px;color:var(--text-dim)">Al 5052 3mm</div></td>
                  <td><div style="font-size:13px;font-weight:500">PrecisionCut SG</div><div style="font-size:11px;color:var(--text-dim)"><i class="fas fa-map-marker-alt"></i> Singapore</div></td>
                  <td>10</td>
                  <td class="text-success text-mono">$198</td>
                  <td><span style="font-size:12px">5 days left</span></td>
                  <td><span class="badge badge-ready">QC Check</span></td>
                  <td><button class="btn btn-sm"><i class="fas fa-eye"></i></button></td>
                </tr>
                <tr>
                  <td class="text-mono text-sm">#SF-0038</td>
                  <td><div class="file-name">panel_chassis</div><div style="font-size:11px;color:var(--text-dim)">SS304 1.5mm</div></td>
                  <td><div style="font-size:13px;font-weight:500">SheetWorks IN</div><div style="font-size:11px;color:var(--text-dim)"><i class="fas fa-map-marker-alt"></i> India</div></td>
                  <td>100</td>
                  <td class="text-success text-mono">$780</td>
                  <td><span style="font-size:12px">18 days left</span></td>
                  <td><span class="badge badge-pending">Confirmed</span></td>
                  <td><button class="btn btn-sm"><i class="fas fa-eye"></i></button></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="tab-content" id="tab-completed">
          <div style="text-align:center;padding:60px;color:var(--text-dim)">
            <i class="fas fa-check-circle" style="font-size:48px;margin-bottom:16px;display:block;color:var(--success)"></i>
            8 completed orders — load from server
          </div>
        </div>
        <div class="tab-content" id="tab-cancelled">
          <div style="text-align:center;padding:60px;color:var(--text-dim)">
            <i class="fas fa-times-circle" style="font-size:48px;margin-bottom:16px;display:block;color:var(--danger)"></i>
            No cancelled orders
          </div>
        </div>
      </div>
    </div>

    <!-- ===== CLOUDINARY ASSETS ===== -->
    <div class="page" id="page-cloudinary">
      <div class="topbar">
        <div>
          <div class="topbar-title">Cloud Assets</div>
          <div class="topbar-sub">Manage your Cloudinary-hosted design images</div>
        </div>
        <div class="topbar-actions">
          <div style="font-size:12px;color:var(--text-dim);display:flex;align-items:center;gap:8px">
            <i class="fas fa-circle" style="color:var(--success);font-size:8px"></i>
            Connected to Cloudinary
          </div>
          <button class="btn"><i class="fas fa-sync-alt"></i> Sync</button>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-3 mb-24">
          <div class="stat-card">
            <div class="stat-label">Stored Assets</div>
            <div class="stat-value">18</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Storage Used</div>
            <div class="stat-value text-accent" style="color:var(--accent)">24 MB</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">MongoDB Links</div>
            <div class="stat-value" style="color:var(--accent2)">18</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fas fa-images text-accent"></i> Saved Designs</div>
          </div>
          <div class="card-body" style="padding:0">
            <table class="data-table">
              <thead><tr>
                <th>Design</th><th>Cloudinary URL</th><th>MongoDB ID</th><th>Size</th><th>Date Saved</th><th></th>
              </tr></thead>
              <tbody id="cloudinary-table">
                <!-- populated by JS -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== SETTINGS ===== -->
    <div class="page" id="page-settings">
      <div class="topbar">
        <div>
          <div class="topbar-title">Settings</div>
          <div class="topbar-sub">Configure your account and integrations</div>
        </div>
        <div class="topbar-actions">
          <button class="btn btn-accent" onclick="saveSettings()"><i class="fas fa-save"></i> Save Changes</button>
        </div>
      </div>
      <div style="padding:28px">
        <div class="grid-2">
          <div>
            <div class="card mb-20">
              <div class="card-header"><div class="card-title"><i class="fas fa-user text-accent"></i> Profile</div></div>
              <div class="card-body">
                <div class="form-row">
                  <div class="form-group">
                    <label class="form-label">First Name</label>
                    <input class="form-input" value="Demo">
                  </div>
                  <div class="form-group">
                    <label class="form-label">Last Name</label>
                    <input class="form-input" value="Designer">
                  </div>
                </div>
                <div class="form-group">
                  <label class="form-label">Company</label>
                  <input class="form-input" value="Acme Engineering">
                </div>
                <div class="form-group">
                  <label class="form-label">Email</label>
                  <input class="form-input" value="demo@sheetforge.io">
                </div>
                <div class="form-group">
                  <label class="form-label">Country</label>
                  <select class="form-select">
                    <option>Kenya</option>
                    <option>Germany</option>
                    <option>United States</option>
                    <option>Singapore</option>
                    <option>India</option>
                  </select>
                </div>
              </div>
            </div>
            <div class="card">
              <div class="card-header"><div class="card-title"><i class="fas fa-bell text-accent"></i> Notifications</div></div>
              <div class="card-body">
                <div style="display:flex;flex-direction:column;gap:12px">
                  <div class="checkbox-row" onclick="toggleCheck(this)">
                    <div class="checkbox checked"><i class="fas fa-check"></i></div>
                    <div><div style="font-size:13px;font-weight:500">Quote received</div><div class="text-dim text-sm">When a provider sends a quote</div></div>
                  </div>
                  <div class="checkbox-row" onclick="toggleCheck(this)">
                    <div class="checkbox checked"><i class="fas fa-check"></i></div>
                    <div><div style="font-size:13px;font-weight:500">DWG conversion complete</div><div class="text-dim text-sm">When AI finishes processing</div></div>
                  </div>
                  <div class="checkbox-row" onclick="toggleCheck(this)">
                    <div class="checkbox checked"><i class="fas fa-check"></i></div>
                    <div><div style="font-size:13px;font-weight:500">Order status updates</div><div class="text-dim text-sm">Production milestones</div></div>
                  </div>
                  <div class="checkbox-row" onclick="toggleCheck(this)">
                    <div class="checkbox"><i class="fas fa-check"></i></div>
                    <div><div style="font-size:13px;font-weight:500">Marketing emails</div><div class="text-dim text-sm">News and feature updates</div></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div>
            <div class="card mb-20">
              <div class="card-header"><div class="card-title"><i class="fas fa-cloud text-accent"></i> Integrations</div></div>
              <div class="card-body">
                <div style="display:flex;flex-direction:column;gap:16px">
                  <div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg);border-radius:var(--r);border:1px solid var(--border)">
                    <div style="width:36px;height:36px;background:#3448c5;border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:14px">C</div>
                    <div style="flex:1"><div style="font-weight:500;font-size:13px">Cloudinary</div><div class="text-dim text-sm">Cloud: cloud-sheetforge-demo</div></div>
                    <span class="badge badge-ready">Connected</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg);border-radius:var(--r);border:1px solid var(--border)">
                    <div style="width:36px;height:36px;background:#00ed64;border-radius:8px;display:flex;align-items:center;justify-content:center;color:black;font-weight:700;font-size:14px">M</div>
                    <div style="flex:1"><div style="font-weight:500;font-size:13px">MongoDB Atlas</div><div class="text-dim text-sm">Cluster: sheetforge-prod.a1b2.mongodb.net</div></div>
                    <span class="badge badge-ready">Connected</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg);border-radius:var(--r);border:1px solid var(--border)">
                    <div style="width:36px;height:36px;background:#0070f3;border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:12px">AI</div>
                    <div style="flex:1"><div style="font-weight:500;font-size:13px">Anthropic Claude API</div><div class="text-dim text-sm">Used for DWG conversion analysis</div></div>
                    <span class="badge badge-ready">Active</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg);border-radius:var(--r);border:1px solid var(--border)">
                    <div style="width:36px;height:36px;background:#6366f1;border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:12px">ST</div>
                    <div style="flex:1"><div style="font-weight:500;font-size:13px">Stripe Payments</div><div class="text-dim text-sm">For provider payments</div></div>
                    <span class="badge badge-pending">Setup Required</span>
                  </div>
                </div>

                <div class="divider"></div>
                <div class="form-group mb-0">
                  <label class="form-label">Cloudinary Cloud Name</label>
                  <input class="form-input" id="cloud-name" value="cloud-sheetforge-demo">
                </div>
                <div class="form-group mb-0" style="margin-top:12px">
                  <label class="form-label">Cloudinary API Key</label>
                  <input class="form-input" id="cloud-api-key" type="password" value="••••••••••••••">
                </div>
                <div class="form-group mb-0" style="margin-top:12px">
                  <label class="form-label">MongoDB URI</label>
                  <input class="form-input" id="mongo-uri" type="password" value="••••••••••••••">
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div><!-- end main-content -->
</div><!-- end app -->

<!-- CLOUD SAVE MODAL -->
<div class="modal-overlay" id="cloud-save-modal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title"><i class="fas fa-cloud-upload-alt text-accent" style="margin-right:10px"></i>Save to Cloudinary & MongoDB</div>
      <button class="modal-close" onclick="closeModal('cloud-save-modal')"><i class="fas fa-times"></i></button>
    </div>
    <div class="modal-body">
      <div class="analysis-card mb-16">
        <div class="analysis-header" style="margin-bottom:10px">
          <i class="fas fa-info-circle"></i> What will be saved
        </div>
        <ul style="list-style:none;display:flex;flex-direction:column;gap:8px;font-size:13px">
          <li class="flex-center gap-8"><i class="fas fa-check-circle text-success"></i> DWG drawing exported as high-res PNG (300dpi)</li>
          <li class="flex-center gap-8"><i class="fas fa-check-circle text-success"></i> Uploaded to Cloudinary under <code style="font-family:var(--font-mono);font-size:11px;background:var(--bg);padding:2px 6px;border-radius:4px">sheetforge/designs/</code></li>
          <li class="flex-center gap-8"><i class="fas fa-check-circle text-success"></i> Cloudinary URL + metadata saved to MongoDB</li>
          <li class="flex-center gap-8"><i class="fas fa-check-circle text-success"></i> Design linked to your account for quote requests</li>
        </ul>
      </div>

      <div class="form-group">
        <label class="form-label">Asset Name</label>
        <input class="form-input" id="save-asset-name" value="bracket_v3_approved" placeholder="e.g. bracket_v3_approved">
      </div>
      <div class="form-group">
        <label class="form-label">Folder</label>
        <select class="form-select">
          <option>sheetforge/designs/2026</option>
          <option>sheetforge/approved</option>
          <option>custom...</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Tags (comma-separated)</label>
        <input class="form-input" value="bracket, mild-steel, approved, CNC">
      </div>

      <div id="save-progress-area" style="display:none">
        <div class="divider"></div>
        <div style="font-size:13px;font-weight:600;margin-bottom:12px">Upload Progress</div>
        <div class="cloud-status" id="save-step-1">
          <div class="spinner" id="save-spinner-1" style="width:16px;height:16px;border-width:2px"></div>
          <div>Rendering DWG to PNG...</div>
        </div>
        <div style="height:8px"></div>
        <div class="cloud-status" id="save-step-2" style="opacity:0.4">
          <i class="fas fa-circle" style="font-size:8px;color:var(--text-muted)"></i>
          <div>Uploading to Cloudinary...</div>
        </div>
        <div style="height:8px"></div>
        <div class="cloud-status" id="save-step-3" style="opacity:0.4">
          <i class="fas fa-circle" style="font-size:8px;color:var(--text-muted)"></i>
          <div>Saving link to MongoDB...</div>
        </div>
        <div style="height:16px"></div>
        <div id="save-result" style="display:none">
          <div style="font-size:13px;font-weight:600;color:var(--success);margin-bottom:8px"><i class="fas fa-check-circle"></i> Saved Successfully!</div>
          <div class="cloud-url">
            <span style="flex:1" id="saved-url">https://res.cloudinary.com/cloud-sheetforge-demo/image/upload/v1748000000/sheetforge/designs/2026/bracket_v3_approved.png</span>
            <button class="copy-btn" onclick="copyURL()">Copy</button>
          </div>
          <div style="margin-top:8px;font-size:11px;color:var(--text-dim)">MongoDB ID: <span class="text-mono" style="color:var(--accent2)">683f4a9c2d8e5b1c7a0f9e32</span></div>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="closeModal('cloud-save-modal')">Cancel</button>
      <button class="btn btn-accent" id="do-save-btn" onclick="doCloudSave()"><i class="fas fa-cloud-upload-alt"></i> Upload & Save</button>
    </div>
  </div>
</div>

<!-- PROVIDER DETAIL MODAL -->
<div class="modal-overlay" id="provider-modal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title" id="modal-provider-name">MetalTech GmbH</div>
      <button class="modal-close" onclick="closeModal('provider-modal')"><i class="fas fa-times"></i></button>
    </div>
    <div class="modal-body" id="modal-provider-body">
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="closeModal('provider-modal')">Close</button>
      <button class="btn btn-accent" onclick="placeOrder()"><i class="fas fa-shopping-cart"></i> Place Order</button>
    </div>
  </div>
</div>

<script>
// ================================================================
// APP STATE
// ================================================================
const APP = {
  user: null,
  currentPage: 'dashboard',
  dwgScale: 1,
  dwgOffX: 0, dwgOffY: 0,
  uploadedFile: null,
  currentDwg: 'bracket',
  approvedDwgs: new Set(['bracket', 'enclosure']),
  cloudAssets: [
    { name: 'bracket_v2_final', url: 'https://res.cloudinary.com/demo/image/upload/v1748000000/sheetforge/bracket_v2_final.png', mongoId: '683f4a9c2d8e5b1c7a0f9e30', size: '1.2 MB', date: 'May 20, 2026' },
    { name: 'enclosure_sheet1', url: 'https://res.cloudinary.com/demo/image/upload/v1748000000/sheetforge/enclosure_sheet1.png', mongoId: '683f4a9c2d8e5b1c7a0f9e31', size: '980 KB', date: 'May 18, 2026' },
  ],
  selectedProvider: null,
  selectedMaterial: 'steel',
  activities: [
    { icon: 'fas fa-check-circle', color: 'var(--success)', text: '<strong>bracket_v3.dwg</strong> approved and saved to Cloudinary', time: '2 min ago' },
    { icon: 'fas fa-robot', color: 'var(--accent)', text: 'AI conversion completed for <strong>enclosure_front.png</strong>', time: '1 hr ago' },
    { icon: 'fas fa-tag', color: 'var(--purple)', text: 'New quote received from <strong>MetalTech GmbH</strong> — $312 for 50 units', time: '3 hrs ago' },
    { icon: 'fas fa-box', color: 'var(--accent2)', text: 'Order <strong>#SF-0041</strong> entered production at MetalTech GmbH', time: 'Yesterday' },
    { icon: 'fas fa-cloud-upload-alt', color: 'var(--accent)', text: '<strong>panel_chassis.png</strong> uploaded and queued for conversion', time: '2 days ago' },
  ]
};

const PROVIDERS = [
  { id: 1, name: 'MetalTech GmbH', country: 'Germany', flag: '🇩🇪', region: 'Europe', price: 312, perUnit: 6.24, rating: 4.9, reviews: 214, lead: '10-14d', capacity: '500t/mo', certs: 'ISO 9001, CE', color: '#1a1a2e', text: 'MT', bg: 'linear-gradient(135deg,#232b4a,#1a2038)', orders: 1420, specialty: 'Precision laser cutting, tight tolerances' },
  { id: 2, name: 'PrecisionCut SG', country: 'Singapore', flag: '🇸🇬', region: 'Asia Pacific', price: 285, perUnit: 5.70, rating: 4.8, reviews: 189, lead: '7-10d', capacity: '200t/mo', certs: 'ISO 9001', color: '#1a2a1a', text: 'PC', bg: 'linear-gradient(135deg,#1a3a2a,#122a1a)', orders: 870, specialty: 'Aluminium fabrication, anodizing' },
  { id: 3, name: 'SheetWorks IN', country: 'India', flag: '🇮🇳', region: 'Asia Pacific', price: 198, perUnit: 3.96, rating: 4.7, reviews: 301, lead: '14-21d', capacity: '1000t/mo', certs: 'ISO 9001, IATF 16949', color: '#2a1a1a', text: 'SW', bg: 'linear-gradient(135deg,#3a1a1a,#2a1010)', orders: 2310, specialty: 'High-volume production, competitive pricing', bestValue: true },
  { id: 4, name: 'AlphaBend AU', country: 'Australia', flag: '🇦🇺', region: 'Asia Pacific', price: 344, perUnit: 6.88, rating: 4.7, reviews: 98, lead: '5-7d', capacity: '150t/mo', certs: 'AS/NZS 1665', color: '#1a1a1a', text: 'AB', bg: 'linear-gradient(135deg,#2a2a1a,#1a1a10)', orders: 540, specialty: 'Express delivery, complex profiles' },
  { id: 5, name: 'EuroForm PL', country: 'Poland', flag: '🇵🇱', region: 'Europe', price: 267, perUnit: 5.34, rating: 4.6, reviews: 156, lead: '10-12d', capacity: '400t/mo', certs: 'ISO 9001, ISO 14001', color: '#1a1a2a', text: 'EF', bg: 'linear-gradient(135deg,#1a1a3a,#101028)', orders: 982, specialty: 'Steel & stainless, powder coating' },
  { id: 6, name: 'CutMaster CN', country: 'China', flag: '🇨🇳', region: 'Asia Pacific', price: 156, perUnit: 3.12, rating: 4.4, reviews: 512, lead: '14-28d', capacity: '5000t/mo', certs: 'ISO 9001', color: '#1a1a1a', text: 'CM', bg: 'linear-gradient(135deg,#2a1a1a,#1a1010)', orders: 8420, specialty: 'Ultra-high volume, all materials' },
  { id: 7, name: 'Fabricate ZA', country: 'South Africa', flag: '🇿🇦', region: 'Africa', price: 224, perUnit: 4.48, rating: 4.5, reviews: 67, lead: '12-16d', capacity: '300t/mo', certs: 'SANS 10147', color: '#1a2a1a', text: 'FZ', bg: 'linear-gradient(135deg,#1a3a2a,#0e2a1a)', orders: 340, specialty: 'Structural steel, galvanizing' },
  { id: 8, name: 'ProSheet US', country: 'United States', flag: '🇺🇸', region: 'North America', price: 398, perUnit: 7.96, rating: 4.8, reviews: 244, lead: '5-8d', capacity: '250t/mo', certs: 'ISO 9001, ASME', color: '#1a1a2a', text: 'PS', bg: 'linear-gradient(135deg,#1a1a3a,#0e0e28)', orders: 1150, specialty: 'Aerospace-grade quality, ITAR compliant' },
  { id: 9, name: 'GulfMetal AE', country: 'UAE', flag: '🇦🇪', region: 'Middle East', price: 289, perUnit: 5.78, rating: 4.6, reviews: 88, lead: '8-12d', capacity: '400t/mo', certs: 'ISO 9001', color: '#2a1a1a', text: 'GM', bg: 'linear-gradient(135deg,#3a2a1a,#281810)', orders: 420, specialty: 'Structural fabrication, site delivery' },
];

const DRAWINGS = [
  { id: 1, name: 'bracket_v3.dwg', original: 'bracket_v3.pdf', material: 'Mild Steel', thickness: '2mm', status: 'approved', date: 'Today' },
  { id: 2, name: 'enclosure_front.dwg', original: 'enclosure_front.png', material: 'Al 5052', thickness: '3mm', status: 'approved', date: 'Yesterday' },
  { id: 3, name: 'panel_chassis.dwg', original: 'panel_chassis.jpg', material: 'SS 304', thickness: '1.5mm', status: 'processing', date: '2 days ago' },
  { id: 4, name: 'hinge_bracket.dwg', original: 'hinge_sketch.jpg', material: 'Mild Steel', thickness: '3mm', status: 'ready', date: 'May 18' },
  { id: 5, name: 'cover_plate.dwg', original: 'cover.png', material: 'Al 6061', thickness: '1mm', status: 'pending', date: 'May 17' },
  { id: 6, name: 'support_arm.dwg', original: 'arm_rough.pdf', material: 'Stainless 316', thickness: '4mm', status: 'approved', date: 'May 15' },
];

// ================================================================
// AUTH
// ================================================================
function switchAuthTab(tab) {
  document.getElementById('login-form').style.display = tab === 'login' ? 'block' : 'none';
  document.getElementById('signup-form').style.display = tab === 'signup' ? 'block' : 'none';
  document.querySelectorAll('.auth-tab').forEach((t, i) => {
    t.classList.toggle('active', (i === 0 && tab === 'login') || (i === 1 && tab === 'signup'));
  });
}

function selectRole(r) {
  document.getElementById('role-designer').classList.toggle('active', r === 'designer');
  document.getElementById('role-provider').classList.toggle('active', r === 'provider');
}

function doLogin() {
  const email = document.getElementById('login-email').value;
  if (!email) { showNotif('error', 'Error', 'Please enter your email'); return; }
  loginUser({ name: 'Demo Designer', email, role: 'designer', initials: 'DD' });
}

function demoLogin(role) {
  if (role === 'designer') loginUser({ name: 'Alex Kimani', email: 'alex@acmeeng.co.ke', role: 'Designer', initials: 'AK' });
  else loginUser({ name: 'Marcus Weber', email: 'm.weber@metaltech.de', role: 'CNC Provider', initials: 'MW' });
}

function doSignup() {
  showNotif('success', 'Account Created', 'Welcome to SheetForge!');
  setTimeout(() => demoLogin('designer'), 800);
}

function loginUser(user) {
  APP.user = user;
  document.getElementById('auth-screen').style.display = 'none';
  document.getElementById('app').classList.add('visible');
  document.getElementById('user-avatar-sidebar').textContent = user.initials;
  document.getElementById('sidebar-user-name').textContent = user.name;
  document.getElementById('sidebar-user-role').textContent = user.role;
  document.getElementById('dashboard-user-name').textContent = user.name.split(' ')[0];
  initApp();
}

function doLogout() {
  APP.user = null;
  document.getElementById('app').classList.remove('visible');
  document.getElementById('auth-screen').style.display = 'flex';
}

// ================================================================
// INIT
// ================================================================
function initApp() {
  renderActivityFeed();
  renderDrawingsGrid();
  renderProvidersGrid();
  renderTopProviders();
  renderCloudinaryTable();
  renderLayerList();
  initDwgCanvas();
  setupDragDrop();
}

// ================================================================
// NAVIGATION
// ================================================================
function showPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  const nav = document.getElementById('nav-' + page);
  if (nav) nav.classList.add('active');
  APP.currentPage = page;
  if (page === 'viewer') setTimeout(() => { initDwgCanvas(); loadDwg(APP.currentDwg); }, 100);
}

// ================================================================
// ACTIVITY FEED
// ================================================================
function renderActivityFeed() {
  const feed = document.getElementById('activity-feed');
  feed.innerHTML = APP.activities.map(a => \`
    <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 0;border-bottom:1px solid var(--border)">
      <div style="width:30px;height:30px;border-radius:8px;background:rgba(255,255,255,0.05);display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <i class="\${a.icon}" style="font-size:12px;color:\${a.color}"></i>
      </div>
      <div style="flex:1">
        <div style="font-size:13px;line-height:1.5">\${a.text}</div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:2px">\${a.time}</div>
      </div>
    </div>
  \`).join('');
}

// ================================================================
// DRAWINGS GRID
// ================================================================
function renderDrawingsGrid(filter = '') {
  const grid = document.getElementById('drawings-grid');
  const filtered = DRAWINGS.filter(d => !filter || d.name.toLowerCase().includes(filter.toLowerCase()));
  grid.innerHTML = filtered.map(d => {
    const statusClass = \`badge-\${d.status}\`;
    const statusText = d.status.charAt(0).toUpperCase() + d.status.slice(1);
    return \`
      <div class="card" style="cursor:pointer;transition:all 0.2s" onmouseenter="this.style.borderColor='var(--border-hi)'" onmouseleave="this.style.borderColor='var(--border)'">
        <div style="height:140px;background:var(--bg);display:flex;align-items:center;justify-content:center;border-radius:var(--r-lg) var(--r-lg) 0 0;border-bottom:1px solid var(--border);overflow:hidden">
          <canvas id="thumb-\${d.id}" width="200" height="140"></canvas>
        </div>
        <div class="card-body">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px">
            <div>
              <div style="font-size:13px;font-weight:600">\${d.name}</div>
              <div style="font-size:11px;color:var(--text-dim)">\${d.original}</div>
            </div>
            <span class="badge \${statusClass}">\${statusText}</span>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
            <span class="format-tag">\${d.material}</span>
            <span class="format-tag">\${d.thickness}</span>
          </div>
          <div style="display:flex;gap-8:0;gap:8px">
            <button class="btn btn-sm flex-1" onclick="APP.currentDwg='\${d.name.replace('.dwg','')}';showPage('viewer')"><i class="fas fa-eye"></i> View</button>
            \${d.status === 'approved' ? \`<button class="btn btn-sm" onclick="openCloudSaveModal()"><i class="fas fa-cloud"></i></button>\` : ''}
            <button class="btn btn-sm"><i class="fas fa-ellipsis-h"></i></button>
          </div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:8px">\${d.date}</div>
        </div>
      </div>\`;
  }).join('');
  filtered.forEach(d => setTimeout(() => drawThumb(d.id, d.status), 50));
}

function filterDrawings(val) { renderDrawingsGrid(val); }

function drawThumb(id, status) {
  const c = document.getElementById('thumb-' + id);
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, 200, 140);
  ctx.fillStyle = '#0f1217';
  ctx.fillRect(0, 0, 200, 140);
  ctx.strokeStyle = status === 'approved' ? '#10b981' : status === 'processing' ? '#3d7eff' : '#4a5568';
  ctx.lineWidth = 1;
  // Draw simple part outline
  ctx.beginPath();
  ctx.rect(30, 20, 140, 100);
  ctx.stroke();
  ctx.beginPath();
  ctx.rect(50, 35, 30, 30);
  ctx.stroke();
  ctx.beginPath();
  ctx.rect(120, 35, 30, 30);
  ctx.stroke();
  // Bend lines
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(30, 60); ctx.lineTo(170, 60); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(90, 20); ctx.lineTo(90, 120); ctx.stroke();
  ctx.setLineDash([]);
  // Status indicator
  ctx.fillStyle = status === 'approved' ? '#10b981' : status === 'processing' ? '#3d7eff' : '#f59e0b';
  ctx.beginPath(); ctx.arc(185, 15, 5, 0, Math.PI * 2); ctx.fill();
}

// ================================================================
// PROVIDERS
// ================================================================
function renderProvidersGrid(list = PROVIDERS) {
  const grid = document.getElementById('providers-grid');
  grid.innerHTML = list.map(p => \`
    <div class="provider-card \${p.bestValue ? 'best-value' : ''}" id="prov-\${p.id}" onclick="openProvider(\${p.id})">
      <div class="provider-head">
        <div class="provider-avatar" style="background:\${p.bg};color:white">\${p.text}</div>
        <div>
          <div class="provider-name">\${p.flag} \${p.name}</div>
          <div class="provider-location"><i class="fas fa-map-marker-alt"></i> \${p.country} • \${p.region}</div>
        </div>
      </div>
      <div class="star-rating">\${'★'.repeat(Math.round(p.rating))}\${'☆'.repeat(5 - Math.round(p.rating))} <span style="color:var(--text-dim);font-size:11px">(\${p.reviews})</span></div>
      <div style="margin-top:12px;display:flex;align-items:flex-end;justify-content:space-between">
        <div>
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">Est. for 50 units</div>
          <div class="provider-price">$\${p.price} <span>total</span></div>
          <div style="font-size:11px;color:var(--text-dim);margin-top:2px">$\${p.perUnit.toFixed(2)}/unit</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:12px;color:var(--text-dim)"><i class="fas fa-clock"></i> \${p.lead}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:2px">\${p.certs.split(',')[0]}</div>
        </div>
      </div>
      <div class="provider-stats">
        <div class="provider-stat">
          <div class="provider-stat-val">\${p.orders.toLocaleString()}</div>
          <div class="provider-stat-label">Orders</div>
        </div>
        <div class="provider-stat">
          <div class="provider-stat-val">\${p.capacity}</div>
          <div class="provider-stat-label">Capacity</div>
        </div>
        <div class="provider-stat">
          <div class="provider-stat-val">\${p.rating}</div>
          <div class="provider-stat-label">Rating</div>
        </div>
      </div>
    </div>\`).join('');
}

function renderTopProviders() {
  const el = document.getElementById('top-providers-preview');
  const top = [...PROVIDERS].sort((a, b) => a.price - b.price).slice(0, 3);
  el.innerHTML = top.map((p, i) => \`
    <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)">
      <div style="width:32px;height:32px;border-radius:8px;background:\${p.bg};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:white">\${p.text}</div>
      <div style="flex:1">
        <div style="font-size:13px;font-weight:500">\${p.flag} \${p.name}</div>
        <div style="font-size:11px;color:var(--text-dim)">\${p.lead} • \${p.rating}★</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:14px;font-weight:700;color:var(--accent2)">$\${p.price}</div>
        <div style="font-size:10px;color:var(--text-muted)">50 units</div>
      </div>
    </div>\`).join('');
}

function filterProviders(val) {
  const list = val ? PROVIDERS.filter(p => p.name.toLowerCase().includes(val.toLowerCase()) || p.country.toLowerCase().includes(val.toLowerCase())) : PROVIDERS;
  renderProvidersGrid(list);
}

function filterByRegion(region) {
  const list = region ? PROVIDERS.filter(p => p.region === region) : PROVIDERS;
  renderProvidersGrid(list);
}

function sortProviders(by) {
  const sorted = [...PROVIDERS].sort((a, b) => {
    if (by === 'price') return a.price - b.price;
    if (by === 'rating') return b.rating - a.rating;
    if (by === 'lead') return parseInt(a.lead) - parseInt(b.lead);
    return 0;
  });
  renderProvidersGrid(sorted);
}

function openProvider(id) {
  const p = PROVIDERS.find(x => x.id === id);
  APP.selectedProvider = p;
  document.getElementById('modal-provider-name').innerHTML = \`\${p.flag} \${p.name}\`;
  document.getElementById('modal-provider-body').innerHTML = \`
    <div class="analysis-card mb-16">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div>
          <div style="font-size:22px;font-weight:800;font-family:var(--font-display);color:var(--accent2)">$\${p.price}</div>
          <div style="font-size:12px;color:var(--text-dim)">for 50 units • $\${p.perUnit.toFixed(2)}/unit</div>
        </div>
        <div class="star-rating" style="font-size:16px">\${'★'.repeat(Math.round(p.rating))}<span style="font-size:13px;color:var(--text-dim);font-family:var(--font-main);margin-left:6px">\${p.rating} (\${p.reviews} reviews)</span></div>
      </div>
      <div class="form-row">
        <div style="text-align:center;padding:12px;background:var(--bg);border-radius:var(--r)">
          <div style="font-size:20px;font-weight:700;font-family:var(--font-display)">\${p.lead}</div>
          <div style="font-size:11px;color:var(--text-muted)">Lead Time</div>
        </div>
        <div style="text-align:center;padding:12px;background:var(--bg);border-radius:var(--r)">
          <div style="font-size:20px;font-weight:700;font-family:var(--font-display)">\${p.capacity}</div>
          <div style="font-size:11px;color:var(--text-muted)">Capacity</div>
        </div>
      </div>
    </div>
    <table class="data-table">
      <tr><td class="text-dim">Country</td><td>\${p.flag} \${p.country}</td></tr>
      <tr><td class="text-dim">Region</td><td>\${p.region}</td></tr>
      <tr><td class="text-dim">Certifications</td><td>\${p.certs}</td></tr>
      <tr><td class="text-dim">Specialty</td><td>\${p.specialty}</td></tr>
      <tr><td class="text-dim">Total Orders</td><td>\${p.orders.toLocaleString()}</td></tr>
    </table>
    <div class="divider"></div>
    <div style="background:var(--bg);border-radius:var(--r);padding:14px;font-size:13px">
      <strong>Quality Assurance:</strong> All parts include CMM inspection report, material certificates (EN 10204 3.1 / Mill cert), and photographic documentation.
    </div>\`;
  document.getElementById('provider-modal').classList.add('open');
}

// ================================================================
// CLOUDINARY TABLE
// ================================================================
function renderCloudinaryTable() {
  const tbody = document.getElementById('cloudinary-table');
  tbody.innerHTML = APP.cloudAssets.map(a => \`
    <tr>
      <td>
        <div style="display:flex;align-items:center;gap:10px">
          <div class="file-icon img"><i class="fas fa-image"></i></div>
          <div>
            <div class="file-name">\${a.name}.png</div>
            <div class="file-size">\${a.size}</div>
          </div>
        </div>
      </td>
      <td>
        <div class="cloud-url" style="font-size:10px;padding:5px 8px;max-width:280px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;display:block">
          \${a.url}
        </div>
      </td>
      <td class="text-mono" style="font-size:11px;color:var(--accent2)">\${a.mongoId}</td>
      <td>\${a.size}</td>
      <td class="text-dim text-sm">\${a.date}</td>
      <td>
        <div style="display:flex;gap:6px">
          <button class="btn btn-sm btn-icon" onclick="navigator.clipboard.writeText('\${a.url}');showNotif('success','Copied','URL copied to clipboard')"><i class="fas fa-copy"></i></button>
          <button class="btn btn-sm btn-icon"><i class="fas fa-external-link-alt"></i></button>
        </div>
      </td>
    </tr>\`).join('');
}

// ================================================================
// UPLOAD & CONVERSION
// ================================================================
function setupDragDrop() {
  const zone = document.getElementById('upload-zone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); });
}

function handleFileSelect(e) {
  if (e.target.files[0]) processFile(e.target.files[0]);
}

function processFile(file) {
  APP.uploadedFile = file;
  const ext = file.name.split('.').pop().toLowerCase();
  document.getElementById('file-info-panel').style.display = 'block';
  document.getElementById('design-details-form').style.display = 'block';
  document.getElementById('upload-file-name').textContent = file.name;
  document.getElementById('upload-file-size').textContent = \`\${(file.size/1024/1024).toFixed(1)} MB • \${ext.toUpperCase()}\`;
  document.getElementById('dwg-filename').textContent = file.name.replace(/\\.[^.]+$/, '') + '.dwg';

  const icon = document.getElementById('upload-file-icon');
  icon.className = \`file-icon \${ext === 'pdf' ? 'pdf' : 'img'}\`;
  icon.innerHTML = \`<i class="fas fa-file-\${ext === 'pdf' ? 'pdf' : 'image'}"></i>\`;

  const previewImg = document.getElementById('upload-preview-img');
  const placeholder = document.getElementById('upload-preview-placeholder');
  if (['png','jpg','jpeg','gif','webp','bmp','svg'].includes(ext)) {
    const reader = new FileReader();
    reader.onload = e => { previewImg.src = e.target.result; previewImg.style.display = 'block'; placeholder.style.display = 'none'; };
    reader.readAsDataURL(file);
  } else {
    previewImg.style.display = 'none';
    placeholder.style.display = 'block';
    placeholder.innerHTML = \`<i class="fas fa-file-\${ext === 'pdf' ? 'pdf' : 'alt'}" style="font-size:32px;margin-bottom:8px;display:block;color:\${ext === 'pdf' ? '#f87171' : 'var(--accent)'}"></i>\${ext.toUpperCase()} file uploaded\`;
  }
  // Advance step
  setStep(1);
  showNotif('success', 'File Ready', \`\${file.name} loaded successfully\`);
}

function setStep(n) {
  for (let i = 1; i <= 5; i++) {
    const dot = document.getElementById('step' + i);
    const lbl = dot.nextElementSibling;
    dot.className = 'step-dot' + (i < n ? ' done' : i === n ? ' active' : '');
    if (lbl) lbl.className = 'step-label' + (i < n ? ' done' : i === n ? ' active' : '');
  }
  for (let i = 1; i <= 4; i++) {
    const line = document.getElementById('line' + i);
    if (line) line.className = 'step-line' + (i < n ? ' done' : '');
  }
}

function startConversion() {
  if (!APP.uploadedFile) { showNotif('error', 'No File', 'Please upload a file first'); return; }
  document.getElementById('convert-btn').disabled = true;
  document.getElementById('convert-btn').innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Analyzing...';
  document.getElementById('ai-analysis-panel').style.display = 'block';
  document.getElementById('conversion-progress-panel').style.display = 'none';
  document.getElementById('dwg-result-panel').style.display = 'none';
  setStep(2);

  // Simulate AI analysis
  let step = 0;
  const analysisSteps = [
    'Detecting part geometry and edges...',
    'Identifying bend lines and fold directions...',
    'Measuring dimensions from reference scale...',
    'Extracting hole positions and diameters...',
    'Analyzing material thickness indicators...',
    'Generating DWG entities...',
  ];
  const content = document.getElementById('ai-analysis-content');
  content.innerHTML = '';
  const interval = setInterval(() => {
    step++;
    const div = document.createElement('div');
    div.style.cssText = 'font-size:12px;color:var(--text-dim);padding:3px 0;display:flex;align-items:center;gap:8px';
    div.innerHTML = \`<i class="fas fa-check-circle" style="color:var(--success);font-size:10px"></i>\${analysisSteps[step - 1]}\`;
    content.appendChild(div);
    if (step >= analysisSteps.length) {
      clearInterval(interval);
      document.getElementById('analysis-spinner').style.display = 'none';
      content.innerHTML += \`
        <div style="margin-top:12px;padding:10px;background:var(--bg);border-radius:var(--r);font-size:12px">
          <div style="color:var(--accent2);font-weight:600;margin-bottom:6px">Analysis Complete</div>
          <div style="color:var(--text-dim)">Detected: rectangular bracket with flanges, 4 bend lines (90°), 6× Ø8mm holes, uniform 2mm steel thickness</div>
        </div>\`;
      setTimeout(() => runConversion(), 600);
    }
  }, 450);
}

function runConversion() {
  setStep(3);
  document.getElementById('conversion-progress-panel').style.display = 'block';
  const convList = document.getElementById('conversion-steps-list');
  const convSteps = [
    { label: 'Vectorizing raster edges', detail: 'Using Canny edge detection' },
    { label: 'Normalizing coordinate system', detail: 'ISO A4 reference frame' },
    { label: 'Generating LINE entities', detail: '48 entities created' },
    { label: 'Adding CIRCLE entities', detail: '6 holes @ Ø8mm' },
    { label: 'Inserting dimension annotations', detail: 'ISO 2768-m tolerance' },
    { label: 'Writing DWG R2018 binary', detail: 'Finalizing file structure' },
  ];
  convList.innerHTML = convSteps.map(s => \`
    <div class="conv-step" style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px;opacity:0.3;transition:all 0.3s">
      <div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0;display:none"></div>
      <i class="fas fa-circle" style="font-size:8px;color:var(--text-muted);flex-shrink:0"></i>
      <div><div>\${s.label}</div><div style="font-size:10px;color:var(--text-muted)">\${s.detail}</div></div>
    </div>\`).join('');

  let i = 0;
  const rows = convList.querySelectorAll('.conv-step');
  const stepInterval = setInterval(() => {
    if (i > 0) {
      rows[i - 1].querySelector('.spinner').style.display = 'none';
      rows[i - 1].querySelector('i').style.cssText = 'font-size:10px;color:var(--success);flex-shrink:0';
      rows[i - 1].querySelector('i').className = 'fas fa-check-circle';
    }
    if (i < rows.length) {
      rows[i].style.opacity = '1';
      rows[i].querySelector('.spinner').style.display = 'block';
      rows[i].querySelector('i').style.display = 'none';
      i++;
    } else {
      clearInterval(stepInterval);
      document.getElementById('conv-status-badge').className = 'badge badge-ready';
      document.getElementById('conv-status-badge').textContent = 'Complete';
      const mat = document.getElementById('material-select') ? document.getElementById('material-select').value : 'Mild Steel';
      const thickness = document.getElementById('thickness-input') ? document.getElementById('thickness-input').value || '2' : '2';
      document.getElementById('dim-t').textContent = thickness + 'mm';
      document.getElementById('dwg-filename').textContent = (APP.uploadedFile?.name || 'part').replace(/\\.[^.]+$/, '') + '.dwg';
      document.getElementById('dwg-result-panel').style.display = 'block';
      document.getElementById('convert-btn').disabled = false;
      document.getElementById('convert-btn').innerHTML = '<i class="fas fa-magic"></i> Re-convert';
      setStep(4);
      showNotif('success', 'DWG Ready', 'Your drawing is ready for review!');
    }
  }, 700);
}

function approveDWG() {
  APP.approvedDwgs.add('new');
  document.getElementById('approve-btn').innerHTML = '<i class="fas fa-check-circle"></i> Approved!';
  document.getElementById('approve-btn').className = 'btn btn-success flex-1';
  setStep(5);
  showNotif('success', 'Design Approved', 'Ready to save to Cloudinary');
  setTimeout(() => openCloudSaveModal(), 1500);
}

function approveFromViewer() {
  APP.approvedDwgs.add(APP.currentDwg);
  document.getElementById('viewer-approve-btn').innerHTML = '<i class="fas fa-check-circle"></i> Approved';
  document.getElementById('viewer-approve-btn').className = 'btn btn-success';
  document.getElementById('viewer-save-btn').style.display = 'flex';
  document.getElementById('prop-status').className = 'badge badge-approved';
  document.getElementById('prop-status').textContent = 'Approved';
  showNotif('success', 'Drawing Approved', 'You can now save to Cloudinary');
}

// ================================================================
// CLOUD SAVE
// ================================================================
function openCloudSaveModal() {
  document.getElementById('save-progress-area').style.display = 'none';
  document.getElementById('save-result').style.display = 'none';
  document.getElementById('do-save-btn').disabled = false;
  document.getElementById('do-save-btn').innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload & Save';
  document.getElementById('cloud-save-modal').classList.add('open');
}

async function doCloudSave() {
  const name = document.getElementById('save-asset-name').value || 'design_export';
  document.getElementById('do-save-btn').disabled = true;
  document.getElementById('do-save-btn').innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Uploading...';
  document.getElementById('save-progress-area').style.display = 'block';

  // Step 1: Render PNG
  await delay(1000);
  const s1 = document.getElementById('save-step-1');
  s1.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>DWG rendered to PNG (1200×800px, 300dpi)</div>';
  s1.classList.add('done');

  // Step 2: Cloudinary upload
  const s2 = document.getElementById('save-step-2');
  s2.style.opacity = '1';
  s2.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0"></div><div>Uploading to Cloudinary...</div>';
  s2.classList.add('uploading');
  await delay(1500);
  s2.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Uploaded to Cloudinary successfully</div>';
  s2.classList.remove('uploading'); s2.classList.add('done');

  // Step 3: MongoDB
  const s3 = document.getElementById('save-step-3');
  s3.style.opacity = '1';
  s3.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0"></div><div>Saving link to MongoDB...</div>';
  await delay(900);
  s3.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Link saved to MongoDB Atlas</div>';
  s3.classList.add('done');

  // Show result
  const url = \`https://res.cloudinary.com/cloud-sheetforge-demo/image/upload/v\${Date.now()}/sheetforge/designs/2026/\${name}.png\`;
  const mongoId = Math.random().toString(16).slice(2, 26);
  document.getElementById('saved-url').textContent = url;
  document.getElementById('save-result').style.display = 'block';
  document.getElementById('do-save-btn').innerHTML = '<i class="fas fa-check"></i> Saved!';

  // Add to assets list
  APP.cloudAssets.unshift({ name, url, mongoId, size: '1.2 MB', date: 'Just now' });
  renderCloudinaryTable();
  showNotif('success', 'Saved to Cloud', 'Design URL stored in MongoDB');
}

function copyURL() {
  navigator.clipboard.writeText(document.getElementById('saved-url').textContent);
  showNotif('info', 'Copied', 'URL copied to clipboard');
}

// ================================================================
// DWG CANVAS VIEWER
// ================================================================
function initDwgCanvas() {
  const canvas = document.getElementById('dwg-canvas');
  if (!canvas) return;
  const container = canvas.parentElement;
  canvas.width = container.offsetWidth || 800;
  canvas.height = 480;
  APP.dwgScale = 1; APP.dwgOffX = 0; APP.dwgOffY = 0;
  loadDwg(APP.currentDwg || 'bracket');

  // Mouse interactions
  let dragging = false, lastX, lastY;
  canvas.onmousedown = e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.style.cursor = 'grabbing'; };
  canvas.onmouseup = () => { dragging = false; canvas.style.cursor = 'crosshair'; };
  canvas.onmousemove = e => {
    if (!dragging) return;
    APP.dwgOffX += e.clientX - lastX;
    APP.dwgOffY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    drawDWG();
  };
  canvas.onwheel = e => { e.preventDefault(); APP.dwgScale *= e.deltaY < 0 ? 1.1 : 0.9; APP.dwgScale = Math.max(0.2, Math.min(8, APP.dwgScale)); drawDWG(); document.getElementById('info-scale').textContent = APP.dwgScale.toFixed(2) + ':1'; };
}

function loadDwg(name) {
  APP.currentDwg = name;
  APP.dwgScale = 1; APP.dwgOffX = 0; APP.dwgOffY = 0;
  drawDWG();
  // Update info
  const configs = {
    bracket: { w: 240, h: 160, entities: 48 },
    enclosure: { w: 380, h: 280, entities: 72 },
    panel: { w: 600, h: 200, entities: 95 },
  };
  const cfg = configs[name] || configs.bracket;
  const iw = document.getElementById('info-w'), ih = document.getElementById('info-h');
  if (iw) iw.textContent = cfg.w;
  if (ih) ih.textContent = cfg.h;
  const vs = document.getElementById('viewer-approve-btn');
  if (vs && APP.approvedDwgs.has(name)) {
    vs.innerHTML = '<i class="fas fa-check-circle"></i> Approved';
    vs.className = 'btn btn-success';
    document.getElementById('viewer-save-btn').style.display = 'flex';
  } else if (vs) {
    vs.innerHTML = '<i class="fas fa-check"></i> Approve';
    vs.className = 'btn';
    document.getElementById('viewer-save-btn').style.display = 'none';
  }
}

function drawDWG() {
  const canvas = document.getElementById('dwg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const s = APP.dwgScale;
  const ox = W/2 + APP.dwgOffX;
  const oy = H/2 + APP.dwgOffY;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#e0e8f0';
  ctx.lineWidth = 0.5;
  const gs = 20 * s;
  const gox = ((ox % gs) + gs) % gs;
  const goy = ((oy % gs) + gs) % gs;
  for (let x = gox; x < W; x += gs) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = goy; y < H; y += gs) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

  // Origin cross
  ctx.strokeStyle = '#ccd4dc';
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(ox - 10, oy); ctx.lineTo(ox + 10, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, oy - 10); ctx.lineTo(ox, oy + 10); ctx.stroke();

  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(s, s);

  // Draw bracket shape
  const shapes = getDwgShapes(APP.currentDwg);
  shapes.forEach(shape => {
    ctx.strokeStyle = shape.color || '#1a2a3a';
    ctx.lineWidth = (shape.lw || 1.5) / s;
    if (shape.dash) ctx.setLineDash(shape.dash.map(d => d / s));
    else ctx.setLineDash([]);
    if (shape.fill) { ctx.fillStyle = shape.fill; }

    if (shape.type === 'rect') {
      ctx.beginPath();
      ctx.rect(shape.x, shape.y, shape.w, shape.h);
      if (shape.fill) ctx.fill();
      ctx.stroke();
    } else if (shape.type === 'line') {
      ctx.beginPath(); ctx.moveTo(shape.x1, shape.y1); ctx.lineTo(shape.x2, shape.y2); ctx.stroke();
    } else if (shape.type === 'circle') {
      ctx.beginPath(); ctx.arc(shape.cx, shape.cy, shape.r, 0, Math.PI * 2);
      if (shape.fill) ctx.fill();
      ctx.stroke();
    } else if (shape.type === 'arc') {
      ctx.beginPath(); ctx.arc(shape.cx, shape.cy, shape.r, shape.a1, shape.a2); ctx.stroke();
    } else if (shape.type === 'text') {
      ctx.fillStyle = shape.color || '#1a2a3a';
      ctx.font = \`\${(shape.size || 8) / s}px DM Mono, monospace\`;
      ctx.fillText(shape.text, shape.x, shape.y);
    }
  });
  ctx.restore();
  ctx.setLineDash([]);
}

function getDwgShapes(name) {
  const shapes = [];
  if (name === 'bracket') {
    // Outer profile
    shapes.push({ type: 'rect', x: -120, y: -80, w: 240, h: 160, lw: 1.5, color: '#1a2a3a' });
    // Flanges
    shapes.push({ type: 'rect', x: -120, y: -80, w: 40, h: 40, lw: 1.5, color: '#1a2a3a' });
    shapes.push({ type: 'rect', x: 80, y: -80, w: 40, h: 40, lw: 1.5, color: '#1a2a3a' });
    shapes.push({ type: 'rect', x: -120, y: 40, w: 40, h: 40, lw: 1.5, color: '#1a2a3a' });
    shapes.push({ type: 'rect', x: 80, y: 40, w: 40, h: 40, lw: 1.5, color: '#1a2a3a' });
    // Holes
    [[-90,-60],[90,-60],[-90,60],[90,60],[-20,0],[20,0]].forEach(([cx,cy]) => {
      shapes.push({ type: 'circle', cx, cy, r: 8, lw: 1, color: '#1a2a3a' });
      shapes.push({ type: 'circle', cx, cy, r: 1.5, lw: 0.5, color: '#1a2a3a', fill: '#1a2a3a' });
    });
    // Bend lines
    shapes.push({ type: 'line', x1: -120, y1: -40, x2: 120, y2: -40, lw: 0.7, color: '#e67700', dash: [6, 4] });
    shapes.push({ type: 'line', x1: -120, y1: 40, x2: 120, y2: 40, lw: 0.7, color: '#e67700', dash: [6, 4] });
    shapes.push({ type: 'line', x1: -80, y1: -80, x2: -80, y2: 80, lw: 0.7, color: '#e67700', dash: [6, 4] });
    shapes.push({ type: 'line', x1: 80, y1: -80, x2: 80, y2: 80, lw: 0.7, color: '#e67700', dash: [6, 4] });
    // Dimensions
    shapes.push({ type: 'line', x1: -120, y1: 95, x2: 120, y2: 95, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'line', x1: -120, y1: 90, x2: -120, y2: 100, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'line', x1: 120, y1: 90, x2: 120, y2: 100, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'text', text: '240', x: -12, y: 110, size: 9, color: '#2a4060' });
    shapes.push({ type: 'line', x1: -135, y1: -80, x2: -135, y2: 80, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'text', text: '160', x: -160, y: 5, size: 9, color: '#2a4060' });
    // Title block
    shapes.push({ type: 'rect', x: -120, y: -105, w: 240, h: 20, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'text', text: 'BRACKET_V3  |  MILD STEEL 2mm  |  ISO 2768-m  |  1:1', x: -115, y: -90, size: 7, color: '#2a4060' });
  } else if (name === 'enclosure') {
    // Main outer
    shapes.push({ type: 'rect', x: -190, y: -140, w: 380, h: 280, lw: 2, color: '#1a2a3a' });
    // Panel cutout
    shapes.push({ type: 'rect', x: -150, y: -100, w: 300, h: 200, lw: 1, color: '#1a2a3a' });
    // Mounting holes
    [[-170,-120],[170,-120],[-170,120],[170,120]].forEach(([cx,cy]) => {
      shapes.push({ type: 'circle', cx, cy, r: 6, lw: 1, color: '#1a2a3a' });
    });
    // Vent slots
    for (let i = 0; i < 8; i++) {
      shapes.push({ type: 'rect', x: -60 + i*18, y: -120, w: 10, h: 20, lw: 0.8, color: '#1a2a3a' });
    }
    shapes.push({ type: 'text', text: 'ENCLOSURE FRONT PANEL  |  AL 5052 H32 3mm  |  1:1', x: -185, y: -155, size: 7, color: '#2a4060' });
    shapes.push({ type: 'line', x1: -190, y1: 155, x2: 190, y2: 155, lw: 0.5, color: '#4a6480' });
    shapes.push({ type: 'text', text: '380', x: -12, y: 168, size: 9, color: '#2a4060' });
  } else if (name === 'panel') {
    // Long panel
    shapes.push({ type: 'rect', x: -300, y: -100, w: 600, h: 200, lw: 2, color: '#1a2a3a' });
    // Repeating holes pattern
    for (let i = 0; i < 12; i++) {
      shapes.push({ type: 'circle', cx: -270 + i*50, cy: 0, r: 10, lw: 1, color: '#1a2a3a' });
    }
    shapes.push({ type: 'line', x1: -300, y1: -40, x2: 300, y2: -40, lw: 0.7, color: '#e67700', dash: [8, 5] });
    shapes.push({ type: 'line', x1: -300, y1: 40, x2: 300, y2: 40, lw: 0.7, color: '#e67700', dash: [8, 5] });
    shapes.push({ type: 'text', text: 'CHASSIS PANEL  |  SS304 1.5mm  |  1:1', x: -295, y: -115, size: 7, color: '#2a4060' });
  }
  return shapes;
}

function dwgZoom(factor) {
  APP.dwgScale *= factor;
  APP.dwgScale = Math.max(0.2, Math.min(8, APP.dwgScale));
  drawDWG();
  document.getElementById('info-scale').textContent = APP.dwgScale.toFixed(2) + ':1';
}

function dwgFitView() {
  APP.dwgScale = 1; APP.dwgOffX = 0; APP.dwgOffY = 0;
  drawDWG();
  document.getElementById('info-scale').textContent = '1:1';
}

function dwgPan() { showNotif('info', 'Pan Mode', 'Click and drag to pan'); }
function dwgMeasure() { showNotif('info', 'Measure', 'Click two points to measure distance'); }
function dwgLayers() { showNotif('info', 'Layers', 'Layer visibility control'); }

function downloadDWG() {
  showNotif('success', 'Download Started', \`\${APP.currentDwg}.dwg downloading...\`);
}

// ================================================================
// LAYER LIST
// ================================================================
function renderLayerList() {
  const el = document.getElementById('layer-list');
  if (!el) return;
  const layers = [
    { name: '0_OUTLINE', color: '#1a2a3a', vis: true },
    { name: '1_BENDS', color: '#e67700', vis: true },
    { name: '2_HOLES', color: '#1a2a3a', vis: true },
    { name: '3_DIMENSIONS', color: '#4a6480', vis: true },
    { name: '4_ANNOTATIONS', color: '#2a4060', vis: true },
  ];
  el.innerHTML = layers.map(l => \`
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
      <div style="width:12px;height:12px;border-radius:3px;background:\${l.color};flex-shrink:0"></div>
      <div style="flex:1;font-size:12px;font-family:var(--font-mono)">\${l.name}</div>
      <div style="width:18px;height:18px;border-radius:4px;border:1.5px solid var(--border-med);display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:10px;color:var(--success)" onclick="this.innerHTML=this.innerHTML?'':'✓'">✓</div>
    </div>\`).join('');
}

// ================================================================
// QUOTE
// ================================================================
const MAT_MULT = { steel: 1, ss: 1.8, al: 1.3, cu: 2.5 };
function selectMaterial(m) {
  APP.selectedMaterial = m;
  document.querySelectorAll('.material-card').forEach(c => c.classList.remove('active'));
  document.getElementById('mat-' + m).classList.add('active');
  recalcQuote();
}

function recalcQuote() {
  const l = parseFloat(document.getElementById('q-length').value) || 240;
  const w = parseFloat(document.getElementById('q-width').value) || 160;
  const t = parseFloat(document.getElementById('q-thickness').value) || 2;
  const qty = parseInt(document.getElementById('q-qty').value) || 50;
  const mult = MAT_MULT[APP.selectedMaterial] || 1;
  const area = l * w / 1e6; // m²
  const matCost = Math.round(area * t * 7800 * 0.8 * mult * qty);
  const cutCost = Math.round(area * 120 * qty);
  const bendCost = Math.round(qty * 0.8 * mult);
  const setup = 28;
  const total = matCost + cutCost + bendCost + setup;
  const low = Math.round(total * 0.9);
  const high = Math.round(total * 1.35);

  document.getElementById('price-low').textContent = low;
  document.getElementById('price-high').textContent = high;
  document.getElementById('price-qty').textContent = qty;
  document.getElementById('bd-mat').textContent = matCost;
  document.getElementById('bd-cut').textContent = cutCost;
  document.getElementById('bd-bend').textContent = bendCost;
  document.getElementById('bd-total').textContent = total;
  document.getElementById('per-unit').textContent = (total / qty).toFixed(2);
}

function requestQuotes() {
  showNotif('success', 'Quote Sent', \`Request sent to \${PROVIDERS.length} verified providers\`);
  APP.activities.unshift({ icon: 'fas fa-paper-plane', color: 'var(--accent)', text: '<strong>Quote request</strong> sent to global CNC providers', time: 'Just now' });
  renderActivityFeed();
  setTimeout(() => {
    showNotif('info', 'Quote Received', 'SheetWorks IN quoted $198 for your order');
    APP.activities.unshift({ icon: 'fas fa-tag', color: 'var(--purple)', text: 'New quote from <strong>SheetWorks IN</strong>', time: 'Just now' });
    renderActivityFeed();
  }, 3000);
}

function placeOrder() {
  if (!APP.selectedProvider) return;
  closeModal('provider-modal');
  showNotif('success', 'Order Placed', \`Order placed with \${APP.selectedProvider.name}!\`);
  APP.activities.unshift({ icon: 'fas fa-box', color: 'var(--accent2)', text: \`Order placed with <strong>\${APP.selectedProvider.name}</strong>\`, time: 'Just now' });
  renderActivityFeed();
}

// ================================================================
// TABS
// ================================================================
function switchTab(id) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.currentTarget.classList.add('active');
}

// ================================================================
// NOTIFICATIONS
// ================================================================
function showNotif(type, title, sub) {
  const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle', warning: 'fa-exclamation-triangle' };
  const stack = document.getElementById('notif-stack');
  const notif = document.createElement('div');
  notif.className = \`notif \${type}\`;
  notif.innerHTML = \`<i class="fas \${icons[type]} notif-icon"></i><div class="notif-text"><div class="notif-title">\${title}</div>\${sub ? \`<div class="notif-sub">\${sub}</div>\` : ''}</div>\`;
  stack.appendChild(notif);
  setTimeout(() => { notif.classList.add('out'); setTimeout(() => notif.remove(), 300); }, 3500);
}

// ================================================================
// MODALS
// ================================================================
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}
document.querySelectorAll('.modal-overlay').forEach(o => {
  o.addEventListener('click', e => { if (e.target === o) o.classList.remove('open'); });
});

// ================================================================
// MISC
// ================================================================
function toggleCheck(row) {
  const cb = row.querySelector('.checkbox');
  cb.classList.toggle('checked');
  cb.innerHTML = cb.classList.contains('checked') ? '<i class="fas fa-check"></i>' : '';
}

function saveSettings() { showNotif('success', 'Settings Saved', 'Your preferences have been updated'); }

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ================================================================
// API CLIENT — connects to server.js backend
// ================================================================
const API_BASE = window.location.origin; // same-origin in production

const api = {
  token: localStorage.getItem('sf_token') || null,

  headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = \`Bearer \${this.token}\`;
    return h;
  },

  async request(method, path, body) {
    try {
      const opts = { method, headers: this.headers() };
      if (body && method !== 'GET') opts.body = JSON.stringify(body);
      const res = await fetch(\`\${API_BASE}\${path}\`, opts);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || \`HTTP \${res.status}\`);
      return data;
    } catch (err) {
      // In demo mode (no backend), silently return null
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) return null;
      throw err;
    }
  },

  get  : (path)         => api.request('GET',    path),
  post : (path, body)   => api.request('POST',   path, body),
  patch: (path, body)   => api.request('PATCH',  path, body),
  del  : (path)         => api.request('DELETE', path),

  async upload(formData) {
    try {
      const res = await fetch(\`\${API_BASE}/api/designs/upload\`, {
        method: 'POST',
        headers: this.token ? { Authorization: \`Bearer \${this.token}\` } : {},
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      return data;
    } catch (err) {
      if (err.message.includes('Failed to fetch')) return null;
      throw err;
    }
  }
};

// ================================================================
// BACKEND-CONNECTED AUTH (falls back to demo mode if no server)
// ================================================================
const _originalDoLogin = doLogin;
window.doLogin = async function() {
  const email = document.getElementById('login-email').value;
  const pass  = document.getElementById('login-pass').value;
  if (!email || !pass) { showNotif('error', 'Error', 'Please enter email and password'); return; }

  try {
    const data = await api.post('/api/auth/login', { email, password: pass });
    if (data) {
      api.token = data.token;
      localStorage.setItem('sf_token', data.token);
      loginUser({
        name: \`\${data.user.firstName} \${data.user.lastName}\`,
        email: data.user.email,
        role: data.user.role === 'provider' ? 'CNC Provider' : 'Designer',
        initials: \`\${data.user.firstName[0]}\${data.user.lastName[0]}\`.toUpperCase(),
      });
    } else {
      // Demo fallback
      loginUser({ name: 'Demo Designer', email, role: 'Designer', initials: 'DD' });
    }
  } catch (err) {
    showNotif('error', 'Login Failed', err.message);
  }
};

window.doSignup = async function() {
  const firstName = document.querySelector('#signup-form input[placeholder="John"]')?.value;
  const lastName  = document.querySelector('#signup-form input[placeholder="Smith"]')?.value;
  const email     = document.querySelector('#signup-form input[placeholder="you@company.com"]')?.value;
  const password  = document.querySelector('#signup-form input[placeholder="Min. 8 characters"]')?.value;
  const company   = document.querySelector('#signup-form input[placeholder="Acme Engineering"]')?.value;
  const role      = document.getElementById('role-provider')?.classList.contains('active') ? 'provider' : 'designer';

  try {
    const data = await api.post('/api/auth/register', { firstName, lastName, email, password, company, role });
    if (data) {
      api.token = data.token;
      localStorage.setItem('sf_token', data.token);
      loginUser({
        name: \`\${firstName} \${lastName}\`,
        email, role: role === 'provider' ? 'CNC Provider' : 'Designer',
        initials: \`\${(firstName||'D')[0]}\${(lastName||'U')[0]}\`.toUpperCase(),
      });
      showNotif('success', 'Account Created', 'Welcome to SheetForge!');
    } else {
      showNotif('success', 'Account Created', 'Welcome to SheetForge! (demo mode)');
      setTimeout(() => demoLogin('designer'), 800);
    }
  } catch (err) {
    showNotif('error', 'Signup Failed', err.message);
  }
};

// ================================================================
// BACKEND-CONNECTED UPLOAD
// ================================================================
const _origProcessFile = processFile;
window.processFile = function(file) {
  _origProcessFile(file);

  // Also upload to backend if available
  const formData = new FormData();
  formData.append('file', file);
  formData.append('partName', file.name.replace(/\\.[^.]+$/, ''));
  api.upload(formData).then(data => {
    if (data?.design) {
      APP.currentDesignId = data.design._id;
      console.log('Design registered:', data.design._id);
    }
  });
};

// ================================================================
// BACKEND-CONNECTED CONVERSION
// ================================================================
const _origStartConversion = startConversion;
window.startConversion = async function() {
  if (!APP.uploadedFile) { showNotif('error', 'No File', 'Please upload a file first'); return; }

  // If we have a backend design ID, use real conversion
  if (APP.currentDesignId && api.token) {
    // Update material/thickness
    const mat = document.getElementById('material-select')?.value;
    const thick = document.getElementById('thickness-input')?.value;
    const name  = document.getElementById('part-name')?.value;
    if (mat || thick || name) {
      await api.patch(\`/api/designs/\${APP.currentDesignId}\`, {
        material: mat, thickness: parseFloat(thick), partName: name,
      }).catch(() => {});
    }

    // Trigger AI conversion
    document.getElementById('convert-btn').disabled = true;
    document.getElementById('convert-btn').innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Analyzing...';
    document.getElementById('ai-analysis-panel').style.display = 'block';

    try {
      const data = await api.post(\`/api/convert/\${APP.currentDesignId}\`);
      if (data?.design) {
        APP.currentDesign = data.design;
        document.getElementById('dim-w').textContent = data.analysis?.width?.toFixed(0) + 'mm';
        document.getElementById('dim-h').textContent = data.analysis?.height?.toFixed(0) + 'mm';
        document.getElementById('dim-b').textContent = data.analysis?.bendLines;
        document.getElementById('dim-holes').textContent = data.analysis?.holes;
        document.getElementById('dwg-result-panel').style.display = 'block';
        setStep(4);
        showNotif('success', 'DWG Ready', 'AI conversion complete!');
        document.getElementById('convert-btn').disabled = false;
        document.getElementById('convert-btn').innerHTML = '<i class="fas fa-magic"></i> Re-convert';
        return;
      }
    } catch (err) {
      console.warn('Backend conversion failed, using demo mode:', err.message);
    }
  }

  // Fallback to demo simulation
  _origStartConversion();
};

// ================================================================
// BACKEND-CONNECTED APPROVE
// ================================================================
window.approveDWG = async function() {
  if (APP.currentDesignId && api.token) {
    await api.patch(\`/api/designs/\${APP.currentDesignId}/approve\`).catch(() => {});
  }
  APP.approvedDwgs.add('new');
  document.getElementById('approve-btn').innerHTML = '<i class="fas fa-check-circle"></i> Approved!';
  document.getElementById('approve-btn').className = 'btn btn-success flex-1';
  setStep(5);
  showNotif('success', 'Design Approved', 'Ready to save to Cloudinary');
  setTimeout(() => openCloudSaveModal(), 1500);
};

// ================================================================
// BACKEND-CONNECTED CLOUD SAVE
// ================================================================
window.doCloudSave = async function() {
  const name = document.getElementById('save-asset-name').value || 'design_export';
  document.getElementById('do-save-btn').disabled = true;
  document.getElementById('do-save-btn').innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></div>Uploading...';
  document.getElementById('save-progress-area').style.display = 'block';

  // Try real backend save
  if (APP.currentDesignId && api.token) {
    try {
      await delay(600);
      const s1 = document.getElementById('save-step-1');
      s1.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>DWG rendered to PNG (1200×800px, 300dpi)</div>';

      const s2 = document.getElementById('save-step-2');
      s2.style.opacity = '1';
      s2.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0"></div><div>Uploading to Cloudinary...</div>';

      const data = await api.post(\`/api/cloud/save/\${APP.currentDesignId}\`, {
        assetName: name,
        folder: 'sheetforge/designs/2026',
        tags: 'sheetforge,cad,approved',
      });

      if (data) {
        s2.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Uploaded to Cloudinary</div>';
        const s3 = document.getElementById('save-step-3');
        s3.style.opacity = '1';
        s3.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Link saved to MongoDB</div>';

        document.getElementById('saved-url').textContent = data.cloudinary.secureUrl;
        document.getElementById('save-result').style.display = 'block';
        document.getElementById('do-save-btn').innerHTML = '<i class="fas fa-check"></i> Saved!';

        APP.cloudAssets.unshift({ name, url: data.cloudinary.secureUrl, mongoId: data.mongoId, size: \`\${Math.round(data.cloudinary.bytes/1024)} KB\`, date: 'Just now' });
        renderCloudinaryTable();
        showNotif('success', 'Saved to Cloud', 'Design URL stored in MongoDB');
        return;
      }
    } catch (err) {
      console.warn('Cloud save error, using demo mode:', err.message);
    }
  }

  // Demo fallback (original function body)
  await delay(1000);
  const s1 = document.getElementById('save-step-1');
  s1.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>DWG rendered to PNG (1200×800px, 300dpi)</div>';
  s1.classList.add('done');
  const s2 = document.getElementById('save-step-2');
  s2.style.opacity = '1';
  s2.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0"></div><div>Uploading to Cloudinary...</div>';
  await delay(1400);
  s2.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Uploaded to Cloudinary successfully</div>';
  const s3 = document.getElementById('save-step-3');
  s3.style.opacity = '1';
  s3.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:1.5px;flex-shrink:0"></div><div>Saving link to MongoDB...</div>';
  await delay(900);
  s3.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success)"></i><div>Link saved to MongoDB Atlas</div>';
  const url = \`https://res.cloudinary.com/cloud-sheetforge-demo/image/upload/v\${Date.now()}/sheetforge/designs/2026/\${name}.png\`;
  const mongoId = Math.random().toString(16).slice(2, 26);
  document.getElementById('saved-url').textContent = url;
  document.getElementById('save-result').style.display = 'block';
  document.getElementById('do-save-btn').innerHTML = '<i class="fas fa-check"></i> Saved!';
  APP.cloudAssets.unshift({ name, url, mongoId, size: '1.2 MB', date: 'Just now' });
  renderCloudinaryTable();
  showNotif('success', 'Saved to Cloud', 'Design URL stored in MongoDB');
};

// ================================================================
// BACKEND-CONNECTED QUOTE REQUEST
// ================================================================
window.requestQuotes = async function() {
  if (APP.currentDesignId && api.token) {
    const specs = {
      length   : parseFloat(document.getElementById('q-length').value) || 240,
      width    : parseFloat(document.getElementById('q-width').value) || 160,
      thickness: parseFloat(document.getElementById('q-thickness').value) || 2,
      quantity : parseInt(document.getElementById('q-qty').value) || 50,
      material : APP.selectedMaterial,
      leadTime : document.getElementById('q-leadtime').value,
      operations: Array.from(document.querySelectorAll('#ops-checklist .checkbox.checked'))
        .map(c => c.nextElementSibling?.textContent?.trim()),
    };
    try {
      const data = await api.post('/api/quotes', { designId: APP.currentDesignId, specs });
      if (data) {
        showNotif('success', 'Quote Sent', \`\${data.quote.bids.length} providers received your request\`);
        APP.activities.unshift({ icon: 'fas fa-paper-plane', color: 'var(--accent)', text: \`<strong>Quote request</strong> sent to \${data.quote.bids.length} CNC providers\`, time: 'Just now' });
        renderActivityFeed();
        setTimeout(() => {
          const best = data.quote.bids.sort((a,b) => a.price - b.price)[0];
          if (best) showNotif('info', 'Quote Received', \`Best price: $\${best.price} — \${best.leadDays} days lead time\`);
        }, 2000);
        return;
      }
    } catch (err) { console.warn('Quote API error:', err.message); }
  }
  // Demo fallback
  showNotif('success', 'Quote Sent', \`Request sent to \${PROVIDERS.length} verified providers\`);
  APP.activities.unshift({ icon: 'fas fa-paper-plane', color: 'var(--accent)', text: '<strong>Quote request</strong> sent to global CNC providers', time: 'Just now' });
  renderActivityFeed();
  setTimeout(() => {
    showNotif('info', 'Quote Received', 'SheetWorks IN quoted $198 for your order');
    APP.activities.unshift({ icon: 'fas fa-tag', color: 'var(--purple)', text: 'New quote from <strong>SheetWorks IN</strong>', time: 'Just now' });
    renderActivityFeed();
  }, 3000);
};

// ================================================================
// BACKEND-CONNECTED LOAD DESIGNS
// ================================================================
async function loadDesignsFromAPI() {
  if (!api.token) return;
  try {
    const data = await api.get('/api/designs');
    if (data?.designs?.length) {
      // Merge with local demo data
      const apiDrawings = data.designs.map(d => ({
        id: d._id, name: d.partName + '.dwg', original: d.originalFile?.filename || 'unknown',
        material: d.material || 'Unknown', thickness: (d.thickness || '?') + 'mm',
        status: d.status === 'ready' ? 'ready' : d.status === 'approved' ? 'approved' : d.status === 'saved' ? 'approved' : 'pending',
        date: new Date(d.createdAt).toLocaleDateString(),
      }));
      // Prepend API data to demo data
      DRAWINGS.unshift(...apiDrawings.slice(0, 3));
      renderDrawingsGrid();
    }
  } catch (_) {}
}

// ================================================================
// KEYBOARD SHORTCUTS
// ================================================================
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const shortcuts = {
    'd': () => showPage('dashboard'),
    'u': () => showPage('upload'),
    'v': () => showPage('viewer'),
    'q': () => showPage('quote'),
    'p': () => showPage('providers'),
    'Escape': () => document.querySelectorAll('.modal-overlay.open').forEach(m => m.classList.remove('open')),
  };
  if (shortcuts[e.key]) shortcuts[e.key]();
});

// ================================================================
// INIT LISTENERS
// ================================================================
window.addEventListener('resize', () => {
  if (APP.currentPage === 'viewer') { initDwgCanvas(); loadDwg(APP.currentDwg); }
});

// Auto-login if token in localStorage
window.addEventListener('load', async () => {
  const savedToken = localStorage.getItem('sf_token');
  if (savedToken) {
    api.token = savedToken;
    try {
      const data = await api.get('/api/auth/me');
      if (data?.user) {
        loginUser({
          name: \`\${data.user.firstName} \${data.user.lastName}\`,
          email: data.user.email,
          role: data.user.role === 'provider' ? 'CNC Provider' : 'Designer',
          initials: \`\${data.user.firstName[0]}\${data.user.lastName[0]}\`.toUpperCase(),
        });
        loadDesignsFromAPI();
        return;
      }
    } catch (_) {
      localStorage.removeItem('sf_token');
    }
  }
});

const _origDoLogout = doLogout;
window.doLogout = function() {
  localStorage.removeItem('sf_token');
  api.token = null;
  _origDoLogout();
};

// Initial recalc
setTimeout(() => { if (document.getElementById('price-low')) recalcQuote(); }, 200);

// Keyboard hint
setTimeout(() => {
  if (document.getElementById('app')?.classList.contains('visible')) return;
  // Will show after login
}, 2000);
</script>

</body>
</html>
`;

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
