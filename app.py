import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AirPinch", layout="wide")

st.markdown("""
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    background: #000 !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    overflow: hidden;
  }
  [data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], footer { display: none !important; }
  iframe { display: block; border: none; }
</style>
""", unsafe_allow_html=True)

components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body {
    width:100vw; height:100vh;
    background:#000; overflow:hidden;
    font-family: 'Courier New', monospace;
  }
  #drawCanvas {
    position:fixed; top:0; left:0;
    width:100vw; height:100vh;
    z-index:1;
  }
  #ui {
    position:fixed; bottom:20px; left:50%;
    transform:translateX(-50%);
    z-index:10;
    display:flex; gap:16px; align-items:center;
  }
  .pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 6px 16px;
    color: rgba(255,255,255,0.5);
    font-size: 11px;
    letter-spacing: 0.08em;
    backdrop-filter: blur(8px);
    white-space: nowrap;
  }
  .pill span { color: rgba(255,255,255,0.85); }
  #status {
    position:fixed; top:16px; left:50%;
    transform:translateX(-50%);
    color:rgba(255,255,255,0.25);
    font-size:11px; letter-spacing:0.1em;
    z-index:10; pointer-events:none;
  }
  #colorDot {
    position:fixed; top:14px; right:20px;
    width:12px; height:12px; border-radius:50%;
    z-index:10; transition: background 0.3s;
    box-shadow: 0 0 8px 3px currentColor;
  }
  video { display:none; }
</style>
</head>
<body>

<canvas id="drawCanvas"></canvas>
<video id="video" playsinline></video>
<div id="status">INITIALISING…</div>
<div id="colorDot"></div>
<div id="ui">
  <div class="pill">☝️ index only → <span>DRAW</span></div>
  <div class="pill">🤏 pinch + move → <span>DRAG</span></div>
  <div class="pill">✌️ two fingers → <span>NEXT COLOR</span></div>
  <div class="pill">🔄 refresh → <span>CLEAR</span></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

<script>
// ─── Canvas setup ───────────────────────────────────────────────────────────
const canvas  = document.getElementById('drawCanvas');
const ctx     = canvas.getContext('2d');
const video   = document.getElementById('video');
const statusEl= document.getElementById('status');
const colorDot= document.getElementById('colorDot');

let W, H;
function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

// ─── Neon color palette ──────────────────────────────────────────────────────
const PALETTES = [
  { stroke:'#ff2d78', glow:'#ff2d78', name:'MAGENTA' },
  { stroke:'#00ffe5', glow:'#00ffe5', name:'CYAN'    },
  { stroke:'#aaff00', glow:'#aaff00', name:'LIME'    },
  { stroke:'#ff9500', glow:'#ff9500', name:'AMBER'   },
  { stroke:'#c87dff', glow:'#c87dff', name:'VIOLET'  },
  { stroke:'#ffffff', glow:'#88ccff', name:'WHITE'   },
];
let colorIdx = 0;
function currentPalette() { return PALETTES[colorIdx % PALETTES.length]; }
function nextColor() {
  colorIdx++;
  const p = currentPalette();
  colorDot.style.background = p.stroke;
  colorDot.style.boxShadow  = `0 0 10px 4px ${p.glow}`;
  statusEl.textContent      = `COLOR → ${p.name}`;
  setTimeout(()=>{ statusEl.textContent = ''; }, 1200);
}
// init dot
colorDot.style.background = PALETTES[0].stroke;
colorDot.style.boxShadow  = `0 0 10px 4px ${PALETTES[0].glow}`;

// ─── State ───────────────────────────────────────────────────────────────────
const PINCH_THRESH  = 0.055;  // normalised distance
const SELECT_THRESH = 60;     // canvas pixels
const MIN_MOVE      = 3;

let strokes       = [];   // [{pts:[{x,y}], color, glow}]
let currentStroke = [];
let currentColor  = null;
let selectedIdx   = -1;
let prevPinchPos  = null;
let prevFingerPos = null;

// Two-finger (color switch) debounce
let twoFingerActive = false;

// ─── Landmark helpers ────────────────────────────────────────────────────────
function lc(lm) {
  // Mirror x so it feels like a mirror
  return { x: (1 - lm.x) * W, y: lm.y * H };
}
function dst(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }
function ndst(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

function isIndexOnly(L) {
  return L[8].y < L[6].y  &&
         L[12].y > L[10].y &&
         L[16].y > L[14].y &&
         L[20].y > L[18].y;
}
function isTwoFingers(L) {
  return L[8].y  < L[6].y  &&
         L[12].y < L[10].y &&
         L[16].y > L[14].y &&
         L[20].y > L[18].y;
}
function isPinch(L) {
  const d = ndst(L[8], L[4]);
  return { on: d < PINCH_THRESH, pos: lc(L[8]) };
}

// ─── Glow stroke drawing ─────────────────────────────────────────────────────
function drawGlowStroke(pts, color, glow, lineW, selected) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap  = 'round';

  // outer glow layers
  const layers = selected
    ? [[18, 0.10], [10, 0.22], [5, 0.40]]
    : [[16, 0.08], [8,  0.18], [4,  0.35]];

  for (const [blur, alpha] of layers) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.strokeStyle = hexAlpha(glow, alpha);
    ctx.lineWidth   = lineW + blur;
    ctx.filter      = `blur(${blur*0.6}px)`;
    ctx.stroke();
  }

  // sharp core line
  ctx.filter = 'none';
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.strokeStyle = selected ? '#ffffff' : color;
  ctx.lineWidth   = selected ? lineW + 1 : lineW;
  ctx.stroke();
  ctx.restore();
}

function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ─── Sparkles ────────────────────────────────────────────────────────────────
let sparklePool = [];
function addSparkles(cx, cy, color, glow) {
  for (let i=0; i<6; i++) {
    const angle = Math.random()*Math.PI*2;
    const speed = 1.5 + Math.random()*3;
    sparklePool.push({
      x: cx, y: cy,
      vx: Math.cos(angle)*speed,
      vy: Math.sin(angle)*speed,
      r:  1.5 + Math.random()*2.5,
      life: 1.0,
      decay: 0.06 + Math.random()*0.06,
      color, glow
    });
  }
}
function tickSparkles() {
  sparklePool = sparklePool.filter(s => s.life > 0);
  for (const s of sparklePool) {
    ctx.save();
    ctx.globalAlpha = s.life * 0.9;
    ctx.shadowColor = s.glow;
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
    ctx.fillStyle = s.color;
    ctx.fill();
    ctx.restore();
    s.x   += s.vx; s.y += s.vy;
    s.vy  += 0.05; // slight gravity
    s.life -= s.decay;
    s.r   *= 0.97;
  }
}

// ─── Landmark dots ───────────────────────────────────────────────────────────
function drawLandmarks(L) {
  for (const lm of L) {
    const p = lc(lm);
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI*2);
    ctx.fillStyle   = 'rgba(0,255,120,0.7)';
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur  = 6;
    ctx.fill();
    ctx.restore();
  }
}

// ─── Main render loop ────────────────────────────────────────────────────────
// We keep a persistent off-screen image for finalised strokes
// so re-drawing doesn't flicker
let bgImage = null; // ImageData of all finalised strokes

function rebuildBg() {
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  const t = tmp.getContext('2d');
  t.fillStyle = '#000';
  t.fillRect(0,0,W,H);
  strokes.forEach((s,i) => {
    if (i !== selectedIdx) drawGlowStrokeOn(t, s.pts, s.color, s.glow, 3, false);
  });
  bgImage = tmp;
}

function drawGlowStrokeOn(c2d, pts, color, glow, lineW, selected) {
  if (pts.length < 2) return;
  c2d.save();
  c2d.lineJoin='round'; c2d.lineCap='round';
  const layers = [[16,0.08],[8,0.18],[4,0.35]];
  for (const [blur,alpha] of layers) {
    c2d.beginPath();
    c2d.moveTo(pts[0].x,pts[0].y);
    for (let i=1;i<pts.length;i++) c2d.lineTo(pts[i].x,pts[i].y);
    c2d.strokeStyle=hexAlpha(glow,alpha);
    c2d.lineWidth=lineW+blur;
    c2d.filter=`blur(${blur*0.6}px)`;
    c2d.stroke();
  }
  c2d.filter='none';
  c2d.beginPath();
  c2d.moveTo(pts[0].x,pts[0].y);
  for (let i=1;i<pts.length;i++) c2d.lineTo(pts[i].x,pts[i].y);
  c2d.strokeStyle=color; c2d.lineWidth=lineW; c2d.stroke();
  c2d.restore();
}

let needRebuild = false;
let lastStrokeCount = 0;
let lastSelected    = -1;

function frame(landmarks) {
  // Detect if we need to rebuild bg cache
  if (strokes.length !== lastStrokeCount || selectedIdx !== lastSelected) {
    needRebuild = true;
    lastStrokeCount = strokes.length;
    lastSelected    = selectedIdx;
  }
  if (needRebuild) { rebuildBg(); needRebuild = false; }

  // Draw bg
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H);
  if (bgImage) ctx.drawImage(bgImage, 0, 0);

  // Draw selected stroke on top (yellow glow)
  if (selectedIdx >= 0 && strokes[selectedIdx]) {
    const s = strokes[selectedIdx];
    drawGlowStroke(s.pts, '#ffff00', '#ffee00', 4, true);
  }

  // Draw current in-progress stroke
  if (currentStroke.length > 1 && currentColor) {
    drawGlowStroke(currentStroke, currentColor.stroke, currentColor.glow, 3, false);
  }

  // Sparkles
  tickSparkles();

  if (!landmarks) return;

  drawLandmarks(landmarks);

  const ip = lc(landmarks[8]);
  const { on: pinching, pos: pinchPos } = isPinch(landmarks);
  const indexOnly  = isIndexOnly(landmarks);
  const twoFing    = isTwoFingers(landmarks);

  // Two-finger → change color
  if (twoFing && !twoFingerActive) { twoFingerActive=true; nextColor(); }
  if (!twoFing) twoFingerActive = false;

  if (pinching) {
    // Finalise open stroke
    if (currentStroke.length > 1) {
      strokes.push({ pts:[...currentStroke], color:currentColor.stroke, glow:currentColor.glow });
      needRebuild = true;
    }
    currentStroke = []; prevFingerPos = null;

    if (selectedIdx === -1) {
      let minD=SELECT_THRESH, best=-1;
      strokes.forEach((s,i)=>s.pts.forEach(pt=>{
        const d=dst(pt,pinchPos);
        if(d<minD){minD=d;best=i;}
      }));
      selectedIdx=best; prevPinchPos=pinchPos;
    } else {
      if (prevPinchPos) {
        const dx=pinchPos.x-prevPinchPos.x, dy=pinchPos.y-prevPinchPos.y;
        strokes[selectedIdx].pts = strokes[selectedIdx].pts.map(p=>({
          x:Math.max(0,Math.min(W-1,p.x+dx)),
          y:Math.max(0,Math.min(H-1,p.y+dy))
        }));
        needRebuild=true;
      }
      prevPinchPos=pinchPos;
    }

    // Pinch indicator
    ctx.save();
    ctx.beginPath(); ctx.arc(pinchPos.x,pinchPos.y,12,0,Math.PI*2);
    ctx.strokeStyle='rgba(255,140,0,0.8)'; ctx.lineWidth=2;
    ctx.shadowColor='#ff8800'; ctx.shadowBlur=10;
    ctx.stroke(); ctx.restore();

  } else {
    if (selectedIdx !== -1) { selectedIdx=-1; needRebuild=true; }
    prevPinchPos=null;

    if (indexOnly && !twoFing) {
      if (!currentColor) currentColor = currentPalette();
      if (prevFingerPos && dst(ip,prevFingerPos)>MIN_MOVE) {
        currentStroke.push({...ip});
        addSparkles(ip.x, ip.y, currentColor.stroke, currentColor.glow);
      } else if (!prevFingerPos) {
        currentColor  = currentPalette();
        currentStroke = [{...ip}];
      }
      prevFingerPos = {...ip};
    } else {
      if (currentStroke.length > 1 && currentColor) {
        strokes.push({ pts:[...currentStroke], color:currentColor.stroke, glow:currentColor.glow });
        needRebuild=true;
      }
      currentStroke=[]; prevFingerPos=null; currentColor=null;
    }
  }
}

// ─── MediaPipe ───────────────────────────────────────────────────────────────
const hands = new Hands({
  locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
});
hands.setOptions({
  maxNumHands:1, modelComplexity:1,
  minDetectionConfidence:0.7, minTrackingConfidence:0.5
});
hands.onResults(results=>{
  if (results.multiHandLandmarks && results.multiHandLandmarks[0]) {
    frame(results.multiHandLandmarks[0]);
    statusEl.textContent='';
  } else {
    frame(null);
    statusEl.textContent='SHOW YOUR HAND';
  }
});

const camera = new Camera(video, {
  onFrame: async()=>{ await hands.send({image:video}); },
  width:1280, height:720
});
camera.start()
  .then(()=>{ statusEl.textContent='READY'; setTimeout(()=>statusEl.textContent='',2000); })
  .catch(e=>{ statusEl.textContent='CAMERA ERROR: '+e.message; });

// Kick off render loop for sparkle animation even without new frames
function loop() { tickSparkles(); requestAnimationFrame(loop); }
// (sparkles drawn inside frame() which is called by mediapipe, this is a fallback)
</script>
</body>
</html>
""", height=900, scrolling=False)
