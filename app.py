import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AirPinch – Draw with Gestures", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #000; }
    [data-testid="stHeader"] { background: transparent; }
    .stMarkdown p { color: #aaa; font-family: monospace; font-size:13px; }
    h1 { color: #fff; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🖐️ AirPinch – Draw in the Air")
st.markdown("✏️ **Draw** — Extend only your index finger · 🤏 **Drag** — Pinch index + thumb near a line · 🔄 Refresh to clear")

components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#000; display:flex; flex-direction:column; align-items:center; font-family:monospace; }
  #container {
    position: relative;
    width: 960px;
    max-width: 100vw;
    aspect-ratio: 960/720;
  }
  canvas { position:absolute; top:0; left:0; width:100%; height:100%; }
  #drawCanvas { z-index:2; }
  #vidCanvas  { z-index:1; opacity:0.15; }
  video { display:none; }
  #status { color:#555; font-size:12px; margin-top:6px; height:20px; }
</style>
</head>
<body>

<div id="container">
  <video id="video" playsinline></video>
  <canvas id="vidCanvas" width="960" height="720"></canvas>
  <canvas id="drawCanvas" width="960" height="720"></canvas>
</div>
<div id="status">⏳ Loading hand tracking…</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
        crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"
        crossorigin="anonymous"></script>

<script>
const CW = 960, CH = 720;
const PINCH_THRESH  = 32;
const SELECT_THRESH = 44;

const video      = document.getElementById('video');
const drawCanvas = document.getElementById('drawCanvas');
const vidCanvas  = document.getElementById('vidCanvas');
const dc         = drawCanvas.getContext('2d');
const vc         = vidCanvas.getContext('2d');
const statusEl   = document.getElementById('status');

let strokes       = [];
let currentStroke = [];
let selectedIdx   = -1;
let prevPinchPos  = null;
let prevFingerPos = null;

function lmC(lm) { return { x: Math.round((1-lm.x)*CW), y: Math.round(lm.y*CH) }; }
function dst(a,b) { return Math.hypot(a.x-b.x, a.y-b.y); }

function isIndexOnly(L) {
  return L[8].y < L[6].y && L[12].y > L[10].y && L[16].y > L[14].y && L[20].y > L[18].y;
}
function getPinch(L) {
  const a = lmC(L[8]), b = lmC(L[4]);
  return { on: dst(a,b) < PINCH_THRESH, pos: a };
}

function sparkles(cx, cy) {
  for (let i=0; i<10; i++) {
    const a = Math.random()*Math.PI*2, d = 5+Math.random()*14, r = 2+Math.random()*2;
    const g = 200+Math.floor(Math.random()*55);
    dc.beginPath();
    dc.arc(cx+Math.cos(a)*d, cy+Math.sin(a)*d, r, 0, Math.PI*2);
    dc.fillStyle = `rgb(${g},${g},${Math.floor(Math.random()*60)})`;
    dc.fill();
  }
}

function renderAll(landmarks) {
  dc.fillStyle = '#000';
  dc.fillRect(0,0,CW,CH);

  strokes.forEach((s,idx) => {
    if (s.length < 2) return;
    dc.beginPath(); dc.moveTo(s[0].x,s[0].y);
    for (let i=1;i<s.length;i++) dc.lineTo(s[i].x,s[i].y);
    dc.strokeStyle = idx===selectedIdx ? '#ffff00' : '#ffffff';
    dc.lineWidth   = idx===selectedIdx ? 4 : 3;
    dc.lineJoin='round'; dc.lineCap='round'; dc.stroke();
  });

  if (currentStroke.length > 1) {
    dc.beginPath(); dc.moveTo(currentStroke[0].x,currentStroke[0].y);
    for (let i=1;i<currentStroke.length;i++) dc.lineTo(currentStroke[i].x,currentStroke[i].y);
    dc.strokeStyle='#ffffff'; dc.lineWidth=3;
    dc.lineJoin='round'; dc.lineCap='round'; dc.stroke();
  }

  if (!landmarks) return;

  landmarks.forEach(lm => {
    const p = lmC(lm);
    dc.beginPath(); dc.arc(p.x,p.y,3,0,Math.PI*2);
    dc.fillStyle='#00ff44'; dc.fill();
  });

  const ip = lmC(landmarks[8]);
  const { on: isPinching, pos: pinchPos } = getPinch(landmarks);
  const indexOnly = isIndexOnly(landmarks);

  if (isPinching) {
    if (currentStroke.length > 1) strokes.push([...currentStroke]);
    currentStroke = []; prevFingerPos = null;

    if (selectedIdx === -1) {
      let minD = SELECT_THRESH, best = -1;
      strokes.forEach((s,i) => s.forEach(pt => {
        const d = dst(pt, pinchPos);
        if (d < minD) { minD=d; best=i; }
      }));
      selectedIdx  = best;
      prevPinchPos = pinchPos;
    } else {
      if (prevPinchPos) {
        const dx = pinchPos.x-prevPinchPos.x, dy = pinchPos.y-prevPinchPos.y;
        strokes[selectedIdx] = strokes[selectedIdx].map(pt=>({
          x: Math.max(0,Math.min(CW-1,pt.x+dx)),
          y: Math.max(0,Math.min(CH-1,pt.y+dy))
        }));
      }
      prevPinchPos = pinchPos;
    }
    // Pinch dot
    dc.beginPath(); dc.arc(pinchPos.x,pinchPos.y,10,0,Math.PI*2);
    dc.strokeStyle='#ff8800'; dc.lineWidth=2; dc.stroke();
  } else {
    selectedIdx==-1 ? null : (selectedIdx=-1);
    prevPinchPos=null;

    if (indexOnly) {
      if (prevFingerPos && dst(ip,prevFingerPos)>2) currentStroke.push(ip);
      else if (!prevFingerPos) currentStroke=[ip];
      prevFingerPos=ip;
      sparkles(ip.x,ip.y);
    } else {
      if (currentStroke.length>1) strokes.push([...currentStroke]);
      currentStroke=[]; prevFingerPos=null;
    }
  }
}

// MediaPipe Hands — pure JavaScript, zero Python dependencies
const hands = new Hands({ locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands:1, modelComplexity:1, minDetectionConfidence:0.7, minTrackingConfidence:0.5 });
hands.onResults(results => {
  vc.save(); vc.scale(-1,1);
  vc.drawImage(results.image,-CW,0,CW,CH);
  vc.restore();
  if (results.multiHandLandmarks && results.multiHandLandmarks[0]) {
    renderAll(results.multiHandLandmarks[0]);
    statusEl.textContent='✅ Hand detected';
  } else {
    renderAll(null);
    statusEl.textContent='⚠️ No hand visible — check lighting';
  }
});

const camera = new Camera(video, {
  onFrame: async () => { await hands.send({image:video}); },
  width:CW, height:CH
});
camera.start()
  .then(()=>{ statusEl.textContent='✅ Camera ready — show your hand!'; })
  .catch(e=>{ statusEl.textContent='❌ Camera error: '+e.message; });
</script>
</body>
</html>
""", height=820, scrolling=False)
