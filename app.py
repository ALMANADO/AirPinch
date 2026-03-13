import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import streamlit as st
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AirPinch", layout="wide")

st.markdown("""
<style>
  * { margin:0 !important; padding:0 !important; box-sizing:border-box; }

  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  .main, .block-container,
  section[data-testid="stMain"] > div {
    background: #000 !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100vw !important;
    width: 100vw !important;
    overflow: hidden !important;
  }

  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  #MainMenu, footer, header { display: none !important; }

  [data-testid="stVerticalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
  }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #000 !important;
    padding: 0 !important;
  }

  /* Hide the ugly connection warning banner */
  [data-testid="stAlert"],
  .stAlert { display: none !important; }

  /* Video fills full screen */
  video {
    width: 100vw !important;
    height: 100vh !important;
    object-fit: cover !important;
    background: #000 !important;
    display: block !important;
  }

  .streamlit-webrtc-component,
  .streamlit-webrtc-component > div,
  .streamlit-webrtc-component > div > div {
    width: 100vw !important;
    height: 100vh !important;
    background: #000 !important;
    padding: 0 !important;
  }

  /* START/STOP button */
  button[kind="primary"] {
    position: fixed !important;
    bottom: 68px !important;
    left: 20px !important;
    z-index: 9999 !important;
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: rgba(255,255,255,0.8) !important;
    border-radius: 999px !important;
    font-family: monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    padding: 6px 20px !important;
    backdrop-filter: blur(8px) !important;
  }

  /* Hide SELECT DEVICE */
  button[kind="secondary"] { display: none !important; }

  .pills {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
    z-index: 9999;
    pointer-events: none;
  }
  .pill {
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 999px;
    padding: 5px 14px;
    color: rgba(255,255,255,0.4);
    font-family: monospace;
    font-size: 11px;
    backdrop-filter: blur(10px);
    white-space: nowrap;
  }
  .pill b { color: rgba(255,255,255,0.85); }
</style>

<div class="pills">
  <div class="pill">☝️ index only → <b>DRAW</b></div>
  <div class="pill">🤏 pinch + move → <b>DRAG</b></div>
  <div class="pill">✌️ two fingers → <b>COLOR</b></div>
  <div class="pill">🔄 refresh → <b>CLEAR</b></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
CW, CH        = 1280, 720
PINCH_THRESH  = 40
SELECT_THRESH = 60
MIN_MOVE      = 3

PALETTES = [
    ((255, 45,  120), (255, 45,  120)),   # magenta
    ((0,  255,  220), (0,  255,  220)),   # cyan
    ((170, 255,  0),  (170, 255,   0)),   # lime
    ((255, 149,   0), (255, 149,   0)),   # amber
    ((200, 125, 255), (200, 125, 255)),   # violet
    ((255, 255, 255), (140, 200, 255)),   # ice white
]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGERTIPS = {4, 8, 12, 16, 20}

# ─────────────────────────────────────────────────────────────
# TURN + STUN config — fixes "connection taking longer" error
# Open Relay is a free public TURN service
# ─────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────
def lm_to_px(lm):
    x = int((1.0 - lm.x) * CW)
    y = int(lm.y * CH)
    return (np.clip(x, 0, CW-1), np.clip(y, 0, CH-1))

def dist(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def is_index_only(lms):
    return (lms[8].y  < lms[6].y  and
            lms[12].y > lms[10].y and
            lms[16].y > lms[14].y and
            lms[20].y > lms[18].y)

def is_two_fingers(lms):
    return (lms[8].y  < lms[6].y  and
            lms[12].y < lms[10].y and
            lms[16].y > lms[14].y and
            lms[20].y > lms[18].y)

def is_pinch(lms):
    tip8 = lm_to_px(lms[8])
    tip4 = lm_to_px(lms[4])
    return dist(tip8, tip4) < PINCH_THRESH, tip8

def draw_glow_line(draw, pts, color, glow, width, alpha_core=220):
    if len(pts) < 2:
        return
    r, g, b = glow
    for i in range(1, len(pts)):
        draw.line([pts[i-1], pts[i]], fill=(r, g, b, 28),  width=width+18)
        draw.line([pts[i-1], pts[i]], fill=(r, g, b, 65),  width=width+8)
        draw.line([pts[i-1], pts[i]], fill=(r, g, b, 130), width=width+3)
    cr, cg, cb = color
    for i in range(1, len(pts)):
        draw.line([pts[i-1], pts[i]], fill=(cr, cg, cb, alpha_core), width=width)

def draw_glow_dot(draw, cx, cy, r, color, glow):
    rr, rg, rb = glow
    draw.ellipse((cx-r-8, cy-r-8, cx+r+8, cy+r+8), fill=(rr, rg, rb, 18))
    draw.ellipse((cx-r-4, cy-r-4, cx+r+4, cy+r+4), fill=(rr, rg, rb, 50))
    draw.ellipse((cx-r-2, cy-r-2, cx+r+2, cy+r+2), fill=(rr, rg, rb, 105))
    cr, cg, cb = color
    draw.ellipse((cx-r,   cy-r,   cx+r,   cy+r),   fill=(cr, cg, cb, 230))

def draw_hand_skeleton(draw, lms):
    pts        = [lm_to_px(lm) for lm in lms]
    bone_color = (0, 220, 255)
    bone_glow  = (0, 160, 255)
    for a, b in CONNECTIONS:
        p1, p2 = pts[a], pts[b]
        draw.line([p1, p2], fill=(*bone_glow,  25),  width=14)
        draw.line([p1, p2], fill=(*bone_glow,  60),  width=6)
        draw.line([p1, p2], fill=(*bone_color, 170), width=2)
    for i, p in enumerate(pts):
        r = 6 if i in FINGERTIPS else (5 if i == 0 else 3)
        draw_glow_dot(draw, p[0], p[1], r, bone_color, bone_glow)

def draw_sparkles(draw, cx, cy, color, glow, rng):
    cr, cg, cb = color
    gr, gg, gb = glow
    for _ in range(8):
        angle = rng.uniform(0, 2*np.pi)
        d     = rng.uniform(6, 18)
        rx    = int(cx + d * np.cos(angle))
        ry    = int(cy + d * np.sin(angle))
        r     = int(rng.integers(2, 5))
        alpha = int(rng.integers(150, 235))
        draw.ellipse((rx-r,   ry-r,   rx+r,   ry+r),   fill=(cr, cg, cb, alpha))
        draw.ellipse((rx-r-2, ry-r-2, rx+r+2, ry+r+2), fill=(gr, gg, gb, 40))

# ─────────────────────────────────────────────────────────────
# Video Processor
# ─────────────────────────────────────────────────────────────
class AirPinchProcessor(VideoProcessorBase):
    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.canvas       = Image.new("RGBA", (CW, CH), (0, 0, 0, 255))
        self.strokes      = []
        self.cur_stroke   = []
        self.cur_color    = None
        self.color_idx    = 0
        self.selected_idx = -1
        self.prev_pinch   = None
        self.prev_finger  = None
        self.two_active   = False
        self.rng          = np.random.default_rng()

    def _palette(self):
        return PALETTES[self.color_idx % len(PALETTES)]

    def _rebuild_canvas(self):
        self.canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 255))
        draw = ImageDraw.Draw(self.canvas, "RGBA")
        for i, s in enumerate(self.strokes):
            if i == self.selected_idx or len(s["pts"]) < 2:
                continue
            draw_glow_line(draw, s["pts"], s["color"], s["glow"], 3)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img     = frame.to_ndarray(format="rgb24")
        results = self._hands.process(img)

        overlay = self.canvas.copy()
        draw    = ImageDraw.Draw(overlay, "RGBA")

        pinching  = False
        pinch_pos = None
        index_pos = None
        lms       = None

        if results.multi_hand_landmarks:
            lms                 = results.multi_hand_landmarks[0].landmark
            index_pos           = lm_to_px(lms[8])
            pinching, pinch_pos = is_pinch(lms)
            index_only          = is_index_only(lms)
            two_fing            = is_two_fingers(lms)

            # Cycle color with two-finger gesture
            if two_fing and not self.two_active:
                self.two_active  = True
                self.color_idx  += 1
            if not two_fing:
                self.two_active = False

            if pinching:
                # Finalise current stroke
                if self.cur_stroke and len(self.cur_stroke) > 1 and self.cur_color:
                    self.strokes.append({
                        "pts":   list(self.cur_stroke),
                        "color": self.cur_color[0],
                        "glow":  self.cur_color[1],
                    })
                    self._rebuild_canvas()
                self.cur_stroke  = []
                self.prev_finger = None

                if self.selected_idx == -1:
                    best, best_d = -1, SELECT_THRESH
                    for i, s in enumerate(self.strokes):
                        for pt in s["pts"]:
                            d = dist(pt, pinch_pos)
                            if d < best_d:
                                best_d = d; best = i
                    self.selected_idx = best
                    self.prev_pinch   = pinch_pos
                else:
                    if self.prev_pinch:
                        dx = pinch_pos[0] - self.prev_pinch[0]
                        dy = pinch_pos[1] - self.prev_pinch[1]
                        s  = self.strokes[self.selected_idx]
                        s["pts"] = [
                            (np.clip(x+dx, 0, CW-1), np.clip(y+dy, 0, CH-1))
                            for x, y in s["pts"]
                        ]
                        self._rebuild_canvas()
                    self.prev_pinch = pinch_pos

            elif index_only and not two_fing:
                if self.selected_idx != -1:
                    self.selected_idx = -1
                    self.prev_pinch   = None
                palette = self._palette()
                if self.prev_finger:
                    if dist(index_pos, self.prev_finger) > MIN_MOVE:
                        if not self.cur_stroke:
                            self.cur_stroke = [self.prev_finger]
                            self.cur_color  = palette
                        self.cur_stroke.append(index_pos)
                else:
                    self.cur_stroke = [index_pos]
                    self.cur_color  = palette
                self.prev_finger = index_pos

            else:
                self.selected_idx = -1
                self.prev_pinch   = None
                if self.cur_stroke and len(self.cur_stroke) > 1 and self.cur_color:
                    self.strokes.append({
                        "pts":   list(self.cur_stroke),
                        "color": self.cur_color[0],
                        "glow":  self.cur_color[1],
                    })
                    self._rebuild_canvas()
                self.cur_stroke  = []
                self.prev_finger = None

        else:
            # No hand — finalise stroke
            self.selected_idx = -1
            self.prev_pinch   = None
            if self.cur_stroke and len(self.cur_stroke) > 1 and self.cur_color:
                self.strokes.append({
                    "pts":   list(self.cur_stroke),
                    "color": self.cur_color[0],
                    "glow":  self.cur_color[1],
                })
                self._rebuild_canvas()
            self.cur_stroke  = []
            self.prev_finger = None

        # ── Render ──

        # Selected stroke highlighted yellow
        if 0 <= self.selected_idx < len(self.strokes):
            s = self.strokes[self.selected_idx]
            draw_glow_line(draw, s["pts"], (255,255,60), (255,220,0), 5, alpha_core=240)

        # Current in-progress stroke
        if self.cur_stroke and len(self.cur_stroke) > 1 and self.cur_color:
            draw_glow_line(draw, self.cur_stroke,
                           self.cur_color[0], self.cur_color[1], 3)

        # Sparkles at fingertip
        if (index_pos and not pinching and lms and
                is_index_only(lms) and self.cur_stroke and self.cur_color):
            draw_sparkles(draw, index_pos[0], index_pos[1],
                          self.cur_color[0], self.cur_color[1], self.rng)

        # Pinch indicator ring
        if pinching and pinch_pos:
            draw.ellipse(
                (pinch_pos[0]-14, pinch_pos[1]-14,
                 pinch_pos[0]+14, pinch_pos[1]+14),
                outline=(255, 140, 0, 200), width=2
            )

        # Glowing hand skeleton
        if lms:
            draw_hand_skeleton(draw, lms)

        # Flatten RGBA → RGB for output
        out_rgb = Image.new("RGB", (CW, CH), (0, 0, 0))
        out_rgb.paste(overlay, mask=overlay.split()[3])

        return av.VideoFrame.from_ndarray(
            np.array(out_rgb, dtype=np.uint8), format="rgb24"
        )

# ─────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────
webrtc_streamer(
    key="airpinch",
    video_processor_factory=AirPinchProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
        "audio": False,
    },
    async_processing=True,
)
