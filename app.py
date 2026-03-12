import streamlit as st
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image, ImageDraw

# ------------------------------
# Configuration
# ------------------------------
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720
SELECT_THRESHOLD = 40          # pixels – distance to select a stroke
PINCH_DIST_THRESHOLD = 30      # pixels – max distance between index and thumb to consider pinched

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ------------------------------
# Helper functions
# ------------------------------
def is_pinch(landmarks, frame_w, frame_h):
    """Return True if index tip and thumb tip are close (in canvas pixels)"""
    h8 = landmarks[8]   # index tip
    h4 = landmarks[4]   # thumb tip
    x1, y1 = int(h8.x * CANVAS_WIDTH), int(h8.y * CANVAS_HEIGHT)
    x2, y2 = int(h4.x * CANVAS_WIDTH), int(h4.y * CANVAS_HEIGHT)
    dist = np.hypot(x1 - x2, y1 - y2)
    return dist < PINCH_DIST_THRESHOLD, (x1, y1)

def is_index_only(landmarks):
    """Check if only index finger is extended (others folded)"""
    index_extended = landmarks[8].y < landmarks[6].y
    middle_folded = landmarks[12].y > landmarks[10].y
    ring_folded = landmarks[16].y > landmarks[14].y
    pinky_folded = landmarks[20].y > landmarks[18].y
    return index_extended and middle_folded and ring_folded and pinky_folded

def map_to_canvas(x, y, frame_w, frame_h):
    """Map normalized hand coordinates to canvas pixel coordinates"""
    cx = int(x * CANVAS_WIDTH)
    cy = int(y * CANVAS_HEIGHT)
    return np.clip(cx, 0, CANVAS_WIDTH-1), np.clip(cy, 0, CANVAS_HEIGHT-1)

def draw_sparkles(draw, center, count=12):
    """Draw sparkle effect around a point using PIL"""
    for _ in range(count):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(5, 15)
        dx = int(dist * np.cos(angle))
        dy = int(dist * np.sin(angle))
        color = tuple(np.random.randint(200, 255, 3).tolist())
        radius = np.random.randint(2, 4)
        draw.ellipse(
            (center[0]+dx-radius, center[1]+dy-radius,
             center[0]+dx+radius, center[1]+dy+radius),
            fill=color
        )

# ------------------------------
# Video Transformer (PIL-based)
# ------------------------------
class HandDrawingTransformer(VideoTransformerBase):
    def _init_(self):
        self.canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), color="black")
        self.strokes = []                     # list of strokes, each stroke = list of points
        self.current_stroke = []               # points being drawn now
        self.selected_stroke_idx = None        # index of stroke being dragged
        self.drag_offset = (0, 0)              # offset between pinch point and stroke center
        self.prev_pinch_pos = None
        self.prev_finger_pos = None

    def transform(self, frame):
        # Get PIL image from frame
        img_pil = frame.to_image()
        img_rgb = np.array(img_pil)
        results = hands.process(img_rgb)

        # Start with a copy of the canvas
        overlay = self.canvas.copy()
        draw = ImageDraw.Draw(overlay)

        index_pos = None
        pinch_pos = None
        pinching = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img_rgb.shape
                # Draw all landmarks on overlay (palm trace) as small green circles
                for lm in hand_landmarks.landmark:
                    cx, cy = map_to_canvas(lm.x, lm.y, w, h)
                    draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill="green")

                # Index tip position
                index_tip = hand_landmarks.landmark[8]
                ix, iy = map_to_canvas(index_tip.x, index_tip.y, w, h)
                index_pos = (ix, iy)

                # Check pinch
                pinching, p_pos = is_pinch(hand_landmarks.landmark, w, h)
                if pinching:
                    pinch_pos = p_pos

                # --- Mode handling ---
                if pinching:
                    # PINCH mode: select/drag
                    if self.selected_stroke_idx is None:
                        # Try to select a stroke near pinch point
                        min_dist = SELECT_THRESHOLD
                        selected = None
                        for i, stroke in enumerate(self.strokes):
                            for pt in stroke:
                                dist = np.hypot(pt[0]-p_pos[0], pt[1]-p_pos[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    selected = i
                        if selected is not None:
                            self.selected_stroke_idx = selected
                            # Compute offset from stroke's center
                            stroke = self.strokes[selected]
                            cx = int(np.mean([p[0] for p in stroke]))
                            cy = int(np.mean([p[1] for p in stroke]))
                            self.drag_offset = (cx - p_pos[0], cy - p_pos[1])
                            self.prev_pinch_pos = p_pos
                    else:
                        # Drag the selected stroke
                        if self.prev_pinch_pos is not None:
                            dx = p_pos[0] - self.prev_pinch_pos[0]
                            dy = p_pos[1] - self.prev_pinch_pos[1]
                            stroke = self.strokes[self.selected_stroke_idx]
                            self.strokes[self.selected_stroke_idx] = [(x+dx, y+dy) for (x,y) in stroke]
                        self.prev_pinch_pos = p_pos
                else:
                    # No pinch: deselect if any
                    self.selected_stroke_idx = None
                    self.prev_pinch_pos = None

                    # Check for drawing mode (only index extended)
                    if is_index_only(hand_landmarks.landmark):
                        # Drawing
                        if self.prev_finger_pos is not None:
                            dist = np.hypot(index_pos[0]-self.prev_finger_pos[0],
                                            index_pos[1]-self.prev_finger_pos[1])
                            if dist > 5:  # avoid too many points
                                if len(self.current_stroke) == 0:
                                    self.current_stroke.append(index_pos)
                                else:
                                    self.current_stroke.append(index_pos)
                        else:
                            self.current_stroke = [index_pos]
                        self.prev_finger_pos = index_pos
                    else:
                        # Not drawing: finalize current stroke if any
                        if self.current_stroke:
                            self.strokes.append(self.current_stroke)
                            self.current_stroke = []
                        self.prev_finger_pos = None

        # Draw all stored strokes on overlay
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                draw.line([stroke[i-1], stroke[i]], fill="white", width=3)

        # Draw current stroke
        if self.current_stroke and len(self.current_stroke) > 1:
            for i in range(1, len(self.current_stroke)):
                draw.line([self.current_stroke[i-1], self.current_stroke[i]], fill="white", width=3)

        # Sparkle effect at finger position if drawing
        if index_pos and self.current_stroke and not pinching:
            draw_sparkles(draw, index_pos)

        # Highlight selected stroke
        if self.selected_stroke_idx is not None:
            stroke = self.strokes[self.selected_stroke_idx]
            for i in range(1, len(stroke)):
                draw.line([stroke[i-1], stroke[i]], fill="yellow", width=4)

        # Update canvas
        self.canvas = overlay

        # Return as numpy array (RGB)
        return np.array(self.canvas)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AirPinch – Draw with Gestures", layout="wide")

st.markdown("""
<style>
    .stVideo { width: 100%; }
    .stVideo > div { width: 100%; }
    .stVideo video { width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

st.title("🖐️ AirPinch – Draw in the Air with Sparkles")
st.markdown("""
*Instructions*:
- *Draw: Extend only your **index finger* (others folded). Move it to draw. ✨ Sparkles appear!
- *Select & Drag: Pinch **index and thumb* together near a drawn line, then move your hand while keeping the pinch. The line will follow.
- *Green dots* = your hand landmarks.
""")

webrtc_streamer(
    key="hand-draw-pinch",
    video_transformer_factory=HandDrawingTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.warning("⚠️ Ensure good lighting and that your hand is clearly visible. The canvas may take a moment to appear.")
