import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# ------------------------------
# Configuration
# ------------------------------
CANVAS_WIDTH = 960          # Larger canvas
CANVAS_HEIGHT = 720
SELECT_THRESHOLD = 40       # pixels – distance to select a stroke
PINCH_DIST_THRESHOLD = 30   # pixels – max distance between index tip and thumb tip to consider pinched

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ------------------------------
# Helper functions
# ------------------------------
def is_pinch(landmarks, frame_w, frame_h):
    """Return True if index tip and thumb tip are close (in canvas pixels)"""
    h8 = landmarks[8]   # index tip
    h4 = landmarks[4]   # thumb tip
    # Convert normalized to canvas pixels
    x1, y1 = int(h8.x * CANVAS_WIDTH), int(h8.y * CANVAS_HEIGHT)
    x2, y2 = int(h4.x * CANVAS_WIDTH), int(h4.y * CANVAS_HEIGHT)
    dist = np.hypot(x1 - x2, y1 - y2)
    return dist < PINCH_DIST_THRESHOLD, (x1, y1)  # return pinch status and index tip position

def is_index_only(landmarks):
    """Check if only index finger is extended (others folded)"""
    # Thumb extended if tip (4) is to the right of IP (3) for right hand? We'll simplify: just check if other fingers are folded.
    # We'll use a more robust method: compare y-coordinates of tip vs pip for each finger.
    tips_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    # For index, we want tip above pip
    index_extended = landmarks[8].y < landmarks[6].y
    # For others, we want tip below pip (folded)
    others_folded = all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(tips_ids[1:], pip_ids[1:]))
    # Thumb: we consider thumb extended if tip is to the left of IP? But for simplicity, ignore thumb for now.
    # We'll require thumb not pinched (i.e., not too close) – but pinch detection is separate.
    # For drawing mode, we want only index extended and no pinch.
    return index_extended and others_folded

def map_to_canvas(x, y, frame_w, frame_h):
    """Map normalized hand coordinates to canvas pixel coordinates"""
    cx = int(x * CANVAS_WIDTH)
    cy = int(y * CANVAS_HEIGHT)
    return np.clip(cx, 0, CANVAS_WIDTH-1), np.clip(cy, 0, CANVAS_HEIGHT-1)

def draw_sparkles(img, center, count=12):
    """Draw sparkle effect around a point"""
    for _ in range(count):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(5, 15)
        dx = int(dist * np.cos(angle))
        dy = int(dist * np.sin(angle))
        color = (np.random.randint(200,255), np.random.randint(200,255), np.random.randint(200,255))
        cv2.circle(img, (center[0]+dx, center[1]+dy), np.random.randint(2,4), color, -1)

# ------------------------------
# Video Transformer
# ------------------------------
class HandDrawingTransformer(VideoTransformerBase):
    def _init_(self):
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        self.strokes = []                     # list of strokes, each stroke = list of points
        self.current_stroke = []               # points being drawn now
        self.selected_stroke_idx = None        # index of stroke being dragged
        self.drag_offset = (0, 0)              # offset between pinch point and stroke center
        self.prev_pinch_pos = None              # previous pinch position for delta calculation
        self.prev_finger_pos = None              # for drawing continuity

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Create a fresh overlay for hand landmarks and strokes
        overlay = self.canvas.copy()
        index_pos = None
        pinch_pos = None
        pinching = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img.shape
                # Draw all landmarks on overlay (palm trace)
                for lm in hand_landmarks.landmark:
                    cx, cy = map_to_canvas(lm.x, lm.y, w, h)
                    cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1)

                # Get index tip position
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
                            # Translate all points
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
                            # Add point and draw line (if moved enough)
                            dist = np.hypot(index_pos[0]-self.prev_finger_pos[0], index_pos[1]-self.prev_finger_pos[1])
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
                cv2.line(overlay, stroke[i-1], stroke[i], (255, 255, 255), 3)

        # Draw current stroke
        if self.current_stroke and len(self.current_stroke) > 1:
            for i in range(1, len(self.current_stroke)):
                cv2.line(overlay, self.current_stroke[i-1], self.current_stroke[i], (255, 255, 255), 3)

        # Sparkle effect at finger position if drawing
        if index_pos and self.current_stroke and not pinching:
            draw_sparkles(overlay, index_pos)

        # Highlight selected stroke
        if self.selected_stroke_idx is not None:
            stroke = self.strokes[self.selected_stroke_idx]
            for i in range(1, len(stroke)):
                cv2.line(overlay, stroke[i-1], stroke[i], (0, 255, 255), 4)

        # Update canvas
        self.canvas = overlay

        # Return the canvas as BGR for display
        return cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)

# ------------------------------
# Streamlit UI with full-width canvas
# ------------------------------
st.set_page_config(page_title="Air Draw with Pinch Drag", layout="wide")

st.markdown("""
<style>
    .stVideo { width: 100%; }
    .stVideo > div { width: 100%; }
    .stVideo video { width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

st.title("🖐️ Air Draw & Pinch to Drag")
st.markdown("""
*Instructions*:
- *Draw: Extend only your **index finger* (others folded). Move it to draw. ✨ Sparkles appear!
- *Select & Drag: Pinch **index and thumb* together near a drawn line, then move your hand while keeping the pinch. The line will follow.
- *Green dots* = your hand landmarks.
""")

webrtc_streamer(
    key="hand-draw-pinch",
    video_transformer_factory=HandDrawingTransformer,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
    async_processing=True,
)

st.warning("⚠️ Ensure good lighting and that your hand is clearly visible. The canvas may take a moment to appear.")
