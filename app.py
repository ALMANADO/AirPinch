import streamlit as st
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image, ImageDraw
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import time

# ------------------------------
# Configuration
# ------------------------------
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720
SELECT_THRESHOLD = 40
PINCH_DIST_THRESHOLD = 30

# ------------------------------
# Helper functions
# ------------------------------
def is_pinch(landmarks):
    h8 = landmarks[8]
    h4 = landmarks[4]
    x1, y1 = int(h8.x * CANVAS_WIDTH), int(h8.y * CANVAS_HEIGHT)
    x2, y2 = int(h4.x * CANVAS_WIDTH), int(h4.y * CANVAS_HEIGHT)
    dist = np.hypot(x1 - x2, y1 - y2)
    return dist < PINCH_DIST_THRESHOLD, (x1, y1)

def is_index_only(landmarks):
    index_extended = landmarks[8].y < landmarks[6].y
    middle_folded = landmarks[12].y > landmarks[10].y
    ring_folded = landmarks[16].y > landmarks[14].y
    pinky_folded = landmarks[20].y > landmarks[18].y
    return index_extended and middle_folded and ring_folded and pinky_folded

def map_to_canvas(x, y):
    cx = int(x * CANVAS_WIDTH)
    cy = int(y * CANVAS_HEIGHT)
    return np.clip(cx, 0, CANVAS_WIDTH-1), np.clip(cy, 0, CANVAS_HEIGHT-1)

def draw_sparkles(draw, center, count=12):
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
# Video Transformer
# ------------------------------
class HandDrawingTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), color="black")
        self.strokes = []
        self.current_stroke = []
        self.selected_stroke_idx = None
        self.drag_offset = (0, 0)
        self.prev_pinch_pos = None
        self.prev_finger_pos = None
        self.timestamp_ms = 0  # For video mode timestamps

        # Set up MediaPipe Hand Landmarker (Tasks API)
        BaseOptions = mp_tasks.BaseOptions
        HandLandmarker = mp_vision.HandLandmarker
        HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
        VisionRunningMode = mp_vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def transform(self, frame):
        img_pil = frame.to_image()
        img_rgb = np.array(img_pil)
        h, w, _ = img_rgb.shape

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Perform hand landmark detection
        results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += 33  # Increment timestamp (assuming ~30 FPS)

        overlay = self.canvas.copy()
        draw = ImageDraw.Draw(overlay)

        index_pos = None
        pinch_pos = None
        pinching = False

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]  # First (only) hand

            # Draw landmarks
            for lm in hand_landmarks:
                cx, cy = map_to_canvas(lm.x, lm.y)
                draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill="green")

            # Get index tip position
            index_tip = hand_landmarks[8]
            ix, iy = map_to_canvas(index_tip.x, index_tip.y)
            index_pos = (ix, iy)

            # Check for pinch
            pinching, p_pos = is_pinch(hand_landmarks)
            if pinching:
                pinch_pos = p_pos

            if pinching:
                if self.selected_stroke_idx is None:
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
                        stroke = self.strokes[selected]
                        cx = int(np.mean([p[0] for p in stroke]))
                        cy = int(np.mean([p[1] for p in stroke]))
                        self.drag_offset = (cx - p_pos[0], cy - p_pos[1])
                        self.prev_pinch_pos = p_pos
                else:
                    if self.prev_pinch_pos is not None:
                        dx = p_pos[0] - self.prev_pinch_pos[0]
                        dy = p_pos[1] - self.prev_pinch_pos[1]
                        stroke = self.strokes[self.selected_stroke_idx]
                        self.strokes[self.selected_stroke_idx] = [(x+dx, y+dy) for (x,y) in stroke]
                    self.prev_pinch_pos = p_pos
            else:
                self.selected_stroke_idx = None
                self.prev_pinch_pos = None
                if is_index_only(hand_landmarks):
                    if self.prev_finger_pos is not None:
                        dist = np.hypot(index_pos[0]-self.prev_finger_pos[0],
                                        index_pos[1]-self.prev_finger_pos[1])
                        if dist > 5:
                            if len(self.current_stroke) == 0:
                                self.current_stroke.append(index_pos)
                            else:
                                self.current_stroke.append(index_pos)
                    else:
                        self.current_stroke = [index_pos]
                    self.prev_finger_pos = index_pos
                else:
                    if self.current_stroke:
                        self.strokes.append(self.current_stroke)
                        self.current_stroke = []
                    self.prev_finger_pos = None

        # Draw persisted strokes
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                draw.line([stroke[i-1], stroke[i]], fill="white", width=3)

        # Draw current stroke
        if self.current_stroke and len(self.current_stroke) > 1:
            for i in range(1, len(self.current_stroke)):
                draw.line([self.current_stroke[i-1], self.current_stroke[i]], fill="white", width=3)

        # Draw sparkles if drawing
        if index_pos and self.current_stroke and not pinching:
            draw_sparkles(draw, index_pos)

        # Highlight selected stroke
        if self.selected_stroke_idx is not None:
            stroke = self.strokes[self.selected_stroke_idx]
            for i in range(1, len(stroke)):
                draw.line([stroke[i-1], stroke[i]], fill="yellow", width=4)

        self.canvas = overlay
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
**Instructions**:
- **Draw**: Extend only your **index finger** (others folded). Move it to draw. ✨ Sparkles appear!
- **Select & Drag**: Pinch **index and thumb** together near a drawn line, then move your hand while keeping the pinch. The line will follow.
- **Green dots** = your hand landmarks.
""")
webrtc_streamer(
    key="hand-draw-pinch",
    video_transformer_factory=HandDrawingTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
st.warning("⚠️ Ensure good lighting and that your hand is clearly visible. The canvas may take a moment to appear.")
