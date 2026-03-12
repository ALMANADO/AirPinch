import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image, ImageDraw
import mediapipe as mp

# ------------------------------
# Configuration
# ------------------------------
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720
SELECT_THRESHOLD = 40
PINCH_DIST_THRESHOLD = 30

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
    return np.clip(cx, 0, CANVAS_WIDTH - 1), np.clip(cy, 0, CANVAS_HEIGHT - 1)


def draw_sparkles(draw, center, count=12):
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(5, 15)
        dx = int(dist * np.cos(angle))
        dy = int(dist * np.sin(angle))
        color = tuple(np.random.randint(200, 255, 3).tolist())
        radius = np.random.randint(2, 4)
        draw.ellipse(
            (
                center[0] + dx - radius,
                center[1] + dy - radius,
                center[0] + dx + radius,
                center[1] + dy + radius,
            ),
            fill=color,
        )


# ------------------------------
# Video Processor (updated API)
# ------------------------------
class HandDrawingProcessor(VideoProcessorBase):
    def __init__(self):
        self.canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), color="black")
        self.strokes = []
        self.current_stroke = []
        self.selected_stroke_idx = None
        self.drag_offset = (0, 0)
        self.prev_pinch_pos = None
        self.prev_finger_pos = None
        # Each instance gets its own MediaPipe Hands to avoid threading issues
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert av.VideoFrame → numpy RGB array
        img_rgb = frame.to_ndarray(format="rgb24")

        results = self._mp_hands.process(img_rgb)

        overlay = self.canvas.copy()
        draw = ImageDraw.Draw(overlay)

        index_pos = None
        pinch_pos = None
        pinching = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # first hand only

            # Draw landmarks
            for lm in hand_landmarks.landmark:
                cx, cy = map_to_canvas(lm.x, lm.y)
                draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="green")

            index_tip = hand_landmarks.landmark[8]
            ix, iy = map_to_canvas(index_tip.x, index_tip.y)
            index_pos = (ix, iy)

            pinching, p_pos = is_pinch(hand_landmarks.landmark)
            if pinching:
                pinch_pos = p_pos

            if pinching:
                # Finalise any active stroke when pinch starts
                if self.current_stroke:
                    self.strokes.append(self.current_stroke)
                    self.current_stroke = []
                self.prev_finger_pos = None

                if self.selected_stroke_idx is None:
                    # Try to select a stroke
                    min_dist = SELECT_THRESHOLD
                    selected = None
                    for i, stroke in enumerate(self.strokes):
                        for pt in stroke:
                            dist = np.hypot(pt[0] - p_pos[0], pt[1] - p_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                selected = i
                    if selected is not None:
                        self.selected_stroke_idx = selected
                        stroke = self.strokes[selected]
                        scx = int(np.mean([p[0] for p in stroke]))
                        scy = int(np.mean([p[1] for p in stroke]))
                        self.drag_offset = (scx - p_pos[0], scy - p_pos[1])
                        self.prev_pinch_pos = p_pos
                else:
                    # Drag selected stroke
                    if self.prev_pinch_pos is not None:
                        dx = p_pos[0] - self.prev_pinch_pos[0]
                        dy = p_pos[1] - self.prev_pinch_pos[1]
                        stroke = self.strokes[self.selected_stroke_idx]
                        self.strokes[self.selected_stroke_idx] = [
                            (
                                np.clip(x + dx, 0, CANVAS_WIDTH - 1),
                                np.clip(y + dy, 0, CANVAS_HEIGHT - 1),
                            )
                            for (x, y) in stroke
                        ]
                    self.prev_pinch_pos = p_pos
            else:
                # Not pinching — release selection
                self.selected_stroke_idx = None
                self.prev_pinch_pos = None

                if is_index_only(hand_landmarks.landmark):
                    if self.prev_finger_pos is not None:
                        dist = np.hypot(
                            index_pos[0] - self.prev_finger_pos[0],
                            index_pos[1] - self.prev_finger_pos[1],
                        )
                        if dist > 2:
                            self.current_stroke.append(index_pos)
                    else:
                        self.current_stroke = [index_pos]
                    self.prev_finger_pos = index_pos
                else:
                    # Finger no longer extended — finalise stroke
                    if self.current_stroke and len(self.current_stroke) > 1:
                        self.strokes.append(self.current_stroke)
                    self.current_stroke = []
                    self.prev_finger_pos = None
        else:
            # No hand visible — finalise any open stroke
            if self.current_stroke and len(self.current_stroke) > 1:
                self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self.prev_finger_pos = None
            self.selected_stroke_idx = None
            self.prev_pinch_pos = None

        # ---- Render all strokes ----
        for i, stroke in enumerate(self.strokes):
            if len(stroke) < 2:
                continue
            color = "yellow" if i == self.selected_stroke_idx else "white"
            width = 4 if i == self.selected_stroke_idx else 3
            for j in range(1, len(stroke)):
                draw.line([stroke[j - 1], stroke[j]], fill=color, width=width)

        # Render current (in-progress) stroke
        if len(self.current_stroke) > 1:
            for j in range(1, len(self.current_stroke)):
                draw.line(
                    [self.current_stroke[j - 1], self.current_stroke[j]],
                    fill="white",
                    width=3,
                )

        # Sparkles while actively drawing
        if index_pos and is_index_only_safe(results) and not pinching:
            draw_sparkles(draw, index_pos)

        self.canvas = overlay

        # Convert PIL → numpy → av.VideoFrame
        out_array = np.array(self.canvas, dtype=np.uint8)
        return av.VideoFrame.from_ndarray(out_array, format="rgb24")


def is_index_only_safe(results):
    """Safe wrapper — returns False if no hand detected."""
    if not results.multi_hand_landmarks:
        return False
    return is_index_only(results.multi_hand_landmarks[0].landmark)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AirPinch – Draw with Gestures", layout="wide")

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a0a; }
    [data-testid="stHeader"] { background: transparent; }
    h1 { color: #ffffff; font-family: monospace; }
    p, li { color: #cccccc; }
    .stAlert { background: #1a1a1a; border: 1px solid #333; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🖐️ AirPinch – Draw in the Air")

st.markdown(
    """
**How to use:**
- ✏️ **Draw** — Extend only your **index finger** (fold thumb, middle, ring, pinky). Move to draw. Sparkles appear!
- 🤏 **Select & Drag** — **Pinch** index + thumb together near a drawn line, then drag while pinching. Line turns yellow.
- 🟢 **Green dots** = hand landmarks detected.
- 🔄 Refresh the page to clear the canvas.
"""
)

webrtc_streamer(
    key="airpinch",
    video_processor_factory=HandDrawingProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.warning(
    "⚠️ Ensure good lighting and your hand is clearly visible to the camera. "
    "Allow camera access when prompted by your browser."
)
