"""
HazardLens — Live Dashboard
-------------------------------
Streamlit app: upload a video, see PPE and zone violations flagged
in real time as it plays, and view a running violation log.

HOW TO RUN:
    streamlit run app.py

BEFORE RUNNING:
    Make sure best.pt (custom YOLO weights) and yolov8n.pt (pretrained)
    are in the same folder as this app.py.
"""

import time
import os
import importlib
import cv2
import pandas as pd
import streamlit as st

import detector
import zone_utils

importlib.reload(detector)
importlib.reload(zone_utils)

from detector import SafeZoneDetector
from zone_utils import ZONE_PRESETS


# ─────────────────────────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────────────────────────

# Get the folder where this app.py file is located.
# This makes the paths work correctly on Streamlit Cloud.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PPE_WEIGHTS = os.path.join(BASE_DIR, "best.pt")
PERSON_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")


# ─────────────────────────────────────────────────────────────
# Streamlit page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HazardLens",
    page_icon="🦺",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────
# Custom styling
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    :root {
        --safety-orange: #FF6B35;
        --navy: #1A1A2E;
        --navy-light: #22223A;
    }

    .stApp {
        background-color: #F5F5F7;
    }

    .sz-header {
        background: linear-gradient(
            90deg,
            var(--navy) 0%,
            var(--navy-light) 100%
        );
        padding: 1.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border-left: 6px solid var(--safety-orange);
    }

    .sz-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }

    .sz-header p {
        color: #C9C9D4;
        margin: 0.2rem 0 0 0;
        font-size: 0.95rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .status-live {
        background-color: #E8F8EE;
        color: #1B7F3E;
    }

    .status-stopped {
        background-color: #F0F0F2;
        color: #6B6B76;
    }

    .pulse-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 0 0 rgba(34,197,94, 0.7);
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(34,197,94, 0.6);
        }

        70% {
            box-shadow: 0 0 0 8px rgba(34,197,94, 0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(34,197,94, 0);
        }
    }

    .stat-card {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        color: white;
    }

    .stat-card .label {
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stat-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .card-ppe {
        background: linear-gradient(
            135deg,
            #D7263D 0%,
            #A31621 100%
        );
    }

    .card-zone {
        background: linear-gradient(
            135deg,
            #FF9F1C 0%,
            #E8720C 100%
        );
    }

    section[data-testid="stSidebar"] {
        background-color: var(--navy);
    }

    section[data-testid="stSidebar"] * {
        color: #EDEDF2 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="sz-header">
    <h1>🦺 HazardLens</h1>
    <p>Real-time PPE compliance and restricted-zone monitoring</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"]
)

conf_threshold = st.sidebar.slider(
    "Detection confidence",
    0.1,
    0.9,
    0.4,
    0.05
)

zone_preset_name = st.sidebar.selectbox(
    "📐 Restricted Zone Area",
    list(ZONE_PRESETS.keys()),
    index=0,
)

selected_zone = ZONE_PRESETS[zone_preset_name]

start_button = st.sidebar.button(
    "▶  Start monitoring",
    use_container_width=True
)

stop_button = st.sidebar.button(
    "⏹  Stop",
    use_container_width=True
)


# ─────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────

if "violations" not in st.session_state:
    st.session_state.violations = []

if "running" not in st.session_state:
    st.session_state.running = False


if start_button:
    st.session_state.running = True

    # Clear old violations when starting a new monitoring session
    st.session_state.violations = []


if stop_button:
    st.session_state.running = False


# ─────────────────────────────────────────────────────────────
# Status badge
# ─────────────────────────────────────────────────────────────

if st.session_state.running:

    st.markdown(
        '<span class="status-badge status-live">'
        '<span class="pulse-dot"></span> MONITORING ACTIVE</span>',
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        '<span class="status-badge status-stopped">'
        '⏸ STOPPED</span>',
        unsafe_allow_html=True,
    )


st.write("")


# ─────────────────────────────────────────────────────────────
# Layout: video on the left, metrics + log on the right
# ─────────────────────────────────────────────────────────────

col_video, col_stats = st.columns([2, 1])


with col_video:

    video_placeholder = st.empty()


with col_stats:

    st.subheader("Live stats")

    stat_col1, stat_col2 = st.columns(2)

    metric_ppe_placeholder = stat_col1.empty()
    metric_zone_placeholder = stat_col2.empty()


    def render_stat_cards(ppe_count: int, zone_count: int):
        """Render the PPE and zone violation count cards."""

        metric_ppe_placeholder.markdown(
            f"""
            <div class="stat-card card-ppe">
                <div class="label">
                    🪖 PPE Violations
                </div>
                <div class="value">
                    {ppe_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        metric_zone_placeholder.markdown(
            f"""
            <div class="stat-card card-zone">
                <div class="label">
                    🚧 Zone Intrusions
                </div>
                <div class="value">
                    {zone_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    render_stat_cards(0, 0)

    st.subheader("Violation log")

    log_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────
# Violation log
# ─────────────────────────────────────────────────────────────

def render_log(violations: list):
    """
    Render the violation log as a native Streamlit dataframe.

    Uses st.dataframe() instead of raw HTML to avoid rendering
    issues where HTML tags appear as text.
    """

    if not violations:

        log_placeholder.info(
            "No violations logged yet."
        )

        return


    # Show newest violations first, limit to 50
    display_data = violations[::-1][:50]

    df = pd.DataFrame(display_data)

    log_placeholder.dataframe(
        df,
        hide_index=True,
        height=320,
    )


render_log(st.session_state.violations)


# ─────────────────────────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────────────────────────

if st.session_state.running:

    # ── Load detector ──────────────────────────────────────────

    try:

        detector = SafeZoneDetector(
            ppe_weights=PPE_WEIGHTS,
            person_weights=PERSON_WEIGHTS,
            conf_threshold=conf_threshold,
            zone_coords=selected_zone,
        )

    except Exception as e:

        st.error(
            f"Could not load model weights.\n\n"
            f"PPE model path:\n{PPE_WEIGHTS}\n\n"
            f"Person model path:\n{PERSON_WEIGHTS}\n\n"
            f"Error:\n{e}"
        )

        st.stop()


    # ── Load video ─────────────────────────────────────────────

    if uploaded_file is None:

        # Check for previously uploaded temporary video
        temp_path = os.path.join(
            BASE_DIR,
            "temp_uploaded_video.mp4"
        )

        # Check for demo video
        demo_path = os.path.join(
            BASE_DIR,
            "demo.mp4"
        )

        if os.path.exists(temp_path):

            video_path = temp_path

        elif os.path.exists(demo_path):

            video_path = demo_path

        else:

            st.warning(
                "Please upload a video file in the sidebar first."
            )

            st.stop()

    else:

        video_path = os.path.join(
            BASE_DIR,
            "temp_uploaded_video.mp4"
        )

        uploaded_file.seek(0)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())


    # ── Open video ─────────────────────────────────────────────

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        st.error(
            "Could not open the uploaded video. "
            "Please try another MP4, MOV, or AVI file."
        )

        st.session_state.running = False

        st.stop()


    ppe_count = 0
    zone_count = 0


    # ─────────────────────────────────────────────────────────
    # Process frames
    # ─────────────────────────────────────────────────────────

    while cap.isOpened() and st.session_state.running:

        ret, frame = cap.read()

        if not ret:
            break


        # Process current frame through YOLO detector
        annotated_frame, violations = detector.process_frame(frame)


        # ── Record new violation events ────────────────────────

        for v in violations:

            st.session_state.violations.append(
                {
                    "Time": time.strftime(
                        "%H:%M:%S",
                        time.localtime(v.timestamp)
                    ),
                    "Type": v.type,
                    "Confidence": f"{v.confidence:.2f}",
                    "Details": v.details,
                }
            )


            if v.type == "INTRUSION":

                zone_count += 1
                ppe_count += 1

            elif v.type == "PPE":

                ppe_count += 1

            elif v.type == "ZONE":

                zone_count += 1


        # ── Update video display ──────────────────────────────

        video_placeholder.image(
            cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            ),
            channels="RGB",
            use_container_width=True,
        )


        # ── Update statistics ─────────────────────────────────

        render_stat_cards(
            ppe_count,
            zone_count
        )


        # ── Update violation log ──────────────────────────────

        render_log(
            st.session_state.violations
        )


        # Small delay to control processing/display speed
        time.sleep(0.03)


    # ─────────────────────────────────────────────────────────
    # Release video
    # ─────────────────────────────────────────────────────────

    cap.release()


    # ─────────────────────────────────────────────────────────
    # End of video
    # ─────────────────────────────────────────────────────────

    if ppe_count == 0 and zone_count == 0:

        st.info(
            "Video processing complete. "
            "No violations detected."
        )

    else:

        st.success(
            f"Video processing complete. "
            f"PPE violations: {ppe_count} | "
            f"Zone intrusions: {zone_count}"
        )


else:

    video_placeholder.info(
        "Click **Start monitoring** in the sidebar to begin."
    )