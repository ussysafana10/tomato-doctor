
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Tomato Doctor",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================
# CUSTOM DESIGN
# =========================
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top left, #263b2c 0%, transparent 35%),
            radial-gradient(circle at bottom right, #3d1717 0%, transparent 30%),
            #0d1117;
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 20px 10px 10px 10px;
    }

    .logo {
        font-size: 70px;
        margin-bottom: -10px;
    }

    .hero h1 {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
    }

    .hero h1 span {
        color: #ef4444;
    }

    .hero p {
        color: #a7b0bd;
        font-size: 17px;
        margin-top: 8px;
    }

    /* Upload card */
    .upload-card {
        background: rgba(31, 41, 55, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .upload-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .upload-subtitle {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 14px;
        min-height: 52px;
        font-size: 17px;
        font-weight: 700;
        border: none;
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(239,68,68,0.35);
    }

    /* Image */
    [data-testid="stImage"] {
        border-radius: 20px;
        overflow: hidden;
        margin-top: 15px;
    }

    /* Result card */
    .result-card {
        background: rgba(22, 28, 36, 0.95);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 24px;
        margin-top: 25px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .result-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .diagnosis {
        background: linear-gradient(
            135deg,
            rgba(34,197,94,0.18),
            rgba(22,163,74,0.05)
        );
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .diagnosis-name {
        font-size: 20px;
        font-weight: 700;
        color: #86efac;
    }

    .confidence {
        color: #d1d5db;
        margin-top: 5px;
    }

    /* Info boxes */
    .info-box {
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(59,130,246,0.20);
        padding: 15px;
        border-radius: 15px;
        margin-top: 20px;
        color: #bfdbfe;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="logo">🍅</div>
    <h1>Tomato <span>Doctor</span></h1>
    <p>Smart tomato leaf disease detection</p>
</div>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load():
    labels = [
        l.strip()
        for l in open("labels.txt", encoding="utf-8")
        if l.strip()
    ]

    it = Interpreter(model_path="tomato.tflite")
    it.allocate_tensors()

    return labels, it


labels, it = load()

inp = it.get_input_details()[0]
out = it.get_output_details()[0]

size = int(inp["shape"][1])


# =========================
# UPLOAD SECTION
# =========================
st.markdown("""
<div class="upload-card">
    <div class="upload-title">📷 Upload Tomato Leaf</div>
    <div class="upload-subtitle">
        Take a clear photo of the tomato leaf for analysis.
    </div>
</div>
""", unsafe_allow_html=True)

f = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================
# IMAGE + PREDICTION
# =========================
if f:

    img = Image.open(f).convert("RGB")

    st.markdown("### 🔍 Leaf Preview")

    st.image(
        img,
        use_container_width=True
    )

    st.markdown(
        "<div style='height:8px'></div>",
        unsafe_allow_html=True
    )

    # Analyze button
    analyze = st.button(
        "🔬 Analyze Leaf",
        use_container_width=True
    )

    if analyze:

        with st.spinner("Analyzing tomato leaf..."):

            arr = np.array(
                img.resize((size, size))
            )

            if inp["dtype"] == np.uint8:
                x = arr.astype(np.uint8)
            else:
                x = arr.astype(np.float32) / 127.5 - 1.0

            it.set_tensor(
                inp["index"],
                x[None]
            )

            it.invoke()

            p = it.get_tensor(
                out["index"]
            )[0]

            top = np.argsort(p)[::-1][:3]

        # =========================
        # RESULTS
        # =========================

        st.markdown("""
        <div class="result-card">
            <div class="result-title">
                🩺 Diagnosis Result
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top result
        best = top[0]
        confidence = float(p[best])

        st.markdown(f"""
        <div class="diagnosis">
            <div class="diagnosis-name">
                🍅 {labels[best]}
            </div>
            <div class="confidence">
                Confidence: <b>{confidence * 100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )

        # Other predictions
        st.markdown("### 📊 Other Possible Results")

        for i in top[1:]:
            score = float(p[i])

            st.write(
                f"**{labels[i]}** — {score * 100:.1f}%"
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

        # Low confidence
        if confidence < 0.7:

            st.warning(
                "⚠️ Low confidence. "
                "Please retake the photo using good lighting "
                "and make sure the leaf is clearly visible."
            )

        else:

            st.success(
                "✅ The image has been analyzed successfully."
            )

        # Advice
        st.markdown("""
        <div class="info-box">
            💡 <b>Tip:</b> For better results, use a clear photo,
            good lighting, and make sure the tomato leaf fills
            most of the camera frame.
        </div>
        """, unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="app-footer">
    🍅 Tomato Doctor • AI-powered tomato leaf analysis
</div>
""", unsafe_allow_html=True)
