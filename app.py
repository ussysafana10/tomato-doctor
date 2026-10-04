import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Tomato Doctor", page_icon="🍅", layout="centered")


# ================= STYLE =================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, #263b2c 0%, transparent 35%),
        radial-gradient(circle at bottom right, #3d1717 0%, transparent 30%),
        #0d1117;
    color: white;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 850px; padding-top: 2rem; padding-bottom: 4rem; }

.hero { text-align: center; padding: 20px 10px 10px 10px; }
.logo { font-size: 70px; margin-bottom: -10px; }
.hero h1 { font-size: 48px; font-weight: 800; margin: 0; color: #fff; }
.hero h1 span { color: #ef4444; }
.hero p { color: #a7b0bd; font-size: 17px; margin-top: 8px; }

.upload-card {
    background: rgba(31,41,55,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px; padding: 25px; margin-top: 25px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
}
.upload-title { font-size: 22px; font-weight: 700; }
.upload-subtitle { color: #9ca3af; font-size: 14px; margin-top: 6px; }

.stButton > button {
    width: 100%; border-radius: 14px; min-height: 52px;
    font-size: 17px; font-weight: 700; border: none;
    background: linear-gradient(90deg, #ef4444, #dc2626); color: white;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(239,68,68,0.35);
}

[data-testid="stImage"] { border-radius: 20px; overflow: hidden; margin-top: 15px; }

.result-card {
    background: rgba(22,28,36,0.95);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px; padding: 24px; margin-top: 25px;
}
.result-title { font-size: 24px; font-weight: 800; }

.diagnosis {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(22,163,74,0.05));
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 18px; padding: 18px; margin-bottom: 12px;
}
.diagnosis-name { font-size: 20px; font-weight: 700; color: #86efac; }
.confidence { color: #d1d5db; margin-top: 5px; }

.advice-card {
    border-radius: 20px; padding: 20px; margin-top: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
.treatment { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.25); }
.prevention { background: rgba(59,130,246,0.10); border-color: rgba(59,130,246,0.25); }
.recommendation { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.25); }
.advice-title { font-size: 21px; font-weight: 800; margin-bottom: 10px; }
.advice-text { color: #d1d5db; line-height: 1.7; }

.app-footer { text-align: center; color: #6b7280; font-size: 13px; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("""
<div class="hero">
    <div class="logo">🍅</div>
    <h1>Tomato <span>Doctor</span></h1>
    <p>Smart tomato leaf disease detection</p>
</div>
""", unsafe_allow_html=True)


# ================= DISEASE INFORMATION =================
# Keys are matched against the model label with "_" replaced by spaces.
disease_info = {
    "late blight": {
        "treatment": "Remove and safely dispose of badly affected leaves and plants. Avoid working with wet foliage because moisture helps the disease spread. For severe infections, consult a local agricultural extension officer about a registered fungicide and how to use it.",
        "prevention": "Keep good spacing between plants for air circulation. Avoid overhead watering, remove infected plant material, and do not leave diseased tomato debris in the field.",
        "recommendation": "Act quickly because late blight can spread rapidly in cool, humid conditions. Monitor nearby tomato plants regularly for new symptoms."
    },
    "early blight": {
        "treatment": "Remove severely affected leaves and dispose of infected material. Improve air circulation and avoid prolonged leaf wetness. If it keeps spreading, ask an agricultural extension officer about suitable registered products.",
        "prevention": "Use clean planting material, provide adequate spacing, reduce leaf wetness, and remove plant debris after harvest. Crop rotation helps reduce disease pressure.",
        "recommendation": "Check lower leaves regularly because symptoms often appear there first. Remove affected foliage early."
    },
    "target spot": {
        "treatment": "Remove heavily affected leaves and improve airflow around plants. Avoid prolonged leaf wetness. Consult an agricultural extension officer for locally registered control options if it spreads.",
        "prevention": "Maintain spacing, avoid overhead irrigation, remove crop debris after harvest, and rotate crops where possible.",
        "recommendation": "Inspect leaves regularly, especially in warm, humid weather, and remove affected foliage early."
    },
    "septoria": {
        "treatment": "Remove infected leaves where practical and keep them away from healthy plants. Improve airflow and avoid splashing water onto leaves. For significant disease, consult an agricultural extension officer.",
        "prevention": "Avoid overhead irrigation, maintain plant spacing, remove infected debris, and rotate crops where possible.",
        "recommendation": "Inspect plants frequently and remove newly infected leaves early. Keep foliage as dry as possible."
    },
    "yellow leaf curl virus": {
        "treatment": "There is no direct cure for an infected plant. Remove severely affected plants when appropriate to reduce the source of infection, and manage whiteflies (the insect vector) with locally recommended practices.",
        "prevention": "Use healthy planting material, control whiteflies, remove heavily infected plants, and control weeds that can host the virus.",
        "recommendation": "Check nearby plants because the virus can spread. If symptoms are widespread, contact an agricultural extension officer."
    },
    "mosaic virus": {
        "treatment": "Viral diseases generally have no direct cure. Remove severely infected plants where appropriate and avoid spreading sap between plants through contaminated tools or hands.",
        "prevention": "Use healthy seeds or seedlings, disinfect tools, control insect vectors, and remove infected plants promptly.",
        "recommendation": "Avoid touching healthy plants right after handling infected ones. Monitor the whole field for similar symptoms."
    },
    "bacterial spot": {
        "treatment": "Remove heavily infected leaves or plants where practical and avoid working with wet plants. Consult an agricultural extension officer for locally registered treatment options.",
        "prevention": "Use clean seed and planting material, avoid overhead watering, improve airflow, and remove infected crop debris.",
        "recommendation": "Inspect leaves and fruit regularly. Early removal of infected material helps reduce spread."
    },
    "bacterial speck": {
        "treatment": "Remove severely affected plant material and avoid spreading water between plants. Seek local agricultural advice on registered control options when necessary.",
        "prevention": "Use healthy planting material, maintain good spacing, avoid unnecessary leaf wetness, and remove infected debris.",
        "recommendation": "Monitor new leaves and fruit closely, especially during cool, wet periods."
    },
    "leaf mold": {
        "treatment": "Remove badly affected leaves and improve ventilation. Reduce humidity and prolonged leaf wetness. Consult an agricultural extension officer if more control is needed.",
        "prevention": "Improve airflow, avoid excessive humidity, provide adequate spacing, and avoid overhead irrigation.",
        "recommendation": "Keep greenhouse or field conditions well ventilated and inspect the lower leaves regularly."
    },
    "spider mites": {
        "treatment": "Inspect the underside of leaves carefully. Remove badly affected leaves where practical and seek local advice on suitable pest-control methods.",
        "prevention": "Reduce plant stress, monitor leaves regularly, and encourage beneficial insects. Avoid unnecessary broad-spectrum pesticides.",
        "recommendation": "Check the underside of leaves because mites are hard to notice early."
    },
    "healthy": {
        "treatment": "No disease treatment is indicated based on this prediction. Continue normal crop care and monitoring.",
        "prevention": "Maintain good spacing, adequate nutrition, proper watering, field hygiene, and regular inspection for early signs of pests or disease.",
        "recommendation": "Keep monitoring the plant regularly. Early detection matters if symptoms appear later."
    },
}


# ================= HELPERS =================
def pretty(label):
    return (label.replace("___", " — ")
                 .replace("__", " — ")
                 .replace("_", " ")
                 .strip())


# ================= LOAD MODEL =================
@st.cache_resource
def load():
    labels = [l.strip() for l in open("labels.txt", encoding="utf-8") if l.strip()]
    it = Interpreter(model_path="tomato.tflite")
    it.allocate_tensors()
    return labels, it


labels, it = load()
inp = it.get_input_details()[0]
out = it.get_output_details()[0]
size = int(inp["shape"][1])


# ================= UPLOAD =================
st.markdown("""
<div class="upload-card">
    <div class="upload-title">📷 Upload Tomato Leaf</div>
    <div class="upload-subtitle">Take a clear photo of the tomato leaf for analysis.</div>
</div>
""", unsafe_allow_html=True)

f = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ================= PREDICTION =================
if f:
    img = Image.open(f).convert("RGB")

    st.markdown("### 🔍 Leaf Preview")
    st.image(img, use_container_width=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    analyze = st.button("🔬 Analyze Leaf", use_container_width=True)

    if analyze:
        with st.spinner("Analyzing tomato leaf..."):
            arr = np.array(img.resize((size, size)))

            # IMPORTANT: the model already applies mobilenet_v2.preprocess_input
            # internally, so pass raw 0-255 pixels. Do NOT divide by 255 or 127.5.
            if inp["dtype"] == np.uint8:
                x = arr.astype(np.uint8)
            else:
                x = arr.astype(np.float32)

            it.set_tensor(inp["index"], x[None])
            it.invoke()
            p = it.get_tensor(out["index"])[0]
            top = np.argsort(p)[::-1][:3]

        best = top[0]
        confidence = float(p[best])
        raw_label = labels[best]
        clean_label = pretty(raw_label)

        st.markdown("""
        <div class="result-card">
            <div class="result-title">🩺 Diagnosis Result</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="diagnosis">
            <div class="diagnosis-name">🍅 {clean_label}</div>
            <div class="confidence">Confidence: <b>{confidence * 100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(confidence, 0.0), 1.0))

        # Find disease info ("_" -> space so keys like "early blight" match)
        label_lower = raw_label.lower().replace("_", " ")
        selected_info = None
        for disease_name, info in disease_info.items():
            if disease_name in label_lower:
                selected_info = info
                break

        if selected_info:
            for css, title, key in [
                ("treatment", "💊 Treatment", "treatment"),
                ("prevention", "🛡️ Prevention", "prevention"),
                ("recommendation", "🌱 Recommendation", "recommendation"),
            ]:
                st.markdown(f"""
                <div class="advice-card {css}">
                    <div class="advice-title">{title}</div>
                    <div class="advice-text">{selected_info[key]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                "ℹ️ No specific treatment information is available for this "
                "diagnosis yet. Please consult a local agricultural extension officer."
            )

        # Other results
        st.markdown("### 📊 Other Possible Results")
        for i in top[1:]:
            score = float(p[i])
            st.write(f"**{pretty(labels[i])}** — {score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

        # Confidence message
        if confidence < 0.7:
            st.warning(
                "⚠️ Low confidence. Please retake the photo using good lighting "
                "and make sure the leaf is clearly visible."
            )
        else:
            st.success("✅ The image has been analyzed successfully.")

        st.markdown("""
        <div class="advice-card recommendation">
            <div class="advice-title">💡 Photo Tip</div>
            <div class="advice-text">
                For better results, use a clear photo, good lighting, and make sure
                the tomato leaf fills most of the camera frame.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================= FOOTER =================
st.markdown("""
<div class="app-footer">🍅 Tomato Doctor • AI-powered tomato leaf analysis</div>
""", unsafe_allow_html=True)
