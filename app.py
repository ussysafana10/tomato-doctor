import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tomato Doctor",
    page_icon="🍅",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b0f14;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aeb7c2;
    font-size: 16px;
    margin-bottom: 25px;
}

.card {
    background: linear-gradient(135deg, #10251d, #123c2b);
    padding: 22px;
    border-radius: 18px;
    margin-top: 18px;
    border: 1px solid #1c6848;
}

.card-title {
    font-size: 24px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.card-text {
    color: #dce5df;
    font-size: 16px;
    line-height: 1.6;
}

.diagnosis {
    background: linear-gradient(135deg, #123b2b, #0e241b);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #238657;
    margin-top: 20px;
}

.diagnosis-title {
    color: #63e695;
    font-size: 18px;
    font-weight: 700;
}

.diagnosis-name {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence {
    color: #b9c7c0;
    font-size: 17px;
    margin-top: 5px;
}

.advice-box {
    background: #151a21;
    padding: 20px;
    border-radius: 16px;
    margin-top: 15px;
}

.advice-title {
    color: #65e694;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

.advice-text {
    color: #d8dee5;
    font-size: 16px;
    line-height: 1.7;
}

.other-title {
    color: white;
    font-size: 25px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.result-name {
    color: white;
    font-size: 17px;
    font-weight: 600;
    margin-top: 12px;
}

.footer {
    text-align: center;
    color: #68717b;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL + LABELS
# ============================================================

MODEL_PATH = "tomato.tflite"
LABELS_PATH = "labels.txt"


@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ============================================================
# LOAD MODEL
# ============================================================

try:
    interpreter = load_model()
    labels = load_labels()

except Exception as e:
    st.error("Unable to load the AI model.")
    st.error(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🍅 Tomato Doctor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered tomato leaf disease analysis</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        use_container_width=True
    )

    analyze = st.button(
        "🔍 Analyze Leaf",
        use_container_width=True
    )

    if analyze:

        # ====================================================
        # MODEL INPUT
        # ====================================================

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]

        height = int(input_shape[1])
        width = int(input_shape[2])

        resized = image.resize((width, height))

        img_array = np.array(resized).astype(np.float32)

        # Normalize image
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # ====================================================
        # PREDICTION
        # ====================================================

        interpreter.set_tensor(
            input_details[0]["index"],
            img_array
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        # Convert probabilities if necessary
        if np.max(output) > 1.0:
            output = output / np.sum(output)

        # ====================================================
        # TOP RESULTS
        # ====================================================

        top_indices = np.argsort(output)[::-1][:3]

        top_results = []

        for index in top_indices:

            if index < len(labels):
                label = labels[index]
            else:
                label = f"Class {index}"

            confidence = float(output[index]) * 100

            top_results.append(
                (label, confidence)
            )

        diagnosis = top_results[0][0]
        confidence = top_results[0][1]


        # ====================================================
        # DIAGNOSIS
        # ====================================================

        st.markdown(
            f"""
            <div class="diagnosis">

                <div class="diagnosis-title">
                    🩺 Diagnosis
                </div>

                <div class="diagnosis-name">
                    {diagnosis}
                </div>

                <div class="confidence">
                    Confidence: {confidence:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # TREATMENT / PREVENTION / RECOMMENDATION
        # ====================================================

        disease_lower = diagnosis.lower()


        # Default information
        treatment = (
            "Remove severely affected leaves and keep the plant area clean. "
            "Use an appropriate treatment recommended by a qualified "
            "agricultural professional."
        )

        prevention = (
            "Keep good spacing between plants, avoid unnecessary leaf wetness, "
            "remove infected plant material, and maintain good field hygiene."
        )

        recommendation = (
            "Continue monitoring the plant and take a clear image of the "
            "affected leaf if symptoms change or become more severe."
        )


        # ----------------------------------------------------
        # LATE BLIGHT
        # ----------------------------------------------------

        if "late blight" in disease_lower:

            treatment = (
                "Remove and destroy severely infected leaves. Improve air "
                "circulation around plants and use an appropriate fungicide "
                "according to the product label and local agricultural advice."
            )

            prevention = (
                "Avoid prolonged leaf wetness, provide adequate spacing, "
                "remove infected plant debris, and monitor plants regularly."
            )

            recommendation = (
                "Act early because Late Blight can spread quickly. Check "
                "nearby tomato plants for similar symptoms."
            )


        # ----------------------------------------------------
        # EARLY BLIGHT
        # ----------------------------------------------------

        elif "early blight" in disease_lower:

            treatment = (
                "Remove affected leaves and dispose of infected plant debris. "
                "Apply a suitable fungicide when recommended."
            )

            prevention = (
                "Avoid overhead watering, improve air circulation, and rotate "
                "crops where possible."
            )

            recommendation = (
                "Inspect lower leaves regularly because Early Blight often "
                "starts on older leaves."
            )


        # ----------------------------------------------------
        # YELLOW LEAF CURL VIRUS
        # ----------------------------------------------------

        elif "yellow leaf curl" in disease_lower:

            treatment = (
                "There is no direct cure for the virus. Remove heavily "
                "infected plants when appropriate and control whiteflies, "
                "which can spread the virus."
            )

            prevention = (
                "Control whiteflies, remove infected plants, use healthy "
                "planting material, and keep the growing area clean."
            )

            recommendation = (
                "Check nearby plants for similar symptoms and monitor for "
                "whitefly activity."
            )


        # ----------------------------------------------------
        # SEPTORIA
        # ----------------------------------------------------

        elif "septoria" in disease_lower:

            treatment = (
                "Remove infected leaves and plant debris. Improve airflow "
                "and use an appropriate fungicide when necessary."
            )

            prevention = (
                "Avoid overhead irrigation, maintain good spacing, and remove "
                "infected leaves from the growing area."
            )

            recommendation = (
                "Monitor the lower leaves frequently and remove new infected "
                "leaves early."
            )


        # ----------------------------------------------------
        # BACTERIAL SPOT
        # ----------------------------------------------------

        elif "bacterial" in disease_lower:

            treatment = (
                "Remove badly affected plant material and avoid working with "
                "wet plants. Follow local agricultural recommendations for "
                "appropriate control measures."
            )

            prevention = (
                "Use clean planting material, avoid overhead watering, and "
                "maintain good field sanitation."
            )

            recommendation = (
                "Inspect nearby plants because bacterial diseases can spread "
                "through water splash and contaminated material."
            )


        # ----------------------------------------------------
        # HEALTHY
        # ----------------------------------------------------

        elif "healthy" in disease_lower:

            treatment = (
                "No disease treatment is indicated from this image. "
                "Continue normal plant care."
            )

            prevention = (
                "Maintain proper watering, sunlight, nutrition, spacing, "
                "and regular inspection of the leaves."
            )

            recommendation = (
                "Keep monitoring the plant and upload another clear image "
                "if new symptoms appear."
            )


        # ====================================================
        # TREATMENT
        # ====================================================

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    💊 Treatment
                </div>

                <div class="card-text">
                    {treatment}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PREVENTION
        # ====================================================

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    🛡️ Prevention
                </div>

                <div class="card-text">
                    {prevention}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    💡 Recommendation
                </div>

                <div class="card-text">
                    {recommendation}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # OTHER POSSIBLE RESULTS
        # ====================================================

        if len(top_results) > 1:

            st.markdown(
                '<div class="other-title">📊 Other Possible Results</div>',
                unsafe_allow_html=True
            )

            for label, score in top_results[1:]:

                st.markdown(
                    f"""
                    <div class="result-name">
                        {label} — {score:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(score / 100, 0.0), 1.0)
                )


        # ====================================================
        # LOW CONFIDENCE WARNING
        # ====================================================

        if confidence < 70:

            st.warning(
                "⚠️ The confidence is relatively low. "
                "For better results, upload a clear photo of the tomato leaf."
            )

        else:

            st.success(
                "✅ The image has been analyzed successfully."
            )


        # ====================================================
        # PHOTO TIP
        # ====================================================

        st.markdown(
            """
            <div class="advice-box">

                <div class="advice-title">
                    💡 Photo Tip
                </div>

                <div class="advice-text">
                    For better results, use a clear image with good lighting
                    and make sure the tomato leaf is clearly visible.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">🍅 Tomato Doctor • AI-powered tomato leaf analysis</div>',
    unsafe_allow_html=True
)
