import numpy as np, streamlit as st
from PIL import Image
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(page_title="Tomato Doctor", page_icon="🍅")
st.title("🍅 Tomato Doctor")

@st.cache_resource
def load():
    labels = [l.strip() for l in open("labels.txt") if l.strip()]
    it = Interpreter(model_path="tomato.tflite")
    it.allocate_tensors()
    return labels, it

labels, it = load()
inp = it.get_input_details()[0]
out = it.get_output_details()[0]
size = int(inp["shape"][1])

f = st.file_uploader("Take / choose leaf photo", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, use_container_width=True)
    arr = np.array(img.resize((size, size)))
    if inp["dtype"] == np.uint8:
        x = arr.astype(np.uint8)
    else:
        x = arr.astype(np.float32) / 127.5 - 1.0
    it.set_tensor(inp["index"], x[None])
    it.invoke()
    p = it.get_tensor(out["index"])[0]
    top = np.argsort(p)[::-1][:3]
    for i in top:
        st.write(f"**{labels[i]}**: {p[i]*100:.1f}%")
        st.progress(float(p[i]))
    if p[top[0]] < 0.7:
        st.warning("Low confidence. Retake the photo or ask an extension officer.")
