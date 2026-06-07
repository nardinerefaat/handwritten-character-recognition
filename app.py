import streamlit as st
import string
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Character Recognition",
    page_icon="✍️",
    layout="wide",
)

# Custom CSS for a more modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bolder;
    }
    button[kind="primary"] {
        background-color: var(--primary-color) !important;
        border: none !important;
        color: white !important;
        transition: 0.3s ease;
    }
    button[kind="primary"]:hover {
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        # background-color: #ffffff;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_trained_model(path="model.h5"):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        import traceback
        st.error(f"Error type: {type(e).__name__}")
        st.code(traceback.format_exc())
        raise
@st.cache_data
def get_character_classes():
    return list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)

# import h5py

model = load_trained_model()
# model = h5py.File("model.h5", "r")
classes = get_character_classes()

bg_color = "#000000"

with st.sidebar:
    st.header("Canvas Controls")
    tool = st.selectbox("Tool", ["Brush", "Eraser"])
    if tool == "Brush":
        stroke_color = "#FFFFFF"
        stroke_width = st.slider("Brush Size", 1, 25, 10)
    else:
        stroke_color = bg_color
        stroke_width = st.slider("Eraser Size", 5, 50, 20)

    st.markdown("---")
    st.header("How to Use")
    st.markdown(
        "1. Use the brush to draw a single character on the black canvas.\n"
        "2. Use the eraser to fix mistakes.\n"
        "3. Click `Predict` to display the result."
    )
    st.markdown("---")
    with st.expander("Supported Characters"):
        st.write(f"Digits: 0-9\nUppercase: A-Z\nLowercase: a-z")
    st.caption("The model expects 28x28 grayscale input similar to EMNIST.")

st.title("✍️ Handwritten Character Recognition")
st.write("Draw a digit or letter on the canvas and see the AI's prediction in real-time.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✖ Clear", use_container_width=True):
            st.rerun()
    with btn_col2:
        predict_clicked = st.button(" ֎ Predict", type="primary", use_container_width=True)

with col2:
    st.subheader("Result")
    if canvas_result.image_data is not None:
        if predict_clicked:
             # Check if there are any strokes on the canvas
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                with st.spinner("Analyzing stroke patterns..."):
                    # Image processing
                    image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                    image = image.resize((28, 28))
                    image_np = np.array(image).astype("float32") / 255.0
                    image_np = np.expand_dims(image_np, axis=(0, -1))

                    # Prediction
                    prediction = model.predict(image_np, verbose=0)[0]
                    class_index = int(np.argmax(prediction))
                    predicted_character = classes[class_index]
                    confidence = float(np.max(prediction)) * 100
                    
                    # Get top 5 predictions for the graph
                    top_5_indices = np.argsort(prediction)[-5:][::-1]
                    top_5_values = prediction[top_5_indices]
                    top_5_labels = [classes[i] for i in top_5_indices]
                    chart_data = pd.DataFrame({"Character": top_5_labels, "Confidence": top_5_values})

                    # Direct text output for Best Guess
                    st.write(f"### Best Guess: {predicted_character}")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    
                    st.write("### 📊 Probability Distribution")
                    st.bar_chart(chart_data, x="Character", y="Confidence", color="Confidence")
                    
                    with st.expander("See Preprocessed Input"):
                        st.image(image, caption=f"28x28 grayscale (Index: {class_index})", width=150)
            else:
                st.warning("⚠️ The canvas is empty! Please draw a character before clicking Predict.")
        else:
            st.info("Draw something on the left and click 'Predict'.")
