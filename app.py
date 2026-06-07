
import streamlit as st
import string
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

st.set_page_config(
    page_title="Character Recognition",
    page_icon="✍️",
    layout="wide",
)


st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
}

button[kind="primary"] {
    background-color: var(--primary-color) !important;
    border: none !important;
    color: white !important;
}

.stMetric {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


def build_model():
    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D())

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(62, activation="softmax"))

    return model


@st.cache_resource
def load_trained_model():
    model = build_model()
    model.load_weights("model.weights.h5")
    return model

@st.cache_data
def get_character_classes():
    return (
        list(string.digits)
        + list(string.ascii_uppercase)
        + list(string.ascii_lowercase)
    )

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

classes = get_character_classes()

bg_color = "#000000"

with st.sidebar:
    st.header("Canvas Controls")

    tool = st.selectbox(
        "Tool",
        ["Brush", "Eraser"]
    )

    if tool == "Brush":
        stroke_color = "#FFFFFF"
        stroke_width = st.slider(
            "Brush Size",
            1,
            25,
            10
        )
    else:
        stroke_color = bg_color
        stroke_width = st.slider(
            "Eraser Size",
            5,
            50,
            20
        )

    st.markdown("---")

    st.header("How to Use")
    st.markdown("""
1. Draw one character.
2. Click Predict.
3. View prediction results.
""")

    st.markdown("---")

    with st.expander("Supported Characters"):
        st.write(
            "Digits: 0-9\n\n"
            "Uppercase: A-Z\n\n"
            "Lowercase: a-z"
        )


st.title("✍️ Handwritten Character Recognition")
st.write(
    "Draw a digit or letter and let the CNN predict it."
)


col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=350,
        height=350,
        drawing_mode="freedraw",
        key="canvas",
    )

    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button(
            "✖ Clear",
            use_container_width=True,
        ):
            st.rerun()

    with btn2:
        predict_clicked = st.button(
            "֎ Predict",
            type="primary",
            use_container_width=True,
        )

with col2:

    st.subheader("Result")

    if predict_clicked:

        if (
            canvas_result.json_data is None
            or len(canvas_result.json_data["objects"]) == 0
        ):
            st.warning(
                "⚠️ Please draw a character first."
            )

        else:

            with st.spinner("Analyzing..."):

                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")

                image = image.resize((28, 28))

                image_np = (
                    np.array(image)
                    .astype("float32")
                    / 255.0
                )

                image_np = np.expand_dims(
                    image_np,
                    axis=(0, -1)
                )

                prediction = model.predict(
                    image_np,
                    verbose=0
                )[0]

                class_index = int(
                    np.argmax(prediction)
                )

                predicted_character = (
                    classes[class_index]
                )

                confidence = (
                    float(np.max(prediction))
                    * 100
                )

                top_5_indices = (
                    np.argsort(prediction)[-5:][::-1]
                )

                top_5_values = prediction[
                    top_5_indices
                ]

                top_5_labels = [
                    classes[i]
                    for i in top_5_indices
                ]

                chart_data = pd.DataFrame(
                    {
                        "Character": top_5_labels,
                        "Confidence": top_5_values,
                    }
                )

                st.success(
                    f"Prediction: {predicted_character}"
                )

                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

                st.subheader(
                    "📊 Top 5 Predictions"
                )

                st.bar_chart(
                    chart_data,
                    x="Character",
                    y="Confidence",
                )

                with st.expander(
                    "See Preprocessed Input"
                ):
                    st.image(
                        image,
                        width=150,
                        caption="28×28 grayscale image",
                    )
