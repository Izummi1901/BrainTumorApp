import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


IMG_SIZE = 128
MODEL_PATH = "best_cnn_multiclass.h5"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        /* tighten top padding */
        .block-container { padding-top: 1.5rem; }
        /* nicer sidebar width */
        [data-testid="stSidebar"] { min-width: 260px; max-width: 260px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("Upload MRI")
        st.caption("Upload a brain MRI slice (JPG / PNG).")

        uploaded_file = st.file_uploader(
            "MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        st.markdown("---")
        st.subheader("Model info")
        st.write(
            "- CNN, input size: **128×128 RGB**  \n"
            "- Classes: **glioma, meningioma, pituitary, no tumor**"
        )
        st.markdown("---")
        st.caption("⚠️ For academic use only – not for clinical diagnosis.")

    st.title("🧠 Brain Tumor Detection from MRI")
    st.caption(
        "The CNN model classifies a single MRI slice into "
        "**glioma**, **meningioma**, **pituitary**, or **no tumor**."
    )

    model = load_cnn_model()

    if uploaded_file is None:
        st.info("Left side se MRI image upload karo to see predictions.")
        return

    image = Image.open(uploaded_file)
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    probs = preds[0]
    pred_index = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])

    prob_df = pd.DataFrame(
        {
            "Tumor Type": CLASS_NAMES,
            "Probability": [float(p) for p in probs],
        }
    ).sort_values("Probability", ascending=False)

    col_img, col_pred = st.columns([1.2, 1])

    with col_img:
        st.subheader("Input MRI")
        st.image(image, use_container_width=True)

    with col_pred:
        st.subheader("Prediction")

        st.markdown(
            f"""
            <div style="
                padding: 0.8rem 1rem;
                border-radius: 0.75rem;
                background-color: #111827;
                border: 1px solid #374151;
                display: inline-block;
                margin-bottom: 0.5rem;">
                <span style="font-size:0.85rem; color:#9CA3AF;">Predicted class</span><br>
                <span style="font-size:1.4rem; font-weight:600; color:#10B981;">
                    {pred_class.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confidence", f"{confidence:.3f}")

        st.markdown("#### Class probabilities")
        st.bar_chart(
            prob_df.set_index("Tumor Type")["Probability"],
        )

        st.markdown("#### Detailed probabilities")
        st.dataframe(
            prob_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Result Interpretation ")
    st.write(
        f"- The model predicts this MRI slice belongs to **{pred_class}** "
        f"with a confidence of **{confidence:.3f}**.  \n"
        "- The bar chart and table show the probability distribution "
        "over all four classes.  \n"
        "- In a real clinical workflow, this would be used **only as a decision support tool**, "
        "not a standalone diagnostic system."
    )


if __name__ == "__main__":
    main()
