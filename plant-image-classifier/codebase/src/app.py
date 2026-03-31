from pathlib import Path
import json
import pickle
import tempfile

import streamlit as st
from PIL import Image

from feature_extraction import FeatureExtractor


# =========================
# Config
# =========================
MODEL_PATH = Path("models/experiments/20260322_174659__rf__hsv__aug__tuned/model.pkl")
CLASS_NAMES_PATH = Path("models/class_names.json")

# Change this to match the final selected feature setting
FEATURE_SET = ["hsv"]
IMG_SIZE = (128, 128)


# =========================
# Utils
# =========================
@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


@st.cache_resource
def load_extractor():
    return FeatureExtractor(
        img_size=IMG_SIZE,
        feature_set=FEATURE_SET
    )


def predict_image(uploaded_file, model, extractor, class_names):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    feat = extractor.extract(tmp_path)
    pred_idx = model.predict([feat])[0]

    pred_class = class_names[pred_idx]

    # For models that support predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([feat])[0]
        confidence = float(probs[pred_idx])

    try:
        tmp_path.unlink()
    except Exception:
        pass

    return pred_class, confidence


# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="Plant Classifier", layout="centered")

    st.title("Campus Plant Classifier")
    st.write("Upload a plant image to predict its class.")

    model, class_names = load_model_and_classes()
    extractor = load_extractor()

    st.markdown("**Supported classes:**")
    st.write(", ".join(class_names))

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            pred_class, confidence = predict_image(uploaded_file, model, extractor, class_names)

            st.success(f"Predicted class: **{pred_class}**")

            if confidence is not None:
                st.write(f"Confidence: **{confidence:.4f}**")


if __name__ == "__main__":
    main()