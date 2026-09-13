import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cyclone AI",
    page_icon="🌀",
    layout="wide"
)


# ==========================================
# PATHS
# ==========================================

PROJECT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT
    / "models"
    / "best_model_v2.pth"
)


# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# HEADER
# ==========================================

st.title("🌀 Cyclone AI Monitoring System")

st.markdown(
    """
    ### AI-based Cyclone Intensity Estimation

    Upload an **INSAT-3D infrared satellite image** and
    the AI model will estimate cyclone intensity.
    """
)


st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("System Information")

st.sidebar.write(
    "Model: ResNet18"
)

st.sidebar.write(
    "Input: INSAT-3D IR"
)

st.sidebar.write(
    "Prediction: Cyclone intensity"
)

st.sidebar.write(
    "Unit: knots"
)

st.sidebar.write(
    "Model version: V2"
)


# ==========================================
# UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload INSAT-3D IR satellite image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")


    # Show image

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Satellite Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # Prepare image

    input_tensor = transform(
        image
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # Prediction

    with torch.no_grad():

        output = model(
            input_tensor
        )

        normalized_prediction = (
            output.item()
        )


    # Convert back to knots

    min_intensity = 25.0
    max_intensity = 128.0

    prediction = (
        normalized_prediction
        * (max_intensity - min_intensity)
        + min_intensity
    )


    # Keep within dataset range

    prediction = max(
        min_intensity,
        min(max_intensity, prediction)
    )


    with col2:

        st.subheader(
            "AI Prediction"
        )

        st.metric(
            "Estimated Cyclone Intensity",
            f"{prediction:.1f} knots"
        )


        # ==================================
        # INTENSITY CATEGORY
        # ==================================

        if prediction < 34:

            category = "Depression / Weak System"
            risk = "LOW"

        elif prediction < 48:

            category = "Tropical Storm"
            risk = "MODERATE"

        elif prediction < 64:

            category = "Severe Cyclone"
            risk = "HIGH"

        elif prediction < 90:

            category = "Very Severe Cyclone"
            risk = "VERY HIGH"

        else:

            category = "Extremely Severe Cyclone"
            risk = "EXTREME"


        st.write(
            "**Estimated Category:**",
            category
        )

        st.write(
            "**Risk Level:**",
            risk
        )


        st.info(
            "This is an experimental AI estimate "
            "for demonstration purposes."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "SIH26070 — AI-based Cyclone Monitoring Prototype"
)