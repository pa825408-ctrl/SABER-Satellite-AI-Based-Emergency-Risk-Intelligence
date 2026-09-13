import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import plotly.express as px
import pandas as pd


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

# ==========================================
# HISTORICAL CYCLONE TRACK & INTENSITY
# ==========================================

st.divider()

st.header("🗺️ Historical Cyclone Track & Intensity")

TRACK_FILE = (
    PROJECT
    / "data"
    / "processed"
    / "cyclone_tracks.csv"
)

try:

    tracks = pd.read_csv(TRACK_FILE)

    # Convert columns
    tracks["ISO_TIME"] = pd.to_datetime(
        tracks["ISO_TIME"],
        errors="coerce"
    )

    tracks["LAT"] = pd.to_numeric(
        tracks["LAT"],
        errors="coerce"
    )

    tracks["LON"] = pd.to_numeric(
        tracks["LON"],
        errors="coerce"
    )

    tracks["USA_WIND"] = pd.to_numeric(
        tracks["USA_WIND"],
        errors="coerce"
    )

    tracks = tracks.dropna(
        subset=["LAT", "LON"]
    )

    # --------------------------------------
    # CYCLONE SELECTION
    # --------------------------------------

    cyclone_names = sorted(
        tracks["NAME"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_cyclone = st.selectbox(
        "Select a historical cyclone",
        cyclone_names
    )

    cyclone_data = tracks[
        tracks["NAME"].astype(str)
        == selected_cyclone
    ].copy()

    cyclone_data = cyclone_data.sort_values(
        "ISO_TIME"
    )

    # --------------------------------------
    # STATISTICS
    # --------------------------------------

    max_wind = cyclone_data["USA_WIND"].max()

    min_wind = cyclone_data["USA_WIND"].min()

    first_date = cyclone_data["ISO_TIME"].min()

    last_date = cyclone_data["ISO_TIME"].max()

    last_point = cyclone_data.iloc[-1]

    # --------------------------------------
    # DISPLAY STATISTICS
    # --------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌪️ Cyclone",
            selected_cyclone
        )

    with col2:
        if pd.notna(max_wind):
            st.metric(
                "💨 Maximum Wind",
                f"{max_wind:.0f} knots"
            )
        else:
            st.metric(
                "💨 Maximum Wind",
                "N/A"
            )

    with col3:
        st.metric(
            "📍 Track Points",
            len(cyclone_data)
        )

    with col4:
        st.metric(
            "📅 Duration",
            f"{first_date.strftime('%d %b')} – "
            f"{last_date.strftime('%d %b')}"
        )

    # --------------------------------------
    # MAP
    # --------------------------------------

    st.subheader(
        "Cyclone Movement and Intensity"
    )

    fig = px.scatter_map(
        cyclone_data,
        lat="LAT",
        lon="LON",
        color="USA_WIND",
        size="USA_WIND",
        hover_name="NAME",
        hover_data={
            "ISO_TIME": True,
            "USA_WIND": True,
            "LAT": True,
            "LON": True
        },
        zoom=3,
        height=600
    )

    # Add connecting track line
    fig.add_scattermap(
        lat=cyclone_data["LAT"],
        lon=cyclone_data["LON"],
        mode="lines",
        line=dict(
            width=3
        ),
        name="Cyclone Track"
    )

    fig.update_layout(
        map=dict(
            style="open-street-map"
        ),
        margin=dict(
            r=0,
            t=0,
            l=0,
            b=0
        ),
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------
    # LAST POSITION
    # --------------------------------------

    st.subheader(
        "📍 Last Recorded Position"
    )

    position_col1, position_col2 = st.columns(2)

    with position_col1:

        st.write(
            f"**Latitude:** "
            f"{last_point['LAT']:.2f}"
        )

        st.write(
            f"**Longitude:** "
            f"{last_point['LON']:.2f}"
        )

    with position_col2:

        if pd.notna(last_point["ISO_TIME"]):

            st.write(
                f"**Time:** "
                f"{last_point['ISO_TIME']}"
            )

        if pd.notna(last_point["USA_WIND"]):

            st.write(
                f"**Wind:** "
                f"{last_point['USA_WIND']:.0f} knots"
            )

    st.success(
        f"Historical cyclone analysis loaded: "
        f"{selected_cyclone}"
    )

except Exception as e:

    st.error(
        f"Could not load cyclone track data: {e}"
    )