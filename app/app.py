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
    st.session_state["predicted_wind"] = float(prediction)


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

# Clean cyclone data before plotting
cyclone_data = cyclone_data.copy()

cyclone_data["LAT"] = pd.to_numeric(
    cyclone_data["LAT"],
    errors="coerce"
)

cyclone_data["LON"] = pd.to_numeric(
    cyclone_data["LON"],
    errors="coerce"
)

cyclone_data["USA_WIND"] = pd.to_numeric(
    cyclone_data["USA_WIND"],
    errors="coerce"
)

cyclone_data = cyclone_data.dropna(
    subset=["LAT", "LON", "USA_WIND"]
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

    

# ==========================================
# CYCLONE RISK / IMPACT ZONES
# ==========================================
# ============================================
# CURRENT CYCLONE MONITORING
# ============================================

st.divider()

st.header("🛰️ AI-Assisted Cyclone Monitoring Simulation")

st.info(
    "This module replays historical cyclone data to demonstrate how "
    "SABER can combine satellite-based AI intensity estimation with "
    "cyclone-track information for risk assessment. This is a "
    "prototype simulation and not a live warning service."
)

# Use the existing AI prediction variable
predicted_intensity = float(
    st.session_state.get("predicted_wind", 0.0)
)
monitor_wind = predicted_intensity


# Determine monitoring risk level
if monitor_wind < 34:
    monitor_risk = "LOW"
elif monitor_wind < 48:
    monitor_risk = "MODERATE"
elif monitor_wind < 64:
    monitor_risk = "HIGH"
else:
    monitor_risk = "VERY HIGH"

monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)

with monitor_col1:
    st.metric(
        "Cyclone",
        selected_cyclone
    )

with monitor_col2:
    st.metric(
        "AI-Estimated Intensity",
        f"{monitor_wind:.1f} kt"
    )

with monitor_col3:
    st.metric(
        "Historical Recorded Wind",
        f"{float(last_point['USA_WIND']):.1f} kt"
    )

with monitor_col4:
    st.metric(
        "Risk Level",
        monitor_risk
    )

st.markdown("### 📍 Last Recorded Historical Position")

position_col1, position_col2, position_col3 = st.columns(3)

with position_col1:
    st.metric(
        "Latitude",
        f"{float(last_point['LAT']):.2f}°"
    )

with position_col2:
    st.metric(
        "Longitude",
        f"{float(last_point['LON']):.2f}°"
    )

with position_col3:
    st.metric(
        "Wind",
        f"{float(last_point['USA_WIND']):.1f} kt"
    )

st.caption(
    f"Last recorded observation: {last_point['ISO_TIME']}"
)

st.divider()

st.header("⚠️ Estimated Cyclone Impact Zones")

st.info(
    "Risk zones are experimental estimates for "
    "prototype demonstration and should not be "
    "used for real-world evacuation decisions."
)


# ------------------------------------------
# USE MAXIMUM WIND
# ------------------------------------------

if pd.notna(max_wind):

    wind = float(max_wind)

else:

    wind = 30.0


# ------------------------------------------
# DETERMINE ZONE RADII
# ------------------------------------------

if monitor_wind < 34:

    inner_radius = 30
    middle_radius = 60
    outer_radius = 100

elif monitor_wind  < 64:

    inner_radius = 50
    middle_radius = 100
    outer_radius = 150

elif monitor_wind < 90:

    inner_radius = 75
    middle_radius = 150
    outer_radius = 250

else:

    inner_radius = 100
    middle_radius = 200
    outer_radius = 300


# ------------------------------------------
# CREATE CIRCLE FUNCTION
# ------------------------------------------

import math


def create_circle(
    latitude,
    longitude,
    radius_km
):

    points = []

    earth_radius = 6371.0

    for angle in range(0, 361, 5):

        angle_rad = math.radians(angle)

        lat_rad = math.radians(latitude)

        lon_rad = math.radians(longitude)

        distance = radius_km / earth_radius

        new_lat = math.asin(
            math.sin(lat_rad)
            * math.cos(distance)
            +
            math.cos(lat_rad)
            * math.sin(distance)
            * math.cos(angle_rad)
        )

        new_lon = (
            lon_rad
            +
            math.atan2(
                math.sin(angle_rad)
                * math.sin(distance)
                * math.cos(lat_rad),
                math.cos(distance)
                -
                math.sin(lat_rad)
                * math.sin(new_lat)
            )
        )

        points.append(
            (
                math.degrees(new_lat),
                math.degrees(new_lon)
            )
        )

    return points


# ------------------------------------------
# LAST POSITION
# ------------------------------------------

risk_lat = float(
    last_point["LAT"]
)

risk_lon = float(
    last_point["LON"]
)


# ------------------------------------------
# CREATE ZONES
# ------------------------------------------

outer_zone = create_circle(
    risk_lat,
    risk_lon,
    outer_radius
)

middle_zone = create_circle(
    risk_lat,
    risk_lon,
    middle_radius
)

inner_zone = create_circle(
    risk_lat,
    risk_lon,
    inner_radius
)


# ------------------------------------------
# RISK MAP
# ------------------------------------------

risk_fig = px.scatter_map(
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
    zoom=4,
    height=650
)


# ------------------------------------------
# OUTER ZONE
# ------------------------------------------

outer_lat = [
    point[0]
    for point in outer_zone
]

outer_lon = [
    point[1]
    for point in outer_zone
]

risk_fig.add_scattermap(
    lat=outer_lat,
    lon=outer_lon,
    mode="lines",
    fill="toself",
    fillcolor="rgba(255, 200, 0, 0.15)",
    line=dict(width=2),
    name=f"LOW / WATCH ZONE ({outer_radius} km)"
)


# ------------------------------------------
# MIDDLE ZONE
# ------------------------------------------

middle_lat = [
    point[0]
    for point in middle_zone
]

middle_lon = [
    point[1]
    for point in middle_zone
]

risk_fig.add_scattermap(
    lat=middle_lat,
    lon=middle_lon,
    mode="lines",
    fill="toself",
    fillcolor="rgba(255, 140, 0, 0.20)",
    line=dict(width=2),
    name=f"MODERATE ZONE ({middle_radius} km)"
)


# ------------------------------------------
# INNER ZONE
# ------------------------------------------

inner_lat = [
    point[0]
    for point in inner_zone
]

inner_lon = [
    point[1]
    for point in inner_zone
]

risk_fig.add_scattermap(
    lat=inner_lat,
    lon=inner_lon,
    mode="lines",
    fill="toself",
    fillcolor="rgba(255, 0, 0, 0.25)",
    line=dict(width=3),
    name=f"HIGH RISK ZONE ({inner_radius} km)"
)


# ------------------------------------------
# CYCLONE CENTER
# ------------------------------------------

risk_fig.add_scattermap(
    lat=[risk_lat],
    lon=[risk_lon],
    mode="markers",
    marker=dict(
        size=18
    ),
    name="Cyclone Center"
)


# ------------------------------------------
# MAP SETTINGS
# ------------------------------------------

risk_fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(
            lat=risk_lat,
            lon=risk_lon
        ),
        zoom=4
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
    risk_fig,
    use_container_width=True
)


# ------------------------------------------
# RISK SUMMARY
# ------------------------------------------

risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
        "🟡 Watch Zone",
        f"{outer_radius} km"
    )


with risk_col2:

    st.metric(
        "🟠 Moderate Zone",
        f"{middle_radius} km"
    )


with risk_col3:

    st.metric(
        "🔴 High Risk Zone",
        f"{inner_radius} km"
    )


st.warning(
    f"Maximum historical wind: "
    f"{wind:.0f} knots. "
    f"Impact zones are prototype estimates."
)
 # ==========================================
# AI-BASED SAFETY RECOMMENDATIONS
# ==========================================

st.divider()

st.header("🚨 AI-Based Safety Recommendations")

# Use the AI prediction from the satellite image
try:
    
    predicted_wind = float(predicted_intensity)
except:
    predicted_wind = wind
    st.session_state["predicted_wind"] = predicted_wind


# ------------------------------------------
# DETERMINE RISK LEVEL
# ------------------------------------------

if predicted_wind < 34:

    risk_level = "LOW"
    risk_icon = "🟢"

    recommendations = [
        "Continue monitoring official weather updates.",
        "Keep basic emergency supplies ready.",
        "Check communication and emergency contact information.",
        "Avoid unnecessary travel if weather conditions deteriorate."
    ]

elif predicted_wind < 48:

    risk_level = "MODERATE"
    risk_icon = "🟡"

    recommendations = [
        "Monitor official cyclone warnings regularly.",
        "Secure loose objects around homes and buildings.",
        "Keep emergency supplies, food, water and medicines ready.",
        "Avoid coastal areas and unnecessary travel.",
        "Keep phones and emergency communication devices charged."
    ]

elif predicted_wind < 64:

    risk_level = "HIGH"
    risk_icon = "🟠"

    recommendations = [
        "Follow official cyclone warnings and advisories.",
        "Secure doors, windows, roofs and outdoor objects.",
        "Move away from exposed coastal and low-lying areas when advised.",
        "Prepare emergency food, drinking water, medicines and flashlights.",
        "Avoid travelling during severe weather.",
        "Follow evacuation instructions issued by local authorities."
    ]

else:

    risk_level = "VERY HIGH"
    risk_icon = "🔴"

    recommendations = [
        "Follow official emergency and cyclone warnings immediately.",
        "Evacuate when instructed by local authorities.",
        "Move to designated shelters or safer elevated locations.",
        "Stay away from beaches, flooded roads and exposed coastal areas.",
        "Keep emergency supplies, important documents and medicines ready.",
        "Do not attempt to travel through floodwater or severe storm conditions.",
        "Maintain communication with emergency services and local authorities."
    ]


# ------------------------------------------
# DISPLAY RISK
# ------------------------------------------

st.subheader(
    f"{risk_icon} Current AI Risk Level: {risk_level}"
)

st.metric(
    "AI Estimated Wind Intensity",
    f"{predicted_wind:.1f} knots"
)


# ------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------

st.write("### Recommended Actions")

for recommendation in recommendations:

    st.write(
        f"• {recommendation}"
    )


# ------------------------------------------
# DISCLAIMER
# ------------------------------------------

st.info(
    "⚠️ These recommendations are prototype AI-generated "
    "guidance. Always follow official warnings and "
    "instructions from authorized disaster-management "
    "and weather authorities."
)

# ==========================================
# SIH EXECUTIVE SUMMARY
# ==========================================

st.divider()

st.header("📊 SIH Cyclone Intelligence Summary")

# ------------------------------------------
# SUMMARY VALUES
# ------------------------------------------

final_ai_wind = float(
    st.session_state.get("predicted_wind", 0.0)
)

try:
    final_historical_wind = float(max_wind)
except:
    final_historical_wind = 0.0

try:
    final_lat = float(risk_lat)
    final_lon = float(risk_lon)
except:
    final_lat = 0.0
    final_lon = 0.0


# ------------------------------------------
# SUMMARY CARDS
# ------------------------------------------

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.metric(
        "🤖 AI Intensity",
        f"{final_ai_wind:.1f} kt"
    )

with summary2:
    st.metric(
        "🚨 Risk Level",
        risk_level
    )

with summary3:
    st.metric(
        "🔴 High Risk Radius",
        f"{inner_radius} km"
    )

with summary4:
    st.metric(
        "📍 Position",
        f"{final_lat:.2f}, {final_lon:.2f}"
    )


# ------------------------------------------
# SYSTEM PIPELINE
# ------------------------------------------

st.subheader("🔄 AI Decision Pipeline")

st.write(
    "🛰️ INSAT-3D IR Satellite Image"
)

st.write(
    "↓"
)

st.write(
    "🤖 ResNet18 V2 Cyclone Intensity Estimation"
)

st.write(
    "↓"
)

st.write(
    "💨 Wind Intensity → Risk Assessment"
)

st.write(
    "↓"
)

st.write(
    "🗺️ Historical Cyclone Track Analysis"
)

st.write(
    "↓"
)

st.write(
    "⚠️ Estimated Impact Zones"
)

st.write(
    "↓"
)

st.write(
    "🚨 Safety Recommendations"
)


# ------------------------------------------
# REPORT GENERATION
# ------------------------------------------

report = f"""
SIH26070 – AI-BASED CYCLONE MONITORING SYSTEM
================================================

SYSTEM SUMMARY
---------------

Model:
ResNet18 V2

Input:
INSAT-3D Infrared Satellite Image

AI Prediction:
{final_ai_wind:.1f} knots

Risk Level:
{risk_level}

Historical Cyclone:
{selected_cyclone}

Maximum Historical Wind:
{final_historical_wind:.1f} knots

Last Recorded Position:
Latitude: {final_lat:.2f}
Longitude: {final_lon:.2f}

Estimated Impact Zones
----------------------

Watch Zone:
{outer_radius} km

Moderate Risk Zone:
{middle_radius} km

High Risk Zone:
{inner_radius} km


SYSTEM PIPELINE
---------------

1. INSAT-3D satellite imagery
2. AI-based cyclone intensity estimation
3. Risk-level assessment
4. Historical cyclone track analysis
5. Geographic impact-zone visualization
6. Safety recommendation generation


SAFETY GUIDANCE
---------------

"""

for item in recommendations:
    report += f"- {item}\n"

report += """

IMPORTANT DISCLAIMER
--------------------

This system is an experimental SIH prototype.
AI predictions and impact zones are intended for
demonstration and decision-support research only.

Real-world emergency decisions must follow official
weather warnings and instructions from authorized
disaster-management authorities.
"""


# ------------------------------------------
# DOWNLOAD BUTTON
# ------------------------------------------

st.subheader("📥 Download Analysis Report")

st.download_button(
    label="⬇️ Download SIH Cyclone Report",
    data=report,
    file_name="SIH26070_Cyclone_Analysis_Report.txt",
    mime="text/plain"
)

st.success(
    "SIH prototype analysis report is ready."
)