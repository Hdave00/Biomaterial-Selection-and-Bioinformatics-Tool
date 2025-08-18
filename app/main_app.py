import streamlit as st
import pandas as pd
import sys
import os
from model_interface import predict_materials
from visualization import display_material_results

st.set_page_config(page_title="Bioimplant Compatibility Tool", layout="wide")

# adding title and writing prompts
st.title("Bioimplant Compatibility & Material Suggestion Tool")

# title welcome text, this is where the basic info will go
st.write("Welcome! Select your input parameters from the sidebar to begin.")

# Sidebar Input
st.sidebar.header("Input Specifications")
body_area = st.sidebar.selectbox("Select Body Area", ["Hip", "Knee", "Skull", "Spine"])
youngs_modulus = st.sidebar.slider("Desired Young's Modulus (GPa)", 20, 200, 100)
tensile_strength = st.sidebar.slider("Desired Tensile Strength (MPa)", 100, 2000, 900)
density = st.sidebar.number_input("Target Density (g/cm³)", min_value=1.0, max_value=20.0, value=4.5)
budget = st.sidebar.number_input("Budget per Unit (€)", value=300)

if st.sidebar.button("Find Materials"):
    user_input = {
        "body_area": body_area,
        "youngs_modulus": youngs_modulus,
        "tensile_strength": tensile_strength,
        "density": density,
        "budget": budget
    }
    results = predict_materials(user_input)
    display_material_results(results)

# adding more features here
@st.cache_data
def load_dataset():
    return pd.read_csv("data/materials.csv")