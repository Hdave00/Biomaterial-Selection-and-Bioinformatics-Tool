 # Streamlit app + simple UI glue

import streamlit as st
import pandas as pd
import sys
import os
from src.inference.model_interface import predict_materials
from visualization import display_material_results

from src.features.feature_extractor import FeatureExtractor
from src.utils.master_index import create_material_index

sys.path.append('../utils')

from src.inference.model_interface import BiocompatibilityModel


# Load all datasets once
mechanical_df = pd.read_csv('master_data/unified_mechanical.csv')
chemical_df = pd.read_csv('master_data/unified_polymer_data.csv')
biological_df = pd.read_csv('master_data/biological/biocompatibility_master.csv')

# Create master index
master_index = create_material_index(mechanical_df, chemical_df, biological_df)

# Initialize extractor
extractor = FeatureExtractor(mechanical_df, chemical_df, biological_df, master_index)

# Example usage
features = extractor.get_material_features("PMMA")


st.set_page_config(page_title="Bioimplant Compatibility Tool", layout="wide")

# adding title and writing prompts
st.title("Bioimplant Compatibility & Material Suggestion Tool")

st.markdown("### Select your target application and constraints")

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

# adding more features here, ideally we want to be able to add datasets as we see fit and load each one at will
@st.cache_data
def load_dataset():
    return pd.read_csv("data/materials.csv")



# Second approach
def predict_biocompatibility(material_name, body_part, requirements):
    # Get real-time data from APIs
    mp_api = ...
    hra_api = ...
    
    # Material properties from Materials Project
    material_props = mp_api.get_material_properties(material_name)
    
    # Biological context from Human Reference Atlas
    biological_context = hra_api.get_tissue_properties(body_part)

    # pseudo vars
    data_manager = ...
    model = ...
    biological_features = ...
    analyze_compatibility = ...
    
    # Get ML features from our datasets
    mechanical_features = data_manager.extract_mechanical_features(material_name)
    chemical_features = data_manager.extract_chemical_features(material_name)
    
    # Predict biocompatibility score
    score = model.predict([mechanical_features, chemical_features, biological_features])
    
    return {
        'biocompatibility_score': score,
        'material_properties': material_props,
        'biological_context': biological_context,
        'compatibility_analysis': analyze_compatibility(material_props, biological_context)
    }



# price index integration prototype

def main():
    st.title("Biocompatible Material Selection Tool")
    
    # User inputs (mechanical requirements, body part, etc.)
    user_requirements = ... # get_user_inputs()
    
    # ML model predicts biocompatibility scores
    recommended_materials = ... # model.predict_biocompatibility(user_requirements)
    
    # Add pricing information from API (POST-ML)
    priced_materials = add_pricing_to_recommendations(recommended_materials)
    
    # Display results with pricing
    display_results_with_pricing(priced_materials)

def add_pricing_to_recommendations(materials):
    """Add real-time pricing to ML recommendations"""
    price_api = ...
    
    priced_materials = []
    for material in materials:
        # Get current market price
        current_price = price_api.get_current_price(material['name'])
        price_trend = price_api.get_price_trend(material['name'])
        
        priced_materials.append({
            **material,  # Keep all ML prediction data
            'current_price_eur_kg': current_price,
            'price_trend': price_trend,
            'budget_compatibility': ... # calculate_budget_compatibility(current_price, material)
        })
    
    return sorted(priced_materials, key=lambda x: x['biocompatibility_score'], reverse=True)

def display_results_with_pricing(materials):
    st.header("Recommended Materials")
    
    for material in materials[:10]:  # Top 10 results
        with st.expander(f"{material['name']} - Score: {material['biocompatibility_score']:.2f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Properties")
                st.metric("Young's Modulus", f"{material['youngs_modulus']} GPa")
                st.metric("Biocompatibility", f"{material['biocompatibility_score']:.1%}")
                
            with col2:
                st.subheader("Biological Context")
                st.metric("Estimated Lifespan", f"{material['lifespan']} years")
                st.metric("Corrosion Risk", material['corrosion_risk'])
                
            with col3:
                st.subheader("Economic Factors")
                st.metric("Current Price", f"€{material['current_price_eur_kg']}/kg")
                st.metric("Price Trend", material['price_trend'])
                st.metric("Budget Match", f"{material['budget_compatibility']:.1%}")