import streamlit as st
import plotly.express as px

# TODO function to show the compatibility of materials and to show comparisons, along with a biocompatibility index score
def display_material_results(df):
    st.subheader("Compatible Materials")
    st.dataframe(df)

    fig = px.scatter(df, x="Young’s Modulus (GPa)", y="Tensile Strength (MPa)",
                 color="Biocompatibility Score", size="Estimated Lifespan (yrs)",
                 hover_name="Material")
    st.plotly_chart(fig, use_container_width=True)