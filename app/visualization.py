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

    # TODO there will be a section for 3d visualisation of the body

    # TODO visualization for the part to be determined

    # TODO UI/UX to be used, here as well 