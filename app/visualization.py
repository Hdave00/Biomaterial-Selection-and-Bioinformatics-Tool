import streamlit as st
import plotly.express as px

def display_material_results(df):
    st.subheader("Compatible Materials")
    st.dataframe(df)

    fig = px.bar(df, x="Material", y="Biocompatibility Score", title="Biocompatibility Comparison")
    st.plotly_chart(fig, use_container_width=True)