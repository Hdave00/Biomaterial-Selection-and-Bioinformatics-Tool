import streamlit as st
import sys
import os
sys.dont_write_bytecode = True

def run_selection_app():
    st.title("Material Selection Module - MINIMAL TEST")
    st.write("If this loads, the basic module structure works")
    
    # Test basic functionality
    if st.button("Test Button"):
        st.success("✅ Basic functionality works!")
    
    # Test if we can load simple data
    try:
        import pandas as pd
        df = pd.DataFrame({"test": [1, 2, 3]})
        st.write("✅ Pandas import works")
    except Exception as e:
        st.error(f"❌ Pandas import failed: {e}")
    
    # Test if we can load your utils
    try:
        from src.utils.csv_database_loader import query_table, get_db_path
        st.write("✅ Database loader import works")
    except Exception as e:
        st.error(f"❌ Database loader import failed: {e}")
    
    # Test if we can load MP integration
    try:
        from app.mp_integration import query_materials_project
        st.write("✅ MP integration import works")
    except Exception as e:
        st.error(f"❌ MP integration import failed: {e}")
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

if __name__ == '__main__':
    run_selection_app()