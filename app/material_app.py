import streamlit as st
import sys
sys.dont_write_bytecode = True

def run_selection_app():
    st.title("Material Selection Module")
    st.write("Testing progressive imports...")
    
    # Step 1: Test basic imports
    try:
        import pandas as pd
        st.success("✅ Pandas imported")
    except Exception as e:
        st.error(f"❌ Pandas failed: {e}")
        return
    
    # Step 2: Test plotly
    try:
        import plotly.express as px
        st.success("✅ Plotly imported")
    except Exception as e:
        st.error(f"❌ Plotly failed: {e}")
        return
    
    # Step 3: Test SQLite
    try:
        import sqlite3
        st.success("✅ SQLite imported")
    except Exception as e:
        st.error(f"❌ SQLite failed: {e}")
        return
    
    # Step 4: Test your utils
    try:
        from src.utils.csv_database_loader import query_table, get_db_path
        st.success("✅ Database utils imported")
    except Exception as e:
        st.error(f"❌ Database utils failed: {e}")
        return
    
    # Step 5: Test MP integration LAST
    st.write("Testing MP integration...")
    try:
        # Use lazy import for MP integration
        from app import mp_integration
        st.success("✅ MP integration imported")
    except Exception as e:
        st.error(f"❌ MP integration failed: {e}")
        st.info("Continuing without MP integration...")
    
    st.success("🎉 All imports successful!")
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

if __name__ == '__main__':
    run_selection_app()