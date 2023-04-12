import streamlit as st

with st.sidebar:
    choice = st.radio("Navigation", ['CSV Query', 'Interactive FAQ'])