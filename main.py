import streamlit as st 
from views import home, plot, cbot
import pandas as pd
import os

st.set_page_config(
    page_title="DataChat.AI",
    layout="wide", 
    initial_sidebar_state="collapsed" 
)

encoding_options = ['utf-8', 'ISO-8859-1', 'Windows-1252', 'utf-16']

def load_csv_with_auto_encoding(file_path, encodings):
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            # st.success(f"File successfully loaded using encoding: {encoding}")
            return data
        except Exception as e:
            # st.warning(f"Failed with encoding {encoding}: {e}")
            st.warning("")
    st.error("Unable to load the file with any of the tested encodings.")
    st.stop()

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Home Page'

shared_directory = "./shared_files"
os.makedirs(shared_directory, exist_ok=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Home Page"):
        st.session_state.selected_tab = 'Home Page'
with col2:
    if st.button("Chatbot Page"):
        st.session_state.selected_tab = 'Chatbot Page'
with col3:
    if st.button("Plot Page"):
        st.session_state.selected_tab = 'Plot Page'

if st.session_state.selected_tab == "Home Page":    
    home.app()

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        file_path = os.path.join(shared_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # st.success(f"File saved to: {file_path}")
        # st.write(f"The uploaded file is now available to other pages.")

        try:
            data = load_csv_with_auto_encoding(file_path, encoding_options)
            st.subheader("Preview of the Uploaded CSV File")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

elif st.session_state.selected_tab == "Plot Page":    
    plot.app1(shared_directory)
elif st.session_state.selected_tab == "Chatbot Page":    
    cbot.app1(shared_directory)
