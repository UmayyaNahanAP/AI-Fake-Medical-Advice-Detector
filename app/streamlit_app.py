import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import detect_fake_medical_advice


st.title("AI Fake Medical Advice Detector")

st.write("Analyze whether medical advice is reliable or misinformation.")

text = st.text_area("Enter Health Advice")


if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter health advice")

    else:

        result = detect_fake_medical_advice(text)

        st.subheader("Analysis Result")

        st.write("Classification:", result["classification"])
        st.write("Confidence:", result["confidence"])
        st.write("Severity:", result["severity"])

        st.subheader("Medical Entities")
        st.write(result["entities"])

        st.subheader("Explanation")
        st.write(result["explanation"])
