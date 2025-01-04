import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_text_from_pdf, preprocess_text, extract_skills

st.title("AI-Powered Resume Analyzer")

# File uploader for resumes
uploaded_resume = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])

# File uploader for skills CSV
uploaded_skills_file = st.file_uploader("Upload Skills CSV", type=["csv"])

if st.button("Analyze"):
    if uploaded_resume and uploaded_skills_file:
        # Load skills from the uploaded CSV file
        try:
            skills_df = pd.read_csv(uploaded_skills_file)
            skills_list = skills_df['skills'].dropna().tolist()  # Adjust the column name as necessary
        except Exception as e:
            st.error(f"Error loading skills from CSV: {e}")
            skills_list = []

        # Process the uploaded resume
        resume_text = extract_text_from_pdf(uploaded_resume)
        if resume_text:
            resume_cleaned = preprocess_text(resume_text)

            # Extract matched skills
            matched_skills = extract_skills(resume_cleaned, skills_list)

            # Display results
            st.subheader("Matched Skills")
            if matched_skills:
                st.write(f"Your resume contains the following skills: {', '.join(matched_skills)}")
            else:
                st.write("No skills matched from the uploaded skills dataset.")
        else:
            st.error("Failed to extract text from the resume.")
    else:
        st.error("Please upload your resume and a skills CSV file before analyzing.")
