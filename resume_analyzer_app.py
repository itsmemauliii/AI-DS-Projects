import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_text_from_pdf, preprocess_text

st.title("AI-Powered Resume Analyzer")

# File uploader for resumes
uploaded_resume = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])

# File uploader for categories CSV
uploaded_categories_file = st.file_uploader("Upload Categories CSV", type=["csv"])

if st.button("Analyze"):
    if uploaded_resume and uploaded_categories_file:
        # Load categories and resumes from the uploaded CSV file
        try:
            categories_df = pd.read_csv(uploaded_categories_file)
            # Ensure the required columns exist
            if 'category' not in categories_df.columns or 'resumes' not in categories_df.columns:
                st.error("CSV must contain 'category' and 'resumes' columns.")
            else:
                # Process the uploaded resume
                resume_text = extract_text_from_pdf(uploaded_resume)
                if resume_text:
                    resume_cleaned = preprocess_text(resume_text)

                    # Check for matches in the resumes column
                    matched_categories = []
                    for index, row in categories_df.iterrows():
                        if row['resumes'] in resume_cleaned:
                            matched_categories.append(row['category'])

                    # Display results
                    st.subheader("Matched Categories")
                    if matched_categories:
                        st.write(f"Your resume matches the following categories: {', '.join(matched_categories)}")
                    else:
                        st.write("No categories matched from the uploaded dataset.")
                else:
                    st.error("Failed to extract text from the resume.")
        except Exception as e:
            st.error(f"Error loading categories from CSV: {e}")
    else:
        st.error("Please upload your resume and a categories CSV file before analyzing.")
