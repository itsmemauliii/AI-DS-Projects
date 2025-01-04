import streamlit as st
import pandas as pd
from utils import extract_text_from_pdf, preprocess_text

st.title("AI-Powered Resume Analyzer")

# File uploader for categories CSV
uploaded_categories_file = st.file_uploader("Upload Categories CSV", type=["csv"])

if st.button("Analyze"):
    if uploaded_categories_file:
        # Load categories and resumes from the uploaded CSV file
        try:
            categories_df = pd.read_csv(uploaded_categories_file)

            # Check if the required columns exist
            if 'Category' not in categories_df.columns or 'Resume' not in categories_df.columns:
                st.error("CSV must contain 'Category' and 'Resume' columns.")
            else:
                # Initialize a list to store results
                results = []

                # Process each resume in the CSV
                for index, row in categories_df.iterrows():
                    resume_text = row['Resume']
                    if resume_text:
                        # Preprocess the resume text
                        resume_cleaned = preprocess_text(resume_text)

                        # Check for matches in the resumes column
                        matched_categories = []
                        if resume_cleaned:  # Ensure resume_cleaned is not empty
                            matched_categories.append(row['Category'])

                        # Store the result
                        results.append({
                            'resume': resume_text,
                            'matched_categories': matched_categories
                        })

                # Display results
                st.subheader("Analysis Results")
                for result in results:
                    st.write(f"Resume: {result['resume']}")
                    if result['matched_categories']:
                        st.write(f"Matched Categories: {', '.join(result['matched_categories'])}")
                    else:
                        st.write("No categories matched.")
        except Exception as e:
            st.error(f"Error loading categories from CSV: {e}")
    else:
        st.error("Please upload a categories CSV file
