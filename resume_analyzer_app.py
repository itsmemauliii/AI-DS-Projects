import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_text_from_pdf, preprocess_text, extract_skills, calculate_similarity, plot_scores, create_results_table, filter_results

st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")

if st.button("Analyze"):
    if uploaded_resumes and job_description:
        resumes = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
        resumes_cleaned = [preprocess_text(text) for text in resumes]
        job_desc_cleaned = preprocess_text(job_description)

        scores = calculate_similarity(resumes_cleaned, job_desc_cleaned)

        top_n = st.slider("Top Matches to Display", min_value=1, max_value=len(scores), value=5)
        min_score = st.slider("Minimum Similarity Score (%)", min_value=0, max_value=100, value=50)
        filtered_resumes, filtered_scores = filter_results(scores, uploaded_resumes, top_n=top_n, min_score=min_score / 100)

        st.subheader("Matching Results")
        create_results_table(filtered_scores, filtered_resumes)
        plot_scores(filtered_scores, filtered_resumes)

        st.subheader("Matched Skills")
        for i, resume in enumerate(filtered_resumes):
            matched_skills = extract_skills(resumes_cleaned[i])
            st.write(f"Resume: {resume.name} | Matched Skills: {', '.join(matched_skills)}")
    else:
        st.error("Please upload resumes and enter a job description before analyzing.")
