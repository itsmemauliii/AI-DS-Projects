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
