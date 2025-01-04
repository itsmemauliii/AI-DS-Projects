import PyPDF2
import re
import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

# Function to extract skills from text
def extract_skills(text, skills_list):
    return [skill for skill in skills_list if skill.lower() in text.lower()]

# Function to calculate similarity (example placeholder)
def calculate_similarity(resumes_cleaned, job_desc_cleaned):
    # Placeholder logic for calculating similarity
    return [len(set(resume.split()) & set(job_desc_cleaned.split())) / len(set(job_desc_cleaned.split())) for resume in resumes_cleaned]

# Function to plot similarity scores
def plot_scores(scores, resumes):
    plt.figure(figsize=(10, 6))
    plt.barh([resume.name for resume in resumes], scores, color='skyblue')
    plt.xlabel("Similarity Score")
    plt.ylabel("Resume")
    plt.title("Resume Matching Scores")
    st.pyplot(plt)

# Function to highlight matched skills
def highlight_skills(resume, job_desc, skills_list):
    resume_skills = set(extract_skills(resume, skills_list))
    job_skills = set(extract_skills(job_desc, skills_list))
    matched = resume_skills & job_skills
    return matched

# Function to create a results table
def create_results_table(scores, resumes):
    data = {
        "Resume": [resume.name for resume in resumes],
        "Similarity Score (%)": [round(score * 100, 2) for score in scores]
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="Similarity Score (%)", ascending=False)
    st.dataframe(df)

# Function to filter results based on user input
def filter_results(scores, resumes, top_n=None, min_score=None):
    results = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
    if min_score is not None:
        results = [(res, score) for res, score in results if score >= min_score]
    if top_n is not None:
        results = results[:top_n]
    return zip(*results) if results else ([], [])

# Streamlit UI
st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")
skills_list = ["python", "sql", "tableau", "machine learning", "deep learning"]

if st.button("Analyze"):
    if uploaded_resumes and job_description:
        resumes = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
        resumes_cleaned = [preprocess_text(resume) for resume in resumes]
        job_desc_cleaned = preprocess_text(job_description)

        # Calculate similarity scores
        scores = calculate_similarity(resumes_cleaned, job_desc_cleaned)

        # Filtering options
        top_n = st.slider("Top Matches to Display", min_value=1, max_value=len(scores), value=5)
        min_score = st.slider("Minimum Similarity Score (%)", min_value=0, max_value=100, value=50)
        filtered_resumes, filtered_scores = filter_results(scores, uploaded_resumes, top_n=top_n, min_score=min_score / 100)

        # Display results
        st.subheader("Matching Results")
        create_results_table(filtered_scores, filtered_resumes)
        plot_scores(filtered_scores, filtered_resumes)

        # Highlight matched skills for the top resume
        if filtered_resumes:
            matched_skills = highlight_skills(resumes_cleaned[0], job_desc_cleaned, skills_list)
            st.write(f"Matched Skills: {', '.join(matched_skills)}")
    else:
        st.warning("Please upload at least one resume and enter a job description.")
