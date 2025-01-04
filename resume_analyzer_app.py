import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load stopwords in English
stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize and remove stopwords
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    
import streamlit as st

st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")

if st.button("Analyze"):
    resumes = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
    resumes_cleaned = [preprocess_text(resume) for resume in resumes]
    job_desc_cleaned = preprocess_text(job_description)

    
    scores = calculate_similarity(resumes_cleaned, job_desc_cleaned)
    st.pyplot(plot_scores(scores, uploaded_resumes))
    results = sorted(zip(uploaded_resumes, scores), key=lambda x: x[1], reverse=True)
    
    st.subheader("Results")
    for resume, score in results:
        st.write(f"Resume: {resume.name} | Similarity Score: {score:.2f}")

skills = ["python", "sql", "tableau", "machine learning", "deep learning"]

def extract_skills(text):
    return [skill for skill in skills if skill.lower() in text.lower()]

import matplotlib.pyplot as plt

def plot_scores(scores, resumes):
    plt.figure(figsize=(10, 6))
    plt.barh([resume.name for resume in resumes], scores, color='skyblue')
    plt.xlabel("Similarity Score")
    plt.ylabel("Resume")
    plt.title("Resume Matching Scores")
    plt.show()
    
def highlight_skills(resume, job_desc):
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job_description))
    matched = resume_skills & job_skills
    return matched

matched_skills = highlight_skills(resumes_cleaned[0], job_desc_cleaned)
st.write(f"Matched Skills: {', '.join(matched_skills)}")

import pandas as pd

def create_results_table(scores, resumes):
    data = {
        "Resume": [resume.name for resume in resumes],
        "Similarity Score (%)": [round(score * 100, 2) for score in scores]
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="Similarity Score (%)", ascending=False)  # Sort by score
    st.dataframe(df)

def filter_results(scores, resumes, top_n=None, min_score=None):
   
    results = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
    
    if min_score is not None:
        results = [(res, score) for res, score in results if score >= min_score]
    
    if top_n is not None:
        results = results[:top_n]
    
    return zip(*results) if results else ([], [])

st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze"):
    resumes = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
    resumes_cleaned = [preprocess_text(text) for text in resumes]
    job_desc_cleaned = preprocess_text(job_description)
    
    # Calculate scores
    scores = calculate_similarity(resumes_cleaned, job_desc_cleaned)
    
    # Filtering options
    top_n = st.slider("Top Matches to Display", min_value=1, max_value=len(scores), value=5)
    min_score = st.slider("Minimum Similarity Score (%)", min_value=0, max_value=100, value=50)
    filtered_resumes, filtered_scores = filter_results(scores, uploaded_resumes, top_n=top_n, min_score=min_score / 100)

    # Display results
    st.subheader("Matching Results")
    create_results_table(filtered_scores, filtered_resumes)
    plot_scores(filtered_scores, filtered_resumes)

import nltk

# Check if the 'stopwords' resource is available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')  # Download the 'stopwords' resource
    stop_words = set(stopwords.words("english"))
