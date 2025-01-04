import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize and remove stopwords
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
    
import streamlit as st

st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze"):
    resumes_text = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
    resumes_cleaned = [preprocess_text(resume) for resume in resumes_text]
    job_desc_cleaned = preprocess_text(job_desc)
    
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
    job_skills = set(extract_skills(job_desc))
    matched = resume_skills & job_skills
    return matched

matched_skills = highlight_skills(resumes_cleaned[0], job_desc_cleaned)
st.write(f"Matched Skills: {', '.join(matched_skills)}")

