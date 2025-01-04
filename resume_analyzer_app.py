import PyPDF2
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required nltk resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to extract text from uploaded PDF files
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

# Function to extract skills from text
def extract_skills(text):
    skills = ["python", "sql", "tableau", "machine learning", "deep learning"]
    return [skill for skill in skills if skill.lower() in text.lower()]

# Function to plot similarity scores
def plot_scores(scores, resumes):
    plt.figure(figsize=(10, 6))
    plt.barh([resume.name for resume in resumes], scores, color='skyblue')
    plt.xlabel("Similarity Score")
    plt.ylabel("Resume")
    plt.title("Resume Matching Scores")
    st.pyplot(plt)

# Function to create results table
def create_results_table(scores, resumes):
    data = {
        "Resume": [resume.name for resume in resumes],
        "Similarity Score (%)": [round(score * 100, 2) for score in scores]
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="Similarity Score (%)", ascending=False)
    st.dataframe(df)

# Function to filter results based on criteria
def filter_results(scores, resumes, top_n=None, min_score=None):
    results = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
    if min_score is not None:
        results = [(res, score) for res, score in results if score >= min_score]
    if top_n is not None:
        results = results[:top_n]
    return zip(*results) if results else ([], [])

# Function to calculate similarity (dummy implementation)
def calculate_similarity(resumes_cleaned, job_desc_cleaned):
    return [0.5 for _ in resumes_cleaned]  # Placeholder similarity score for demonstration

# Streamlit app layout
st.title("AI-Powered Resume Analyzer")

uploaded_resumes = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")

if st.button("Analyze"):
    if uploaded_resumes and job_description:
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

        # Highlight matched skills
        st.subheader("Matched Skills")
        for i, resume in enumerate(filtered_resumes):
            matched_skills = extract_skills(resumes_cleaned[i])
            st.write(f"Resume: {resume.name} | Matched Skills: {', '.join(matched_skills)}")
    else:
        st.error("Please upload resumes and enter a job description before analyzing.")
st.write("Uploaded Resumes:", uploaded_resumes)
st.write("Job Description:", job_description)
if not uploaded_resumes:
    st.error("No resumes uploaded. Please upload at least one PDF.")
if not job_description:
    st.error("Job description is empty. Please enter a job description.")

uploaded_file = st.file_uploader("Upload Resumes (PDFs only):", type=["pdf"])
uploaded_file = st.file_uploader("Upload CSV File (containing resumes and categories):", type=["csv"])
import nltk
nltk.download('punkt')
