import PyPDF2
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required nltk resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return ""

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def extract_skills(text, skill_list_path='skills.txt'):
    with open(skill_list_path, 'r') as file:
        skills = [line.strip().lower() for line in file.readlines()]
    return [skill for skill in skills if skill.lower() in text.lower()]

def calculate_similarity(resumes_cleaned, job_desc_cleaned):
    all_texts = resumes_cleaned + [job_desc_cleaned]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarity_scores.flatten().tolist()

def plot_scores(scores, resumes):
    plt.figure(figsize=(10, 6))
    plt.barh([resume.name for resume in resumes], scores, color='skyblue')
    plt.xlabel("Similarity Score")
    plt.ylabel("Resume")
    plt.title("Resume Matching Scores")
    st.pyplot(plt)

def create_results_table(scores, resumes):
    data = {"Resume": [resume.name for resume in resumes], "Similarity Score (%)": [round(score * 100, 2) for score in scores]}
    df = pd.DataFrame(data)
    df = df.sort_values(by="Similarity Score (%)", ascending=False)
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
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import nltk
nltk.data.path.append('./nltk_data')  # Point to the downloaded resources folder

# Your existing NLTK usage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources are loaded
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('punkt_tab', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')
