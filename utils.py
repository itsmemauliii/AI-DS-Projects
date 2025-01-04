import PyPDF2
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
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
        return ""

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def extract_skills(text, skill_list_path='skills.txt'):
    try:
        with open(skill_list_path, 'r') as file:
            skills = [line.strip().lower() for line in file.readlines()]
    except FileNotFoundError:
        return []
    return [skill for skill in skills if skill in text.lower()]

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
    plt.tight_layout()
    plt.show()

def create_results_table(scores, resumes):
    data = {"Resume": [resume.name for resume in resumes], "Similarity Score (%)": [round(score * 100, 2) for score in scores]}
    df = pd.DataFrame(data)
    df = df.sort_values(by="Similarity Score (%)", ascending=False)
    return df

def filter ```python
def filter_resumes_by_score(df, threshold=50):
    return df[df["Similarity Score (%)"] >= threshold]
