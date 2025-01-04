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

skills = ["python", "sql", "tableau", "machine learning", "deep learning"]

def extract_skills(text):
    return [skill for skill in skills if skill.lower() in text.lower()]

