import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

if __name__ == "__main__":
    df = pd.read_csv('data/imdb_dataset.csv')
    df['cleaned_review'] = df['review'].apply(clean_text)
    df.to_csv('data/processed_reviews.csv', index=False)
