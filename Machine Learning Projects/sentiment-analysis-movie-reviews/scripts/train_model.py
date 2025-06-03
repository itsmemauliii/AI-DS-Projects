import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load processed data
df = pd.read_csv('data/processed_reviews.csv')

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment'].map({'positive':1, 'negative':0})

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save artifacts
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
