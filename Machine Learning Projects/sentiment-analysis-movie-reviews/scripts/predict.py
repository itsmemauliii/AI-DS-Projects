import pickle
from preprocess import clean_text

def load_artifacts():
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_sentiment(text):
    model, vectorizer = load_artifacts()
    cleaned_text = clean_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return 'positive' if prediction == 1 else 'negative'

if __name__ == "__main__":
    review = input("Enter a movie review: ")
    sentiment = predict_sentiment(review)
    print(f"The review is {sentiment}")
