# Sentiment Analysis on Movie Reviews
## Project Description
This project performs sentiment analysis on IMDB movie reviews to classify them as positive or negative. It uses Natural Language Processing (NLP) techniques and machine learning to analyze the text content of reviews.

## Features
- Text preprocessing (cleaning, stopword removal, stemming)
- TF-IDF vectorization for feature extraction
- Logistic Regression classifier
- Model evaluation metrics
- Prediction pipeline

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage
1. To preprocess the data:
   ```bash
   python scripts/preprocess.py
   ```

2. To train the model:
   ```bash
   python scripts/train_model.py
   ```

3. To make predictions:
   ```bash
   python scripts/predict.py
   ```
   (Then enter a movie review when prompted)

## Dataset
The dataset contains 50,000 IMDB movie reviews labeled as positive or negative. A sample is included in the `data` folder.

## Results
The logistic regression model achieves approximately 89% accuracy on the test set.

## Future Improvements
- Try different classifiers (Random Forest, SVM, Neural Networks)
- Use word embeddings (Word2Vec, GloVe)
- Implement deep learning models (LSTM, Transformer)
- Create a web interface for the model

## License
This project is licensed under the MIT License.
```

This project provides a complete pipeline for sentiment analysis, from data preprocessing to model deployment. The modular structure makes it easy to extend or modify components. The README file provides clear instructions for setup and usage, making it accessible to other developers.
