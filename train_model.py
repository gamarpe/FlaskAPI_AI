import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import imdb

# 1. Descarga el dataset (top 5,000 palabras)
num_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# 2. Decodifica las secuencias numéricas a texto plano
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(text_sequence):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_sequence])

X_train_text = [decode_review(sequence) for sequence in X_train]
X_test_text = [decode_review(sequence) for sequence in X_test]

# 3. Vectoriza con TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=num_words)
xtrain_tfidf = vectorizer.fit_transform(X_train_text)
xtest_tfidf = vectorizer.transform(X_test_text)

# 4. Entrena y evalúa la regresión logística
model = LogisticRegression(max_iter=200)
model.fit(xtrain_tfidf, y_train)

y_pred = model.predict(xtest_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# 5. Exporta los archivos .pkl para que Flask los lea
joblib.dump(model, 'imdb_sentiment_model.pkl')
joblib.dump(vectorizer, 'imdb_vectorizer.pkl')
print('Archivos .pkl guardados correctamente.')
