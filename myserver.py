import os
from flask import Flask, render_template,request
import joblib

# 1. Carga el modelo y el vectorizador aquí (a nivel global)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR,"imdb_sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "imdb_vectorizer.pkl"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review")
    if not review:
        return "No hay reseña disponoble"
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return f"La reseña es <b>{sentiment}</b>"


if __name__ == "__main__":
    app.run(debug=True)
