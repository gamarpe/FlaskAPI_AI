from flask import jsonify
import os
from flask import Flask, render_template, request
import joblib

# 1. Carga el modelo y el vectorizador aquí (a nivel global)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "imdb_sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "imdb_vectorizer.pkl"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review")
    if not review:
        return "No hay reseña disponible"
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    probabilidades = model.predict_proba(review_tfidf)[0]
    confianza = round(float(max(probabilidades)) * 100, 2)
    sentiment = "positiva" if prediction == 1 else "negativa"
    return f"La reseña es <b>{sentiment}</b> (Confianza: {confianza}%)"


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "review" not in data:
        return jsonify({"error": "Falta el campo 'review' en el JSON"}), 400

    texto = data["review"]
    vector = vectorizer.transform([texto])
    prediccion = model.predict(vector)[0]

    # Obtiene la probabilidad más alta entre negativo [0] y positivo [1]
    probabilidades = model.predict_proba(vector)[0]
    confianza = round(float(max(probabilidades)) * 100, 2)

    return jsonify({
        "review": texto,
        "sentiment": "positiva" if prediccion == 1 else "negativa",
        "confidence_percent": confianza

    })


if __name__ == "__main__":
    app.run(debug=True)
