from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and vectorizer safely
try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print("Error loading model/vectorizer:", e)

@app.route("/")
def home():
    # This loads home.html (your main home page)
    return render_template("home.html")

@app.route("/predict_page")
def predict_page():
    # This loads the text input page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]

        # Transform message
        X = vectorizer.transform([message])

        # Predict
        result = model.predict(X)[0]

        # Convert result to label
        output = "Spam ❌" if result == 1 else "Ham ✅"

        return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
