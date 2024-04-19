from flask import Flask, render_template, request, redirect, url_for
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model("J:\\Janu Projects\\Text_Sentiment_ANN\\pkl-files\\model.h5")

# Load tokenizer for text preprocessing
with open(
    "J:\\Janu Projects\\Text_Sentiment_ANN\\pkl-files\\tokenizer.pkl", "rb"
) as file:
    tokenizer = pickle.load(file)

# Function to preprocess text data
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=32)
    return padded_sequences

# Route for index page
@app.route("/")
def index():
    return render_template("index.html")

# Route for predicting sentiment
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]
        preprocessed_text = preprocess_text(text)
        sentiment_prob = model.predict(preprocessed_text)[0]

        # Determine sentiment based on probability
        if sentiment_prob[0] > 0.5:
            sentiment_text = "Negative"
        else:
            sentiment_text = "Positive"

        return render_template("result.html", sentiment=sentiment_text, text=text)

    # Add a default return statement in case the request method is not POST
    return redirect(url_for("index"))  # Redirect to the index page

if __name__ == "__main__":
    app.run(debug=True)
