from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        new_review = request.form['text']
        
        # Load the model
        model = joblib.load("naive_bayes.pkl")
        
        # Make prediction
        prediction = model.predict([new_review])
        prediction_proba = model.predict_proba([new_review])
        
        # Display the result
        if prediction[0] == 0:
            result = "Negative"
            color = "red"
        else:
            result = "Positive"
            color = "green"
        
        # Display the prediction score
        score = f"Prediction Score: {prediction_proba[0][1]:.2f}"
        
        return render_template('output.html', prediction=result, color=color, score=score)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port  = 5000)