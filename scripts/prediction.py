#!/usr/bin/env python3

from flask import Flask, request, jsonify

from flask_cors import CORS
import pickle
import os



# Load the model and vectorizer
with open('../models/loan_model.pkl', "rb") as f:
    model, dv = pickle.load(f)

# Initialize the Flask app
app = Flask("loan_approval")
CORS(app) 


@app.route("/predict", methods=["GET"])
def loan_predict():
    try:
        data = request.get_json()
        # Transform input data
        X = dv.transform([data])
        # Make prediction
        prediction = model.predict(X)
        loan_approved = prediction[0] >= 0.5
        result = {"loan_approved": bool(loan_approved)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
