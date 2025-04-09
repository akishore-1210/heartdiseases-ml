from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("heart_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        try:
            # Get input values from form
            age = int(request.form["age"])
            high_bp = int(request.form["high_bp"])
            high_cholesterol = int(request.form["high_cholesterol"])
            diabetes = int(request.form["diabetes"])
            smoking = int(request.form["smoking"])
            obesity = int(request.form["obesity"])

            # Ensure input matches model's expected format
            input_data = np.array([[age, high_bp, high_cholesterol, diabetes, smoking, obesity]])

            # Make prediction
            prediction_result = model.predict(input_data)

            # Convert prediction to readable output
            prediction = "High Heart Risk" if prediction_result[0] == 1 else "Low Heart Risk"
        
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
