from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# 👉 Landing Page
@app.route("/")
def landing():
    return render_template("landing.html")


# 👉 Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            name = request.form['name']

            pregnancies = int(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            bp = float(request.form['BloodPressure'])
            skin = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DiabetesPedigreeFunction'])
            age = int(request.form['Age'])

            # Validation
            if not (0 <= pregnancies <= 15):
                return render_template("index.html", error="Pregnancies must be between 0–15")

            if not (50 <= glucose <= 250):
                return render_template("index.html", error="Glucose must be between 50–250")

            if not (40 <= bp <= 180):
                return render_template("index.html", error="Blood Pressure must be between 40–180")

            if not (10 <= skin <= 60):
                return render_template("index.html", error="Skin Thickness must be between 10–60")

            if not (0 <= insulin <= 300):
                return render_template("index.html", error="Insulin must be between 0–300")

            if not (10 <= bmi <= 60):
                return render_template("index.html", error="BMI must be between 10–60")

            if not (0.0 <= dpf <= 3):
                return render_template("index.html", error="DPF must be between 0–3")

            if not (10 <= age <= 100):
                return render_template("index.html", error="Age must be between 10–100")

            data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

            data = scaler.transform(data)

            prediction = model.predict(data)

            result = f"{name}, You are Diabetic 😟" if prediction[0] == 1 else f"{name}, You are Not Diabetic 😊"

            return render_template("index.html", result=result)

        except:
            return render_template("index.html", error="Invalid input!")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)