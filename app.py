from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/", methods=["GET","POST"])
def home():
    prediction = ""

    if request.method == "POST":
        age = float(request.form["age"])
        policy_deductable = float(request.form["policy_deductable"])

        data = pd.DataFrame([[age, policy_deductable]], columns=["age","policy_deductable"])

        scaled = scaler.transform(data)
        result = model.predict(scaled)

        if result[0] == 1:
            prediction = "Fraud Claim"
        else:
            prediction = "Not Fraud"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
  
