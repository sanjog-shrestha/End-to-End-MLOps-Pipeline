from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
app = Flask(__name__)
model = joblib.load("house_price_xgb_model.pkl")
@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    if request.method == "POST":
        try:
            input_data = {
                'area': [float(request.form['area'])],
                'bedrooms': [int(request.form['bedrooms'])],
                'bathrooms': [int(request.form['bathrooms'])],
                'stories': [int(request.form['stories'])],
                'mainroad': [request.form['mainroad']],
                'guestroom': [request.form['guestroom']],
                'basement': [request.form['basement']],
                'hotwaterheating': [request.form['hotwaterheating']],
                'airconditioning': [request.form['airconditioning']],
                'parking': [int(request.form['parking'])],
                'prefarea': [request.form['prefarea']],
                'furnishingstatus': [request.form['furnishingstatus']]
            }

            df = pd.DataFrame(input_data)
            pred_log = model.predict(df)
            predicted_price = np.expm1(pred_log)[0]  # convert log back to price
        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template("index.html", price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)