from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "house_price_model.joblib")
model, model_columns = bundle["model"], bundle["columns"]
cities = sorted(bundle["cities"])

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="/static")

def predict_price(form):
    row = pd.DataFrame([{
        "city": form["city"],
        "area_sqft": int(form["area_sqft"]),
        "bedrooms": int(form["bedrooms"]),
        "bathrooms": int(form["bathrooms"]),
        "property_age": int(form["property_age"]),
        "furnishing": form["furnishing"],
        "parking": int(form["parking"]),
        "locality_type": form["locality_type"],
    }])
    encoded = pd.get_dummies(row, columns=["city", "furnishing", "locality_type"])
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return max(0, float(model.predict(encoded)[0]))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    details = None
    error = None
    if request.method == "POST":
        try:
            area = int(request.form["area_sqft"])
            beds = int(request.form["bedrooms"])
            baths = int(request.form["bathrooms"])
            age = int(request.form["property_age"])
            parking = int(request.form["parking"])
            if not 250 <= area <= 10000: raise ValueError("Area must be between 250 and 10,000 sq. ft.")
            if not 1 <= beds <= 10 or not 1 <= baths <= 10: raise ValueError("Bedrooms and bathrooms must be between 1 and 10.")
            if not 0 <= age <= 100 or not 0 <= parking <= 5: raise ValueError("Please enter a valid property age and parking count.")
            prediction = predict_price(request.form)
            details = dict(request.form)
        except (ValueError, KeyError) as exc:
            error = str(exc) or "Please check the values and try again."
    return render_template("index.html", cities=cities, prediction=prediction, details=details, error=error)

@app.route("/about")
def about():
    return render_template("about.html", metrics=bundle["metrics"])

if __name__ == "__main__":
    app.run(debug=True)
