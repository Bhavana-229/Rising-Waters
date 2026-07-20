from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/flood_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]

        # Column names (must match the training dataset)
        columns = [
            "MonsoonIntensity",
            "TopographyDrainage",
            "RiverManagement",
            "Deforestation",
            "Urbanization",
            "ClimateChange",
            "DamsQuality",
            "Siltation",
            "AgriculturalPractices",
            "Encroachments",
            "IneffectiveDisasterPreparedness",
            "DrainageSystems",
            "CoastalVulnerability",
            "Landslides",
            "Watersheds",
            "DeterioratingInfrastructure",
            "PopulationScore",
            "WetlandLoss",
            "InadequatePlanning",
            "PoliticalFactors"
        ]

        # Convert input into DataFrame
        input_data = pd.DataFrame([features], columns=columns)

        # Predict flood probability
        prediction = model.predict(input_data)[0]

        # Determine risk level
        if prediction < 0.30:
            risk = "🟢 Low Flood Risk"
        elif prediction < 0.60:
            risk = "🟡 Moderate Flood Risk"
        else:
            risk = "🔴 High Flood Risk"

        return render_template(
            "result.html",
            prediction=round(prediction, 3),
            risk=risk
        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)