import joblib

from utils.feature_extractor import extract_features
from utils.recommendations import recommendations

# Load Model
model = joblib.load(
    "model/pomegranate_rf_model.pkl"
)

def predict_disease(image_path):

    # Extract Features
    features = extract_features(
        image_path
    )

    if features is None:
        return {
            "error": "Feature extraction failed"
        }

    # Predict Disease
    prediction = model.predict(
        [features]
    )[0]

    # Confidence
    confidence = max(
        model.predict_proba(
            [features]
        )[0]
    ) * 100

    # Get Recommendation Data
    rec = recommendations.get(
        prediction,
        {}
    )

    # Final Response
    result = {

        "status":
        "Healthy"
        if prediction == "Healthy"
        else "Diseased",

        "disease":
        prediction,

        "confidence":
        round(confidence, 2),

        "severity":
        "High"
        if confidence > 90
        else "Medium",

        "description":
        rec.get("description", ""),

        "symptoms":
        ", ".join(
            rec.get("symptoms", [])
        ),

        "treatment":
        ", ".join(
            rec.get("chemical_spray", [])
        ),

        "spray":
        ", ".join(
            rec.get("chemical_spray", [])
        ),

        "fertilizer":
        ", ".join(
            rec.get("fertilizer", [])
        ),

        "organic_solution":
        ", ".join(
            rec.get(
                "organic_treatment",
                []
            )
        ),

        "prevention":
        ", ".join(
            rec.get("prevention", [])
        ),

        "watering_advice":
        ", ".join(
            rec.get(
                "water_management",
                []
            )
        ),

        "chemical_pesticides":
        rec.get(
            "chemical_spray",
            []
        ),

        "organic_pesticides":
        rec.get(
            "organic_spray",
            []
        ),

        "chemical_fertilizers":
        rec.get(
            "fertilizer",
            []
        ),

        "organic_fertilizers":
        rec.get(
            "organic_treatment",
            []
        ),

        "farmer_advice":
        rec.get(
            "farmer_advice",
            []
        )
    }

    return result