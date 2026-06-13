from flask import Flask, request, render_template, jsonify
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Trained Models")

# --- Load Trained Model ---
print("Loading svc_trained_model.pkl ...")
model = joblib.load(os.path.join(MODEL_DIR, "svc_trained_model.pkl"))
print("Model loaded successfully!")

# --- Define and fit Label Encoders (same as notebook) ---
import pandas as pd

capShape_le             = LabelEncoder().fit(["b", "c", "f", "k", "s", "x"])
capSurface_le           = LabelEncoder().fit(["f", "g", "s", "y"])
capColor_le             = LabelEncoder().fit(["b", "c", "e", "g", "n", "p", "r", "u", "w", "y"])
bruises_le              = LabelEncoder().fit(["f", "t"])
odor_le                 = LabelEncoder().fit(["a", "c", "f", "l", "m", "n", "p", "s", "y"])
gillSize_le             = LabelEncoder().fit(["b", "n"])
gillColor_le            = LabelEncoder().fit(["b", "e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"])
stalkRoot_le            = LabelEncoder().fit(["?", "b", "c", "e", "r"])
stalkSurfaceBelowRing_le = LabelEncoder().fit(["f", "k", "s", "y"])
ringNumber_le           = LabelEncoder().fit(["n", "o", "t"])
ringType_le             = LabelEncoder().fit(["e", "f", "l", "n", "p"])
sporePrintColor_le      = LabelEncoder().fit(["b", "h", "k", "n", "o", "r", "u", "w", "y"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        cap_shape               = request.form.get("cap_shape")
        cap_surface             = request.form.get("cap_surface")
        cap_color               = request.form.get("cap_color")
        bruises                 = request.form.get("bruises")
        odor                    = request.form.get("odor")
        gill_size               = request.form.get("gill_size")
        gill_color              = request.form.get("gill_color")
        stalk_root              = request.form.get("stalk_root")
        stalk_surface_below_ring = request.form.get("stalk_surface_below_ring")
        ring_number             = request.form.get("ring_number")
        ring_type               = request.form.get("ring_type")
        spore_print_color       = request.form.get("spore_print_color")

        # Validation
        if not all([cap_shape, cap_surface, cap_color, bruises, odor,
                    gill_size, gill_color, stalk_root, stalk_surface_below_ring,
                    ring_number, ring_type, spore_print_color]):
            return jsonify({"error": "Missing input fields"}), 400

        # Encode inputs using label encoders (same order as notebook training)
        features = np.array([[
            capShape_le.transform([cap_shape])[0],
            capSurface_le.transform([cap_surface])[0],
            capColor_le.transform([cap_color])[0],
            bruises_le.transform([bruises])[0],
            odor_le.transform([odor])[0],
            gillSize_le.transform([gill_size])[0],
            gillColor_le.transform([gill_color])[0],
            stalkRoot_le.transform([stalk_root])[0],
            stalkSurfaceBelowRing_le.transform([stalk_surface_below_ring])[0],
            ringNumber_le.transform([ring_number])[0],
            ringType_le.transform([ring_type])[0],
            sporePrintColor_le.transform([spore_print_color])[0],
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        result = "Edible" if prediction == 0 else "Poisonous"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))