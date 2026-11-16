import os
import joblib
import numpy as np
import tensorflow as tf
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

MODEL_PATH = os.environ.get("MODEL_PATH", "student_performance_ann.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "student_scaler.pkl")

# These two files must be uploaded into this backend folder before deployment.
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Fuzzy variables and membership functions based on the project's notebook.
marks = ctrl.Antecedent(np.arange(0, 21, 1), "marks")
absences = ctrl.Antecedent(np.arange(0, 76, 1), "absences")
studytime = ctrl.Antecedent(np.arange(1, 5, 1), "studytime")
performance = ctrl.Consequent(np.arange(0, 101, 1), "performance")

marks["poor"] = fuzz.trapmf(marks.universe, [0, 0, 5, 9])
marks["average"] = fuzz.trimf(marks.universe, [6, 10, 14])
marks["good"] = fuzz.trimf(marks.universe, [11, 15, 18])
marks["excellent"] = fuzz.trapmf(marks.universe, [16, 19, 20, 20])

absences["low"] = fuzz.trapmf(absences.universe, [0, 0, 2, 7])
absences["medium"] = fuzz.trimf(absences.universe, [4, 12, 22])
absences["high"] = fuzz.trapmf(absences.universe, [15, 30, 75, 75])

studytime["low"] = fuzz.trapmf(studytime.universe, [1, 1, 1.5, 2])
studytime["medium"] = fuzz.trimf(studytime.universe, [1.5, 2.5, 3.5])
studytime["high"] = fuzz.trapmf(studytime.universe, [3, 3.5, 4, 4])

performance["poor"] = fuzz.trapmf(performance.universe, [0, 0, 20, 40])
performance["average"] = fuzz.trimf(performance.universe, [30, 50, 70])
performance["good"] = fuzz.trimf(performance.universe, [60, 75, 90])
performance["excellent"] = fuzz.trapmf(performance.universe, [80, 90, 100, 100])

rules = [
    ctrl.Rule(marks["poor"], performance["poor"]),
    ctrl.Rule(marks["average"] & absences["high"], performance["average"]),
    ctrl.Rule(marks["average"] & studytime["high"], performance["good"]),
    ctrl.Rule(marks["average"], performance["average"]),
    ctrl.Rule(marks["good"] & absences["high"] & studytime["low"], performance["average"]),
    ctrl.Rule(marks["good"], performance["good"]),
    ctrl.Rule(marks["excellent"], performance["excellent"]),
]
performance_control = ctrl.ControlSystem(rules)


def get_category(score):
    if score < 40:
        return "Poor", "Needs additional academic support. Encourage a regular study routine and monitor progress."
    if score < 60:
        return "Average", "Maintain consistent study habits and focus on improving areas with lower marks."
    if score < 80:
        return "Good", "Good progress. Continue the current study routine and work on strengthening understanding."
    return "Excellent", "Excellent performance. Maintain these effective learning habits and continue progressing."


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Student Performance Neuro-Fuzzy API is running.",
        "endpoint": "POST /predict"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        study = float(data["studytime"])
        absent = float(data["absences"])
        g1 = float(data["G1"])
        g2 = float(data["G2"])

        if study not in [1, 2, 3, 4]:
            return jsonify({"error": "studytime must be 1, 2, 3, or 4."}), 400
        if not 0 <= absent <= 75 or not absent.is_integer():
            return jsonify({"error": "absences must be a whole number from 0 to 75."}), 400
        if not 0 <= g1 <= 20 or not 0 <= g2 <= 20:
            return jsonify({"error": "G1 and G2 must be between 0 and 20."}), 400

        # Feature order must match the training notebook:
        # ["studytime", "absences", "G1", "G2"].
        input_array = np.array([[study, absent, g1, g2]], dtype=float)
        scaled_input = scaler.transform(input_array)
        predicted = float(model.predict(scaled_input, verbose=0).reshape(-1)[0])
        predicted = float(np.clip(predicted, 0, 20))

        fuzzy_sim = ctrl.ControlSystemSimulation(performance_control)
        fuzzy_sim.input["marks"] = predicted
        fuzzy_sim.input["absences"] = absent
        fuzzy_sim.input["studytime"] = study
        fuzzy_sim.compute()
        fuzzy_score = float(fuzzy_sim.output["performance"])
        category, recommendation = get_category(fuzzy_score)

        return jsonify({
            "predicted_grade": round(predicted, 2),
            "fuzzy_score": round(fuzzy_score, 2),
            "category": category,
            "recommendation": recommendation
        })
    except KeyError as exc:
        return jsonify({"error": f"Missing input field: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("Prediction error")
        return jsonify({"error": "Unable to calculate prediction. Check model files and backend logs."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
