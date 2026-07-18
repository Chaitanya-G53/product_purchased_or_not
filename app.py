import os
import pickle
import math

import numpy as np
from flask import Flask, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "naive_purchased_model.pkl")

app = Flask(__name__)

# --------------------------------------------------------------------------------
# LOAD MODEL ONCE AT STARTUP
# --------------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find 'naive_purchased_model.pkl' at {MODEL_PATH}. "
        "Make sure it is committed to the same repo folder as app.py."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(getattr(model, "feature_names_in_", ["Gender", "Age", "EstimatedSalary"]))
CLASS_COUNT = getattr(model, "class_count_", np.array([0, 0]))
TOTAL_SAMPLES = int(CLASS_COUNT.sum())
PURCHASE_RATE = float(CLASS_COUNT[1] / TOTAL_SAMPLES) if TOTAL_SAMPLES else 0.0
THETA = getattr(model, "theta_", None)  # per-class feature means


def gauge_dasharray(pct, radius=70):
    """Return (dash, circumference) for an SVG stroke-dasharray ring."""
    pct = max(0.0, min(1.0, pct))
    circumference = 2 * math.pi * radius
    dash = circumference * pct
    return round(dash, 1), round(circumference, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    # Defaults for the form (used on first load / GET)
    form_values = {"gender": "Female", "age": 35, "salary": 60000}

    if request.method == "POST":
        gender = request.form.get("gender", "Female")
        age = float(request.form.get("age", 35))
        salary = float(request.form.get("salary", 60000))
        form_values = {"gender": gender, "age": int(age), "salary": int(salary)}

        gender_encoded = 1 if gender == "Male" else 0
        raw = {"Gender": gender_encoded, "Age": age, "EstimatedSalary": salary}
        row = np.array([[raw[f] for f in FEATURES]])

        pred_class = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        buy_prob = float(proba[list(model.classes_).index(1)])

        dash, circumference = gauge_dasharray(buy_prob)
        gauge_color = "#2ecc71" if buy_prob >= 0.5 else "#eb5757"

        if buy_prob >= 0.7:
            recommendation = ("success", "🟢 High-intent prospect — prioritize for direct outreach, "
                               "personalized offers, or premium product tiers.")
        elif buy_prob >= 0.4:
            recommendation = ("warning", "🟡 Moderate intent — consider retargeting ads or a "
                               "limited-time discount to nudge conversion.")
        else:
            recommendation = ("danger", "🔴 Low intent — deprioritize for paid outreach; "
                               "nurture via general awareness campaigns instead.")

        result = {
            "pred_class": pred_class,
            "buy_prob": buy_prob,
            "buy_pct": round(buy_prob * 100, 1),
            "nobuy_pct": round((1 - buy_prob) * 100, 1),
            "gauge_color": gauge_color,
            "dash": dash,
            "circumference": circumference,
            "recommendation_kind": recommendation[0],
            "recommendation_text": recommendation[1],
        }

    theta_nonbuy_age = round(float(THETA[0][1]), 1) if THETA is not None else None
    theta_buy_age = round(float(THETA[1][1]), 1) if THETA is not None else None
    theta_nonbuy_salary = round(float(THETA[0][2]), 0) if THETA is not None else None
    theta_buy_salary = round(float(THETA[1][2]), 0) if THETA is not None else None

    return render_template(
        "index.html",
        form_values=form_values,
        result=result,
        total_samples=TOTAL_SAMPLES,
        purchase_rate=round(PURCHASE_RATE * 100, 1),
        class_count=[int(CLASS_COUNT[0]), int(CLASS_COUNT[1])],
        theta_nonbuy_age=theta_nonbuy_age,
        theta_buy_age=theta_buy_age,
        theta_nonbuy_salary=theta_nonbuy_salary,
        theta_buy_salary=theta_buy_salary,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
