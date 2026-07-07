from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
artifacts = joblib.load("model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
le_gender = artifacts["le_gender"]
feature_columns = artifacts["feature_columns"]
_NON_CATEGORY_WORKOUT_COLUMNS = {"Workout_Frequency (days/week)"}
VALID_WORKOUT_TYPES = sorted(
    col.replace("Workout_", "", 1)
    for col in feature_columns
    if col.startswith("Workout_") and col not in _NON_CATEGORY_WORKOUT_COLUMNS
)
VALID_GENDERS = sorted(le_gender.classes_)

LABEL_MAP = {1: "Pemula (Level 1)", 2: "Menengah (Level 2)"}

REQUIRED_FIELDS = [
    "Age", "Gender", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM",
    "Resting_BPM", "Session_Duration (hours)", "Calories_Burned",
    "Workout_Type", "Fat_Percentage", "Water_Intake (liters)",
    "Workout_Frequency (days/week)", "BMI",
]


def validate_input(data: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, "")]
    if missing:
        return f"Field wajib hilang: {missing}"

    if data["Gender"] not in VALID_GENDERS:
        return f"Gender '{data['Gender']}' tidak valid. Pilihan: {VALID_GENDERS}"

    if data["Workout_Type"] not in VALID_WORKOUT_TYPES:
        return f"Workout_Type '{data['Workout_Type']}' tidak valid. Pilihan: {VALID_WORKOUT_TYPES}"

    numeric_fields = [f for f in REQUIRED_FIELDS if f not in ("Gender", "Workout_Type")]
    for f in numeric_fields:
        try:
            float(data[f])
        except (TypeError, ValueError):
            return f"Field '{f}' harus berupa angka, diterima: {data[f]!r}"

    return None


def predict_level(input_dict: dict) -> int:
    """Prediksi label (1 atau 2). Mengikuti logika predict_level() di notebook
    section 19, tapi hanya memakai objek yang di-load dari model.pkl."""
    df_in = pd.DataFrame([input_dict])

    # Cast field numerik ke float (form HTML mengirim semuanya sebagai string)
    numeric_fields = [f for f in REQUIRED_FIELDS if f not in ("Gender", "Workout_Type")]
    for f in numeric_fields:
        df_in[f] = df_in[f].astype(float)

    df_in["Gender"] = le_gender.transform(df_in["Gender"])
    df_in = pd.get_dummies(df_in, columns=["Workout_Type"], prefix="Workout")

    # Samakan kolom dengan urutan saat training
    for col in feature_columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[feature_columns]

    X_scaled = scaler.transform(df_in)
    pred = model.predict(X_scaled)[0]
    return int(pred)


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=VALID_GENDERS,
        workout_types=VALID_WORKOUT_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Dukung baik form HTML (index.html) maupun JSON (API call langsung)
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    err = validate_input(data)
    if err:
        if request.is_json:
            return jsonify(error=err), 400
        return render_template(
            "index.html", genders=VALID_GENDERS, workout_types=VALID_WORKOUT_TYPES,
            error=err, form_data=data,
        )

    try:
        pred = predict_level(data)
    except Exception as e:
        # Tangkap error tak terduga (mis. mismatch versi sklearn saat unpickle)
        # supaya user dapat pesan yang jelas, bukan HTML 500 generik.
        err_msg = f"Gagal melakukan prediksi: {e}"
        if request.is_json:
            return jsonify(error=err_msg), 500
        return render_template(
            "index.html", genders=VALID_GENDERS, workout_types=VALID_WORKOUT_TYPES,
            error=err_msg, form_data=data,
        )

    if request.is_json:
        return jsonify(prediction=pred, label=LABEL_MAP[pred])

    return render_template(
        "index.html", genders=VALID_GENDERS, workout_types=VALID_WORKOUT_TYPES,
        result=LABEL_MAP[pred], form_data=data,
    )

if __name__ == "__main__":
    app.run(debug=True)
