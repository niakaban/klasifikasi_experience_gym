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

# Tips berbasis feature_importances_ model (dicek 2026-07-07, dari model.pkl kamu).
# Age/Gender/Height/BMI/Weight sengaja dikeluarkan meski ada importance-nya,
# karena bukan sesuatu yang bisa "ditingkatkan" user untuk naik level.
# Calories_Burned dikeluarkan karena turunan dari frekuensi/durasi (sirkular).
# Workout_Type dikeluarkan karena importance-nya sangat rendah (~0.002-0.004).
# [Medium confidence] urutan importance ini belum dicek silang dengan korelasi
# antar fitur — feature_importances_ RandomForest bisa bias ke fitur kontinu.
TIPS = {
    1: {
        "judul": "Rekomendasi untuk Level Pemula",
        "poin": [
            "Frekuensi latihan (hari/minggu) adalah faktor paling berpengaruh dalam data ini — tingkatkan secara bertahap, bukan lompat drastis.",
            "Durasi sesi latihan adalah faktor kedua paling berpengaruh — perpanjang secara gradual sambil menjaga kualitas gerakan.",
            "Perhatikan asupan cairan (water intake) selama dan setelah latihan.",
        ],
    },
    2: {
        "judul": "Rekomendasi untuk Level Menengah",
        "poin": [
            "Pertahankan konsistensi frekuensi dan durasi latihan yang sudah baik.",
            "Resting BPM yang menurun seiring waktu jadi indikator kebugaran kardio membaik — bisa jadi metrik tambahan untuk dipantau.",
        ],
    },
}


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
    numeric_fields = [f for f in REQUIRED_FIELDS if f not in ("Gender", "Workout_Type")]
    for f in numeric_fields:
        df_in[f] = df_in[f].astype(float)
    df_in["Gender"] = le_gender.transform(df_in["Gender"])
    df_in = pd.get_dummies(df_in, columns=["Workout_Type"], prefix="Workout")
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
        err_msg = f"Gagal melakukan prediksi: {e}"
        if request.is_json:
            return jsonify(error=err_msg), 500
        return render_template(
            "index.html", genders=VALID_GENDERS, workout_types=VALID_WORKOUT_TYPES,
            error=err_msg, form_data=data,
        )

    tips = TIPS.get(pred)

    if request.is_json:
        return jsonify(prediction=pred, label=LABEL_MAP[pred], tips=tips)
    return render_template(
        "index.html", genders=VALID_GENDERS, workout_types=VALID_WORKOUT_TYPES,
        result=LABEL_MAP[pred], form_data=data, tips=tips,
    )


if __name__ == "__main__":
    app.run(debug=True)
