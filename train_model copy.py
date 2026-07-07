import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
 
DATASET_PATH = "gym_members_exercise_tracking.csv"
 
def main():
    # --- 1. Load dataset -----------------------------------------------
    df_raw = pd.read_csv(DATASET_PATH)
    print("Ukuran data asli:", df_raw.shape)
    print(df_raw["Experience_Level"].value_counts())
 
    # --- 2. Hapus Level 3 (sesuai arahan dosen, notebook section 3) ----
    df = df_raw[df_raw["Experience_Level"].isin([1, 2])].copy()
    df = df.reset_index(drop=True)
    print("Ukuran data setelah hapus Level 3:", df.shape)
 
    # --- 3. Missing value handling (notebook section 5) -----------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
 
    # --- 4. Encoding (notebook section 7) --------------------------------
    # Gender -> Label Encoding (biner, tidak menambah dimensi)
    # Workout_Type -> One-Hot Encoding (nominal, tidak ada urutan)
    df_encoded = df.copy()
    le_gender = LabelEncoder()
    df_encoded["Gender"] = le_gender.fit_transform(df_encoded["Gender"])
    print("Mapping Gender:", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
 
    df_encoded = pd.get_dummies(df_encoded, columns=["Workout_Type"], prefix="Workout", drop_first=False)
    for c in df_encoded.columns:
        if str(df_encoded[c].dtype) == "bool":
            df_encoded[c] = df_encoded[c].astype(int)
 
    # --- 5. Feature / target split (notebook section 8) ------------------
    X = df_encoded.drop(columns=["Experience_Level"])
    y = df_encoded["Experience_Level"]
    print("Jumlah fitur:", X.shape[1])
    print("Daftar fitur:", list(X.columns))
 
    # --- 6. Train-test split 80:20, stratified (notebook section 9) -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # --- 7. Scaling (notebook section 10) --------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    # --- 8. SMOTE hanya di train set (notebook section 11) ---------------
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    print("Distribusi y_train setelah SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())
 
    # --- 9. GridSearchCV — param grid identik dengan notebook section 15 -
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
    }
 
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_sm, y_train_sm)
    print("Parameter terbaik:", grid_search.best_params_)
 
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_scaled)
    print(f"Akurasi model final (test set): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Level 1", "Level 2"]))
 
    # --- 10. Simpan SATU dict artifact -----------------------------------
    artifacts = {
        "model": best_rf,
        "scaler": scaler,
        "le_gender": le_gender,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifacts, "model.pkl")
    print("\nmodel.pkl berhasil disimpan.")
    print("Keys:", list(artifacts.keys()))
 
 
if __name__ == "__main__":
    main()