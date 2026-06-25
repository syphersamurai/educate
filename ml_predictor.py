import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'student_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'student_performance.csv')

# Stress and motivation level maps (for converting string inputs from the form)
STRESS_MAP = {'low': 0, 'medium': 1, 'high': 2}
MOTIVATION_MAP = {'low': 0, 'medium': 1, 'high': 2}

# Friendly labels for the numeric codes stored in the CSV.
# FinalGrade ordering verified against mean ExamScore: code 0 = highest.
LEARNING_STYLE_LABELS = {
    0: 'Visual',
    1: 'Auditory',
    2: 'Reading/Writing',
    3: 'Kinesthetic',
}
FINAL_GRADE_LABELS = {
    0: 'A (Excellent)',
    1: 'B (Good)',
    2: 'C (Average)',
    3: 'D (Needs Improvement)',
}


def train_and_save_model():
    """
    Loads the CSV, encodes categorical columns,
    trains two RandomForest models (LearningStyle + FinalGrade),
    and saves them with their encoders.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Please place the student_performance.csv "
            f"file in the same folder as this script."
        )

    df = pd.read_csv(CSV_PATH)

    print(f"[ML] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[ML] Columns found: {list(df.columns)}")

    # --- Flexible column name matching ---
    # The CSV may use different capitalisation or spacing
    col_map = {col.lower().replace(' ', '').replace('_', ''): col for col in df.columns}

    def get_col(name):
        key = name.lower().replace(' ', '').replace('_', '')
        if key in col_map:
            return col_map[key]
        raise KeyError(
            f"Column '{name}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    study_col      = get_col('StudyHours')
    attend_col     = get_col('Attendance')
    stress_col     = get_col('StressLevel')
    motivation_col = get_col('Motivation')
    style_col      = get_col('LearningStyle')
    grade_col      = get_col('FinalGrade')

    features = [study_col, attend_col, stress_col, motivation_col]

    # --- Encode categorical feature columns if needed ---
    encoders = {}

    for col in features:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"[ML] Encoded feature column: {col}")

    X = df[features].fillna(0)

    # --- Encode target: LearningStyle ---
    le_style = LabelEncoder()
    y_style = le_style.fit_transform(df[style_col].astype(str))
    encoders['LearningStyle'] = le_style
    print(f"[ML] LearningStyle classes: {list(le_style.classes_)}")

    # --- Encode target: FinalGrade ---
    le_grade = LabelEncoder()
    y_grade = le_grade.fit_transform(df[grade_col].astype(str))
    encoders['FinalGrade'] = le_grade
    print(f"[ML] FinalGrade classes: {list(le_grade.classes_)}")

    # --- Train models ---
    # Bounded depth keeps the saved file small (~MB instead of ~100MB)
    # and reduces overfitting on this dataset.
    clf_style = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    clf_style.fit(X, y_style)

    clf_grade = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    clf_grade.fit(X, y_grade)

    # --- Save everything ---
    joblib.dump(
        {
            'style_model': clf_style,
            'grade_model': clf_grade,
            'feature_cols': features
        },
        MODEL_PATH
    )
    joblib.dump(encoders, ENCODER_PATH)

    print(f"[ML] Models saved to {MODEL_PATH}")
    print(f"[ML] Encoders saved to {ENCODER_PATH}")


_cached_models = None
_cached_encoders = None


def _load_models():
    """Load (and lazily train) the models — cached after the first call."""
    global _cached_models, _cached_encoders
    if _cached_models is None or _cached_encoders is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            print("[ML] Model not found. Training now...")
            train_and_save_model()
        _cached_models = joblib.load(MODEL_PATH)
        _cached_encoders = joblib.load(ENCODER_PATH)
        print("[ML] Models loaded into memory")
    return _cached_models, _cached_encoders


def predict_student_profile(
    study_hours,
    attendance,
    stress_level,
    motivation
) -> str:
    """
    Predicts learning style and grade category from student metrics.

    Args:
        study_hours  : float or str — weekly study hours (e.g. 12)
        attendance   : float or str — attendance percentage (e.g. 75)
        stress_level : str — 'low', 'medium', or 'high'  (from dropdown)
        motivation   : str — 'low', 'medium', or 'high'  (from dropdown)

    Returns:
        A formatted string describing the predicted profile, ready for the LLM prompt.
    """

    models, encoders = _load_models()

    clf_style    = models['style_model']
    clf_grade    = models['grade_model']
    feature_cols = models['feature_cols']

    le_style = encoders['LearningStyle']
    le_grade = encoders['FinalGrade']

    # --- Convert string inputs to numeric ---
    stress_num     = STRESS_MAP.get(str(stress_level).lower(), 1)
    motivation_num = MOTIVATION_MAP.get(str(motivation).lower(), 1)

    # --- Build input DataFrame matching training columns ---
    input_values = {
        feature_cols[0]: float(study_hours),    # StudyHours
        feature_cols[1]: float(attendance),     # Attendance
        feature_cols[2]: stress_num,            # StressLevel
        feature_cols[3]: motivation_num         # Motivation
    }

    input_df = pd.DataFrame([input_values])

    # --- Predict ---
    style_idx  = clf_style.predict(input_df)[0]
    grade_idx  = clf_grade.predict(input_df)[0]

    learning_style_code = le_style.inverse_transform([style_idx])[0]
    grade_code = le_grade.inverse_transform([grade_idx])[0]

    # Translate numeric codes into readable labels for the LLM
    learning_style = LEARNING_STYLE_LABELS.get(int(learning_style_code), f'Style {learning_style_code}')
    grade_category = FINAL_GRADE_LABELS.get(int(grade_code), f'Grade {grade_code}')

    # --- Format stress/motivation back to readable ---
    stress_str     = str(stress_level).capitalize()
    motivation_str = str(motivation).capitalize()

    # --- Build profile string for LLM ---
    profile_string = (
        f"Predicted Learning Style: {learning_style}. "
        f"Predicted Grade Category: {grade_category}. "
        f"Stress Level: {stress_str}. "
        f"Motivation Level: {motivation_str}. "
        f"Weekly Study Hours: {study_hours}. "
        f"Attendance Rate: {attendance}%."
    )

    print(f"[ML] Prediction: {profile_string}")
    return profile_string


if __name__ == "__main__":
    print("=" * 50)
    print("Training and testing ML Predictor...")
    print("=" * 50)

    train_and_save_model()

    print("\nRunning test prediction...")
    result = predict_student_profile(
        study_hours=15,
        attendance=80,
        stress_level='medium',
        motivation='high'
    )
    print(f"\nTest Result:\n{result}")