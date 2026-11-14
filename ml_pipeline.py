import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
 
FEATURES = ["study_hours", "attendance", "previous_marks"]
 
def clean_data(df):
    """Check required columns and remove rows with missing target labels."""
    required = FEATURES + ["passed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    cleaned = df.copy().dropna(subset=["passed"])
    for col in FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=FEATURES)
    return cleaned
 
def validate_data(df):
    """Check data ranges and target labels."""
    if df.empty:
        raise ValueError("Dataset is empty")
    if not df["study_hours"].between(0, 24).all():
        raise ValueError("study_hours must be between 0 and 24")
    if not df["attendance"].between(0, 100).all():
        raise ValueError("attendance must be between 0 and 100")
    if not df["previous_marks"].between(0, 100).all():
        raise ValueError("previous_marks must be between 0 and 100")
    if not df["passed"].isin([0, 1]).all():
        raise ValueError("passed must contain only 0 or 1")
    return True
 
def build_model():
    """Build preprocessing and Logistic Regression as one pipeline."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
 
def train_model(df):
    """Clean, validate and train the classifier."""
    cleaned = clean_data(df)
    validate_data(cleaned)
    X = cleaned[FEATURES]
    y = cleaned["passed"]
    if y.nunique() < 2:
        raise ValueError("Training data must contain both classes: 0 and 1")
    model = build_model()
    model.fit(X, y)
    return model
 
def predict(model, rows):
    """Predict results for rows containing the required features."""
    missing = [col for col in FEATURES if col not in rows.columns]
    if missing:
        raise ValueError(f"Missing prediction features: {missing}")
    return model.predict(rows[FEATURES])