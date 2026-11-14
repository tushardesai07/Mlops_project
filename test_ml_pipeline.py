import pandas as pd
import pytest
from ml_pipeline import clean_data, validate_data, train_model, predict
 
@pytest.fixture
def sample_data():
    # Small artificial dataset for learning only.
    return pd.DataFrame({
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "attendance": [40, 45, 50, 55, 65, 75, 85, 95],
        "previous_marks": [20, 28, 35, 42, 60, 70, 82, 92],
        "passed": [0, 0, 0, 0, 1, 1, 1, 1]
    })
 
def test_clean_data_removes_missing_feature(sample_data):
    sample_data.loc[0, "study_hours"] = None
    result = clean_data(sample_data)
    assert len(result) == 7
 
def test_clean_data_rejects_missing_column(sample_data):
    sample_data = sample_data.drop(columns=["attendance"])
    with pytest.raises(ValueError):
        clean_data(sample_data)
 
def test_validate_data_accepts_valid_data(sample_data):
    assert validate_data(sample_data) is True
 
def test_validate_data_rejects_invalid_attendance(sample_data):
    sample_data.loc[0, "attendance"] = 120
    with pytest.raises(ValueError, match="attendance"):
     validate_data(sample_data)
 
def test_training_pipeline_fits_model(sample_data):
    model = train_model(sample_data)
    assert hasattr(model, "predict")
 
def test_model_outputs_are_valid(sample_data):
    model = train_model(sample_data)
    rows = sample_data[["study_hours", "attendance", "previous_marks"]].iloc[:3]
    outputs = predict(model, rows)
    assert len(outputs) == 3
    assert set(outputs).issubset({0, 1})
 
def test_predict_rejects_missing_feature(sample_data):
    model = train_model(sample_data)
    rows = sample_data[["study_hours", "previous_marks"]]
    with pytest.raises(ValueError, match="attendance"):
        predict(model, rows)