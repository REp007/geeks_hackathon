import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ENCODERS_DIR = BASE_DIR / "encoders"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


salary_model = _load_pickle(MODELS_DIR / "linear_regression_model.pkl")
salary_encoder_info: Dict[str, Any] = _load_pickle(ENCODERS_DIR / "salary_encoder.pkl")

priority_model = _load_pickle(MODELS_DIR / "decision_tree_model.pkl")
priority_encoders: Dict[str, Any] = _load_pickle(ENCODERS_DIR / "label_encoders.pkl")
priority_target_encoder = priority_encoders.get("priority")

app = Flask(__name__)


SALARY_FIELDS = [
    "years_experience",
    "role",
    "degree",
    "company_size",
    "location",
    "level",
]

PRIORITY_FIELDS = [
    "years_exp_band",
    "skills_coverage_band",
    "referral_flag",
    "english_level",
    "location_match",
]


def _validate_json(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _prepare_salary_features(payload: Dict[str, Any]) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for field in SALARY_FIELDS:
        if field not in payload:
            raise ValueError(f"Missing field: {field}")
        value = payload[field]
        if field == "years_experience":
            try:
                data[field] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("years_experience must be numeric") from exc
        else:
            data[field] = value

    df = pd.DataFrame([data])
    categorical: List[str] = salary_encoder_info.get("categorical", [])
    encoded = pd.get_dummies(df, columns=categorical, drop_first=True)

    expected_columns: List[str] = list(salary_encoder_info.get("columns", []))
    if not expected_columns:
        raise ValueError("Salary encoder is missing expected columns")

    encoded = encoded.reindex(columns=expected_columns, fill_value=0)
    return encoded.astype(float)


def _prepare_priority_features(payload: Dict[str, Any]) -> List[Any]:
    encoded_row: List[Any] = []
    missing = [field for field in PRIORITY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")

    for field in PRIORITY_FIELDS:
        value = payload[field]
        if field == "referral_flag":
            if isinstance(value, bool):
                value = int(value)
            try:
                encoded_row.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError("referral_flag must be 0 or 1") from exc
            continue

        encoder = priority_encoders.get(field)
        if encoder is None:
            raise ValueError(f"Encoder not found for field: {field}")
        try:
            encoded_value = encoder.transform([value])[0]
        except ValueError as exc:
            raise ValueError(f"Invalid value for {field}: {value}") from exc
        encoded_row.append(int(encoded_value))

    return encoded_row


@app.route("/api/salary/predict", methods=["POST"])
def predict_salary() -> Any:
    try:
        payload = _validate_json(request.get_json())
        features = _prepare_salary_features(payload)
        prediction = salary_model.predict(features)[0]
        return jsonify({"predicted_salary": float(prediction)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # pragma: no cover - unexpected errors surfaced to the client
        return jsonify({"error": "Failed to generate salary prediction"}), 500


@app.route("/api/priority/predict", methods=["POST"])
def predict_priority() -> Any:
    if priority_target_encoder is None:
        return jsonify({"error": "Priority encoder missing target mapping"}), 500
    try:
        payload = _validate_json(request.get_json())
        features = _prepare_priority_features(payload)
        prediction = priority_model.predict([features])
        label = priority_target_encoder.inverse_transform(prediction)[0]
        return jsonify({"priority": str(label)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to generate priority prediction"}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)
