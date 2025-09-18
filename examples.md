example of using model 1(linear_regression_model)
pkl for model 1
```# Load model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder info (columns)
with open("salary_encoder.pkl", "rb") as f:
    encoder_info = pickle.load(f)

expected_cols = encoder_info["columns"]

# Incoming JSON
data = {
  "years_experience": 8.2,
  "role": "bi engineer",
  "degree": "phd",
  "company_size": "mid",
  "location": "casablanca",
  "level": "junior"
}

# Convert to DataFrame
import pandas as pd
row = pd.DataFrame([data])

# One-hot encode with same categories
row_encoded = pd.get_dummies(row, columns=encoder_info["categorical"], drop_first=True)

# Add missing columns (set them = 0)
for col in expected_cols:
    if col not in row_encoded:
        row_encoded[col] = 0

# Reorder columns to match training
row_encoded = row_encoded[expected_cols]

# Predict
pred = model.predict(row_encoded)[0]
print("Predicted salary MAD:", pred)
```

---- 

example of using model4 (decision_tree_model) 

```
import pickle

# Load model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_target = encoders["priority"]

sample = {
    "years_exp_band": "1-3",
    "skills_coverage_band": "medium", # Corrected typo
    "referral_flag": 1,
    "english_level": "b2",
    "location_match": "local" # Changed to 'local' as 'yes' is not a valid category
}

row = [
    encoders["years_exp_band"].transform([sample["years_exp_band"]])[0],
    encoders["skills_coverage_band"].transform([sample["skills_coverage_band"]])[0],
    sample["referral_flag"],  # already numeric
    encoders["english_level"].transform([sample["english_level"]])[0],
    encoders["location_match"].transform([sample["location_match"]])[0]
]
print("Encoded row:", row)


yhat = model.predict([row])
priority = le_target.inverse_transform(yhat)[0]
print("Predicted priority:", priority)

```