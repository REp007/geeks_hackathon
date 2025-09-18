# 📄 INSTRUCTIONS.md

## 🎯 Project Goal

This project is an **AI-Powered HR Automation System**.
It combines **4 Machine Learning models** with a **Flask API service** to handle HR tasks:

1. **Job Fit Prediction** – Logistic Regression + TF-IDF
2. **Salary Prediction** – Linear Regression + OneHotEncoder
3. **Resume Screening** – Deep Learning (TF-IDF + Neural Network)
4. **Candidate Priority** – Random Forest

Flask exposes REST endpoints to query each model. MongoDB is used as the data store.

---

## 📂 Project Structure

```
hr_automation/
│── app.py                # Flask entrypoint
│── models/               # Saved models (.pkl, .h5)
│   ├── salary_model.pkl
│   ├── jobfit_model.pkl
│   ├── resume_model.h5
│   └── priority_model.pkl
│── routes/               # API endpoints
│   ├── salary.py
│   ├── jobfit.py
│   ├── resume.py
│   └── priority.py
│── training/             # Jupyter Notebooks or Python scripts to train & save models
│   ├── train_salary.ipynb
│   ├── train_jobfit.ipynb
│   ├── train_resume.ipynb
│   └── train_priority.ipynb
│── utils/                # Helper functions
│   ├── preprocess.py
│   └── database.py
│── requirements.txt
│── .env                  # MongoDB URI, API keys
```

---

## 🧠 Codex Instructions

When generating or editing code, follow these rules:

### 1. Training Scripts (`training/`)

* Always load data with **pandas**.
* For text features, use **TfidfVectorizer**.
* For categorical features, use **OneHotEncoder** inside a **Pipeline**.
* Always split into train/test sets with `train_test_split`.
* Save models with **joblib** (`.pkl`) or **keras.save\_model()** (`.h5`).
* Place trained models inside `/models/`.

### 2. Flask Service (`app.py` + `routes/`)

* Each ML model has its own route file under `/routes/`.
* Endpoints must follow REST conventions:

  * `POST /api/salary/predict`
  * `POST /api/job_fit/predict`
  * `POST /api/resume_screen/predict`
  * `POST /api/candidate_priority/predict`
* Load models at the top of each route file (not inside the request handler).
* Always return `jsonify({...})`.

### 3. MongoDB (`utils/database.py`)

* Use `pymongo` for connection.
* Read DB credentials from `.env`.
* Collections:

  * `candidates`
  * `jobs`

### 4. Preprocessing (`utils/preprocess.py`)

* Implement reusable text cleaners, encoders, and feature builders.
* Example: `clean_text(resume_text)` → lowercasing, removing stopwords.

### 5. Coding Style

* Follow **PEP8**.
* Keep functions short and modular.
* Comment critical steps.
* Print shapes of datasets when training models.

---

## 🚀 Workflow

1. **Data Prep**

   * Place datasets in `dataset/`.
   * Clean + preprocess in Jupyter notebooks.

2. **Train Models**

   * Run notebooks (`training/*.ipynb`) to train and save `.pkl` / `.h5`.

3. **Flask API**

   * Run `python app.py` to start backend at `http://localhost:8000/`.

4. **Test Endpoints**

   * Use **Postman** or `curl` to send JSON requests.
   * Example:

     ```bash
     curl -X POST http://localhost:8000/api/salary/predict \
     -H "Content-Type: application/json" \
     -d '{"years_experience": 5, "role": "Data Scientist", "degree": "Masters", "company_size": "Large", "location": "Casablanca", "level": "Senior"}'
     ```

---

## ✅ What Codex Should Do

* Write **training code** (Scikit-learn, TensorFlow).
* Write **Flask routes** to load models & return predictions.
* Help with **MongoDB queries** (insert, find, update).
* Generate **unit tests** for endpoints.
* NEVER hardcode secrets → always load from `.env`.
* Always save trained models into `/models/`.

---

## ❌ What Codex Should NOT Do

* Do not put model training inside Flask routes.
* Do not commit raw dataset files (just reference them).
* Do not hardcode database URIs or API keys.

