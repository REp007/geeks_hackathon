# AI-Powered HR Automation: Transforming Entire Department Workflows
## Getting Started

To run this app, follow these steps:

1. **Install dependencies** using [uv](https://github.com/astral-sh/uv):

    ```bash
    uv pip install -r requirements.txt
    ```

2. **Run the app** with Python 3:

    ```bash
    python3 main.py
    ```
using uv 



```
hr_automation/
│── app.py                # Flask entrypoint
│── models/               # Trained ML model files (.pkl, .h5)
│   ├── salary_model.pkl
│   ├── jobfit_model.pkl
│   ├── resume_model.h5
│   └── priority_model.pkl
│── routes/               # API endpoint blueprints
│   ├── salary.py
│   ├── jobfit.py
│   ├── resume.py
│   └── priority.py
│── training/             # Scripts to train & save models
│   ├── train_salary.py
│   ├── train_jobfit.py
│   ├── train_resume.py
│   └── train_priority.py
│── utils/                # Helpers
│   ├── preprocess.py     # Text/vectorizers, encoders
│   └── database.py       # MongoDB connector
│── requirements.txt
│── .env                  # Environment variables (MongoDB URI, API keys)
```