# Backend Setup & Operations

FastAPI powers the prediction API that the React dashboard (and any other client) talks to. Follow the steps below to create an isolated environment, install dependencies, train the model artifact, and run the service locally.

## 1. Create & activate a virtual environment

```powershell
cd "C:\Users\pc\OneDrive\Desktop\Breast Cancer demo project"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux equivalent:

```bash
cd "/path/to/Breast Cancer demo project"
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install runtime dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

These packages are enough to *serve* the API (FastAPI, Uvicorn, scikit-learn, SHAP, etc.).

## 3. (Optional) Install training extras

Training uses KaggleHub, imbalanced-learn, and XGBoost. Install them only if you plan to regenerate the model:

```bash
python -m pip install -r backend/training_requirements.txt
```

## 4. Train / refresh the model artifact

The script mirrors the notebook pipeline: it downloads the Kaggle dataset (or you can pass `--use-sklearn-only` to skip the download), evaluates every model + sampling combination, retrains the best performer on the full dataset, and writes a pickle the API can load.

```bash
python -m backend.train_model               # saves backend/breast_cancer_model.pkl
# or skip Kaggle:
python -m backend.train_model --use-sklearn-only
```

Notes:

- The output path defaults to `backend/breast_cancer_model.pkl`; override with `--output path/to/file.pkl` if needed.
- If you run the script from inside `backend/`, use the module form (`python -m backend.train_model`) so imports continue to work.

## 5. Run the FastAPI service

From the project root (with the virtual environment active):

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Useful endpoints:

- `GET http://127.0.0.1:8000/health` - readiness check (returns `{"status":"ok"}` once the model is loaded).
- `POST http://127.0.0.1:8000/predict` - send the 10-feature JSON payload to obtain `{prediction, label, probability, reasons, base_value}`.

## 6. Environment variables

| Variable        | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `MODEL_PATH`    | Absolute or relative path to a pickle created by `backend/train_model.py`.  |
| `ALLOWED_ORIGINS` | Comma-separated list of origins FastAPI should allow for CORS (default `*`). |

Set them before launching Uvicorn, e.g. `set MODEL_PATH=C:\models\custom.pkl` on PowerShell or `export MODEL_PATH=/models/custom.pkl` on macOS/Linux.

## 7. Next steps

With the API running on `http://127.0.0.1:8000`, start the React app (inside `frontend/`) via `npm run dev`, optionally setting `VITE_API_URL=http://127.0.0.1:8000`. The refreshed `breast_cancer_model.pkl` plus the modernized UI complete the local stack.
