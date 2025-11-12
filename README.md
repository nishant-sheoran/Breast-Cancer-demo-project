# Breast Cancer Prediction Frontend

This repository now contains two complementary interfaces for the breast cancer prediction pipeline trained in `Untitled13.ipynb`.

1. **FastAPI backend + React dashboard (new)** – a standard web architecture where a REST API serves predictions/SHAP reasoning and a React UI collects feature inputs.
2. **Streamlit notebook-style app (original)** – still available at `streamlit_app.py` if you prefer an all-in-one Python experience.

## Project layout

```
.
├── backend/                # FastAPI service that loads breast_cancer_model.pkl and exposes /predict
│   ├── app.py
│   └── requirements.txt
├── frontend/               # Vite + React SPA that talks to the FastAPI backend
│   ├── package.json
│   └── src/…
├── streamlit_app.py        # Previous Streamlit interface (optional)
├── requirements.txt        # Streamlit UI deps
├── README.md
└── Untitled13.ipynb        # Colab notebook used for training/evaluation
```

## Backend (FastAPI + SHAP)

### Requirements

* Python 3.10+ (tested with 3.12)
* `backend/breast_cancer_model.pkl` (default) – a scikit-learn compatible pipeline that accepts the 10 features below in the given order
  ```
  radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
  compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
  fractal_dimension_mean
  ```

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

Place your trained pipeline inside `backend/` as `breast_cancer_model.pkl` (the default the API now loads automatically), or set the `MODEL_PATH` environment variable to an absolute/relative path before launching Uvicorn if you prefer a different filename or location.

### Run

```bash
uvicorn backend.app:app --reload --port 8000
```

* `GET /health` – quick readiness check.
* `POST /predict` – accepts the 10-feature JSON payload and returns `{ prediction, probability, reasons }`. When SHAP succeeds you receive contribution rows; otherwise the API falls back to a rule-based explanation (deviation from benign medians).

### Retrain / refresh the model

```bash
# Optional dependencies used only for training/evaluation
pip install -r backend/training_requirements.txt

# Train, evaluate, and export backend/breast_cancer_model.pkl
python backend/train_model.py
```

`train_model.py` mirrors the notebook logic: it downloads the Kaggle dataset (or falls back to `sklearn.datasets.load_breast_cancer`), evaluates the supported models across Original/SMOTE/undersampling strategies, retrains the best configuration on the full dataset, and saves a ready-to-serve `ScaledClassifier` wrapper at `backend/breast_cancer_model.pkl`.

## Frontend (React)

### Requirements

* Node.js 18+ (works great with 20.x)

### Setup

```bash
cd frontend
npm install
```

Set `VITE_API_URL` if the API is not at `http://localhost:8000`:

```bash
echo VITE_API_URL=http://127.0.0.1:8000 > .env.local
```

### Run

```bash
npm run dev
```

Open the printed Vite URL (default http://localhost:5173). Enter the ten feature values, submit, and you will see the model prediction plus the ranked reasoning table returned by the API.

## Streamlit app (optional)

If you prefer to stay entirely in Python, the original Streamlit UI is still available:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes & troubleshooting

* Both frontends expect the `.pkl` file to include any preprocessing (scaling, SMOTE, etc.). Save the full pipeline when exporting from the notebook.
* SHAP requires access to at least `predict_proba` or `decision_function` for the most faithful explanations. Without those, the backend still responds using the fallback z-score logic so the React UI always has a reason string to display.
* This project is intended for research support only and **not** for clinical decision making.
