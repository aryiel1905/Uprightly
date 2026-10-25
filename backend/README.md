# Uprightly Random Forest Backend

The live posture result uses `uprightly_random_forest_python_original_data.joblib`, a three-class scikit-learn pipeline retrained from the original corrected development data. It is a separate model from the RapidMiner `.rmmodel`, not a conversion of that artifact. MediaPipe runs in the browser to extract the eight normalized points used by the `uprightly_8point_v1` feature schema; camera images and video are not sent to this API.

## Install and run

Use the pinned Python dependencies compatible with the saved model (scikit-learn 1.9.1 and joblib 1.6.0):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Configure the frontend in `.env`:

```bash
VITE_ML_API_URL=http://127.0.0.1:8000
```

## API contract

- `GET /health` returns service readiness.
- `POST /predict` accepts the 35 numeric features listed in `model/uprightly_random_forest_python_original_data_manifest.json` in a flat JSON object.
- The response contains `label`, `confidence`, class `probabilities`, probability-derived `score`, and `feedback`.

Supported labels are `neutral_posture`, `mild_asymmetry`, and `severe_misalignment`.

The active artifact is `backend/model/uprightly_random_forest_python_original_data.joblib`. Its pipeline includes median imputation, so a separate median CSV is not needed. Startup fails if the model or manifest is missing, if its feature order or classes differ from the API contract, or if the installed model library versions differ from the manifest.

The original 261-image held-out evaluation reported 76.245% accuracy and 56.023% macro F1. Only 2 of 21 neutral-posture images were correctly classified; do not describe the model as reliable at recognizing neutral posture. Those image-based results do not establish live-webcam accuracy. See `MODEL_INTEGRATION_PRESENTATION.md` for the code path and presentation excerpts.

To reproduce the supplied held-out predictions when the handoff folder is available locally:

```powershell
.\.venv\Scripts\python.exe -m backend.verify_original_model "C:\path\to\original_training_20261014"
```

The verifier compares all 261 labels and their three confidence values against `python_joblib/test_predictions.csv`. The older `backend/train.py` is a legacy training script for a different feature schema; it does not produce this supplied model.
