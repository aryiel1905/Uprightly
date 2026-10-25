# Uprightly: system architecture and model integration

## Live prediction path

1. The React app captures webcam frames locally and runs MediaPipe pose and face landmarkers in the browser (`src/App.tsx`).
2. It builds eight normalized points: nose, chin, both eye midpoints, both ears, and both shoulders.
3. `src/features/posture/randomForestFeatures.ts` validates those points and calculates the 35 features of `uprightly_8point_v1`.
4. The app sends those numbers, not the image, as JSON to `POST /predict` (`src/App.tsx`).
5. `backend/api.py` validates the request, orders the features, and calls the saved scikit-learn pipeline. The pipeline contains a median imputer followed by a 300-tree Random Forest.
6. The API returns `label`, `confidence`, `probabilities`, a probability-derived `score`, and brief `feedback`. The app displays and stabilizes the result.

The three model labels are `neutral_posture`, `mild_asymmetry`, and `severe_misalignment`. `GET /health` reports whether the API process is available; model and manifest validation occurs during startup.

## Active model and provenance

- Artifact: `backend/model/uprightly_random_forest_corrected_data.joblib`
- Manifest: `backend/model/uprightly_random_forest_corrected_data_manifest.json`
- Training: a separate Python retraining on 666 original corrected development images. This is not a binary conversion of the RapidMiner `.rmmodel`.
- Held-out evaluation: 261 images from separate participants; accuracy 76.245%, macro F1 56.023%. Neutral-posture recall was 2/21 (9.52%).
- Runtime: `backend/requirements.txt` pins scikit-learn 1.9.1 and joblib 1.6.0, the versions recorded with the artifact.

The held-out image evaluation does not measure webcam performance. The live browser uses MediaPipe task models, while the training feature extraction used MediaPipe Python solutions. The 35 formulas and landmark definitions align, but this difference can affect the values seen by the classifier. Real-session validation remains necessary.

The legacy `backend/train.py` trains a different, older feature schema. It is not the source of the active model.

## Running and verifying

Run the backend from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Set `VITE_ML_API_URL=http://127.0.0.1:8000` in the frontend `.env` and run `npm run dev`. The production frontend needs a deployed API URL in the same setting; deploying frontend assets alone does not host the Python model.

With the original handoff folder available, run:

```powershell
.\.venv\Scripts\python.exe -m backend.verify_original_model "C:\path\to\original_training_20261014"
```

That check compares all labels and class confidences against the supplied prediction record. See `MODEL_INTEGRATION_PRESENTATION.md` for concise code excerpts and a presentation outline.
