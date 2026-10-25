# Presenting the Uprightly model integration

## One-slide explanation

**Webcam frame → MediaPipe landmarks → 35 posture features → FastAPI `/predict` → saved Random Forest pipeline → posture feedback.**

The model is `uprightly_random_forest_corrected_data.joblib`. It is a separately trained scikit-learn pipeline on the original corrected development data, not a conversion of the RapidMiner `.rmmodel`. The pipeline includes median imputation and a 300-tree Random Forest. The backend loads it once at startup; each valid request supplies the 35 named numeric features in the manifest's order.

## Code to copy into slides

These excerpts come from the actual integrated code. Use the source files named below when you need the full surrounding function.

### 1. Calculate model inputs in the browser

Source: `src/App.tsx` and `src/features/posture/randomForestFeatures.ts`.

```ts
const rfFeatures =
  corePoseVisible && frameValidation.valid && rfPoints
    ? extractRandomForestFeatures(rfPoints)
    : null;
```

`extractRandomForestFeatures` calculates 35 angles, distances, and normalized offsets from the eight landmark points. For example:

```ts
shoulder_tilt_angle_degrees: lineAngle(points.LS, points.RS),
eye_span_to_shoulder_ratio: ratio(distance(points.LE, points.RE)),
```

### 2. Send features to the backend

Source: `src/App.tsx`, inside `inferMl`.

```ts
const res = await fetch(`${mlApiUrl}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
```

`payload` is the 35-feature object. The request does not include a camera image.

### 3. Load and validate the saved model

Source: `backend/api.py`, inside `load_model`.

```python
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
if metadata.get("feature_names_in_order") != FEATURES:
    raise RuntimeError("The model manifest feature order is not supported.")
if metadata.get("class_order") != CLASSES:
    raise RuntimeError("The model manifest classes are not supported.")

_model = joblib.load(MODEL_PATH)
validate_model(_model)
```

The code also checks that both files exist, that the schema ID is `uprightly_8point_v1`, and that the installed scikit-learn and joblib versions match the manifest.

### 4. Classify one request

Source: `backend/api.py`, inside `predict`.

```python
row = body.model_dump()
frame = pd.DataFrame([[row[name] for name in FEATURES]], columns=FEATURES)
proba = _model.predict_proba(frame)[0]
classes = [str(value) for value in _model.classes_]
probabilities = {
    class_name: float(proba[index])
    for index, class_name in enumerate(classes)
}
label = str(_model.predict(frame)[0])
```

The API returns the predicted label, its confidence, all class probabilities, a probability-derived posture score, and feedback. `src/App.tsx` displays the result and confirms later state changes across multiple readings to reduce flicker.

## Verification to describe

`backend/verify_original_model.py` loads the shipped artifact and compares it with the supplied `test_predictions.csv`. All **261 predicted labels** matched, and each of the three confidence columns matched to floating-point precision. The held-out test accuracy is **76.245%** and macro F1 is **56.023%**. Neutral-posture recall is **2/21 (9.52%)**, which is a limitation to state plainly. These are image test results, not measured live-webcam accuracy.

To repeat the comparison with the original handoff folder:

```powershell
.\.venv\Scripts\python.exe -m backend.verify_original_model "C:\path\to\original_training_20261014"
```

## Suggested 30-second explanation

“Uprightly detects face and shoulder landmarks in the browser and transforms eight points into 35 numeric posture features. The browser sends those features to our FastAPI endpoint. At startup, the backend loads our separately trained Python Random Forest pipeline and checks the feature schema and class order. For each request, the pipeline returns a posture class and class probabilities; the API maps these to a score and feedback, and the interface displays a stable result. We verified the integrated artifact against all 261 supplied held-out predictions. Its neutral-posture detection remains a limitation.”
