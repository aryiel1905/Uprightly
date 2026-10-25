"""Compare the integrated model with the supplied held-out prediction record."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from backend import api


def verify(handoff: Path) -> None:
    testing = pd.read_csv(handoff / "csv" / "landmarks" / "testing.csv")
    reference = pd.read_csv(
        handoff / "python_joblib" / "test_predictions.csv"
    )
    if len(testing) != len(reference) or not testing["image_name"].equals(
        reference["image_name"]
    ):
        raise AssertionError("The testing rows do not align with the reference predictions.")

    api.load_model()
    features = testing[api.FEATURES]
    predicted = api._model.predict(features)
    probabilities = api._model.predict_proba(features)

    if not np.array_equal(predicted, reference["prediction"].to_numpy()):
        raise AssertionError("Model labels differ from the supplied prediction record.")
    for index, label in enumerate(api.CLASSES):
        expected = reference[f"confidence_{label}"].to_numpy()
        np.testing.assert_allclose(probabilities[:, index], expected, atol=1e-12)

    accuracy = accuracy_score(testing["label"], predicted)
    macro_f1 = f1_score(testing["label"], predicted, average="macro")
    print(f"Verified {len(testing)} reference predictions and class confidences.")
    print(f"Accuracy: {accuracy:.6f}; macro F1: {macro_f1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "handoff",
        type=Path,
        help="Path to the original_training_20261014 handoff folder",
    )
    verify(parser.parse_args().handoff)
