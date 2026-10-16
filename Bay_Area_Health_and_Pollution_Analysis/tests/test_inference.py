import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from bayair.cli import predict


def test_serialized_batch_inference_preserves_rows_and_flags_support(tmp_path):
    train = pd.DataFrame({"PM2.5": [5., 6., 7., 8.], "Poverty": [10., 20., 30., 40.]})
    y = np.array([30., 40., 50., 60.])
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge()).fit(train, y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "model_metadata.json").write_text(json.dumps({
        "features": list(train.columns), "feature_min": train.min().to_dict(),
        "feature_max": train.max().to_dict()}))
    frame = pd.DataFrame({"id": ["second", "first"], "PM2.5": [9., np.nan], "Poverty": [20., 30.]})
    frame.to_csv(tmp_path / "input.csv", index=False)
    predict(tmp_path / "input.csv", tmp_path / "nested" / "output.csv", tmp_path)
    output = pd.read_csv(tmp_path / "nested" / "output.csv")
    assert output.id.tolist() == ["second", "first"]
    assert output.outside_training_range.tolist() == [True, False]
    assert output.missing_feature_count.tolist() == [0, 1]
    np.testing.assert_allclose(output.predicted_asthma_indicator, model.predict(frame[train.columns]))
