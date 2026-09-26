# URL phishing classification

The supplied datasets are not included in this repository. Place `train.csv`,
`test.csv`, and `external_test.csv` in `data/`.

Install the locked dependencies:

```bash
uv sync
```

The saved model in `models/all_engineered_model.joblib` is used for inference.
To rebuild it from the supplied training data:

```bash
uv run python preprocess.py data/train.csv data/train_preprocessed.csv
```

```bash
uv run python train.py data/train_preprocessed.csv
```

To predict one raw URL from the repository root:

```python
from predict import predict

print(predict("https://www.PayPal-secure.login.tk/webscr?cmd=x"))
```

`predict()` returns exactly `label` (`"phishing"` or `"clean"`) and
`probability` (a finite Python float between 0 and 1). It uses the threshold
in `src/url_phishing_classification/config.py`, currently 0.95. The function
requires a string; it raises `TypeError` for other types. Empty and malformed
strings are parsed as far as possible and scored by the same model without
raising a URL-validation error.

Generate the requested `predictions.csv` from the test file:

```bash
uv run python export_predictions.py data/test.csv predictions/predictions.csv
```

Generate a separate CSV for the external test file:

```bash
uv run python export_predictions.py data/external_test.csv predictions/predictions_external.csv
```

Each run reads one CSV and writes one CSV. Both outputs have columns
`url,probability`, preserve input order and duplicate URLs, and omit labels.
