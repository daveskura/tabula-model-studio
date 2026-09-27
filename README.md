# Tabula Model Studio

A guided machine learning workbench that runs entirely in the browser. Load a table, choose a column to predict, and Tabula walks you through preparing the data, training and comparing models, understanding and tuning the best one, and using it for predictions. Saved models can be checked for drift and retrained as new data arrives.

No server, no install, no data upload: files are read and models are trained on your own machine.

## What it does

| Step | What happens |
|---|---|
| 1. Load data | CSV, TSV, Excel (.xlsx), Parquet, JSON, pasted tables, or one of three built-in samples |
| 2. Choose a goal | Pick the target column; Tabula suggests regression, classification or forecasting |
| 3. Prepare columns | Include/exclude inputs, fix types, choose a random or time-based holdout |
| 4. Train models | Baseline, linear/logistic, decision tree, random forest, gradient boosting, k-nearest neighbors |
| 5. Evaluate & tune | Metrics vs. baseline, predicted-vs-actual or confusion matrix, decision threshold, permutation importance, automatic findings (leakage, overfitting, imbalance, useless columns), random-search tuning, save |
| 6. Predict | Single case with a per-feature explanation, batch file with CSV download, or a forecast with an 80% range |
| Model library | Versioned models, drift check (PSI) and accuracy on new data, retrain as a new version, export/import as JSON |

## Run it

It is a static site: `index.html` plus `engine.js`.

```sh
npm start            # or: python3 -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly from disk also works, but training then runs on the main thread (browsers block Web Workers on `file://`).

### GitHub Pages

Settings → Pages → Build and deployment → Source: **Deploy from a branch**, Branch: **main**, folder **/ (root)**. The site appears at `https://<user>.github.io/<repo>/` a minute later.

## Tests

```sh
npm test             # Node 18+, no dependencies
```

## How it works

- `engine.js` holds all the modeling code (parsing, profiling, preprocessing, models, metrics, tuning, forecasting, drift). It runs inside a Web Worker so the page stays responsive, falls back to the main thread when workers are unavailable, and also loads in Node for testing.
- Preprocessing: median imputation with missing-value indicators, standardization, one-hot encoding of the 30 most common categories, date parts.
- Tree models use histogram binning and a single gradient/hessian tree builder shared by the decision tree, random forest (Gini-equivalent) and gradient boosting (squared, logistic and softmax losses).
- Forecasting builds lag, rolling-mean and calendar features, optionally removes a linear trend, and forecasts recursively. It is scored on the most recent periods and compared with a seasonal-naive baseline.
- Parquet files are read in the browser with [hyparquet](https://github.com/hyparam/hyparquet) (Snappy built in) plus [fzstd](https://github.com/101arrowz/fzstd) and [fflate](https://github.com/101arrowz/fflate) for zstd and gzip, loaded from jsDelivr on first use. Brotli and LZ4 compressed files are not supported yet. Up to 100,000 rows are loaded.
- Saved models and their training data are kept in the browser's IndexedDB. They do not sync between browsers; use Export to keep a copy.

## Limits

- Designed for up to about 100,000 rows (training uses up to 20,000 by default).
- Long free-text columns are not used.
- Forecasts use the target's own history and the calendar, not other columns.
- The AI assistant panel only works when the app is opened inside Claude as an artifact. Everything else works anywhere.

## License

MIT
