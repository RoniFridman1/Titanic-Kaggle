# Titanic Survival Classification

An end-to-end classification project using Kaggle's Titanic dataset. It includes exploratory data analysis, feature
engineering, a tuned PyTorch classifier, and a Streamlit interface for inference and evaluation.

Only Kaggle's `train.csv` is used. The official Kaggle `test.csv` and `gender_submission.csv` files are not used. A
reproducible stratified train/validation split is created locally by `train.py`.

## Repository contents

```text
.
├── Titanic_EDA.ipynb    # EDA and Random Forest baseline
├── train.py             # PyTorch tuning, training and artifact generation
├── ds_app.py            # Streamlit inference and evaluation interface
├── requirements.txt     # Python dependencies
├── README.md
└── .gitignore
```

The repository intentionally excludes:

- Kaggle credentials and `.env`
- Downloaded data under `data/`
- Generated model files under `artifacts/`

Provide your own Kaggle API token and run training locally before starting Streamlit or EDA notebook.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Kaggle authentication

Generate an API token from your Kaggle account settings and create a local `.env` file in the repository root:

```env
KAGGLE_API_TOKEN=your_token_here
```

The `.env` file is ignored by Git and must never be committed. You may also need to accept the Titanic competition rules
through the Kaggle website before downloading the data.

## Exploratory data analysis

Start Jupyter:

```bash
jupyter notebook Titanic_EDA.ipynb
```

The notebook covers:

- Dataset structure, target balance and missing values
- Numeric and categorical distributions
- Relationships between passenger attributes and survival
- Age imputation using title and passenger class
- Logarithmic fare transformation
- Title, family, ticket-group and probable-spouse features
- Age groups and interaction features
- Correlation analysis
- Random Forest baseline evaluation
- Permutation feature importance and feature reduction

The Random Forest is used as a classical baseline. The final submitted classifier is implemented in PyTorch as required
by the assignment.

## Training

Run:

```bash
python train.py
```

Optional arguments:

```bash
python train.py \
  --env-file .env \
  --data-dir data \
  --output-dir artifacts \
  --validation-size 0.20 \
  --cv-folds 3 \
  --seed 42
```

On Windows PowerShell, the same command can be written on one line when inside the folder containing the files:

```powershell
python train.py --env-file .env --data-dir data --output-dir artifacts --validation-size 0.20 --cv-folds 3 --seed 42
```

The script:

1. Downloads only Titanic `train.csv` directly from Kaggle, or reuses an existing local copy.
2. Creates a stratified held-out validation split.
3. Fits age-imputation and preprocessing parameters using training rows only.
4. Recreates the features selected during EDA.
5. Tunes a linear PyTorch classifier and several MLP architectures using stratified cross-validation.
6. Uses early stopping, class-weighted binary cross-entropy and AdamW optimization.
7. Selects the configuration by mean cross-validation F1, using ROC-AUC as a tiebreaker.
8. Retrains the selected model and evaluates it on the held-out validation split.

Training creates the following local files:

```text
data/
└── train.csv

artifacts/
├── titanic_model.pt
├── preprocessor.joblib
├── feature_statistics.json
├── metrics.json
├── validation_data.csv
└── validation_predictions.csv
```

These files are generated locally and are intentionally excluded from the repository.

## Streamlit application

Run training first so that the required artifacts exist. Then start the application:

```bash
streamlit run ds_app.py
```

Streamlit normally opens at:

```text
http://localhost:8501
```

The application supports:

- Loading the generated model and preprocessing files
- Uploading a raw Titanic-format CSV or providing a server-side CSV path
- Adjusting the classification threshold
- Viewing survival probabilities and binary predictions
- Accuracy, precision, recall, F1 and ROC-AUC when labels are available
- Confusion-matrix and ROC-curve visualizations
- Downloading prediction results

For immediate evaluation after training, use:

```text
artifacts/validation_data.csv
```

The input CSV must contain these raw fields:

```text
Name, Sex, Pclass, Age, SibSp, Parch, Ticket, Fare
```

`PassengerId` is optional. `Survived` is optional for inference but required for evaluation metrics.

Because ticket-group and relationship features are calculated from the supplied CSV, inference should preferably be run
on a group of passengers rather than one passenger at a time.

## Model design

The final model is a binary PyTorch classifier. The search compares a linear classifier with shallow and deeper MLP
architectures using different hidden layers, dropout, learning rates, weight decay and batch sizes.

The preprocessing and feature pipeline includes:

- Age imputation by title and passenger class
- Fare log transformation
- Passenger-title features
- Ticket-group size and possible-relative indicators
- High-probability spouse matching
- Age categories and selected interaction products
- One-hot encoding of categorical inputs
- Standardization of numeric inputs

The final model uses `BCEWithLogitsLoss` with class weighting, AdamW, early stopping during tuning, and a fixed random
seed.

## Evaluation

The training report includes:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

The held-out validation split is not used for hyperparameter selection. Cross-validation occurs only within the training
portion, and the held-out split is evaluated after model selection.

## Reproducibility and leakage prevention

- All random operations use the configurable `--seed` value.
- The outer split is stratified by `Survived`.
- Hyperparameter tuning uses only the training portion.
- Age-imputation, scaling and encoding parameters are fitted without held-out validation rows.
- Kaggle's official test and submission files are never used.
- Generated credentials, data and model artifacts are excluded from version control.

## Notes

- Do not run `ds_app.py` before `train.py`; the application requires the generated artifacts.
- Only load model and preprocessing files that you trust.
- Results may vary slightly between CPU and GPU environments despite fixed seeds.
