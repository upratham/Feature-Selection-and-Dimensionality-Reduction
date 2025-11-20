# Feature Selection and Dimensionality Reduction for Air Quality Classification

This repository contains an end-to-end machine learning pipeline that compares several feature selection and dimensionality reduction techniques for a multi-class air quality classification problem.

The project:
- Uses an air-pollution dataset with 9 input features and 4 target classes representing air quality levels.
- Compares **Univariate feature selection (Chi-square)**, **Random Forest feature importance**, **PCA**, and **LDA**.
- Trains and evaluates **Support Vector Machine (SVM)** and **Gaussian Naive Bayes (GNB)** classifiers over multiple random train/test splits.
- Logs results, saves intermediate datasets, and generates plots that summarize feature importance, variance explained, and accuracy vs. number of features.

---

## Project Structure

Assuming a layout like:

```text
.
├── data/
│   └── pollution_dataset.csv
├── src/
│   ├── data_import.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_eval.py
├── logs/
├── result/
├── README.md
└── requirements.txt
```

**Main modules (inside `src/`):**

- `data_import.py`  
  Loads the pollution dataset from CSV and configures logging to the `logs/` directory.

- `data_preprocessing.py`  
  - Replaces negative numeric values with 0.  
  - Encodes the `Air Quality` label into integer classes:  
    `Good -> 0`, `Moderate -> 1`, `Poor -> 2`, `Hazardous -> 3`.

- `feature_engineering.py`  
  Implements and visualizes four feature-engineering strategies:
  1. **Univariate Selection (Chi-square)** via `SelectKBest`
  2. **Random Forest feature importance** (top-K features)
  3. **Principal Component Analysis (PCA)** with explained variance plots and class-colored scatter plots
  4. **Linear Discriminant Analysis (LDA)** with explained variance plots and class-colored scatter plots

  Saves:
  - `result/feature_engineering_result.txt` – textual logs/insights
  - `result/feature_engineering_plots/` – PNG plots for each method and K.

- `model_training.py`  
  - Applies each dimensionality reduction method for K ∈ {1, 2, 3, 9}  
    (K=9 uses all original features without reduction).
  - Trains:
    - Linear **SVM**
    - **GaussianNB**
  - Repeats training for a configurable number of random splits (`iter`).
  - Saves all results to `result/model_training_results.csv`.

- `model_eval.py`  
  - Loads `model_training_results.csv`.
  - Computes descriptive statistics and average accuracy per K and per method.
  - Generates line plots of **accuracy vs K for each model** under each dimensionality-reduction method.
  - Logs a summary report to `result/result_eval_report.txt`.
  - Saves evaluation plots to `result/Result_eval_plots/`.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate     # Windows (PowerShell or CMD)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data

- Expected file: `data/pollution_dataset.csv`
- The CSV must include:
  - Input features such as pollutant concentrations and contextual attributes.
  - A target column named **`Air Quality`** with string labels `Good`, `Moderate`, `Poor`, `Hazardous`.

If your data file is located elsewhere or has a different name, update the default path in `load_data()` inside `data_import.py` **or** pass a custom path when calling that function.

---

## How to Run

### 1. Navigate to the `src` folder

From the project root:

```bash
cd src
```

### 2. Run the full workflow

To execute the main evaluation workflow for the whole project, run:

```bash
python model_eval.py
```

> This script loads the training results (if available), performs evaluation, and generates plots and summary reports.

If you want training to also run inside the same workflow, you can modify `model_eval.py` to call the training routine before evaluation (e.g., through a helper function).

### 3. Run model training (optional but recommended for fresh results)

If you want to explicitly generate or regenerate training results:

```bash
python model_training.py
```

This will:
- Load and preprocess the data.
- Run all combinations of dimensionality reduction methods, models, and K values.
- Save a fresh `result/model_training_results.csv`.

Then re-run:

```bash
python model_eval.py
```

to create updated plots and evaluation summaries.

### 4. Run feature-engineering plots only (optional)

To recreate the feature-engineering visualizations (feature importance, PCA, LDA):

```bash
python feature_engineering.py
```

This will populate `result/feature_engineering_plots/` and append insights to `result/feature_engineering_result.txt`.

---

## Outputs

- **Logs**
  - `logs/data_import.log`
  - `logs/data_preprocessing.log`
  - `logs/feature_engineering.log`
  - `logs/model_training.log`
  - `logs/model_eval.log`

- **Intermediate data**
  - `data/data <Method> k=<K>.csv` files created during training for each method and K.

- **Results**
  - `result/model_training_results.csv` – all individual runs and accuracies.
  - `result/result_eval_report.txt` – textual evaluation summary (means, variance, etc.).

- **Plots**
  - `result/feature_engineering_plots/` – feature importance & PCA/LDA plots.
  - `result/Result_eval_plots/` – accuracy vs K line plots for each method and model.

---

## Notes

- The project is written for Python 3.8+.
- Random train/test splits are repeated multiple times (e.g., 50 iterations) to obtain stable estimates of model performance.
- You can adjust:
  - The number of iterations (`iter`) in `model_eval.py` / `model_training.py`.
  - The set of K values (number of features/components) in `model_training.py`.
- Feel free to adapt paths (for data, logs, and results) to suit your environment.
