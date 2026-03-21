# AI Model for Cardiovascular Disease Prediction

A machine learning pipeline that predicts cardiovascular disease (CVD) risk using logistic regression across three publicly available clinical datasets. Built as part of the MS Health Informatics program at Hofstra University.

---

## Project Overview

Cardiovascular disease remains one of the leading causes of morbidity and mortality worldwide. This project explores whether machine learning can identify individuals at high risk **before clinical symptoms emerge** — enabling earlier, more targeted preventative care.

The pipeline implements two logistic regression models (baseline and refined), compares their performance, and discusses the clinical tradeoffs between precision and recall in a healthcare prediction context.

---

## Datasets

| Dataset | Source | Key Features |
|--------|--------|--------------|
| **Cardio Train** *(primary)* | [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | Age, gender, systolic/diastolic BP, cholesterol, BMI, smoking, alcohol, physical activity |
| **Framingham Heart Study** | [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) | Age, cholesterol, BP, lifestyle habits |
| **UCI Heart Disease** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | Chest pain type, ECG results, max heart rate |

> **Note:** Data files are excluded from this repository via `.gitignore`. Download them from the links above and place them in the project root before running.

---

## Project Structure

```
cvd-risk-prediction/
├── cvd_prediction.py       # Main pipeline script
├── cvd_prediction.ipynb    # Jupyter notebook version (step-by-step)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Methods

### Preprocessing
- Missing numerical values imputed with **column median** (robust to outliers)
- `id` column dropped (no predictive value)
- Age converted from days → years for interpretability
- Features standardized with **StandardScaler** (zero mean, unit variance)

### Exploratory Data Analysis
- Statistical summary via `df.describe()`
- Target class distribution plot (seaborn countplot)
- Feature correlation heatmap (to detect multicollinearity)

### Models

| Model | Solver | Max Iterations |
|-------|--------|----------------|
| Baseline | `lbfgs` | 100 |
| Refined | `liblinear` | 200 |

Train/test split: **80% / 20%**, `random_state=42`

---

## Results

| Metric | Baseline | Refined |
|--------|----------|---------|
| Accuracy | ~71% | ~72% |
| Class 0 (No CVD) — Precision | 0.69 | 0.71 |
| Class 0 (No CVD) — Recall | 0.74 | 0.77 |
| Class 1 (CVD) — Precision | 0.72 | 0.75 |
| Class 1 (CVD) — Recall | 0.67 | 0.68 |
| False Positives | 1,785 | 1,625 |
| False Negatives | 2,319 | 2,244 |

Switching from `lbfgs` to `liblinear` and increasing `max_iter` resolved a convergence warning and reduced both false positives and false negatives. In a clinical context, **false negatives are particularly critical** — missing a CVD-positive case can delay diagnosis and worsen patient outcomes.

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/tarajeeclarke/cvd-risk-prediction.git
cd cvd-risk-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
Download `cardio_train.csv` from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in the project root.

### 4. Run the pipeline
```bash
# Python script
python cvd_prediction.py

# Or open the notebook
jupyter notebook cvd_prediction.ipynb
```

---

## Future Improvements

- **SMOTE** to address class imbalance and improve recall for CVD-positive cases
- **Random Forest** and **Gradient Boosting** models to capture non-linear relationships
- Cross-validation for more robust performance estimates
- Feature importance analysis for clinical interpretability
- Multi-dataset unified pipeline with harmonized feature schemas

---

## Technologies

- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

---

## Author

**Tarajee Clarke**  
MS Health Informatics — Hofstra University  
[LinkedIn](https://www.linkedin.com/in/tarajeeclarke) · [GitHub](https://github.com/tarajeeclarke)
