# Parkinson's Disease UPDRS Prediction from Voice Analysis

Machine learning system for predicting Parkinson's disease severity (UPDRS scores) from voice biomarkers, enabling remote patient monitoring.

## 🔗 Project Links

- **GitHub Repository:** https://github.com/EchoRover/aiproject
- **Dataset Source:** [UCI Parkinson's Telemonitoring Database](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)

## 📊 Project Overview

This project develops a machine learning pipeline to predict UPDRS (Unified Parkinson's Disease Rating Scale) scores from voice recordings. Random Forest achieved R²=0.916 (RMSE=3.06 points), exceeding clinical targets by 22%. This enables smartphone-based remote monitoring without clinic visits.

## 📁 Project Structure

```
aiproject/
├── datasets/
│   └── parkinsons_updrs.data          # UCI dataset (5,875 recordings)
├── notebooks/parkinsons/
│   ├── 01_parkinsons_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_parkinsons_preprocessing.ipynb  # Feature engineering & scaling
│   └── 03_parkinsons_regression.ipynb # Model training & evaluation
├── submission/
│   ├── report.tex                     # Technical report (LaTeX)
│   ├── code_documentation.tex         # Code documentation
│   └── figures/                       # All visualizations
├── archive/                           # Previous project iterations
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/EchoRover/aiproject.git
cd aiproject
```

### 2. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Analysis
Execute notebooks in order:
```bash
jupyter notebook notebooks/parkinsons/01_parkinsons_eda.ipynb
jupyter notebook notebooks/parkinsons/02_parkinsons_preprocessing.ipynb
jupyter notebook notebooks/parkinsons/03_parkinsons_regression.ipynb
```

## 📈 Key Results

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.155 | 9.68 | 8.06 |
| Polynomial Regression | 0.281 | 8.92 | 7.43 |
| Decision Tree | 0.875 | 3.72 | 2.55 |
| **Random Forest** | **0.916** | **3.06** | **2.05** |
| Neural Network | 0.731 | 5.46 | 3.85 |

## 🔬 Methodology

1. **EDA:** Analyzed 5,875 voice recordings from 42 patients, identified weak linear correlations (|r|<0.3), detected multicollinearity
2. **Preprocessing:** Created 7 engineered features (kept 3 best), StandardScaler normalization, patient-based 80/20 split
3. **Modeling:** Tested 5 algorithms with GridSearchCV hyperparameter tuning
4. **Result:** Random Forest best for tabular data with 22 features

## 🛠️ Dependencies

- Python 3.8+
- pandas, numpy (data manipulation)
- scikit-learn (ML models, preprocessing)
- matplotlib, seaborn (visualization)
- torch (neural network)

Install all: `pip install -r requirements.txt`

## 📄 Documentation

- **Technical Report:** `submission/report.tex` (compile with LaTeX)
- **Code Documentation:** `submission/code_documentation.tex`
- **Notebooks:** Fully commented with step-by-step explanations

## 👨‍💻 Authors

**Evan Johan Tobias & Mohsin Akram Khan**  
AENL338 - AI Project  
December 2025

## 📝 License

This project is for academic purposes.
- **Large datasets** (>10MB) like `energydata_complete.csv` are NOT in Git
- See `datasets/DATA_SOURCES.md` for download instructions for large files
- This keeps the repository lightweight while documenting all data sources

## Algorithms to Showcase

- Regression (Linear, Polynomial)
- Logistic Regression
- Decision Trees
- K-means Clustering
- Neural Networks (PyTorch)
- Backpropagation

## Evaluation Metrics

- MSE (Mean Squared Error)
- R² Score
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Silhouette Score (for clustering)
- And other metrics learned in course
