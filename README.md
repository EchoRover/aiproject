# AI Final Year Project

Machine Learning and Deep Learning project showcasing various algorithms and techniques.

## Project Structure

```
aiproject/
├── datasets/          # All datasets (raw and processed)
├── notebooks/         # Jupyter notebooks for analysis and training
├── models/           # Saved trained models
├── references/       # Course notes and reference materials
├── reports/          # Final report and presentation
├── project_notes/    # Project tracking and decisions
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd aiproject
   ```

2. **Download large datasets** (not included in Git):
   - See `datasets/DATA_SOURCES.md` for download instructions
   - Download `energydata_complete.csv` and place in `datasets/` folder

3. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR on Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Important Note on Datasets

- **Small datasets** (<10MB) like `ENB2012_data.xlsx` are tracked in Git
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
