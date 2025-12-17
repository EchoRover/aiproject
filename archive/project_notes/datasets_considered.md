# Datasets Considered

## Criteria for Selection:
- Must allow showcasing multiple algorithm types
- Good for demonstrating learned concepts
- Appropriate size and complexity
- Clear business/research question

## Options:
[To be filled as we explore datasets]

## Final Selections:

### ✅ Dataset 1: ENB2012_data.xlsx - Energy Efficiency
- **Source:** UCI Machine Learning Repository (likely)
- **Size:** 128KB
- **Problem Type:** Regression
- **Target Variables:** Heating Load, Cooling Load
- **Features:** Building parameters (surface area, wall area, roof area, height, orientation, glazing area, etc.)
- **Samples:** ~768 buildings
- **Git Status:** ✓ Tracked in repository (small size)
- **Algorithms to Apply:**
  * Linear Regression
  * Polynomial Regression
  * Decision Trees
  * Random Forest
  * Neural Networks (PyTorch)
  * K-means Clustering (building types)

### ✅ Dataset 2: energydata_complete.csv - Appliance Energy Prediction
- **Source:** UCI ML Repository / Kaggle (to be confirmed)
- **Size:** 12MB
- **Problem Type:** Regression (primary), Classification (derived)
- **Target Variable:** Appliances energy consumption
- **Features:** Temperature, humidity, weather conditions, time-series data (~28 features)
- **Samples:** ~19,735 observations
- **Git Status:** ✗ NOT tracked (too large) - Download separately
- **Download Instructions:** See datasets/DATA_SOURCES.md
- **Algorithms to Apply:**
  * Linear Regression
  * Logistic Regression (after creating binary target)
  * Decision Trees
  * Neural Networks (PyTorch) - time-series
  * K-means Clustering (consumption patterns)

## Why These Datasets?
1. **Both are energy-related** - cohesive project theme
2. **Different scales** - small vs large dataset experience
3. **Different problem types** - pure regression vs regression + classification
4. **Rich feature sets** - good for showcasing feature engineering
5. **Clean data** - no missing values, focus on modeling
6. **Cover all algorithms** - suitable for every technique learned in course
7. **Real-world relevance** - practical applications in energy efficiency
