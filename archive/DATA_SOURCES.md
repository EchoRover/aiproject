# Dataset Sources and Download Instructions

## Current Datasets

### 1. ENB2012_data.xlsx (Energy Efficiency Dataset)
- **Size:** 128KB (included in repository)
- **Source:** [To be filled - likely UCI ML Repository]
- **Description:** Energy efficiency data for building characteristics
- **Features:** Building parameters and energy consumption
- **Target:** Heating/Cooling load prediction (Regression problem)

### 2. energydata_complete.csv (Appliance Energy Prediction)
- **Size:** 12MB (NOT included in repository - too large)
- **Source:** [To be filled]
- **Description:** Time-series energy consumption data
- **Download Instructions:**
  ```
  1. Download from: [URL to be added]
  2. Place in: /datasets/energydata_complete.csv
  ```
- **Alternative:** Available in project archive or contact team members
- **Features:** Temperature, humidity, weather data, appliance energy usage
- **Target:** Appliance energy consumption prediction

## How to Download Large Datasets

If you're cloning this repository, large datasets (>10MB) are not included in git.

### Step 1: Download Required Files
- Download `energydata_complete.csv` from [source URL]

### Step 2: Place Files in Correct Location
```bash
# From project root
cp ~/Downloads/energydata_complete.csv datasets/
```

### Step 3: Verify Files
```bash
# Check if files exist
ls -lh datasets/*.csv datasets/*.xlsx
```

## Dataset Summary

| Dataset | Size | Type | Algorithms Applicable |
|---------|------|------|--------------------|
| ENB2012_data.xlsx | 128KB | Regression | Linear Reg, Decision Trees, Neural Networks |
| energydata_complete.csv | 12MB | Regression/Time-series | All algorithms + clustering |

## Notes
- Small datasets (<10MB) are tracked in git for easy setup
- Large datasets (>10MB) require manual download to keep repository lightweight
- All datasets are documented here with sources and download instructions
