# Parkinson's Disease Analysis - Complete Notebook Generator
# This script contains all the code for the comprehensive analysis

notebook_content = """
# LOAD DATA
df = pd.read_csv('../datasets/parkinsons_updrs.data')
print("="*80)
print("PARKINSON'S DATASET LOADED")
print("="*80)
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nFirst 5 rows:")
df.head()

# EDA
print("\\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(df.describe())
print(f"\\nMissing values: {df.isnull().sum().sum()}")

# Feature selection - remove identifiers
feature_cols = [col for col in df.columns if col not in ['subject#', 'motor_UPDRS', 'total_UPDRS']]
X = df[feature_cols].values
y_motor = df['motor_UPDRS'].values
y_total = df['total_UPDRS'].values

print(f"\\nFeatures: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"Targets: motor_UPDRS, total_UPDRS")

# Train-test split
X_train, X_test, y_motor_train, y_motor_test = train_test_split(X, y_motor, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTrain: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# LINEAR REGRESSION
lr_motor = LinearRegression()
lr_motor.fit(X_train_scaled, y_motor_train)
lr_motor_pred = lr_motor.predict(X_test_scaled)
lr_motor_r2 = r2_score(y_motor_test, lr_motor_pred)
lr_motor_rmse = np.sqrt(mean_squared_error(y_motor_test, lr_motor_pred))

lr_total = LinearRegression()
lr_total.fit(X_train_scaled, y_total_train)
lr_total_pred = lr_total.predict(X_test_scaled)
lr_total_r2 = r2_score(y_total_test, lr_total_pred)
lr_total_rmse = np.sqrt(mean_squared_error(y_total_test, lr_total_pred))

print("="*80)
print("LINEAR REGRESSION RESULTS")
print("="*80)
print(f"Motor UPDRS: R²={lr_motor_r2:.4f}, RMSE={lr_motor_rmse:.4f}")
print(f"Total UPDRS: R²={lr_total_r2:.4f}, RMSE={lr_total_rmse:.4f}")

# POLYNOMIAL REGRESSION
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_motor = LinearRegression()
poly_motor.fit(X_train_poly, y_motor_train)
poly_motor_pred = poly_motor.predict(X_test_poly)
poly_motor_r2 = r2_score(y_motor_test, poly_motor_pred)
poly_motor_rmse = np.sqrt(mean_squared_error(y_motor_test, poly_motor_pred))

poly_total = LinearRegression()
poly_total.fit(X_train_poly, y_total_train)
poly_total_pred = poly_total.predict(X_test_poly)
poly_total_r2 = r2_score(y_total_test, poly_total_pred)
poly_total_rmse = np.sqrt(mean_squared_error(y_total_test, poly_total_pred))

print("="*80)
print("POLYNOMIAL REGRESSION RESULTS")
print("="*80)
print(f"Motor UPDRS: R²={poly_motor_r2:.4f}, RMSE={poly_motor_rmse:.4f}")
print(f"Total UPDRS: R²={poly_total_r2:.4f}, RMSE={poly_total_rmse:.4f}")

# DECISION TREE
dt_motor = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_motor.fit(X_train_scaled, y_motor_train)
dt_motor_pred = dt_motor.predict(X_test_scaled)
dt_motor_r2 = r2_score(y_motor_test, dt_motor_pred)
dt_motor_rmse = np.sqrt(mean_squared_error(y_motor_test, dt_motor_pred))

dt_total = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_total.fit(X_train_scaled, y_total_train)
dt_total_pred = dt_total.predict(X_test_scaled)
dt_total_r2 = r2_score(y_total_test, dt_total_pred)
dt_total_rmse = np.sqrt(mean_squared_error(y_total_test, dt_total_pred))

print("="*80)
print("DECISION TREE RESULTS")
print("="*80)
print(f"Motor UPDRS: R²={dt_motor_r2:.4f}, RMSE={dt_motor_rmse:.4f}")
print(f"Total UPDRS: R²={dt_total_r2:.4f}, RMSE={dt_total_rmse:.4f}")

# RANDOM FOREST
rf_motor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_motor.fit(X_train_scaled, y_motor_train)
rf_motor_pred = rf_motor.predict(X_test_scaled)
rf_motor_r2 = r2_score(y_motor_test, rf_motor_pred)
rf_motor_rmse = np.sqrt(mean_squared_error(y_motor_test, rf_motor_pred))

rf_total = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_total.fit(X_train_scaled, y_total_train)
rf_total_pred = rf_total.predict(X_test_scaled)
rf_total_r2 = r2_score(y_total_test, rf_total_pred)
rf_total_rmse = np.sqrt(mean_squared_error(y_total_test, rf_total_pred))

print("="*80)
print("RANDOM FOREST RESULTS")
print("="*80)
print(f"Motor UPDRS: R²={rf_motor_r2:.4f}, RMSE={rf_motor_rmse:.4f}")
print(f"Total UPDRS: R²={rf_total_r2:.4f}, RMSE={rf_total_rmse:.4f}")

# All results summary
results_motor = pd.DataFrame({
    'Model': ['Linear Reg', 'Polynomial Reg', 'Decision Tree', 'Random Forest'],
    'R²': [lr_motor_r2, poly_motor_r2, dt_motor_r2, rf_motor_r2],
    'RMSE': [lr_motor_rmse, poly_motor_rmse, dt_motor_rmse, rf_motor_rmse]
})

results_total = pd.DataFrame({
    'Model': ['Linear Reg', 'Polynomial Reg', 'Decision Tree', 'Random Forest'],
    'R²': [lr_total_r2, poly_total_r2, dt_total_r2, rf_total_r2],
    'RMSE': [lr_total_rmse, poly_total_rmse, dt_total_rmse, rf_total_rmse]
})

print("\\n" + "="*80)
print("MOTOR UPDRS - ALL MODELS")
print("="*80)
print(results_motor.to_string(index=False))

print("\\n" + "="*80)
print("TOTAL UPDRS - ALL MODELS")
print("="*80)
print(results_total.to_string(index=False))

print(f"\\n🏆 Best Motor UPDRS: {results_motor.loc[results_motor['R²'].idxmax(), 'Model']} (R²={results_motor['R²'].max():.4f})")
print(f"🏆 Best Total UPDRS: {results_total.loc[results_total['R²'].idxmax(), 'Model']} (R²={results_total['R²'].max():.4f})")
"""

print("Notebook content prepared. Copy cells as needed.")
