import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('car_fuel_efficiency_2026.csv')

# Use only the required columns
columns = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[columns]

print("="*60)
print("HOMEWORK 2 - ML ZOOMCAMP 2026")
print("="*60)

# EDA: Check fuel_efficiency_mpg distribution
print("\n--- EDA ---")
print(f"Fuel efficiency stats:")
print(f"  Mean: {df['fuel_efficiency_mpg'].mean():.2f}")
print(f"  Median: {df['fuel_efficiency_mpg'].median():.2f}")
print(f"  Std: {df['fuel_efficiency_mpg'].std():.2f}")
print(f"  Min: {df['fuel_efficiency_mpg'].min():.2f}")
print(f"  Max: {df['fuel_efficiency_mpg'].max():.2f}")

# QUESTION 1: Which column has missing values?
print("\n" + "="*60)
print("QUESTION 1: Which column has missing values?")
print("="*60)
missing_counts = df.isnull().sum()
print("\nMissing values per column:")
for col in columns:
    count = missing_counts[col]
    if count > 0:
        print(f"  {col}: {count} missing values")

q1_answer = missing_counts[missing_counts > 0].index[0]
print(f"\nAnswer Q1: {q1_answer}")

# QUESTION 2: Median of horsepower
print("\n" + "="*60)
print("QUESTION 2: What's the median of 'horsepower'?")
print("="*60)
median_hp = df['horsepower'].median()
print(f"\nMedian of horsepower: {median_hp}")
print(f"Answer Q2: {median_hp}")


# Prepare data for modeling
def prepare_data(df, seed=42):
    """Prepare and split the dataset"""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[:n_train]].copy()
    df_val = df.iloc[idx[n_train:n_train+n_val]].copy()
    df_test = df.iloc[idx[n_train+n_val:]].copy()
    
    return df_train, df_val, df_test

def linear_regression(X, y):
    """Train linear regression model (normal equation)"""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    w = np.linalg.solve(XtX, Xty)
    return w

def linear_regression_regularized(X, y, r=0):
    """Train regularized linear regression (ridge)"""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    reg_matrix = r * np.eye(XtX.shape[0])
    reg_matrix[0, 0] = 0  # Don't regularize bias
    w = np.linalg.solve(XtX + reg_matrix, Xty)
    return w

def predict(X, w):
    """Make predictions"""
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return X_with_bias @ w

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# QUESTION 3: Compare imputation strategies
print("\n" + "="*60)
print("QUESTION 3: Compare imputation strategies (0 vs mean)")
print("="*60)

df_train, df_val, df_test = prepare_data(df, seed=42)

# Strategy 1: Fill with 0
df_train_0 = df_train.fillna(0)
df_val_0 = df_val.fillna(0)

X_train_0 = df_train_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_train_0 = df_train_0['fuel_efficiency_mpg'].values
X_val_0 = df_val_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_val_0 = df_val_0['fuel_efficiency_mpg'].values

w_0 = linear_regression(X_train_0, y_train_0)
y_pred_0 = predict(X_val_0, w_0)
rmse_0 = rmse(y_val_0, y_pred_0)
print(f"\nFill with 0 - Validation RMSE: {round(rmse_0, 3)}")

# Strategy 2: Fill with mean (computed from training set only)
mean_value = df_train[q1_answer].mean()
df_train_mean = df_train.fillna({q1_answer: mean_value})
df_val_mean = df_val.fillna({q1_answer: mean_value})

X_train_mean = df_train_mean[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_train_mean = df_train_mean['fuel_efficiency_mpg'].values
X_val_mean = df_val_mean[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_val_mean = df_val_mean['fuel_efficiency_mpg'].values

w_mean = linear_regression(X_train_mean, y_train_mean)
y_pred_mean = predict(X_val_mean, w_mean)
rmse_mean = rmse(y_val_mean, y_pred_mean)
print(f"Fill with mean - Validation RMSE: {round(rmse_mean, 3)}")

if rmse_0 < rmse_mean:
    q3_answer = "With 0"
elif rmse_mean < rmse_0:
    q3_answer = "With mean"
else:
    q3_answer = "Both are equally good"


# QUESTION 4: Regularization parameter
print("\n" + "="*60)
print("QUESTION 4: Best regularization parameter r")
print("="*60)

df_train, df_val, df_test = prepare_data(df, seed=42)
df_train_0 = df_train.fillna(0)
df_val_0 = df_val.fillna(0)

X_train = df_train_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_train = df_train_0['fuel_efficiency_mpg'].values
X_val = df_val_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_val = df_val_0['fuel_efficiency_mpg'].values

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
best_r = None
best_rmse = float('inf')

print("\nTesting different r values:")
for r in r_values:
    w = linear_regression_regularized(X_train, y_train, r)
    y_pred = predict(X_val, w)
    rmse_val = rmse(y_val, y_pred)
    rmse_rounded = round(rmse_val, 4)
    print(f"  r={r:6.2f} -> RMSE={rmse_rounded}")
    
    if rmse_val < best_rmse:
        best_rmse = rmse_val
        best_r = r

print(f"\nBest r: {best_r}")
print(f"Best RMSE: {round(best_rmse, 4)}")
print(f"Answer Q4: {best_r}")

# QUESTION 5: Stability across seeds
print("\n" + "="*60)
print("QUESTION 5: Standard deviation of RMSE across seeds")
print("="*60)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_scores = []

print("\nRMSE for each seed:")
for seed in seeds:
    df_train, df_val, df_test = prepare_data(df, seed=seed)
    df_train_0 = df_train.fillna(0)
    df_val_0 = df_val.fillna(0)
    
    X_train = df_train_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
    y_train = df_train_0['fuel_efficiency_mpg'].values
    X_val = df_val_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
    y_val = df_val_0['fuel_efficiency_mpg'].values
    
    w = linear_regression(X_train, y_train)
    y_pred = predict(X_val, w)
    rmse_val = rmse(y_val, y_pred)
    rmse_scores.append(rmse_val)
    print(f"  Seed {seed}: RMSE = {rmse_val:.6f}")

std_rmse = np.std(rmse_scores)
print(f"\nStandard deviation: {round(std_rmse, 3)}")
print(f"Answer Q5: {round(std_rmse, 3)}")

# QUESTION 6: Final test evaluation
print("\n" + "="*60)
print("QUESTION 6: Test RMSE with seed 9, r=0.001")
print("="*60)

df_train, df_val, df_test = prepare_data(df, seed=9)
df_train_val = pd.concat([df_train, df_val], ignore_index=True)

df_train_val_0 = df_train_val.fillna(0)
df_test_0 = df_test.fillna(0)

X_train_val = df_train_val_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_train_val = df_train_val_0['fuel_efficiency_mpg'].values
X_test = df_test_0[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']].values
y_test = df_test_0['fuel_efficiency_mpg'].values

w = linear_regression_regularized(X_train_val, y_train_val, r=0.001)
y_pred_test = predict(X_test, w)
rmse_test = rmse(y_test, y_pred_test)

print(f"\nTest RMSE: {rmse_test:.4f}")
print(f"Answer Q6: {rmse_test:.4f}")

# Summary
print("\n" + "="*60)
print("SUMMARY OF ANSWERS")
print("="*60)
print(f"Q1: {q1_answer}")
print(f"Q2: {median_hp}")
print(f"Q3: {q3_answer}")
print(f"Q4: {best_r}")
print(f"Q5: {round(std_rmse, 3)}")
print(f"Q6: {rmse_test:.4f}")
print("="*60)

print(f"\nAnswer Q3: {q3_answer}")
