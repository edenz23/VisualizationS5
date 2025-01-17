import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Path to the dataset (update the path as needed)
data_path = 'final_table_with_all_sensors_and_irrigation.csv'
data = pd.read_csv(data_path)

# Feature engineering for the dataset
data['vpd_rolling_mean'] = data['vpd (kPa)'].rolling(window=7, min_periods=1).mean()
data['tdr_salt_trend'] = data['tdr_salt_80'].diff(periods=7)
data['irrigation_vpd_interaction'] = data['irrigation'] * data['vpd (kPa)']
data['irrigation_vpd_ratio'] = data['irrigation'] / (data['vpd (kPa)'] + 1e-6)
data['salt_to_vpd_interaction'] = data['tdr_salt_80'] * data['vpd (kPa)']
data['day_of_year'] = pd.to_datetime(data['date']).dt.dayofyear

# Drop NaN values
data = data.dropna()

# Define feature columns and target column
feature_columns_growth = [
    'irrigation', 'vpd (kPa)', 'tdr_salt_80', 'vpd_rolling_mean',
    'tdr_salt_trend', 'irrigation_vpd_interaction',
    'irrigation_vpd_ratio', 'salt_to_vpd_interaction', 'day_of_year'
]
target_column_growth = 'frond_growth_rate'

# -------------------------------
# Data Preprocessing Functions
# -------------------------------

def preprocess_data(data, relevant_columns):
    """
    Filters out rows with missing values for relevant columns.
    """
    return data.dropna(subset=relevant_columns)

def split_data(data, feature_columns, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X = data[feature_columns]
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# -------------------------------
# Model Training Functions
# -------------------------------

def train_growth_model(data, feature_columns, target_column):
    """
    Train a predictive model for growth rate and return it.
    """
    X_train, X_test, y_train, y_test = split_data(data, feature_columns, target_column)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\n--- Growth Rate Model Performance ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    return model

def train_vpd_model(data):
    """
    Train a predictive model for VPD based on seasonal data.
    """
    feature_columns = ['day_of_year']
    target_column = 'vpd (kPa)'

    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data[target_column], test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    print("\n--- VPD Model Performance ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    return model

def train_tdr_model(data):
    """
    Train a predictive model for TDR Salt based on relevant features.
    """
    feature_columns = [
        'day_of_year', 'irrigation', 'vpd (kPa)',
        'vpd_rolling_mean', 'tdr_salt_trend', 'irrigation_vpd_interaction'
    ]
    target_column = 'tdr_salt_80'

    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data[target_column], test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.8,
        learning_rate=0.03,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    print("\n--- TDR Salt Model Performance ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    return model

# -------------------------------
# Prediction Function
# -------------------------------

def predict_future_with_variable_irrigation(initial_input, irrigation_values, model, model_vpd, model_tdr, future_days=90):
    """
    Predict future growth dynamically with variable irrigation values, VPD, and TDR Salt predictions.
    """
    # Ensure exactly 9 irrigation values by extending the last value
    if len(irrigation_values) < 9:
        irrigation_values += [irrigation_values[-1]] * (9 - len(irrigation_values))
    elif len(irrigation_values) > 9:
        irrigation_values = irrigation_values[:9]

    # Expand irrigation values to cover 90 days (each value repeated 10 times)
    irrigation_schedule = []
    for value in irrigation_values:
        irrigation_schedule.extend([value] * 10)

    predictions = []
    current_input = initial_input.iloc[0].copy()

    for day in range(future_days):
        # Update irrigation dynamically
        current_input["irrigation"] = irrigation_schedule[day]

        # Update day_of_year and predict VPD
        current_input["day_of_year"] = (current_input["date"].timetuple().tm_yday) % 365
        current_input["vpd (kPa)"] = model_vpd.predict(pd.DataFrame({"day_of_year": [current_input["day_of_year"]]}))[0]

        # Calculate additional features for TDR Salt prediction
        current_input["vpd_rolling_mean"] = current_input["vpd (kPa)"]  # Assuming no rolling data for simplicity
        current_input["tdr_salt_trend"] = 0  # No trend for simplicity
        current_input["irrigation_vpd_interaction"] = current_input["irrigation"] * current_input["vpd (kPa)"]

        # Predict TDR Salt
        current_input["tdr_salt_80"] = model_tdr.predict(pd.DataFrame({
            "day_of_year": [current_input["day_of_year"]],
            "irrigation": [current_input["irrigation"]],
            "vpd (kPa)": [current_input["vpd (kPa)"]],
            "vpd_rolling_mean": [current_input["vpd_rolling_mean"]],
            "tdr_salt_trend": [current_input["tdr_salt_trend"]],
            "irrigation_vpd_interaction": [current_input["irrigation_vpd_interaction"]]
        }))[0]

        # Calculate additional features dynamically for growth prediction
        current_input["salt_to_vpd_interaction"] = current_input["tdr_salt_80"] * current_input["vpd (kPa)"]

        # Prepare features for growth prediction
        features = pd.DataFrame([{
            "irrigation": current_input["irrigation"],
            "vpd (kPa)": current_input["vpd (kPa)"],
            "tdr_salt_80": current_input["tdr_salt_80"],
            "vpd_rolling_mean": current_input["vpd_rolling_mean"],
            "tdr_salt_trend": current_input["tdr_salt_trend"],
            "irrigation_vpd_interaction": current_input["irrigation_vpd_interaction"],
            "irrigation_vpd_ratio": current_input["irrigation"] / (current_input["vpd (kPa)"] + 1e-6),
            "salt_to_vpd_interaction": current_input["salt_to_vpd_interaction"],
            "day_of_year": current_input["day_of_year"]
        }])

        predicted_growth = model.predict(features)[0]

        # Add prediction to results
        predictions.append({
            "date": current_input["date"],
            "irrigation": current_input["irrigation"],
            "vpd (kPa)": current_input["vpd (kPa)"],
            "tdr_salt_80": current_input["tdr_salt_80"],
            "Predicted Growth Rate": predicted_growth
        })

        # Update input data for the next day
        current_input["date"] += timedelta(days=1)

    return pd.DataFrame(predictions)


# -------------------------------
# GUI Functions
# -------------------------------

def run_prediction():
    """
    Collect user input, run the prediction, and display results in the table.
    """
    global predictions
    try:
        irrigation_values = list(map(float, irrigation_series_var.get().split(',')))
        vpd = float(vpd_var.get())
        tdr_salt = float(tdr_salt_var.get())
        start_date = datetime.strptime(start_date_var.get(), "%Y-%m-%d")

        initial_input = pd.DataFrame([{
            "date": start_date,
            "irrigation": irrigation_values[0],
            "vpd (kPa)": vpd,
            "tdr_salt_80": tdr_salt
        }])

        model_growth = train_growth_model(data, feature_columns_growth, target_column_growth)
        model_vpd = train_vpd_model(data)
        model_tdr = train_tdr_model(data)

        predictions = predict_future_with_variable_irrigation(initial_input, irrigation_values, model_growth, model_vpd, model_tdr, future_days=90)

        display_results(predictions)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_results(predictions):
    """
    Display predictions in the Treeview table.
    """
    for row in tree.get_children():
        tree.delete(row)

    for _, row in predictions.iterrows():
        tree.insert("", "end", values=list(row))

def save_to_csv():
    """
    Save the predictions to a CSV file.
    """
    global predictions
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            predictions.to_csv(file_path, index=False)
            messagebox.showinfo("Success", "File saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Could not save file: {e}")

# -------------------------------
# GUI Initialization
# -------------------------------

root = tk.Tk()
root.title("Growth Prediction Tool")

irrigation_series_var = tk.StringVar()
vpd_var = tk.StringVar()
tdr_salt_var = tk.StringVar()
start_date_var = tk.StringVar()

tk.Label(root, text="Irrigation Values (9 values, comma-separated):").grid(row=0, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=irrigation_series_var).grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="VPD (kPa):").grid(row=1, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=vpd_var).grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="TDR Salt 80:").grid(row=2, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=tdr_salt_var).grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=3, column=0, padx=5, pady=5)
tk.Entry(root, textvariable=start_date_var).grid(row=3, column=1, padx=5, pady=5)

tk.Button(root, text="Run Prediction", command=run_prediction).grid(row=4, column=0, columnspan=2, pady=10)
tk.Button(root, text="Save to CSV", command=save_to_csv).grid(row=5, column=0, columnspan=2, pady=10)

tree = ttk.Treeview(root, columns=("Date", "Irrigation", "VPD", "TDR Salt", "Predicted Growth Rate"), show="headings")
tree.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

tree.heading("Date", text="Date")
tree.heading("Irrigation", text="Irrigation")
tree.heading("VPD", text="VPD")
tree.heading("TDR Salt", text="TDR Salt")
tree.heading("Predicted Growth Rate", text="Predicted Growth Rate")

root.mainloop()
