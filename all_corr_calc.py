import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# Load the main correlation DataFrame (corr_df)
corr_df_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/corr_df_template.csv"
corr_df = pd.read_csv(corr_df_path)

# Ensure column names are clean (remove extra spaces)
corr_df.columns = corr_df.columns.str.strip()

# Define the sensor array
sens_arr = ['tdr_salt_40', 'tdr_salt_80', 'tdr_water_40', 'tdr_water_80', 'tensiometer_40', 'tensiometer_80', 'frond_growth_rate']

# Load the dataset containing all sensors and irrigation data
file_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/final_table_with_all_sensors_and_irrigation.csv"
all_data = pd.read_csv(file_path)

# Ensure the date column is in datetime format
all_data['date'] = pd.to_datetime(all_data['date'])

# Temporary storage for averages
e_irr_values = []
d_irr_values = []
water_50_values = []
water_100_values = []
e_50_values = []
d_50_values = []
e_100_values = []
d_100_values = []

# Iterate over all combinations of sensors
for sensor1, sensor2 in combinations(sens_arr, 2):
    #print(f"Calculating correlation between {sensor1} and {sensor2}...")

    # Filter rows where both sensors have data
    data_filtered = all_data.dropna(subset=[sensor1, sensor2])

    # Group by tree and calculate correlations
    for tree, group in data_filtered.groupby('tree'):
        correlation = group[sensor1].corr(group[sensor2])

        # Extract the tree tokens
        tree_number = tree.split('_')[0].strip()
        tree_irr_type = tree.split('_')[1].strip()
        tree_water_amount = tree.split('_')[2].strip()

        # Debug: Display tree information
        #print(f"Tree number: '{tree_number}', Irrigation Type: '{tree_irr_type}', Water Amount: '{tree_water_amount}'")

        # Find the column corresponding to the tree number
        tree_column = None
        for col in corr_df.columns:
            if col == tree_number:  # Match the prefix exactly
                tree_column = col
                break

        # Find rows where sensors match in either direction
        row_match = (
            ((corr_df['sensor b'] == sensor1) & (corr_df['sensor a'] == sensor2)) |  # Direction 1
            ((corr_df['sensor b'] == sensor2) & (corr_df['sensor a'] == sensor1))  # Direction 2
        )

        # Update the correlation value in the correct location
        if row_match.any() and tree_column:
            corr_df.loc[row_match, tree_column] = correlation
            #print(f"Updated correlation: {correlation} for sensors {sensor1}, {sensor2} in column {tree_column}")

            # ADD TO IRRIGATION AVERAGE CALCULATIONS
            if tree_irr_type == "E":
                e_irr_values.append(correlation)  # Add to E irr list
            elif tree_irr_type == "D":
                d_irr_values.append(correlation)  # Add to D irr list

            # ADD TO WATER AMOUNT AVERAGE CALCULATIONS
            if tree_water_amount == "50":
                water_50_values.append(correlation)  # Add to 50% water list
            elif tree_water_amount == "100":
                water_100_values.append(correlation)  # Add to 100% water list

            # ADD TO SPECIFIC COMBINATIONS (E_50, D_50, E_100, D_100)
            if tree_water_amount == "50" and tree_irr_type == "E":
                e_50_values.append(correlation)
            elif tree_water_amount == "50" and tree_irr_type == "D":
                d_50_values.append(correlation)
            elif tree_water_amount == "100" and tree_irr_type == "E":
                e_100_values.append(correlation)
            elif tree_water_amount == "100" and tree_irr_type == "D":
                d_100_values.append(correlation)
        else:
            if not row_match.any():
                print(f"No matching row found for sensor pair: {sensor1}, {sensor2}")
            if not tree_column:
                print(f"No matching column found for tree prefix: '{tree_number}'")

#E_100

# Iterate over each row to calculate and update "E_100"
for index, row in corr_df.iterrows():
    # Check if both "T15" and "T7" have non-NaN values in the current row
    if pd.notna(row["T15"]) and pd.notna(row["T7"]):
        t15_value = row["T15"]
        t7_value = row["T7"]

        # Ensure the values in "T15" and "T7" are numeric (int or float)
        if isinstance(t15_value, (int, float)) and isinstance(t7_value, (int, float)):
            # Calculate the average of "T15" and "T7"
            e_100_avg = (t15_value + t7_value) / 2

            # Update the value in the "E_100" column for the current row
            corr_df.at[index, "E_100"] = e_100_avg
            #print(f"Row {index}: Updated E_100 with value {e_100_avg}")
        else:
            # If either "T15" or "T7" is not numeric, print a debug message
            print(f"Row {index}: T15 or T7 is not numeric.")
    else:
        # If either "T15" or "T7" is missing or NaN, print a debug message
        print(f"Row {index}: Missing or NaN values in T15 or T7.")

#E_50
for index, row in corr_df.iterrows():
    if pd.notna(row["T12"]) and pd.notna(row["T4"]):
        t12_value = row["T12"]
        t4_value = row["T4"]
        if isinstance(t12_value, (int, float)) and isinstance(t4_value, (int, float)):
            e_50_avg = (t12_value + t4_value) / 2

            corr_df.at[index, "E_50"] = e_50_avg
            #print(f"Row {index}: Updated E_50 with value {e_50_avg}")
        else:
            print(f"Row {index}: T12 or T4 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T12 or T4.")
#D_100
for index, row in corr_df.iterrows():
    if pd.notna(row["T14"]) and pd.notna(row["T9"]):
        t14_value = row["T14"]
        t9_value = row["T9"]
        if isinstance(t14_value, (int, float)) and isinstance(t9_value, (int, float)):
            d_100_avg = (t14_value + t9_value) / 2

            corr_df.at[index, "D_100"] = d_100_avg
            #print(f"Row {index}: Updated D_100 with value {d_100_avg}")
        else:
            print(f"Row {index}: T14 or T9 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T14 or T9.")
#D_50
for index, row in corr_df.iterrows():
    if pd.notna(row["T2"]) and pd.notna(row["T5"]):
        t2_value = row["T2"]
        t5_value = row["T5"]
        if isinstance(t2_value, (int, float)) and isinstance(t5_value, (int, float)):
            d_50_avg = (t2_value + t5_value) / 2

            corr_df.at[index, "D_50"] = d_50_avg
            #print(f"Row {index}: Updated D_50 with value {d_50_avg}")
        else:
            print(f"Row {index}: T2 or T5 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T2 or T5.")
#50
for index, row in corr_df.iterrows():
    if pd.notna(row["T2"]) and pd.notna(row["T5"]) and pd.notna(row["T12"]) and pd.notna(row["T4"]) :
        t2_value = row["T2"]
        t5_value = row["T5"]
        t12_value = row["T12"]
        t4_value = row["T4"]
        if isinstance(t2_value, (int, float)) and isinstance(t5_value, (int, float))and isinstance(t12_value, (int, float))and isinstance(t4_value, (int, float)):
            all_50_avg = (t2_value + t5_value + t12_value + t4_value) / 4

            corr_df.at[index, "50% water"] = all_50_avg
            #print(f"Row {index}: Updated 50% water with value {all_50_avg}")
        else:
            print(f"Row {index}: T2 or T5 or T12 OR T4 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T2 or T5 or T12 OR T4.")
#100
for index, row in corr_df.iterrows():
    if pd.notna(row["T7"]) and pd.notna(row["T9"]) and pd.notna(row["T14"]) and pd.notna(row["T15"]) :
        t7_value = row["T7"]
        t9_value = row["T9"]
        t14_value = row["T14"]
        t15_value = row["T15"]
        if isinstance(t7_value, (int, float)) and isinstance(t9_value, (int, float))and isinstance(t14_value, (int, float))and isinstance(t15_value, (int, float)):
            all_100_avg = (t7_value + t9_value + t14_value + t15_value) / 4

            corr_df.at[index, "100% water"] = all_100_avg
            #print(f"Row {index}: Updated 100% water with value {all_100_avg}")
        else:
            print(f"Row {index}: T7 or T9 or T14 OR T15 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T7 or T9 or T14 OR T15.")

#E
for index, row in corr_df.iterrows():
    if pd.notna(row["T4"]) and pd.notna(row["T7"]) and pd.notna(row["T12"]) and pd.notna(row["T15"]) :
        t4_value = row["T4"]
        t7_value = row["T7"]
        t12_value = row["T12"]
        t15_value = row["T15"]
        if isinstance(t4_value, (int, float)) and isinstance(t7_value, (int, float))and isinstance(t12_value, (int, float))and isinstance(t15_value, (int, float)):
            e_irr_avg = (t4_value + t7_value + t12_value + t15_value) / 4

            corr_df.at[index, "E irr"] = e_irr_avg
            #print(f"Row {index}: Updated E irr with value {e_irr_avg}")
        else:
            print(f"Row {index}: T4 or T7 or T12 OR T15 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T4 or T7 or T12 OR T15.")
#D
for index, row in corr_df.iterrows():
    if pd.notna(row["T2"]) and pd.notna(row["T5"]) and pd.notna(row["T9"]) and pd.notna(row["T14"]) :
        t2_value = row["T2"]
        t5_value = row["T5"]
        t9_value = row["T9"]
        t14_value = row["T14"]
        if isinstance(t2_value, (int, float)) and isinstance(t5_value, (int, float))and isinstance(t9_value, (int, float))and isinstance(t14_value, (int, float)):
            d_irr_avg = (t2_value + t5_value + t9_value + t14_value) / 4

            corr_df.at[index, "D irr"] = d_irr_avg
            #print(f"Row {index}: Updated D irr with value {d_irr_avg}")
        else:
            print(f"Row {index}: T2 or T5 or T9 OR T14 is not numeric.")
    else:
        print(f"Row {index}: Missing or NaN values in T2 or T5 or T9 OR T14.")


# Save the updated corr_df to a CSV file
output_corr_df_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/updated_corr_df.csv"
corr_df.to_csv(output_corr_df_path, index=False)
print(f"Updated corr_df saved to: {output_corr_df_path}")

# Print the fully updated DataFrame
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', 1000)  # Increase the width of display for better visualization
print("Fully updated corr_df:")
print(corr_df)




# Plot data
"""tdr_salt_40, tdr_salt_80"""
# Filter out missing or non-numeric values
filtered_data = all_data.dropna(subset=['tdr_salt_40', 'tdr_salt_80'])
x = filtered_data['tdr_salt_40']
y = filtered_data['tdr_salt_80']

# Fit a linear regression line (best fit line)
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Generate y-values for the best fit line
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = best_fit_line(x_vals)

# Plot the scatter plot with the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, color='red', label=f'Best Fit Line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})')
plt.title('tdr_salt_40 vs tdr_salt_80 with Best Fit Line')
plt.xlabel('tdr_salt_40')
plt.ylabel('tdr_salt_80')
plt.legend()
plt.grid()
plt.show()

"""tdr_water_40, tdr_water_80"""
# Filter out missing or non-numeric values
filtered_data = all_data.dropna(subset=['tdr_water_40', 'tdr_water_80'])
x = filtered_data['tdr_water_40']
y = filtered_data['tdr_water_80']

# Fit a linear regression line (best fit line)
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Generate y-values for the best fit line
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = best_fit_line(x_vals)

# Plot the scatter plot with the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, color='red', label=f'Best Fit Line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})')
plt.title('tdr_water_40 vs tdr_water_80 with Best Fit Line')
plt.xlabel('tdr_water_40')
plt.ylabel('tdr_water_80')
plt.legend()
plt.grid()
plt.show()


"""tdr_water_40, tensiometer_40"""
# Filter out missing or non-numeric values
filtered_data = all_data.dropna(subset=['tdr_water_40', 'tensiometer_40'])
x = filtered_data['tdr_water_40']
y = filtered_data['tensiometer_40']

# Fit a linear regression line (best fit line)
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Generate y-values for the best fit line
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = best_fit_line(x_vals)

# Plot the scatter plot with the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, color='red', label=f'Best Fit Line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})')
plt.title('tdr_water_40 vs tensiometer_40 with Best Fit Line')
plt.xlabel('tdr_water_40')
plt.ylabel('tensiometer_40')
plt.legend()
plt.grid()
plt.show()


# Filter out missing, non-numeric, and negative values for frond_growth_rate
filtered_data = all_data.dropna(subset=['tensiometer_40', 'frond_growth_rate', 'date'])
filtered_data = filtered_data[filtered_data['frond_growth_rate'] >= 0]

# Convert the 'date' column to a numerical format (e.g., timestamp)
filtered_data['date_numeric'] = pd.to_datetime(filtered_data['date']).map(pd.Timestamp.timestamp)



# Fit a linear regression line (best fit line)
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Generate y-values for the best fit line
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = best_fit_line(x_vals)

# Plot the scatter plot with the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, color='red', label=f'Best Fit Line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})')
plt.title('tensiometer_40 vs frond_growth_rate (Positive Values Only) with Best Fit Line')
plt.xlabel('tensiometer_40')
plt.ylabel('frond_growth_rate')
plt.legend()
plt.grid()
plt.show()


# Extract x, y, and z values
x = filtered_data['tensiometer_40']  # X-axis
y = filtered_data['frond_growth_rate']  # Y-axis
z = filtered_data['date_numeric']  # Z-axis (time)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D
sc = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.7, label='Data Points')

# Add color bar to indicate date progression
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Date (Color-coded)')

# Labels and title
ax.set_title('3D Plot: Tensiometer vs Frond Growth Rate vs Time')
ax.set_xlabel('tensiometer_40')
ax.set_ylabel('frond_growth_rate')
ax.set_zlabel('Date (Numeric)')

# Show plot
plt.show()


# Filter out missing, non-numeric, and negative values for frond_growth_rate
filtered_data = all_data.dropna(subset=['tensiometer_40', 'frond_growth_rate', 'date'])
filtered_data = filtered_data[filtered_data['frond_growth_rate'] >= 0]

# Convert the 'date' column to datetime and normalize to a range of [0, 1]
filtered_data['date'] = pd.to_datetime(filtered_data['date'])
date_norm = (filtered_data['date'] - filtered_data['date'].min()) / (
    filtered_data['date'].max() - filtered_data['date'].min()
)

# Create a color map from green to orange to brown
color_map = mcolors.LinearSegmentedColormap.from_list("DateColor", ["green", "orange", "brown"])
colors = color_map(date_norm)


"""color graph"""
# Filter out missing, non-numeric, and negative values for frond_growth_rate
filtered_data = all_data.dropna(subset=['tensiometer_40', 'frond_growth_rate', 'date'])
filtered_data = filtered_data[filtered_data['frond_growth_rate'] >= 0]

# Convert the 'date' column to datetime
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# Normalize the date column to a range of [0, 1]
date_norm = (filtered_data['date'] - filtered_data['date'].min()) / (
    filtered_data['date'].max() - filtered_data['date'].min()
)

# Create a color map from green to orange to brown
color_map = mcolors.LinearSegmentedColormap.from_list("DateColor", ["green", "orange", "brown"])

# Map the normalized date values to colors
colors = color_map(date_norm)


# Fit a linear regression line (best fit line)
coefficients = np.polyfit(x, y, 1)  # Returns [slope, intercept]
best_fit_line = np.poly1d(coefficients)  # Create a function for the line

# Calculate the correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]  # Correlation coefficient between x and y

# Generate x and y values for the best fit line
x_vals = np.linspace(x.min(), x.max(), 100)  # Generate evenly spaced x values
y_vals = best_fit_line(x_vals)  # Calculate corresponding y values for the best fit line

# Plot the best fit line on top of the scatter plot
plt.plot(x_vals, y_vals, color='red', label=f'Best Fit Line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})')

# Add the correlation value to the plot title
plt.title(f'Tensiometer_40 vs Frond_Growth_Rate (Correlation: {correlation:.2f})')
# Plot the scatter plot
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    filtered_data['tensiometer_40'],
    filtered_data['frond_growth_rate'],
    c=date_norm,  # Use normalized date values for color
    cmap=color_map,  # Use the green-orange-brown colormap
    alpha=0.7
)
# Add labels and title
plt.title('Tensiometer_40 vs Frond_Growth_Rate (Color-Coded by Date)')
plt.xlabel('Tensiometer_40')
plt.ylabel('Frond_Growth_Rate')

# Add a color bar
cbar = plt.colorbar(scatter, pad=0.1)
cbar.set_label('Date Progression (Green → Orange → Brown)')

# Show grid and plot
plt.grid(True)
plt.show()


