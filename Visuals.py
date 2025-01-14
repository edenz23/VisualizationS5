import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Load data
def load_data():
    file_path = 'final_table_with_all_sensors_and_irrigation.csv'  # Replace with your file path
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])  # Ensure date is datetime type
    data['month'] = data['date'].dt.month  # Extract month for analysis
    data = data[data['month'].between(3, 10)]  # Filter months to March-October
    return data

data = load_data()

# User-friendly field names
field_name_mapping = {
    "tensiometer_40": "Tensiometer at 40cm",
    "tensiometer_80": "Tensiometer at 80cm",
    "tdr_water_40": "TDR Water at 40cm",
    "tdr_water_80": "TDR Water at 80cm",
    "tdr_salt_40": "TDR Salt at 40cm",
    "tdr_salt_80": "TDR Salt at 80cm",
    "eto (mm/day)": "Evapotranspiration (mm/day)",
    "vpd (kPa)": "Vapor Pressure Deficit (kPa)",
    "frond_growth_rate": "Frond Growth Rate",
    "irrigation": "Irrigation Amount",
    "D": "Dropper Type D",
    "E": "Dropper Type E",
    "100": "100% Water Supplied",
    "50": "50% Water Supplied"
}

# Define metric groups for Seasonal Trends
metric_groups = {
    "Tensiometer": ["tensiometer_40", "tensiometer_80"],
    "TDR Water": ["tdr_water_40", "tdr_water_80"],
    "TDR Salt": ["tdr_salt_40", "tdr_salt_80"]
}

# Reverse mapping for user-friendly names
reverse_mapping = {v: k for k, v in field_name_mapping.items()}

# Streamlit app
st.title("Interactive Irrigation Data Visualizations")

# Filters
st.sidebar.header("Filters")
selected_month = st.sidebar.multiselect(
    "Select Month(s):",
    options=range(3, 11),
    format_func=lambda x: datetime(1900, x, 1).strftime('%B'),
    default=range(3, 11)
)
selected_season = st.sidebar.selectbox("Select Season:", options=[
    "None", "Spring", "Summer", "Autumn"
])
season_months = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10]
}
if selected_season != "None":
    selected_month = season_months[selected_season]

selected_dropper = st.sidebar.multiselect("Select Dropper Type:", options=['D', 'E'], default=['D', 'E'])
selected_percentage = st.sidebar.multiselect("Select Water Percentage:", options=['100', '50'], default=['100', '50'])

# Apply filters
filtered_data = data.copy()
filtered_data = filtered_data[filtered_data['month'].isin(selected_month)]
filtered_data = filtered_data[(filtered_data['D'].isin([1 if d == 'D' else 0 for d in selected_dropper])) | (filtered_data['E'].isin([1 if e == 'E' else 0 for e in selected_dropper]))]
filtered_data = filtered_data[(filtered_data['100'].isin([1 if p == '100' else 0 for p in selected_percentage])) | (filtered_data['50'].isin([1 if p == '50' else 0 for p in selected_percentage]))]

# Visualization options
st.sidebar.header("Visualization Options")
visualization_type = st.sidebar.selectbox("Select Visualization Type:", options=[
    "Seasonal Trends", "Irrigation Impact", "Heatmap (Correlations)", "Tree Health Visualization"
])

# Visualization logic
if visualization_type == "Seasonal Trends":
    st.header("Seasonal Trends")
    metric_options = ["Tensiometer", "TDR Water", "TDR Salt"] + [field_name_mapping[f] for f in [
        "eto (mm/day)", "vpd (kPa)", "frond_growth_rate", "irrigation", "D", "E", "100", "50"]]
    selected_metric = st.selectbox("Select Metric:", options=metric_options, index=0)

    if selected_metric in field_name_mapping.values():
        metrics = [reverse_mapping[selected_metric]]
    else:
        metrics = metric_groups[selected_metric]

    trend_data = filtered_data.groupby('month')[metrics].mean().reset_index()
    fig = px.line(
        trend_data,
        x='month',
        y=metrics,
        title='Seasonal Trends',
        labels={'month': 'Month'},
        template="plotly_white"
    )
    fig.update_xaxes(tickmode='array', tickvals=list(range(3, 11)), ticktext=[datetime(1900, m, 1).strftime('%B') for m in range(3, 11)])
    st.plotly_chart(fig)

elif visualization_type == "Irrigation Impact":
    st.header("Irrigation Impact")
    x_axis_options = ["100% Water Supplied", "50% Water Supplied", "Dropper Type D", "Dropper Type E", "Combination"]
    x_axis = st.selectbox("Select X-Axis:", options=x_axis_options, index=0)

    if x_axis == "Combination":
        filtered_data['Combination'] = (
            filtered_data['D'].map({1: 'D', 0: ''}) + " & " +
            filtered_data['E'].map({1: 'E', 0: ''}) + " / " +
            filtered_data['100'].map({1: '100%', 0: ''}) + " & " +
            filtered_data['50'].map({1: '50%', 0: ''})
        )
        x_axis = 'Combination'
    else:
        x_axis = reverse_mapping[x_axis]

    y_axis = st.selectbox("Select Y-Axis:", options=[field_name_mapping[k] for k in [
        "frond_growth_rate", "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80"]])
    y_axis = reverse_mapping[y_axis]

    fig = px.box(
        filtered_data,
        x=x_axis,
        y=y_axis,
        color=x_axis if x_axis != 'Combination' else None,
        title=f'Irrigation Impact on {field_name_mapping[y_axis]}',
        labels={x_axis: x_axis, y_axis: field_name_mapping[y_axis]},
        template="plotly_white"
    )
    st.plotly_chart(fig)

elif visualization_type == "Heatmap (Correlations)":
    st.header("Heatmap (Correlations)")
    correlation_data = filtered_data[[
        "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80",
        "tdr_salt_40", "tdr_salt_80", "eto (mm/day)", "vpd (kPa)", "frond_growth_rate", "irrigation"
    ]]
    corr = correlation_data.corr()
    corr.columns = [field_name_mapping.get(c, c) for c in corr.columns]
    corr.index = [field_name_mapping.get(c, c) for c in corr.index]
    fig = px.imshow(
        corr,
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        labels={'color': 'Correlation'},
        template="plotly_white",
        width=1000,
        height=800
    )
    st.plotly_chart(fig)

elif visualization_type == "Tree Health Visualization":
    st.header("Tree Health Visualization")
    tree_metric = st.selectbox("Select Metric for Tree Health:", options=[
        "Frond Growth Rate", "100% Water Supplied", "50% Water Supplied", "Dropper Type D", "Dropper Type E"
    ])
    metric_key = reverse_mapping.get(tree_metric, "frond_growth_rate")
    health_data = filtered_data.groupby('month')[metric_key].mean().reset_index()

    fig = px.scatter(
        health_data,
        x='month',
        y=metric_key,
        size=metric_key,
        color=metric_key,
        title=f'Tree Health Based on {tree_metric}',
      labels={'month': 'Month', metric_key: tree_metric},
        template="plotly_white"
    )
    fig.update_xaxes(tickmode='array', tickvals=list(range(3, 11)), ticktext=[datetime(1900, m, 1).strftime('%B') for m in range(3, 11)])
    st.plotly_chart(fig)

