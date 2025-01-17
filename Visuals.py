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


st.title("Interactive Irrigation Data Visualizations")

# Filters
st.sidebar.header("Filters")
selected_month = st.sidebar.multiselect(
    "Select Month(s):",
    options=range(3, 11),
    format_func=lambda x: datetime(1900, x, 1).strftime('%B'),
    default=range(3, 11)
)
selected_season = st.sidebar.selectbox("Select Season:", options=["None", "Spring", "Summer", "Autumn"])
season_months = {"Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10]}
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
    "Seasonal Trends", "Irrigation Impact", "Heatmap (Correlations)", "Tree Health Visualization", "Combination Comparisons"
])


# Visualization logic
if visualization_type == "Seasonal Trends":
    st.header("Seasonal Trends")

    # Create two columns for side-by-side dropdowns
    col1, col2 = st.columns(2)

    with col1:
        time_scale = st.selectbox("Select Time Scale:", options=["Month", "Week", "Day", "All Data Points"], index=0)

    with col2:
        # Updated metric options to list all metrics individually
        metric_options = [
            "Tensiometer at 40cm", "Tensiometer at 80cm",
            "TDR Water at 40cm", "TDR Water at 80cm",
            "TDR Salt at 40cm", "TDR Salt at 80cm",
            "Evapotranspiration (mm/day)", "Vapor Pressure Deficit (kPa)",
            "Frond Growth Rate", "Irrigation Amount"
        ]
        selected_metrics = st.multiselect(
            "Select Metrics:",
            options=metric_options,
            default=[metric_options[0]]  # Default to the first metric for usability
        )

    # Reverse mapping for the selected metrics
    metrics = [reverse_mapping[selected_metric] for selected_metric in selected_metrics]

    # Group by the selected time scale
    if time_scale == "Month":
        trend_data = filtered_data.groupby('month')[metrics].mean().reset_index()
        x_axis = 'month'
        x_labels = [datetime(1900, m, 1).strftime('%B') for m in range(3, 11)]
    elif time_scale == "Week":
        filtered_data['week'] = filtered_data['date'].dt.isocalendar().week
        trend_data = filtered_data.groupby('week')[metrics].mean().reset_index()
        x_axis = 'week'
        x_labels = None
    elif time_scale == "Day":
        trend_data = filtered_data.groupby('date')[metrics].mean().reset_index()
        x_axis = 'date'
        x_labels = None
    else:  # All Data Points
        trend_data = filtered_data
        x_axis = 'date'
        x_labels = None

    # Create the line plot for multiple metrics
    fig = px.line(
        trend_data,
        x=x_axis,
        y=metrics,
        title=f'Seasonal Trends ({time_scale})',
        labels={x_axis: time_scale, **{metric: field_name_mapping.get(metric, metric) for metric in metrics}},
        template="plotly_white"
    )
    if x_labels:
        fig.update_xaxes(tickmode='array', tickvals=list(range(3, 11)), ticktext=x_labels)
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

elif visualization_type == "Combination Comparisons":
    st.header("Combination Comparisons")

    # Dropdown for plot type (box plot, scatter plot, trend line)
    plot_type = st.selectbox("Select Plot Type:", options=["Box Plot", "Scatter Plot", "Trend Line"])

    # Define possible combinations including all 8 groups
    def create_combinations(row):
        combinations = []
        if row['D'] == 1:
            combinations.append("Only D")
        if row['E'] == 1:
            combinations.append("Only E")
        if row['50'] == 1:
            combinations.append("Only 50")
        if row['100'] == 1:
            combinations.append("Only 100")
        if row['D'] == 1 and row['100'] == 1:
            combinations.append("100% & D")
        if row['E'] == 1 and row['100'] == 1:
            combinations.append("100% & E")
        if row['D'] == 1 and row['50'] == 1:
            combinations.append("50% & D")
        if row['E'] == 1 and row['50'] == 1:
            combinations.append("50% & E")
        return combinations

    # Create a unified "Combination" column as a list of combinations
    filtered_data['Combination'] = filtered_data.apply(create_combinations, axis=1)

    # Expand the data so each row corresponds to a single combination
    expanded_data = filtered_data.explode('Combination')

    # Available combinations for selection
    combinations_map = [
        "100% & D", "100% & E", "50% & D", "50% & E",
        "Only D", "Only E", "Only 50", "Only 100"
    ]

    # Select multiple combinations for scatter plot
    selected_combinations = st.multiselect(
        "Select Combinations:",
        options=combinations_map,
        default=combinations_map
    )

    # Filter data based on selected combinations
    filtered_combinations = expanded_data[expanded_data['Combination'].isin(selected_combinations)]

    # Box plot
    if plot_type == "Box Plot":
        y_axis = st.selectbox("Select Y-Axis:", options=[field_name_mapping[k] for k in [
            "frond_growth_rate", "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80"]])
        y_axis = reverse_mapping[y_axis]

        fig = px.box(
            filtered_combinations,
            x='Combination',
            y=y_axis,
            color='Combination',
            title=f'Comparison of {field_name_mapping[y_axis]} Across Combinations',
            labels={'Combination': 'Combination', y_axis: field_name_mapping[y_axis]},
            template="plotly_white"
        )
        st.plotly_chart(fig)

    # Scatter plot
    elif plot_type == "Scatter Plot":
        # Allow user to select X and Y axes for scatter plot
        x_axis_options = [
            'date', 'tensiometer_40', 'tensiometer_80', 'tdr_water_40',
            'tdr_water_80', 'frond_growth_rate', 'eto (mm/day)', 'vpd (kPa)', 'irrigation'
        ]
        x_axis = st.selectbox("Select X-Axis:", options=x_axis_options, index=0)
        y_axis = st.selectbox("Select Y-Axis:", options=x_axis_options, index=1)

        # Plot the scatter chart
        fig = px.scatter(
            filtered_combinations,
            x=x_axis,
            y=y_axis,
            color='Combination',
            title=f'Scatter Plot: {x_axis} vs {y_axis}',
            labels={x_axis: x_axis, y_axis: y_axis},
            template="plotly_white"
        )
        st.plotly_chart(fig)

    # Trend line plot
    elif plot_type == "Trend Line":
        y_axis = st.selectbox("Select Y-Axis:", options=[field_name_mapping[k] for k in [
            "frond_growth_rate", "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80"]])
        y_axis = reverse_mapping[y_axis]

        # Create a trend line for each combination group over time
        fig = px.line(
            filtered_combinations,
            x='date',
            y=y_axis,
            color='Combination',
            title=f'Trend Line of {field_name_mapping[y_axis]} Over Time',
            labels={'date': 'Date', y_axis: field_name_mapping[y_axis]},
            template="plotly_white"
        )
        st.plotly_chart(fig)




