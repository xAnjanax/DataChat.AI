import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import os
import json
import requests

from streamlit_lottie import st_lottie

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

shared_directory = "./shared_files"
os.makedirs(shared_directory, exist_ok=True)

filename = "dataset_analysis.json"

def app1(shared_directory):
    st.title("Dataset Plot Page")

    col1, col2, col3 = st.columns(3) 

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_url_plot = "https://lottie.host/a3bd440c-75d6-4425-a6b0-00c0d142ac70/GPuI0XxcFm.json"
    lottie_plot = load_lottieurl(lottie_url_plot)

    with col1:
        st_lottie(
        lottie_plot,
        height=300,
        width=300,
        )

    lottie_url_plot1 = "https://lottie.host/61b3ff71-fca8-451e-a11c-d8a77e99bb0b/nhrnODftaO.json"
    lottie_plot1 = load_lottieurl(lottie_url_plot1) 

    with col2: 
        st_lottie(
            lottie_plot1, 
            height = 300, 
            width = 300, 
        )
    
    lottie_url_plot2 = "https://lottie.host/61b18533-9abf-44a4-be1a-f47274ba2aaf/4YOrd97Q4P.json"
    lottie_plot2 = load_lottieurl(lottie_url_plot2) 

    with col3: 
        st_lottie(
            lottie_plot2, 
            height = 300, 
            width = 300, 
        )

    if os.listdir(shared_directory): 
        files = os.listdir(shared_directory)

        st.write("Available files in the shared directory:")
        selected_file = st.selectbox("Select a file to load:", files)

        if st.button("Delete Selected File"):
            try:
                file_path = os.path.join(shared_directory, selected_file)
                os.remove(file_path)
                st.success(f"File '{selected_file}' deleted successfully! Reload to refresh.")
                
                if os.path.exists(filename):
                    with open(filename, 'w') as json_file:
                        json.dump({}, json_file) 
                    # st.success("File cleared successfully!")
            except Exception as e:
                st.error(f"Failed to delete file '{selected_file}': {e}")
        else:
            if selected_file:
                file_path = os.path.join(shared_directory, selected_file)

        if "plot_config" not in st.session_state:
            st.session_state["plot_config"] = {"x_col": None, "y_col": None, "additional_col": "None", "plot_type": "Scatter Plot"}

        if selected_file:
            file_path = os.path.join(shared_directory, selected_file)
            # st.write(f"Using the file: `{file_path}`")
            
            try:
                data = pd.read_csv(file_path, encoding='ISO-8859-1')
                st.subheader("Preview of the Selected CSV File")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
                st.stop()

            if data.shape[1] < 2:
                st.error("The dataset must have at least two columns to create a plot.")
                st.stop()

            cols = data.columns.tolist()

            x_col = st.selectbox(
                "Select X-axis column:", cols, key="x_col",
                index=cols.index(st.session_state["plot_config"].get("x_col", cols[0])) if st.session_state["plot_config"]["x_col"] in cols else 0
            )

            y_col = st.selectbox(
                "Select Y-axis column:", cols, key="y_col",
                index=cols.index(st.session_state["plot_config"].get("y_col", cols[0])) if st.session_state["plot_config"]["y_col"] in cols else 0
            )

            additional_col = st.selectbox(
                "Select additional column for category (optional):",
                ["None"] + cols, key="additional_col",
                index=(["None"] + cols).index(st.session_state["plot_config"].get("additional_col", "None")) if st.session_state["plot_config"]["additional_col"] in ["None"] + cols else 0
            )

            plot_type = st.selectbox(
                "Select Plot Type:",
                ["Scatter Plot", "Line Plot", "Stacked Bar Plot", "Clustered Bar Plot", "Pie Chart", "Donut Chart", "Histogram", "Box Plot", "Heat Map"],
                key="plot_type",
                index=["Scatter Plot", "Line Plot", "Stacked Bar Plot", "Clustered Bar Plot", "Pie Chart", "Donut Chart", "Histogram", "Box Plot", "Heat Map"].index(
                    st.session_state["plot_config"].get("plot_type", "Scatter Plot")
                )
            )

            st.session_state["plot_config"] = {
                "x_col": x_col,
                "y_col": y_col,
                "additional_col": additional_col if additional_col != "None" else None,
                "plot_type": plot_type,
            }

            if x_col and y_col:
                additional_col = st.session_state["plot_config"]["additional_col"]
                if plot_type == "Scatter Plot":
                    fig = px.scatter(data, x=x_col, y=y_col, color=additional_col, title="Scatter Plot")
                    regression_line = st.checkbox("Add Regression Line")
                    if regression_line:
                        slope, intercept = np.polyfit(data[x_col], data[y_col], 1)
                        data['Regression Line'] = slope * data[x_col] + intercept
                        fig.add_scatter(x=data[x_col], y=data['Regression Line'], mode='lines', name="Regression Line")
                elif plot_type == "Line Plot":
                    fig = px.line(data, x=x_col, y=y_col, color=additional_col, title="Line Plot")
                elif plot_type == "Stacked Bar Plot":
                    fig = px.bar(data, x=x_col, y=y_col, color=additional_col, barmode="stack", title="Stacked Bar Plot")
                elif plot_type == "Clustered Bar Plot":
                    fig = px.bar(data, x=x_col, y=y_col, color=additional_col, barmode="group", title="Clustered Bar Plot")
                elif plot_type == "Pie Chart":
                    fig = px.pie(data, values=y_col, names=x_col, title="Pie Chart")
                elif plot_type == "Donut Chart":
                    fig = px.pie(data, values=y_col, names=x_col, hole=0.4, title="Donut Chart")
                elif plot_type == "Histogram":
                    fig = px.histogram(data, x=x_col, color=additional_col, nbins=30, title="Histogram")
                elif plot_type == "Box Plot":
                    fig = px.box(data, x=x_col, y=y_col, color=additional_col, title="Box Plot")
                elif plot_type == "Heat Map":
                    fig = px.density_heatmap(data, x=x_col, y=y_col, z=additional_col, title="Heat Map")

                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Descriptive Statistics")
        col_to_analyze = st.selectbox("Select a column for statistics:", data.columns)
        if col_to_analyze:
            mean_value = data[col_to_analyze].mean()
            median_value = data[col_to_analyze].median()
            mode_value = data[col_to_analyze].mode()[0]
            st.write(f"Mean: {mean_value}")
            st.write(f"Median: {median_value}")
            st.write(f"Mode: {mode_value}")

        st.subheader("Decision Tree Generation")
        decision_tree_feature = st.selectbox("Select feature column(s) for Decision Tree:", data.columns)
        target_column = st.selectbox("Select target column for Decision Tree:", data.columns)

        tree_type = st.radio("Tree Type:", ["Classifier", "Regressor"])
        max_depth = st.slider("Select Maximum Depth of the Tree:", min_value=1, max_value=10, value=3)

        if st.button("Generate Decision Tree"):
            try:
                X = data[[decision_tree_feature]]
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if tree_type == "Classifier":
                    tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                    tree_model.fit(X_train, y_train)

                    class_names = [str(cls) for cls in np.unique(y)]
                else:
                    tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                    tree_model.fit(X_train, y_train)

                    class_names = None

                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(
                    tree_model,
                    feature_names=[decision_tree_feature],
                    class_names=class_names,  # Only for classification
                    filled=True,
                    ax=ax,
                )
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating decision tree: {e}")

        st.subheader("Data Plotting with Specifications") 

        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        user_column = st.selectbox("Select a column to group by (categorical):", categorical_columns)

        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        numeric_column = st.selectbox("Select a numeric column to filter:", numeric_columns)

        range_min = int(data[numeric_column].min())
        range_max = int(data[numeric_column].max())
        selected_range = st.slider(f"Set a range for {numeric_column}:", 
                                min_value=range_min, 
                                max_value=range_max, 
                                value=(range_min, range_max))

        filtered_df = data[(data[numeric_column] >= selected_range[0]) & (data[numeric_column] <= selected_range[1])]
        grouped_data = filtered_df.groupby(user_column)[numeric_column].sum()

        st.write("### Filtered Data")
        st.dataframe(filtered_df)

        st.write("### Plot Filtered Data")

        col1_1, col2_1, col3_1, col4, col5, col6 = st.columns(6)

        with col1_1:
            bar_chart_selected = st.button("Bar Chart")
        with col2_1:
            line_chart_selected = st.button("Line Chart")
        with col3_1:
            pie_chart_selected = st.button("Pie Chart")
        with col4:
            scatter_plot_selected = st.button("Scatter Plot")
        with col5:
            histogram_selected = st.button("Histogram")
        with col6:
            box_plot_selected = st.button("Box Plot")

        if not grouped_data.empty:
            
            if bar_chart_selected:
                fig = px.bar(data, x=user_column, y=numeric_column, title=f"Bar Plot of {numeric_column} vs {user_column}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif line_chart_selected:
                st.line_chart(grouped_data)
                
            elif pie_chart_selected:
                fig = px.pie(data, names=user_column, title=f"Pie Chart of {user_column}")
                st.plotly_chart(fig, use_container_width=True)

            elif scatter_plot_selected: 
                fig = px.scatter(data, x=user_column, y=numeric_column, title=f"Scatter Plot of {numeric_column} vs {user_column}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif histogram_selected:
                fig = px.histogram(data, x=user_column, title=f"Histogram of {user_column}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif box_plot_selected:
                fig = px.box(data, y=numeric_column, title=f"Box Plot of {numeric_column}")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("No data matches the selected criteria.")

    else:
        st.warning("No files available in the shared directory. Please upload a file to proceed.")

if __name__ == "__main__":
    app1(shared_directory)
