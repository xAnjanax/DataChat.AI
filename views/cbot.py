import os
import json
from groq import Groq
from dotenv import load_dotenv 
import streamlit as st 
import pandas as pd
from streamlit_lottie import st_lottie

shared_directory = "./shared_files"
os.makedirs(shared_directory, exist_ok=True)

load_dotenv(".env.local")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def load_json_file(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data
    except Exception as e:
        return {"error": str(e)}

def ask_groq_with_context(user_query, json_data, model="llama3-8b-8192"):
    try:
        json_context = json.dumps(json_data, indent=2)

        query = f"Dataset Information:\n{json_context}\n\nUser Question: {user_query}"

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model=model,
            stream=False,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def app1(shared_directory): 

    st.title("Dataset Chatbot")

    lottie_url_bot = "https://lottie.host/3028b81e-b177-4652-8e09-986e31671ba2/Ve66loFMZE.json"
    lottie_bot = (lottie_url_bot)

    st_lottie(lottie_bot, height=300, width=None)

    if os.listdir(shared_directory): 
        files = os.listdir(shared_directory)

        st.write("Available files in the shared directory:")
        selected_file = st.selectbox("Select a file to load:", files)

        if st.button("Delete Selected File"):
            try:
                file_path = os.path.join(shared_directory, selected_file)
                os.remove(file_path)
                st.success(f"File '{selected_file}' deleted successfully! Reload to refresh.")
                return 
            except Exception as e:
                st.error(f"Failed to delete file '{selected_file}': {e}")
        else:
            if selected_file:
                file_path = os.path.join(shared_directory, selected_file)

        try:
            file_path = os.path.join(shared_directory, selected_file)

            df = pd.read_csv(file_path, encoding='ISO-8859-1')

            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
            missing_values = df.isnull().sum()
            num_rows, num_cols = df.shape

            if len(numeric_columns) > 1:
                correlations = df[numeric_columns].corr().to_dict()
                correlation_text = "Correlation matrix available for numeric columns."
            else:
                correlations = None
                correlation_text = "Not enough numeric columns for a correlation matrix."

            parameter_texts = {
                "Number of Rows and Columns": f"The dataset contains {num_rows} rows and {num_cols} columns.",
                "Numeric Columns": f"Numeric columns: {', '.join(numeric_columns) if numeric_columns else 'None'}",
                "Categorical Columns": f"Categorical columns: {', '.join(categorical_columns) if categorical_columns else 'None'}",
                "Datetime Columns": f"Datetime columns: {', '.join(datetime_columns) if datetime_columns else 'None'}",
                "Missing Values": "\n".join(
                    [f"{col}: {missing_values[col]} missing values" for col in df.columns if missing_values[col] > 0]
                ) if missing_values.sum() > 0 else "No missing values.",
                "Correlations": correlation_text,
            }

            output_filename = "dataset_analysis.json"
            output_data = {
                "dataset_summary": {
                    "num_rows": num_rows,
                    "num_columns": num_cols,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "datetime_columns": datetime_columns,
                    "missing_values": missing_values.to_dict(),
                    "correlations": correlations,
                },
                "explanations": parameter_texts,
            }

            with open(output_filename, "w") as selected_file:
                json.dump(output_data, selected_file, indent=4)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Upload a CSV file to get started.")

    filename = "dataset_analysis.json"

    json_data = load_json_file(filename)

    if "error" not in json_data and os.listdir(shared_directory):

        st.text("You can ask me: ")
        st.text("- What are the key patterns or trends in this dataset?") 
        st.text("- Should I use a scatter plot or a line plot to show the relationship between variable 'X' and 'Y'?") 
        st.text("- How do I create a report summarizing the results of my analysis?") 

        user_query = st.text_input("Enter your query about the dataset:")

        if user_query:
            response = ask_groq_with_context(user_query, json_data)

            st.subheader("Chatbot Response")
            st.write(response)
    else:
        st.info("Chatbot session will be initiated upon uploading a file.")
