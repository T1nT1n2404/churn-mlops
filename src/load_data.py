import os
import pandas as pd

file_path = os.path.join("data", "raw", "churn.csv")

def load_data(path=file_path, output_path=os.path.join("data", "processed", "churn_processed.csv")):
    """Load the churn dataset from a CSV file.

    Args:
        path (str): The file path to the CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")
    
    print(f"Loading data from {path}...")
    data = pd.read_csv(path)
    print("Data loaded successfully.")

    data.columns = [col.lower().replace(" ", "_") for col in data.columns]

    data['totalcharges'] = pd.to_numeric(data['totalcharges'], errors='coerce')
    data['totalcharges'] = data['totalcharges'].fillna(0)

    if 'churn' in data.columns:
        data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}.")


if __name__ == "__main__":
    load_data()