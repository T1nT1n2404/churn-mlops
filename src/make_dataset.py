import pandas as pd
from sklearn.model_selection import train_test_split
import os
input_path = os.path.join("data", "processed", "churn_processed.csv")
output_train_path = os.path.join("data", "processed", "train.csv")
output_test_path = os.path.join("data", "processed", "test.csv")
def process_data(input_path, output_train_path, output_test_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")
    
    print(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path)

    if 'customerid' in data.columns:
        data = data.drop('customerid', axis=1)

    data = pd.get_dummies(data, drop_first=True)

    X = data.drop('churn', axis=1)
    y = data['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)

    print(f"Training data saved to {output_train_path}.")


if __name__ == "__main__":
    process_data(input_path, output_train_path, output_test_path)
    print(f"Testing data saved to {output_test_path}.")