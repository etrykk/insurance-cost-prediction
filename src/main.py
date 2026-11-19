#Author: etrykk
#Date: 11/18/2026
#Insurance Cost Prediction

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    #Read dataset
    insurance_data = pd.read_csv(DATA_DIR / "insurance.csv")
    
    print(insurance_data.head())
    print(insurance_data.shape)
    print(insurance_data.dtypes)
    print(insurance_data.isna().sum())
    
    return insurance_data


def analyze_data(insurance_data):
    print(insurance_data.describe())
    print(insurance_data["smoker"].value_counts())
    print(insurance_data.groupby("smoker")["charges"].mean())
    print(insurance_data["age"].mean())
    print(insurance_data["charges"].max())
    print(insurance_data["charges"].min())
    

def visualize_data(insurance_data):
    plt.figure()
    plt.hist(insurance_data["charges"], bins=30)
    plt.xlabel("Charges in usd")
    plt.ylabel("Frequency")
    plt.title("Charges in insurance data")
    plt.savefig(OUTPUT_DIR / "Charges_in_insurance_data.png", dpi=300)
    plt.show()
    
    plt.figure()
    plt.scatter(insurance_data["age"], insurance_data["charges"])
    plt.xlabel("Age")
    plt.ylabel("Charges in usd")
    plt.title("Insurance charges by age")
    plt.savefig(OUTPUT_DIR / "Charges_by_Age.png", dpi=300)
    plt.show()
    
    smoker = insurance_data[insurance_data["smoker"] == "yes"]
    non_smoker = insurance_data[insurance_data["smoker"] == "no"]
    
    plt.figure()
    plt.scatter(smoker["age"], smoker["charges"], color="red", label="Smokers")
    plt.scatter(non_smoker["age"], non_smoker["charges"], color="blue", label="Non Smokers")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.title("Charges for smokers vs non_smokers")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "Charges_for_smokers_vs_non.png", dpi=300)
    plt.show()
    

def correlation_analysis(insurance_data):
    correlation_matrix = insurance_data[["age", "bmi", "children", "charges"]].corr()
    print(correlation_matrix)


def visualize_correlation(insurance_data):
    correlation_matrix = insurance_data[["age", "bmi", "children", "charges"]].corr()
    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(OUTPUT_DIR / "Correlation_Heatmap.png", dpi=300)
    plt.show()
    
def prepare_data(insurance_data):
    x = insurance_data.drop(columns=["charges"])
    y = insurance_data["charges"]
    
    return x, y

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test
    
    
def identify_features(x_train):
    categorical_columns = x_train.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
    
    for column in categorical_columns:
        print(column, x_train[column].unique())
        
    return categorical_columns, numerical_columns


def encode_features(x_train, x_test, categorical_columns, numerical_columns):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train_encoded = encoder.fit_transform(x_train[categorical_columns])
    x_test_encoded = encoder.transform(x_test[categorical_columns])

    
    x_train_numerical = x_train[numerical_columns]
    x_test_numerical = x_test[numerical_columns]
    x_train_final = np.hstack((x_train_numerical, x_train_encoded))
    x_test_final = np.hstack((x_test_numerical, x_test_encoded))

    
    return x_train_final, x_test_final    


def train_model(x_train_final, y_train):
    model = LinearRegression()
    trained_model = model.fit(x_train_final, y_train)
    return trained_model
    
def predict_charges(model, x_test_final):
    predictions = model.predict(x_test_final)
    
    return predictions

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r_squared:.4f}")
    return mae,rmse,r_squared


def visualize_predictions(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())
    plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--"
    )
    plt.xlabel("Actual Insurance Charges ($)")
    plt.ylabel("Predicted Insurance Charges ($)")
    plt.title("Actual vs. Predicted Insurance Charges")
    plt.savefig(OUTPUT_DIR / "Test_and_Prediction_Outcomes.png", dpi=300)
    plt.show()
        
    
def visualize_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted insurance charges")
    plt.ylabel("Residuals")
    plt.title("Insurance Residuals")
    plt.savefig(OUTPUT_DIR / "Insurance_Residuals.png", dpi=300)
    plt.show()
    

def main():
    insurance_data = load_data()
    analyze_data(insurance_data)
    visualize_data(insurance_data)
    correlation_analysis(insurance_data)
    visualize_correlation(insurance_data)
    x, y = prepare_data(insurance_data)

    x_train, x_test, y_train, y_test = split_data(x, y)

    categorical_columns, numerical_columns = identify_features(x_train)

    x_train_final, x_test_final = encode_features(
        x_train, x_test, categorical_columns, numerical_columns
    )
    model = train_model(x_train_final, y_train)
    y_pred = predict_charges(model, x_test_final)
    evaluate_model(y_test, y_pred)
    visualize_predictions(y_test, y_pred)
    visualize_residuals(y_test, y_pred)


if __name__ == "__main__":
    main()
    