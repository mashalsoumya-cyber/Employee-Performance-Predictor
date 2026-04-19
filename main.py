import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_employee_data
from preprocessing import load_data, preprocess_data
from eda import perform_eda
from train_model import train_and_evaluate_model
from predict import predict_employee_performance


def main():
    print("=" * 60)
    print("EMPLOYEE PERFORMANCE PREDICTOR PROJECT")
    print("=" * 60)

    # Create folders
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Generate dataset
    print("\nStep 1: Generating employee dataset...")
    df_generated = generate_employee_data(n=500)
    df_generated.to_csv("data/employee_data.csv", index=False)
    print("Dataset saved at data/employee_data.csv")

    # Step 2: Load dataset
    print("\nStep 2: Loading dataset...")
    df = load_data("data/employee_data.csv")
    print("Dataset loaded successfully.")
    print(df.head())

    # Step 3: EDA
    print("\nStep 3: Performing EDA...")
    perform_eda(df)

    # Step 4: Preprocessing
    print("\nStep 4: Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_data(df)
    print("Preprocessing completed successfully.")

    # Step 5: Model Training
    print("\nStep 5: Training model...")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, label_encoder)

    # Step 6: Sample Prediction
    print("\nStep 6: Running sample prediction...")
    sample_employee = {
        "Age": 29,
        "Gender": "Male",
        "Department": "Finance",
        "Education": "Bachelor",
        "YearsExperience": 6,
        "MonthlySalary": 52000,
        "HoursWorkedPerWeek": 46,
        "ProjectsHandled": 7,
        "TrainingHours": 25,
        "AttendancePercentage": 91.2,
        "JobSatisfaction": 7,
        "WorkLifeBalance": 6,
        "PromotionLast2Years": 0
    }

    predicted_performance = predict_employee_performance(sample_employee)
    print("Predicted Employee Performance:", predicted_performance)

    print("\nProject executed successfully.")
    print("Check data/, outputs/, and models/ folders.")


if __name__ == "__main__":
    main()