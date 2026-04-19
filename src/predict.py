import pickle
import pandas as pd


def predict_employee_performance(sample_input, model_path="models/employee_performance_model.pkl"):
    with open(model_path, "rb") as f:
        saved_objects = pickle.load(f)

    model = saved_objects["model"]
    preprocessor = saved_objects["preprocessor"]
    label_encoder = saved_objects["label_encoder"]

    sample_df = pd.DataFrame([sample_input])
    sample_processed = preprocessor.transform(sample_df)
    prediction = model.predict(sample_processed)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return predicted_label


if __name__ == "__main__":
    sample_employee = {
        "Age": 30,
        "Gender": "Female",
        "Department": "IT",
        "Education": "Master",
        "YearsExperience": 7,
        "MonthlySalary": 65000,
        "HoursWorkedPerWeek": 44,
        "ProjectsHandled": 8,
        "TrainingHours": 40,
        "AttendancePercentage": 95.5,
        "JobSatisfaction": 8,
        "WorkLifeBalance": 7,
        "PromotionLast2Years": 1
    }

    result = predict_employee_performance(sample_employee)
    print("Predicted Employee Performance:", result)