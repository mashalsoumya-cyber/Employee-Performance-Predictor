import os
import numpy as np
import pandas as pd


def generate_employee_data(n=500, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(21, 50, n)
    gender = np.random.choice(["Male", "Female"], n)
    department = np.random.choice(
        ["HR", "Sales", "IT", "Finance", "Operations"], n
    )
    education = np.random.choice(
        ["Diploma", "Bachelor", "Master"], n, p=[0.2, 0.55, 0.25]
    )
    experience = np.random.randint(0, 20, n)
    monthly_salary = np.random.randint(18000, 120000, n)
    hours_worked_per_week = np.random.randint(30, 65, n)
    projects_handled = np.random.randint(1, 15, n)
    training_hours = np.random.randint(0, 80, n)
    attendance = np.round(np.random.uniform(70, 100, n), 2)
    job_satisfaction = np.random.randint(1, 11, n)
    work_life_balance = np.random.randint(1, 11, n)
    promotion_last_2_years = np.random.choice([0, 1], n, p=[0.75, 0.25])

    # Score-based logic for better target quality
    score = (
        experience * 1.5
        + projects_handled * 2
        + training_hours * 0.25
        + attendance * 0.3
        + job_satisfaction * 3
        + work_life_balance * 2
        + promotion_last_2_years * 8
        + (monthly_salary / 10000) * 1.2
        - (hours_worked_per_week - 45).clip(min=0) * 0.8
    )

    performance = []
    for s in score:
        if s >= 60:
            performance.append("High")
        elif s >= 42:
            performance.append("Medium")
        else:
            performance.append("Low")

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Education": education,
        "YearsExperience": experience,
        "MonthlySalary": monthly_salary,
        "HoursWorkedPerWeek": hours_worked_per_week,
        "ProjectsHandled": projects_handled,
        "TrainingHours": training_hours,
        "AttendancePercentage": attendance,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "PromotionLast2Years": promotion_last_2_years,
        "PerformanceRating": performance
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_employee_data(500)
    df.to_csv("data/employee_data.csv", index=False)
    print("Dataset created successfully: data/employee_data.csv")
    print(df.head())
    print("\nClass distribution:")
    print(df["PerformanceRating"].value_counts())