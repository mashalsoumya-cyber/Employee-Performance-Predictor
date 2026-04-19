# Employee Performance Predictor

## Overview
Employee Performance Predictor is a machine learning project that predicts whether an employee's performance is **Low**, **Medium**, or **High** based on workplace and HR-related features.

This project covers:
- synthetic employee dataset generation
- exploratory data analysis (EDA)
- preprocessing and feature encoding
- machine learning model training
- evaluation and prediction
- saving trained model and outputs

---

## Project Structure

```bash
Employee-Performance-Predictor/
│
├── data/
│   └── employee_data.csv
│
├── models/
│   └── employee_performance_model.pkl
│
├── outputs/
│   ├── class_distribution.png
│   ├── feature_correlation.png
│   ├── performance_by_salary.png
│   └── model_metrics.txt
│
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md