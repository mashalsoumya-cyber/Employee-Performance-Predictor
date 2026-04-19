import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, label_encoder):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Evaluation")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    # Save metrics
    with open("outputs/model_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Employee Performance Predictor - Model Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    # Save full pipeline parts
    with open("models/employee_performance_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder
        }, f)

    print("\nModel saved to models/employee_performance_model.pkl")