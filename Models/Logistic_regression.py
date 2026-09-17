import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_logistic_regression(x_train_log, x_test_log, y_train_log, y_test_log):
    # Đối với multiclass (3 lớp: 0, 1, 2), solver 'lbfgs' hoặc 'liblinear' (OvR) đều được
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_log, y_train_log)
    
    y_predict_log = model.predict(x_test_log)
    accuracy = accuracy_score(y_test_log, y_predict_log)
    
    print("Logistic Regression Model:")
    print("Tỉ lệ phân phối các lớp trong tập Train:")
    print(pd.Series(y_train_log).value_counts(normalize=True))
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_log, y_predict_log))
    print("Classification Report:\n")
    print(classification_report(y_test_log, y_predict_log))
    
    return model