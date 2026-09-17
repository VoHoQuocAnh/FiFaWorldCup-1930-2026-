import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_xgboost(x_train, x_test, y_train, y_test):
    model_xgb = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        eval_metric="mlogloss"
    )
    model_xgb.fit(x_train, y_train)
    pre_xgb = model_xgb.predict(x_test)
    print("XGBoost Classifier Model:")
    print("Accuracy:", round(accuracy_score(y_test, pre_xgb), 2))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pre_xgb))
    print("Classification Report:\n", classification_report(y_test, pre_xgb))
    return model_xgb