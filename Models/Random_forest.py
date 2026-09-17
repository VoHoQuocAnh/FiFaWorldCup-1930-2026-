from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    pre_rf = model.predict(x_test)    
    print("Random Forest Classifier Model:")
    print("Accuracy:", round(accuracy_score(y_test, pre_rf), 2))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pre_rf))
    print("Classification Report:\n", classification_report(y_test, pre_rf))
    
    return model