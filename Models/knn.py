from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def run_knn(x_train, x_test, y_train, y_test):
    # Đóng gói Scaler + KNN vào 1 Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    
    pipeline.fit(x_train, y_train)
    pre_knn = pipeline.predict(x_test)
    acc = accuracy_score(y_test, pre_knn)
    
    print("KNN Model:")
    print("Accuracy:", round(acc, 2))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pre_knn))
    print("Classification Report:\n", classification_report(y_test, pre_knn))

    return pipeline