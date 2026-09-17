from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

def run_decision_tree(x_train_K, x_test_K, y_train_K, y_test_K):
    model = DecisionTreeClassifier(max_depth = 8,random_state = 42)
    model.fit(x_train_K, y_train_K)
    pre_Tree = model.predict(x_test_K)
    print("Decision Tree Classifier Model:")
    print("Accuracy:",round(accuracy_score(y_test_K,pre_Tree), 2))
    print("Confusion Matrix:\n",confusion_matrix(y_test_K,pre_Tree))
    print("Classification Report:\n",classification_report(y_test_K,pre_Tree))
    return model 