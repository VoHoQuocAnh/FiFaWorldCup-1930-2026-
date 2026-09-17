import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Models.Linear_regression import run_linear_regression
from Models.Logistic_regression import run_logistic_regression
from Models.knn import run_knn
from Models.Random_forest import run_random_forest
from Models.Decision_tree import run_decision_tree
from Models.Xgboost import run_xgboost

df = pd.read_csv("data/fifa.csv")
df = df.dropna(subset=["total_goals_team1", "total_goals_team2"])
# 1. One-hot Encoding
features = ["world_cup_year", "team1", "team2", "stage", "group", "host_country", "stadium"]
x = pd.get_dummies(df[features], drop_first=True)

# 2. BÀI TOÁN HỒI QUY: LINEAR REGRESSION (Dự đoán số bàn thắng của Team 1)
print("=" * 50)
print("1. LINEAR REGRESSION")
print("=" * 50)

y_linear = df["total_goals_team1"]
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x, y_linear, test_size=0.2, random_state=42
)
run_linear_regression(x_train_reg, x_test_reg, y_train_reg, y_test_reg)

# 3. BÀI TOÁN PHÂN LOẠI (CLASSIFICATION: Thắng, Hòa, Thua)
# Chuẩn hóa nhãn chung: 2 = Thắng, 1 = Hòa, 0 = Thua 
y_clf = np.select(
    [
        df["total_goals_team1"] > df["total_goals_team2"],  # Thắng
        df["total_goals_team1"] == df["total_goals_team2"]  # Hòa
    ],
    [2, 1],
    default=0                                              # Thua
)

x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
    x, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Dictionary lưu kết quả đánh giá để so sánh
model_accuracies = {}

# Logistic Regression 
print("\n" + "=" * 50)
print("2. LOGISTIC REGRESSION")
print("=" * 50)
log_model = run_logistic_regression(x_train_c, x_test_c, y_train_c, y_test_c)
model_accuracies["Logistic Regression"] = accuracy_score(y_test_c, log_model.predict(x_test_c))

# K-Nearest Neighbors
print("\n" + "=" * 50)
print("3. KNN")
print("=" * 50)
knn_model = run_knn(x_train_c, x_test_c, y_train_c, y_test_c)
model_accuracies["KNN"] = accuracy_score(y_test_c, knn_model.predict(x_test_c))

# Random Forest
print("\n" + "=" * 50)
print("4. RANDOM FOREST")
print("=" * 50)
rf_model = run_random_forest(x_train_c, x_test_c, y_train_c, y_test_c)
if rf_model is not None:
    model_accuracies["Random Forest"] = accuracy_score(y_test_c, rf_model.predict(x_test_c))

# Decision Tree
print("\n" + "=" * 50)
print("5. DECISION TREE")
print("=" * 50)
dt_model = run_decision_tree(x_train_c, x_test_c, y_train_c, y_test_c)
if dt_model is not None:
    model_accuracies["Decision Tree"] = accuracy_score(y_test_c, dt_model.predict(x_test_c))

# XGBoost
print("\n" + "=" * 50)
print("6. XGBOOST")
print("=" * 50)
xgb_model = run_xgboost(x_train_c, x_test_c, y_train_c, y_test_c)
if xgb_model is not None:
    model_accuracies["XGBoost"] = accuracy_score(y_test_c, xgb_model.predict(x_test_c))

# 4. TỔNG KẾT VÀ SO SÁNH ACCURACY

if model_accuracies:
    print("\n" + "=" * 50)
    print("BẢNG TỔNG KẾT ĐỘ CHÍNH XÁC (ACCURACY):")
    print("=" * 50)
    for model_name, acc in model_accuracies.items():
        print(f"- {model_name:20s}: {acc * 100:.2f}%")

# Tạo thư mục saved_models nếu chưa có
os.makedirs("saved_models", exist_ok=True)
# Giả sử Random Forest là mô hình bạn muốn lưu:
joblib.dump(rf_model, "saved_models/best_model.pkl")
print("-> Đã lưu mô hình thành công vào file: saved_models/best_model.pkl")

plt.figure(figsize=(10, 5))
sns.barplot(x=list(model_accuracies.keys()), y=[acc * 100 for acc in model_accuracies.values()], palette="viridis")
plt.title("So sánh độ chính xác giữa các mô hình")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.savefig("model_comparison.png", bbox_inches='tight')
print("-> Đã xuất biểu đồ so sánh thành công: model_comparison.png")