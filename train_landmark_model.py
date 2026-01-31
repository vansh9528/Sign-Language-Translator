import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ================== DATA LOAD ==================
DATASET_DIR = "landmark_dataset"

X = []
y = []

for file in os.listdir(DATASET_DIR):
    if not file.endswith(".csv"):
        continue

    label = file.replace(".csv", "")
    file_path = os.path.join(DATASET_DIR, file)

    data = pd.read_csv(file_path, header=None)

    X.extend(data.values)
    y.extend([label] * len(data))

print("Total samples:", len(X))
print("Gestures:", sorted(set(y)))

# ================== TRAIN / TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================== MODEL PIPELINE ==================
# 🔑 StandardScaler is CRITICAL for landmark-based SVM
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    )
)

# ================== TRAIN ==================
model.fit(X_train, y_train)

# ================== EVALUATION ==================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(acc * 100, 2), "%")

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ================== SAVE MODEL ==================
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as gesture_model.pkl")
