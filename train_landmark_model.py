import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X = []
y = []

DATASET_DIR = "landmark_dataset"

for file in os.listdir(DATASET_DIR):
    if not file.endswith(".csv"):
        continue

    label = file.replace(".csv", "")
    data = pd.read_csv(os.path.join(DATASET_DIR, file), header=None)

    X.extend(data.values)
    y.extend([label] * len(data))

print("Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = SVC(
    kernel="rbf",
    probability=True,
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as gesture_model.pkl")
