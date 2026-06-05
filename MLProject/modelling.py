import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from joblib import dump

# Load dataset
df = pd.read_csv("obesity_preprocessed.csv")

# Feature dan Target
X = df.drop("Obesity", axis=1)
y = df["Obesity"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

# Simpan Model
dump(
    model,
    "obesity_model.pkl"
)

print("Model berhasil disimpan!")