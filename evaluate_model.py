import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. Load model, vectorizer, and test data ───────────────────────────
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_test_vec, y_test = joblib.load("test_data.pkl")

# ── 2. Generate predictions ────────────────────────────────────────────
y_pred = model.predict(X_test_vec)

# ── 3. Accuracy ────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

# ── 4. Classification Report ───────────────────────────────────────────
label_names = ["Sadness", "Joy", "Anger"]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# ── 5. Confusion Matrix ────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# ── 6. Test 3 sample inputs for demo ──────────────────────────────────
emotion_map = {0: "Sadness", 1: "Joy", 3: "Anger"}
samples = [
    "I want to punch someone",
    "I want to dance",
   "I lost my wife"
]
print("\n--- Demo: 3 Sample Predictions ---")
vec_samples = vectorizer.transform(samples)
predictions = model.predict(vec_samples)
for text, label in zip(samples, predictions):
    print(f"Input: '{text}' → Predicted: {emotion_map[label]}")