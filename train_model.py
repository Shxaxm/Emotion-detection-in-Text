from emotion_preprocessing_script import preprocess_text_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ── 1. Load preprocessed data ──────────────────────────────────────────
dataset_path = "./upload/train-00000-of-00001.parquet"
chosen_emotions = [0, 1, 3]  # Sadness, Joy, Anger

df_train, df_test, count_vectorizer, X_train_cv, X_test_cv, y_train, y_test = \
    preprocess_text_data(dataset_path, target_emotions=chosen_emotions)

# ── 2. Vectorize with TF-IDF (fits only on training data) ──────────────
print("\nVectorizing with TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf_vectorizer.fit_transform(df_train["processed_text"])
X_test_vec  = tfidf_vectorizer.transform(df_test["processed_text"])

print(f"TF-IDF training matrix shape: {X_train_vec.shape}")
print(f"TF-IDF testing matrix shape:  {X_test_vec.shape}")

# ── 3. Train Logistic Regression model ─────────────────────────────────
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_vec, y_train)
print("Training complete!")

# ── 4. Save model and vectorizer for Student 3 ─────────────────────────
joblib.dump(model, "emotion_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump((X_test_vec, y_test), "test_data.pkl")   # for Student 3's evaluation
print("\nSaved: emotion_model.pkl, tfidf_vectorizer.pkl, test_data.pkl")
print("\nDone! Hand these 3 .pkl files to Student 3.")