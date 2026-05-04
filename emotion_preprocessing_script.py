
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Note: Make sure you have run the NLTK download commands from the guide first!

def preprocess_text_data(file_path, target_emotions=[0, 1, 3], test_size=0.2, random_state=42):
    """
    Performs all data preprocessing steps on the emotion detection dataset,
    including filtering for target emotions and splitting into train/test sets.

    Args:
        file_path (str): The path to the parquet dataset file.
        target_emotions (list): A list of integer labels for the emotions to include.
                                (0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise)
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with original and processed text for the training set.
            - pd.DataFrame: DataFrame with original and processed text for the testing set.
            - sklearn.feature_extraction.text.CountVectorizer: Fitted CountVectorizer.
            - scipy.sparse.csr_matrix: Vectorized text data for the training set.
            - scipy.sparse.csr_matrix: Vectorized text data for the testing set.
    """

    # 1. Load the dataset
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Please check if the file path is correct and pyarrow is installed.")
        return None, None, None, None, None

    print(f"Dataset loaded. Original shape: {df.shape}")

    # Filter for target emotions
    print(f"\nFiltering for target emotions: {target_emotions}...")
    df_filtered = df[df["label"].isin(target_emotions)].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print("Filtered Label Distribution:")
    print(df_filtered["label"].value_counts())

    # 2. Text Cleaning
    print("\nStarting text cleaning (Lowercasing, Punctuation, URLs, Mentions, Hashtags)...")
    def clean_text(text):
        text = text.lower()  # Lowercasing
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r"@\w+", "", text) # Remove mentions
        text = re.sub(r"#\w+", "", text) # Remove hashtags
        text = re.sub(r"[^a-z\s]", "", text) # Remove punctuation and special characters
        return text

    df_filtered["cleaned_text"] = df_filtered["text"].apply(clean_text)

    # 3. Tokenization
    print("Starting tokenization...")
    df_filtered["tokens"] = df_filtered["cleaned_text"].apply(word_tokenize)

    # 4. Stop Word Removal
    print("Starting stop word removal...")
    stop_words = set(stopwords.words("english"))
    df_filtered["filtered_tokens"] = df_filtered["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # 5. Lemmatization (preferred over stemming for better word forms)
    print("Starting lemmatization...")
    lemmatizer = WordNetLemmatizer()
    df_filtered["lemmatized_tokens"] = df_filtered["filtered_tokens"].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

    # Join tokens back into a single string for vectorization
    df_filtered["processed_text"] = df_filtered["lemmatized_tokens"].apply(lambda tokens: " ".join(tokens))

    print("\nSample of processed text before splitting:")
    print(df_filtered[["text", "processed_text", "label"]].head())

    # 6. Data Splitting (80/20 train-test split)
    print(f"\nSplitting data into training and testing sets (train_size={1-test_size}, test_size={test_size})...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_filtered["processed_text"],
        df_filtered["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df_filtered["label"] # Ensure balanced classes in splits
    )

    # Create DataFrames for train and test sets to keep all processed columns
    df_train = df_filtered.loc[X_train_text.index]
    df_test = df_filtered.loc[X_test_text.index]

    print(f"Training set shape: {df_train.shape}")
    print(f"Testing set shape: {df_test.shape}")
    print("Training set label distribution:")
    print(y_train.value_counts(normalize=True))
    print("Testing set label distribution:")
    print(y_test.value_counts(normalize=True))

    # 7. Vectorization (Fit on training data, transform both train and test)
    print("\nStarting vectorization with CountVectorizer (fitting on training data)...")
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train_text)
    X_test_vectorized = vectorizer.transform(X_test_text)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Vectorized training data shape: {X_train_vectorized.shape}")
    print(f"Vectorized testing data shape: {X_test_vectorized.shape}")

    return df_train, df_test, vectorizer, X_train_vectorized, X_test_vectorized, y_train, y_test

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = "./upload/train-00000-of-00001.parquet"

    # Define the target emotions (Joy:1, Sadness:0, Anger:3)
    chosen_emotions = [0, 1, 3]

    # Run the preprocessing pipeline
    df_train, df_test, count_vectorizer, X_train, X_test, y_train, y_test = preprocess_text_data(
        dataset_path, 
        target_emotions=chosen_emotions
    )

    if df_train is not None:
        print("\nPreprocessing complete. You now have:")
        print("- Training features (X_train) and labels (y_train) for model training.")
        print("- Testing features (X_test) and labels (y_test) for model evaluation.")
        print("- The fitted CountVectorizer (count_vectorizer) to transform new text.")

        print("\nSample of training data (processed text and label):")
        print(df_train[["processed_text", "label"]].head())
        print("\nSample of testing data (processed text and label):")
        print(df_test[["processed_text", "label"]].head())

        # You can save these for later use if needed
        # df_train.to_parquet("./processed_train_data.parquet")
        # df_test.to_parquet("./processed_test_data.parquet")
        # import joblib
        # joblib.dump(count_vectorizer, "./count_vectorizer.joblib")

